"""
CLEAR_CSAU.py — CSAU adapted for CLEAR dataset

CLEAR 数据集模态结构：
  - CAPTION_MODE : image + caption → 多模态分支
  - TEXT_QA_MODE : text QA        → 单模态分支

Forget/Retain 数据集目录：
  {data_dir}/forget{split}+tofu
  {data_dir}/retain{100-split}+tofu

整体流程：
  Phase 1: 加载 pre-unlearning LoRA + 双路显著性
  Phase 2: 按显著性三分类参数（shared / text_only / visual_only）
  Phase 3: 差异化遗忘训练
"""

import os
import sys
import json
import math
import argparse
from itertools import cycle
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoProcessor, get_scheduler, LlavaForConditionalGeneration
from peft import PeftModel

from data_process.CLEAR_process import (
    CAPTION_MODE,
    TEXT_QA_MODE,
    CLEARDataset,
)


# =============================================================================
# LoRA 参数过滤：仅保留 LLM LoRA，排除视觉编码器和 projector
# =============================================================================

_LLM_KW = ["language_model"]
_EXCLUDE_KW = ["vision_tower", "multi_modal_projector"]


def is_llm_lora_param(name: str) -> bool:
    return (
        any(k in name for k in _LLM_KW)
        and not any(k in name for k in _EXCLUDE_KW)
        and ("lora_A" in name or "lora_B" in name)
    )


# =============================================================================
# 从 adapter_config.json 读取 LoRA 配置
# =============================================================================

def get_lora_rank_alpha(lora_dir: str):
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("r", 16), cfg.get("lora_alpha", 16)
    return 16, 16


# =============================================================================
# Cache 工具
# =============================================================================

def build_sal_cache_path(cache_dir: str, lora_r: int, lora_alpha: int) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"saliency_rank{lora_r}_alpha{lora_alpha}.pt")


def _save(path: str, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, path)


def _load(path: str):
    return torch.load(path, map_location="cpu")


# =============================================================================
# CLEAR Collate
# 输出 4-tuple: (input_ids, attention_mask, pixel_values, labels)
# =============================================================================

def _build_clear_batch(examples, processor, mode: str):
    """
    mode: "multimodal" | "unimodal"
    只监督 answer 区域。
    """
    texts = []
    images = []
    prompt_lengths = []

    for ex in examples:
        image = ex.get("image")
        question = ex.get("question", "")
        answer = ex.get("answer", "")

        if mode == "unimodal":
            image = None

        if image is not None:
            user_content = [{"type": "image"}, {"type": "text", "text": question}]
        else:
            user_content = [{"type": "text", "text": question}]

        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        prompt_messages = [
            {"role": "user", "content": user_content},
        ]

        full_text = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
        ).strip()
        texts.append(full_text)

        if image is not None:
            images.append(image)

        prompt_text = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
        ).strip()
        prompt_inputs = processor(
            text=[prompt_text],
            images=[image] if image is not None else None,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

    batch = processor(
        text=texts,
        images=images if len(images) > 0 else None,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    pixel_values = batch.get("pixel_values", None)
    return batch["input_ids"], batch["attention_mask"], pixel_values, labels


def make_collate_fn(processor, mode: str):
    def collate(examples):
        return _build_clear_batch(examples, processor, mode)
    return collate


# =============================================================================
# Phase 1：双路显著性计算
# 固定实现：
#   - 标签替换为 "I don't know."
#   - saliency = grad^2
# =============================================================================

def _get_input_device(model):
    m = model.base_model if hasattr(model, "base_model") else model
    emb = (
        m.get_input_embeddings()
        if hasattr(m, "get_input_embeddings")
        else m.language_model.get_input_embeddings()
    )
    return emb.weight.device if emb is not None else next(model.parameters()).device


def _replace_labels_idk(labels: torch.Tensor, idk_ids: list) -> torch.Tensor:
    new_labels = labels.clone()
    for i in range(labels.shape[0]):
        valid = (labels[i] != -100).nonzero(as_tuple=True)[0]
        for k, pos in enumerate(valid):
            new_labels[i, pos] = idk_ids[k % len(idk_ids)]
    return new_labels


def _forward(batch, model, mode: str):
    input_ids, attention_mask, pixel_values, labels = batch
    if mode == "multimodal" and pixel_values is not None:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def _run_saliency_path(model, loader, mode, idk_ids, target_params, path_name, accelerator):
    """
    单路显著性累积：
      - 标签替换为 IDK
      - saliency = grad^2
    """
    sal = {
        n: torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for n, p in target_params.items()
    }
    device = _get_input_device(model)
    n_valid = 0

    for batch in tqdm(loader, desc=f"[Saliency] {path_name}", disable=not accelerator.is_main_process):
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)

        batch_list = list(batch)
        batch_list[3] = _replace_labels_idk(batch_list[3], idk_ids)
        batch = tuple(batch_list)

        model.zero_grad()
        loss = _forward(batch, model, mode).loss
        if loss is None or loss.item() == 0.0:
            continue

        loss.backward()
        n_valid += 1

        for n, p in target_params.items():
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu()
            sal[n] += g ** 2

    if n_valid == 0:
        raise RuntimeError(f"[Saliency] {path_name} 梯度全为 0，请检查数据和模型。")

    for n in sal:
        sal[n] /= n_valid

    accelerator.print(f"[Saliency] {path_name}: {n_valid} 有效批次")
    return sal


def compute_saliency(model, loader_text, loader_caption, accelerator, processor):
    """
    Text 路    -> TEXT_QA unimodal
    Caption 路 -> CAPTION multimodal
    """
    model.train()
    target_params = {
        n: p for n, p in model.named_parameters()
        if p.requires_grad and is_llm_lora_param(n)
    }
    if not target_params:
        raise RuntimeError("[Saliency] 未找到 LLM LoRA 参数，请检查过滤逻辑。")

    accelerator.print(f"[Saliency] fixed method: IDK label replacement + grad^2")
    accelerator.print(f"[Saliency] trainable_lora_params={len(target_params)}")

    idk_ids = processor.tokenizer.encode("I don't know.", add_special_tokens=False)

    sal_text = _run_saliency_path(
        model=model,
        loader=loader_text,
        mode="unimodal",
        idk_ids=idk_ids,
        target_params=target_params,
        path_name="Text(TEXT_QA)",
        accelerator=accelerator,
    )

    sal_caption = _run_saliency_path(
        model=model,
        loader=loader_caption,
        mode="multimodal",
        idk_ids=idk_ids,
        target_params=target_params,
        path_name="Caption(CAPTION)",
        accelerator=accelerator,
    )

    model.zero_grad()
    return sal_text, sal_caption


# =============================================================================
# Phase 2：参数三分类
# =============================================================================

def partition_params(sal_text, sal_caption, top_k_ratio, modality_margin, accelerator):
    scalar_text = {n: v.mean().item() for n, v in sal_text.items()}
    scalar_caption = {n: v.mean().item() for n, v in sal_caption.items()}

    max_text = max(scalar_text.values()) + 1e-12
    max_caption = max(scalar_caption.values()) + 1e-12

    scalar_text_norm = {n: v / max_text for n, v in scalar_text.items()}
    scalar_caption_norm = {n: v / max_caption for n, v in scalar_caption.items()}

    avg_text_raw = sum(scalar_text.values()) / (len(scalar_text) + 1e-12)
    avg_caption_raw = sum(scalar_caption.values()) / (len(scalar_caption) + 1e-12)
    accelerator.print(
        f"[Partition] raw mean text={avg_text_raw:.6f}, caption={avg_caption_raw:.6f}, "
        f"ratio(text/caption)={avg_text_raw / (avg_caption_raw + 1e-12):.2f}x"
    )

    combined = {
        n: scalar_text_norm[n] + scalar_caption_norm[n]
        for n in scalar_text_norm
    }
    k = max(1, int(len(combined) * top_k_ratio))
    top_names = sorted(combined, key=combined.__getitem__, reverse=True)[:k]

    shared, text_only, visual_only = set(), set(), set()

    for name in top_names:
        r = scalar_text_norm[name] / (scalar_text_norm[name] + scalar_caption_norm[name] + 1e-12)
        if r > 0.5 + modality_margin:
            text_only.add(name)
        elif r < 0.5 - modality_margin:
            visual_only.add(name)
        else:
            shared.add(name)

    accelerator.print(
        f"[Partition] shared={len(shared)}, text_only={len(text_only)}, "
        f"visual_only={len(visual_only)}, frozen={len(combined) - k}"
    )
    return shared, text_only, visual_only


# =============================================================================
# Phase 3 辅助：梯度处理
# 固定使用 gradnorm balance
# =============================================================================

def _grad_norm(grads: dict, param_set: set) -> float:
    return sum(
        grads[n].float().norm().item() ** 2
        for n in param_set if n in grads
    ) ** 0.5


def _balance_grads(grads_caption, grads_text, active, args):
    n_caption = _grad_norm(grads_caption, active)
    n_text = _grad_norm(grads_text, active)
    scale_caption = args.forget_target_ratio * (n_text + 1e-8) / (n_caption + 1e-8)
    return {n: g * scale_caption for n, g in grads_caption.items()}, grads_text


def _apply_modality_scaling(forget_grads, shared, args):
    for n in forget_grads:
        coeff = args.alpha_shared if n in shared else args.beta_specific
        forget_grads[n] = forget_grads[n] * coeff
    return forget_grads


def _apply_symmetry(forget_grads, text_only, visual_only, gamma, min_group=5):
    t_norm = _grad_norm(forget_grads, text_only)
    v_norm = _grad_norm(forget_grads, visual_only)

    if (
        t_norm < 1e-8
        or v_norm < 1e-8
        or len(text_only) < min_group
        or len(visual_only) < min_group
    ):
        return forget_grads

    target = (t_norm + v_norm) / 2.0
    t_scale = max(0.1, 1.0 + gamma * (target / t_norm - 1.0))
    v_scale = max(0.1, 1.0 + gamma * (target / v_norm - 1.0))

    for n in text_only:
        if n in forget_grads:
            forget_grads[n] = forget_grads[n] * t_scale
    for n in visual_only:
        if n in forget_grads:
            forget_grads[n] = forget_grads[n] * v_scale
    return forget_grads


# =============================================================================
# Phase 3：forget / retain 单步
# retain 权重固定为 1.0
# =============================================================================

def csau_forget_step(model, caption_batch, text_batch, shared, text_only, visual_only, args, accelerator, grad_accum_steps):
    raw_model = accelerator.unwrap_model(model)
    active = shared | text_only | visual_only

    raw_model.zero_grad()
    loss_caption = _forward(caption_batch, model, "multimodal").loss
    accelerator.backward(-loss_caption)
    grads_caption = {
        n: p.grad.detach().clone()
        for n, p in raw_model.named_parameters()
        if p.grad is not None
    }
    torch.cuda.empty_cache()

    raw_model.zero_grad()
    loss_text = _forward(text_batch, model, "unimodal").loss
    accelerator.backward(-args.alpha_forget * loss_text)
    grads_text = {
        n: p.grad.detach().clone()
        for n, p in raw_model.named_parameters()
        if p.grad is not None
    }
    torch.cuda.empty_cache()

    grads_caption, grads_text = _balance_grads(grads_caption, grads_text, active, args)

    forget_grads = {}
    for n in active:
        g_caption = grads_caption.get(n)
        g_text = grads_text.get(n)

        if g_caption is None and g_text is None:
            continue
        if g_text is None:
            forget_grads[n] = g_caption
        elif g_caption is None:
            forget_grads[n] = g_text
        else:
            forget_grads[n] = g_caption + g_text

    forget_grads = _apply_modality_scaling(forget_grads, shared, args)
    forget_grads = _apply_symmetry(forget_grads, text_only, visual_only, args.gamma_sym)

    scale = 1.0 / grad_accum_steps
    step_grads = {
        n: (forget_grads.get(n, torch.zeros_like(p.data)) * scale).cpu()
        for n, p in raw_model.named_parameters()
        if p.requires_grad and n in active
    }

    raw_model.zero_grad()
    return (loss_caption.item() + args.alpha_forget * loss_text.item()), step_grads


def csau_retain_step(model, retain_caption, retain_text, active, args, accelerator, grad_accum_steps):
    raw_model = accelerator.unwrap_model(model)
    raw_model.zero_grad()

    loss_caption = _forward(retain_caption, model, "multimodal").loss
    accelerator.backward(loss_caption)

    loss_text = _forward(retain_text, model, "unimodal").loss
    accelerator.backward(loss_text)

    scale = 1.0 / grad_accum_steps
    step_grads = {
        n: (
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p.data))
            * scale
        ).cpu()
        for n, p in raw_model.named_parameters()
        if p.requires_grad and n in active
    }

    raw_model.zero_grad()
    torch.cuda.empty_cache()

    return loss_caption.item() + loss_text.item(), step_grads


def apply_gradients(model, accum_grads, accelerator):
    raw_model = accelerator.unwrap_model(model)

    if accelerator.num_processes > 1 and dist.is_initialized():
        for n in accum_grads:
            g = accum_grads[n].cuda()
            dist.all_reduce(g, op=dist.ReduceOp.AVG)
            accum_grads[n] = g.cpu()

    for n, p in raw_model.named_parameters():
        p.grad = accum_grads[n].to(p.device) if n in accum_grads else None


# =============================================================================
# Main
# =============================================================================

def main(args):
    accelerator = Accelerator()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    accelerator.print(f"\n[{datetime.now()}] CLEAR_CSAU")
    accelerator.print("Command: " + " ".join(sys.argv))
    for k, v in sorted(vars(args).items()):
        accelerator.print(f"  {k}: {v}")

    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    lora_r, lora_alpha = get_lora_rank_alpha(args.lora_dir)
    accelerator.print(f"[LoRA] r={lora_r}, alpha={lora_alpha}")

    sal_path = build_sal_cache_path(
        cache_dir=args.cache_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    retain_ratio = 100 - args.forget_split
    forget_path = os.path.join(args.data_dir, f"forget{args.forget_split}+tofu")
    retain_path = os.path.join(args.data_dir, f"retain{retain_ratio}+tofu")

    accelerator.print(f"[Data] forget path: {forget_path}")
    accelerator.print(f"[Data] retain path: {retain_path}")

    forget_raw = load_dataset(forget_path, split="train")
    retain_raw = load_dataset(retain_path, split="train")
    accelerator.print(f"[Data] forget raw size={len(forget_raw)} | retain raw size={len(retain_raw)}")

    forget_caption_ds = CLEARDataset(forget_raw, mode=CAPTION_MODE)
    forget_text_ds = CLEARDataset(forget_raw, mode=TEXT_QA_MODE)
    retain_caption_ds = CLEARDataset(retain_raw, mode=CAPTION_MODE)
    retain_text_ds = CLEARDataset(retain_raw, mode=TEXT_QA_MODE)

    accelerator.print(
        f"[Data] forget  CAPTION={len(forget_caption_ds)}, TEXT_QA={len(forget_text_ds)}\n"
        f"[Data] retain  CAPTION={len(retain_caption_ds)}, TEXT_QA={len(retain_text_ds)}"
    )

    if len(forget_caption_ds) == 0 or len(forget_text_ds) == 0:
        raise ValueError("[Data] forget dataset contains empty branch.")
    if len(retain_caption_ds) == 0 or len(retain_text_ds) == 0:
        raise ValueError("[Data] retain dataset contains empty branch.")

    collate_caption = make_collate_fn(processor, mode="multimodal")
    collate_text = make_collate_fn(processor, mode="unimodal")

    # =========================================================================
    # Phase 1: Load pre-unlearning LoRA + saliency
    # =========================================================================
    accelerator.print("\n===== Phase 1: Load LoRA + Saliency =====")

    if os.path.exists(sal_path):
        accelerator.print(f"[Phase1] Loading cached saliency from: {sal_path}")
        cache = _load(sal_path)
        sal_text = cache["saliency_text"]
        sal_caption = cache["saliency_caption"]
    else:
        accelerator.print(f"[Phase1] Loading LoRA for saliency computation: {args.lora_dir}")

        model_sal = LlavaForConditionalGeneration.from_pretrained(
            args.base_model_dir,
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

        vocab_size = len(processor.tokenizer)
        if vocab_size > model_sal.get_input_embeddings().weight.shape[0]:
            model_sal.resize_token_embeddings(vocab_size)

        model_sal = PeftModel.from_pretrained(model_sal, args.lora_dir)

        for n, p in model_sal.named_parameters():
            p.requires_grad_(is_llm_lora_param(n))

        model_sal.enable_input_require_grads()
        accelerator.print("[Phase1] Start dual-path saliency computation")

        sal_text, sal_caption = compute_saliency(
            model=model_sal,
            loader_text=DataLoader(forget_text_ds, batch_size=1, shuffle=False, collate_fn=collate_text),
            loader_caption=DataLoader(forget_caption_ds, batch_size=1, shuffle=False, collate_fn=collate_caption),
            accelerator=accelerator,
            processor=processor,
        )

        _save(
            sal_path,
            {
                "saliency_text": sal_text,
                "saliency_caption": sal_caption,
            },
        )
        accelerator.print(f"[Phase1] Saved saliency cache to: {sal_path}")

        del model_sal
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: Partition
    # =========================================================================
    accelerator.print("\n===== Phase 2: Partition =====")
    shared, text_only, visual_only = partition_params(
        sal_text=sal_text,
        sal_caption=sal_caption,
        top_k_ratio=args.top_k_ratio,
        modality_margin=args.modality_margin,
        accelerator=accelerator,
    )

    del sal_text, sal_caption
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 3: Train
    # =========================================================================
    accelerator.print("\n===== Phase 3: Training =====")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    vocab_size = len(processor.tokenizer)
    if vocab_size > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(vocab_size)

    model = PeftModel.from_pretrained(model, args.lora_dir)

    for n, p in model.named_parameters():
        p.requires_grad_(is_llm_lora_param(n))

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    train_lora_names = {
        n for n, p in model.named_parameters()
        if p.requires_grad and is_llm_lora_param(n)
    }

    for label, mask in [
        ("shared", shared),
        ("text_only", text_only),
        ("visual_only", visual_only),
    ]:
        unknown = mask - train_lora_names
        assert not unknown, f"[Assert] {label} contains unknown parameter names: {list(unknown)[:3]}"

    accelerator.print("[Assert] mask parameter names verified")

    loader_forget_caption = DataLoader(
        forget_caption_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_caption,
    )
    loader_forget_text = DataLoader(
        forget_text_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_text,
    )
    loader_retain_caption = DataLoader(
        retain_caption_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_caption,
    )
    loader_retain_text = DataLoader(
        retain_text_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_text,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    n_steps = math.ceil(len(loader_forget_caption) / args.grad_accum_steps) * args.num_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_steps,
    )

    (
        model,
        optimizer,
        loader_forget_caption,
        loader_forget_text,
        loader_retain_caption,
        loader_retain_text,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        loader_forget_caption,
        loader_forget_text,
        loader_retain_caption,
        loader_retain_text,
        lr_scheduler,
    )

    forget_text_iter = cycle(loader_forget_text)
    retain_caption_iter = cycle(loader_retain_caption)
    retain_text_iter = cycle(loader_retain_text)
    active_params = shared | text_only | visual_only

    accelerator.print(
        "\n正确的遗忘应使 forget loss 持续上升，retain loss 尽量稳定。\n"
        "若 forget loss 下降，建议回看 alpha_shared / beta_specific / top_k_ratio。\n"
    )

    prev_avg_f = None

    for epoch in range(args.num_epochs):
        model.train()
        total_f = 0.0
        total_r = 0.0
        n_opt_steps = 0
        accum_grads = {}
        window = 0
        n_mini = len(loader_forget_caption)

        bar = tqdm(
            loader_forget_caption,
            total=n_mini,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, caption_batch in enumerate(bar):
            text_batch = next(forget_text_iter)

            f_loss, grads_f = csau_forget_step(
                model=model,
                caption_batch=caption_batch,
                text_batch=text_batch,
                shared=shared,
                text_only=text_only,
                visual_only=visual_only,
                args=args,
                accelerator=accelerator,
                grad_accum_steps=args.grad_accum_steps,
            )

            grads_r_acc = {}
            r_loss_sum = 0.0
            for _ in range(args.retain_steps_per_forget):
                r_loss_i, grads_ri = csau_retain_step(
                    model=model,
                    retain_caption=next(retain_caption_iter),
                    retain_text=next(retain_text_iter),
                    active=active_params,
                    args=args,
                    accelerator=accelerator,
                    grad_accum_steps=args.grad_accum_steps,
                )
                r_loss_sum += r_loss_i
                for n, g in grads_ri.items():
                    grads_r_acc[n] = grads_r_acc.get(n, 0) + g

            r_loss = r_loss_sum / args.retain_steps_per_forget
            for n in grads_r_acc:
                grads_r_acc[n] /= args.retain_steps_per_forget

            for n in set(grads_f) | set(grads_r_acc):
                gf = grads_f.get(n, torch.zeros(1))
                gr = grads_r_acc.get(n, torch.zeros(1))
                accum_grads[n] = accum_grads.get(n, 0) + gf + gr

            window += 1
            total_f += f_loss
            total_r += r_loss

            is_last = (step + 1 == n_mini)
            is_accum = (window == args.grad_accum_steps or is_last)

            if is_accum:
                if window < args.grad_accum_steps:
                    correction = args.grad_accum_steps / window
                    for n in accum_grads:
                        accum_grads[n] = accum_grads[n] * correction

                apply_gradients(model, accum_grads, accelerator)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad],
                    max_norm=args.clip_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                n_opt_steps += 1
                accum_grads = {}
                window = 0

            bar.set_postfix(forget=f"{f_loss:.4f}", retain=f"{r_loss:.4f}")

        avg_f = total_f / n_mini
        avg_r = total_r / n_mini

        accelerator.print(
            f"Epoch {epoch + 1} | forget={avg_f:.4f} | retain={avg_r:.4f} | opt_steps={n_opt_steps}"
        )

        if prev_avg_f is not None:
            mark = "✓ 上升" if avg_f >= prev_avg_f else "⚠️ 下降"
            accelerator.print(f"  forget: {prev_avg_f:.4f} -> {avg_f:.4f}  {mark}")
        prev_avg_f = avg_f

        accelerator.wait_for_everyone()
        save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(save_path)
        accelerator.print(f"[Save] -> {save_path}")

        if avg_r > args.retain_loss_threshold:
            accelerator.print(
                f"[EarlyStop] retain={avg_r:.4f} > {args.retain_loss_threshold}, "
                f"suggest rolling back to previous epoch."
            )
            break

    accelerator.print("Training complete.")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CLEAR_CSAU")

    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Base LLaVA model directory.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Pre-unlearning LoRA adapter directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save unlearned checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="CLEAR data root containing forget{n}+tofu and retain{100-n}+tofu.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory for saliency cache files.",
    )

    parser.add_argument(
        "--forget_split",
        type=int,
        default=5,
        help="Forget split integer used to build CLEAR split paths.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)

    parser.add_argument("--alpha_shared", type=float, default=2.0)
    parser.add_argument("--beta_specific", type=float, default=1.0)
    parser.add_argument(
        "--alpha_forget",
        type=float,
        default=1.0,
        help="TEXT_QA forget loss weight.",
    )
    parser.add_argument("--gamma_sym", type=float, default=0.5)
    parser.add_argument("--top_k_ratio", type=float, default=0.3)
    parser.add_argument("--modality_margin", type=float, default=0.15)

    parser.add_argument("--retain_steps_per_forget", type=int, default=4)
    parser.add_argument("--retain_loss_threshold", type=float, default=30.0)

    parser.add_argument("--forget_target_ratio", type=float, default=1.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    main(parser.parse_args())

