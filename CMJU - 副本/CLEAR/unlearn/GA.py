import os
import sys
import math
import argparse
from itertools import cycle
from datetime import datetime

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoProcessor, get_scheduler, LlavaForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset

from data_process.CLEAR_process import (
    CAPTION_MODE,
    TEXT_QA_MODE,
    CLEARDataset,
)


_LLM_KW = ["language_model"]
_EXCLUDE_KW = ["vision_tower", "multi_modal_projector"]


def is_llm_lora_param(name: str) -> bool:
    return (
        any(k in name for k in _LLM_KW)
        and not any(k in name for k in _EXCLUDE_KW)
        and ("lora_A" in name or "lora_B" in name)
    )


def _build_clear_batch(examples, processor, mode: str):
    """
    mode: "multimodal" | "unimodal"
    ans_only 始终开启，只监督 answer 区域。
    输出 4-tuple: (input_ids, attn_mask, pixel_values, labels)
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
            full_messages, add_generation_prompt=False
        ).strip()
        texts.append(full_text)

        if image is not None:
            images.append(image)

        prompt_text = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True
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


def _forward(batch, model, mode: str):
    input_ids, attn_mask, pixel_values, labels = batch
    if mode == "multimodal" and pixel_values is not None:
        return model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
    return model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
    )


# =============================================================================
# 主函数
# =============================================================================

def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    accelerator.print(f"\n[{datetime.now()}] CLEAR_GA")
    accelerator.print("Command: " + " ".join(sys.argv))
    for k, v in sorted(vars(args).items()):
        accelerator.print(f"  {k}: {v}")

    # ── processor ──
    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ── 加载 base 模型 + 预训练 LoRA ──
    accelerator.print(f"[Model] 加载 base 模型: {args.base_model_dir}")
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

    accelerator.print(f"[Model] 加载预训练 LoRA: {args.lora_dir}")
    model = PeftModel.from_pretrained(model, args.lora_dir)

    for n, p in model.named_parameters():
        p.requires_grad_(is_llm_lora_param(n))

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ── 拼接 forget 数据集路径 ──
    forget_path = os.path.join(args.data_dir, f"forget{args.forget_split_ratio}+tofu")
    accelerator.print(f"[Data] forget 路径: {forget_path}")

    forget_raw = load_dataset(forget_path, split="train")
    forget_caption_ds = CLEARDataset(forget_raw, mode=CAPTION_MODE)
    forget_text_ds = CLEARDataset(forget_raw, mode=TEXT_QA_MODE)

    accelerator.print(
        f"[Data] forget  CAPTION={len(forget_caption_ds)}, TEXT_QA={len(forget_text_ds)}"
    )

    if len(forget_caption_ds) == 0:
        raise ValueError("[Data] forget CAPTION 为空！请检查数据集路径。")
    if len(forget_text_ds) == 0:
        raise ValueError("[Data] forget TEXT_QA 为空！请检查数据集路径。")

    col_caption = make_collate_fn(processor, mode="multimodal")
    col_text = make_collate_fn(processor, mode="unimodal")

    loader_caption = DataLoader(
        forget_caption_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=col_caption,
    )
    loader_text = DataLoader(
        forget_text_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=col_text,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    n_steps = math.ceil(len(loader_caption) / args.grad_accum_steps) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_steps,
    )

    (
        model,
        optimizer,
        loader_caption,
        loader_text,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        loader_caption,
        loader_text,
        lr_scheduler,
    )

    # CAPTION 为主驱动，TEXT_QA 用 cycle 防止数量不等时中断
    text_iter = cycle(loader_text)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        n_opt_steps = 0
        n_mini = len(loader_caption)

        bar = tqdm(
            loader_caption,
            total=n_mini,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for mini_step, caption_b in enumerate(bar):
            text_b = next(text_iter)
            is_last = (mini_step + 1 == n_mini)
            is_accum = ((mini_step + 1) % args.grad_accum_steps == 0) or is_last

            # ── 多模态 forget（梯度上升）──
            outputs_m = _forward(caption_b, model, "multimodal")
            loss_m = outputs_m.loss
            accelerator.backward(-loss_m / args.grad_accum_steps)

            # ── 单模态 forget（梯度上升）──
            outputs_u = _forward(text_b, model, "unimodal")
            loss_u = outputs_u.loss
            accelerator.backward(-loss_u / args.grad_accum_steps)

            step_loss = loss_m.item() + loss_u.item()
            total_loss += step_loss

            if is_accum:
                # 末尾不足整步时补偿梯度幅度
                actual = (mini_step + 1) % args.grad_accum_steps
                if actual != 0:
                    correction = args.grad_accum_steps / actual
                    for p in accelerator.unwrap_model(model).parameters():
                        if p.grad is not None:
                            p.grad.mul_(correction)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad],
                    max_norm=args.clip_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                n_opt_steps += 1

            bar.set_postfix(
                loss_m=f"{loss_m.item():.4f}",
                loss_u=f"{loss_u.item():.4f}",
                avg=f"{total_loss / (mini_step + 1):.4f}",
                opt=n_opt_steps,
            )

        avg_loss = total_loss / n_mini
        accelerator.print(
            f"Epoch {epoch + 1} | avg_loss={avg_loss:.4f} | opt_steps={n_opt_steps}"
        )

        accelerator.wait_for_everyone()
        save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(save_path)
        accelerator.print(f"[Save] → {save_path}")

    accelerator.print("Training complete.")


# =============================================================================
# 命令行参数
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser("CLEAR_GA")

    # ── 路径 ──
    p.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="LLaVA base 模型路径",
    )
    p.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="CLEAR_finetune.py 输出的预训练 adapter 目录，例如 .../final_adapter",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="遗忘后模型的保存目录",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="CLEAR 数据根目录，包含 forget{n}+tofu 子目录",
    )

    # ── 数据划分 ──
    p.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        help="遗忘比例整数（1/5/10），用于拼接目录名，默认 5",
    )

    # ── 训练 ──
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    main(p.parse_args())

