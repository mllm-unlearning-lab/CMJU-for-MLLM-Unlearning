import os
import sys
import re
import json
import math
import argparse
import statistics
from itertools import cycle
from datetime import datetime

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoProcessor, get_scheduler, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors import safe_open

from unlearn_dataset import (
    Multimodal_Dataset,
    Unimodal_Dataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
)

sys.path.append("../")
sys.path.append("../../")


# =============================================================================
# Cache helpers
# =============================================================================

def save_torch_dict(path: str, obj: dict, accelerator=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    msg = f"[Cache] Saved: {path}"
    (accelerator.print if accelerator else print)(msg)


def load_torch_dict(path: str, accelerator=None):
    obj = torch.load(path, map_location="cpu")
    msg = f"[Cache] Loaded: {path}"
    (accelerator.print if accelerator else print)(msg)
    return obj


def build_cache_paths(args, lora_config):
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_tag = f"rank{lora_config.r}_ratio{args.forget_split_ratio}"

    svd_lora_path = os.path.join(cache_dir, f"phase1_svd_lora_{cache_tag}.pt")
    svd_peft_path = os.path.join(cache_dir, f"phase1_svd_peft_{cache_tag}")
    sal_path = os.path.join(cache_dir, f"phase1_saliency_{cache_tag}.pt")
    return svd_lora_path, svd_peft_path, sal_path


# =============================================================================
# Base parameter loader
# =============================================================================

def load_base_params_from_safetensors_index(base_model_dir: str, need_keys: set, accelerator=None):
    index_path = os.path.join(base_model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_to_keys = {}
    missing = 0
    for k in need_keys:
        shard = weight_map.get(k)
        if shard is None:
            missing += 1
            continue
        shard_to_keys.setdefault(shard, []).append(k)

    if accelerator:
        accelerator.print(
            f"[BaseLoad] Requested={len(need_keys)}, Missing={missing}, Shards={len(shard_to_keys)}"
        )

    loaded = {}
    for shard, keys in shard_to_keys.items():
        shard_path = os.path.join(base_model_dir, shard)
        if accelerator:
            accelerator.print(f"[BaseLoad] Reading {len(keys)} tensors from {shard}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                loaded[k] = f.get_tensor(k).clone()
    return loaded


# =============================================================================
# Utilities
# =============================================================================

def get_input_device(model):
    m = model
    if hasattr(m, "base_model"):
        m = m.base_model

    emb = None
    if hasattr(m, "get_input_embeddings"):
        emb = m.get_input_embeddings()
    if emb is None and hasattr(m, "language_model"):
        emb = m.language_model.get_input_embeddings()
    if emb is None:
        return next(model.parameters()).device
    return emb.weight.device


def load_model_and_processor(args):
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return model, processor


def invoke(batch, model, mode):
    if mode == "multimodal":
        input_ids, attention_mask, pixel_values, labels = batch
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )

    input_ids, attention_mask, _, labels = batch
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def batch_to_device(batch, target_device):
    return tuple(
        t.to(target_device) if isinstance(t, torch.Tensor) else t
        for t in batch
    )


def is_lora_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


LLM_KEYWORDS = ["language_model"]
EXCLUDE_KEYWORDS = ["vision_tower", "multi_modal_projector"]
FULL_TARGET_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)

LORA_TARGET_MODULES = (
    r"^language_model\.model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def is_llm_full_param(name: str) -> bool:
    if not any(kw in name for kw in LLM_KEYWORDS):
        return False
    if any(kw in name for kw in EXCLUDE_KEYWORDS):
        return False
    if not any(name.endswith(sfx) for sfx in FULL_TARGET_SUFFIXES):
        return False
    return True


def is_llm_lora_param(name: str) -> bool:
    if not any(kw in name for kw in LLM_KEYWORDS):
        return False
    if any(kw in name for kw in EXCLUDE_KEYWORDS):
        return False
    if "lora_A" not in name and "lora_B" not in name:
        return False
    return True


def snapshot_ft_params(model_ft) -> dict:
    return {
        name: param.detach().cpu().clone()
        for name, param in model_ft.named_parameters()
        if is_llm_full_param(name)
    }


def load_vanilla_lora_weights(model, vanilla_dir, accelerator):
    accelerator.print(f"[Phase1] Detected LoRA adapter: {vanilla_dir}")
    model = PeftModel.from_pretrained(model, vanilla_dir, is_trainable=True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# =============================================================================
# SVD initialization
# =============================================================================

def svd_init_lora_from_delta(ft_snapshot, base_params_cpu, peft_model, lora_r: int, lora_alpha: int, accelerator):
    scale_compensation = math.sqrt(lora_r / lora_alpha)
    accelerator.print(
        f"[SVD] Initializing LoRA from delta weights (r={lora_r}, alpha={lora_alpha}, scale={scale_compensation:.4f})"
    )

    peft_params = dict(peft_model.named_parameters())
    n_init = 0
    n_skip = 0

    for lora_name, lora_param in peft_model.named_parameters():
        if "lora_A" not in lora_name:
            continue

        stripped = lora_name
        if stripped.startswith("base_model.model."):
            stripped = stripped[len("base_model.model."):]

        full_name = re.sub(r"\.lora_[AB](\.\w+)*\.weight$", ".weight", stripped)
        if full_name not in ft_snapshot or full_name not in base_params_cpu:
            n_skip += 1
            continue

        W_ft = ft_snapshot[full_name].float()
        W_base = base_params_cpu[full_name].float()
        delta = W_ft - W_base

        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        except Exception as e:
            accelerator.print(f"[SVD] Skip {full_name}: {e}")
            n_skip += 1
            continue

        r_actual = min(lora_r, S.shape[0])
        sqrt_S = S[:r_actual].sqrt()
        lora_B_val = (U[:, :r_actual] * sqrt_S.unsqueeze(0)) * scale_compensation
        lora_A_val = (sqrt_S.unsqueeze(1) * Vh[:r_actual, :]) * scale_compensation

        lora_B_name = lora_name.replace("lora_A", "lora_B")
        lora_B_param = peft_params.get(lora_B_name)
        if lora_B_param is None:
            n_skip += 1
            continue

        with torch.no_grad():
            lora_param.data.copy_(lora_A_val.to(dtype=lora_param.dtype, device=lora_param.device))
            lora_B_param.data.copy_(lora_B_val.to(dtype=lora_B_param.dtype, device=lora_B_param.device))
        n_init += 1

    accelerator.print(f"[SVD] Completed: initialized={n_init}, skipped={n_skip}")


# =============================================================================
# Phase 1: dual-path saliency
# =============================================================================

def compute_saliency_scores(model, sal_loader_text, sal_loader_multi, accelerator):
    model.train()
    input_device = get_input_device(model)
    accelerator.print(f"[Phase1] Input embedding device: {input_device}")
    accelerator.print("[Phase1] Saliency method: fisher")

    target_params_dict = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad and is_llm_lora_param(name)
    }
    if not target_params_dict:
        raise RuntimeError("No trainable LLM LoRA parameters found for saliency computation.")

    n_params = len(target_params_dict)
    accelerator.print(f"[Phase1] Tracking {n_params} LoRA parameters")

    saliency_text = {
        n: torch.zeros_like(p.data, dtype=torch.float32, device="cpu")
        for n, p in target_params_dict.items()
    }
    saliency_multi = {
        n: torch.zeros_like(p.data, dtype=torch.float32, device="cpu")
        for n, p in target_params_dict.items()
    }

    def _run_one_path(loader, mode, sal_dict, path_name):
        n_batches = 0
        n_grad = 0
        loss_log = []

        for batch in tqdm(loader, desc=f"[Phase1] {path_name}", disable=not accelerator.is_main_process):
            batch = batch_to_device(batch, input_device)
            model.zero_grad()

            outputs = invoke(batch, model, mode)
            loss = outputs.loss

            if n_batches < 5:
                loss_log.append(round(loss.item(), 4) if loss else None)
            n_batches += 1

            if loss is None or loss.item() == 0.0:
                del outputs
                continue

            loss.backward()

            for name, param in target_params_dict.items():
                if param.grad is not None:
                    sal_dict[name] += param.grad.detach().float().cpu() ** 2

            del outputs, loss
            n_grad += 1

        accelerator.print(f"[Phase1] {path_name}: first_5_loss={loss_log}")
        accelerator.print(f"[Phase1] {path_name}: total_batches={n_batches}, valid_grad_batches={n_grad}")

        n_nonzero = sum(1 for v in sal_dict.values() if v.abs().sum() > 0)
        accelerator.print(f"[Phase1] {path_name}: nonzero_params={n_nonzero}/{n_params}")

        if n_nonzero == 0:
            raise RuntimeError(f"[Phase1] {path_name}: all gradients are zero.")

        if n_grad > 0:
            for name in sal_dict:
                sal_dict[name] /= n_grad

        return n_batches

    n_text = _run_one_path(sal_loader_text, "unimodal", saliency_text, "Text path")
    n_multi = _run_one_path(sal_loader_multi, "multimodal", saliency_multi, "Multimodal path")

    model.zero_grad()
    accelerator.print(f"[Phase1] Saliency finished: text_batches={n_text}, mm_batches={n_multi}")
    return saliency_text, saliency_multi


# =============================================================================
# Phase 2: parameter partition
# =============================================================================

def partition_parameters(saliency_text, saliency_multi, top_k_ratio=0.3, modality_margin=0.15, accelerator=None):
    def aprint(msg):
        (accelerator.print if accelerator else print)(msg)

    scalar_text = {n: v.mean().item() for n, v in saliency_text.items()}
    scalar_multi = {n: v.mean().item() for n, v in saliency_multi.items()}
    all_names = list(scalar_text.keys())
    combined = {n: scalar_text[n] + scalar_multi[n] for n in all_names}

    k = max(1, int(len(all_names) * top_k_ratio))
    top_names = sorted(all_names, key=lambda x: combined[x], reverse=True)[:k]

    mask_shared = set()
    mask_text_preferred = set()
    mask_mm_preferred = set()
    ratios = []

    for name in top_names:
        st = scalar_text[name]
        sm = scalar_multi[name]
        ratio = st / (st + sm + 1e-12)
        ratios.append(ratio)

        if ratio > 0.5 + modality_margin:
            mask_text_preferred.add(name)
        elif ratio < 0.5 - modality_margin:
            mask_mm_preferred.add(name)
        else:
            mask_shared.add(name)

    n_frozen = len(all_names) - len(top_names)
    aprint(
        f"[Partition] shared={len(mask_shared)}, "
        f"text_preferred={len(mask_text_preferred)}, "
        f"mm_preferred={len(mask_mm_preferred)}, "
        f"frozen={n_frozen}"
    )
    aprint(f"[Partition] top_k_ratio={top_k_ratio}, modality_margin={modality_margin}")

    if ratios:
        r_mean = statistics.mean(ratios)
        r_stdev = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        aprint(
            f"[Partition] ratio_stats: min={min(ratios):.3f}, max={max(ratios):.3f}, "
            f"mean={r_mean:.3f}, stdev={r_stdev:.4f}"
        )

    return mask_shared, mask_text_preferred, mask_mm_preferred


# =============================================================================
# Phase 3 helpers
# =============================================================================

def _log_merged_grad_norms(accum_grads, mask_shared, mask_text_preferred, mask_mm_preferred, accelerator):
    def gnorm(mask):
        sq = sum(accum_grads[n].float().norm().item() ** 2 for n in mask if n in accum_grads)
        return sq ** 0.5

    accelerator.print(
        f"[Train] GradNorm shared={gnorm(mask_shared):.6f} "
        f"text_preferred={gnorm(mask_text_preferred):.6f} "
        f"mm_preferred={gnorm(mask_mm_preferred):.6f}"
    )


def csau_forget_step(model, multi_batch, uni_batch, mask_shared, mask_text_preferred, mask_mm_preferred, args, accelerator, grad_accum_steps):
    raw_model = accelerator.unwrap_model(model)
    active_params = mask_shared | mask_text_preferred | mask_mm_preferred

    raw_model.zero_grad()
    out_m = invoke(multi_batch, model, "multimodal")
    loss_m_val = out_m.loss.item()
    accelerator.backward(-out_m.loss)
    forget_grads_m = {
        name: param.grad.detach().clone()
        for name, param in raw_model.named_parameters()
        if param.grad is not None
    }
    del out_m
    torch.cuda.empty_cache()

    raw_model.zero_grad()
    out_u = invoke(uni_batch, model, "unimodal")
    loss_u_val = out_u.loss.item()
    accelerator.backward(-args.alpha_forget * out_u.loss)
    forget_grads_u = {
        name: param.grad.detach().clone()
        for name, param in raw_model.named_parameters()
        if param.grad is not None
    }
    del out_u
    torch.cuda.empty_cache()

    loss_forget_val = loss_m_val + args.alpha_forget * loss_u_val

    def _norm_active(gdict):
        sq = sum(gdict[n].float().norm().item() ** 2 for n in active_params if n in gdict)
        return sq ** 0.5

    nm = _norm_active(forget_grads_m)
    nu = _norm_active(forget_grads_u)
    eps = 1e-8
    scale_m = (nu + eps) / (nm + eps)

    forget_grads = {}
    for n in active_params:
        gm = forget_grads_m.get(n)
        gu = forget_grads_u.get(n)

        if gm is None and gu is None:
            continue
        if gm is None:
            forget_grads[n] = gu
        elif gu is None:
            forget_grads[n] = scale_m * gm
        else:
            forget_grads[n] = scale_m * gm + gu

    for n in list(forget_grads.keys()):
        coeff = args.alpha_shared if n in mask_shared else args.beta_specific
        forget_grads[n] = forget_grads[n] * coeff

    def gnorm_group(mask):
        sq = sum(forget_grads[n].float().norm().item() ** 2 for n in mask if n in forget_grads)
        return sq ** 0.5

    text_norm = gnorm_group(mask_text_preferred)
    mm_norm = gnorm_group(mask_mm_preferred)
    min_group = 5

    if (
        text_norm > 1e-8 and mm_norm > 1e-8
        and len(mask_text_preferred) >= min_group
        and len(mask_mm_preferred) >= min_group
    ):
        target = (text_norm + mm_norm) / 2.0
        text_scale = max(0.1, 1.0 + args.gamma_sym * (target / text_norm - 1.0))
        mm_scale = max(0.1, 1.0 + args.gamma_sym * (target / mm_norm - 1.0))

        for n in mask_text_preferred:
            if n in forget_grads:
                forget_grads[n] = forget_grads[n] * text_scale
        for n in mask_mm_preferred:
            if n in forget_grads:
                forget_grads[n] = forget_grads[n] * mm_scale

    scale = 1.0 / grad_accum_steps
    step_grads = {}
    for name, param in raw_model.named_parameters():
        if not param.requires_grad or name not in active_params:
            continue
        f_grad = forget_grads.get(name, torch.zeros_like(param.data)).to(param.device)
        step_grads[name] = (f_grad * scale).cpu()

    raw_model.zero_grad()
    return loss_forget_val, step_grads


def csau_retain_step(model, retain_multi_batch, retain_uni_batch, active_params, grad_accum_steps, accelerator):
    raw_model = accelerator.unwrap_model(model)

    raw_model.zero_grad()
    out = invoke(retain_multi_batch, model, "multimodal")
    loss_r_multi = out.loss.item()
    accelerator.backward(out.loss)
    del out
    torch.cuda.empty_cache()

    out = invoke(retain_uni_batch, model, "unimodal")
    loss_r_uni = out.loss.item()
    accelerator.backward(out.loss)
    del out
    torch.cuda.empty_cache()

    loss_retain_val = loss_r_multi + loss_r_uni

    scale = 1.0 / grad_accum_steps
    step_grads = {}
    for name, param in raw_model.named_parameters():
        if not param.requires_grad or name not in active_params:
            continue
        r_grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param.data)
        step_grads[name] = (r_grad * scale).cpu()

    raw_model.zero_grad()
    return loss_retain_val, step_grads


def apply_gradients(model, accum_grads, accelerator):
    raw_model = accelerator.unwrap_model(model)

    if accelerator.num_processes > 1 and dist.is_initialized():
        for name in accum_grads:
            g = accum_grads[name].cuda()
            dist.all_reduce(g, op=dist.ReduceOp.AVG)
            accum_grads[name] = g.cpu()

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        param.grad = accum_grads[name].to(param.device) if name in accum_grads else None


# =============================================================================
# Main
# =============================================================================

def main(args):
    print(datetime.now())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator()

    accelerator.print("\n===== Configuration =====")
    accelerator.print("Command:")
    accelerator.print(" ".join(sys.argv))
    accelerator.print("\nArguments:")
    for k, v in sorted(vars(args).items()):
        accelerator.print(f"  {k}: {v}")
    accelerator.print("=========================\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        init_lora_weights="gaussian",
    )

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")

    df_forget = pd.read_parquet(os.path.join(forget_folder, "train-00000-of-00001.parquet"))
    df_retain = pd.read_parquet(os.path.join(retain_folder, "train-00000-of-00001.parquet"))
    accelerator.print(f"[Data] Forget samples={len(df_forget)}, Retain samples={len(df_retain)}")

    forget_multi_ds = Multimodal_Dataset(df=df_forget)
    forget_uni_ds = Unimodal_Dataset(df=df_forget)
    retain_multi_ds = Multimodal_Dataset(df=df_retain)
    retain_uni_ds = Unimodal_Dataset(df=df_retain)

    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    collate_multi = lambda x: train_collate_fn_llava_multimodal(x, processor, args)
    collate_uni = lambda x: train_collate_fn_llava_unimodal(x, processor, args)

    svd_lora_path, svd_peft_path, sal_path = build_cache_paths(args, lora_config)

    # =========================================================================
    # Phase 1
    # =========================================================================
    accelerator.print("\n===== Phase 1: LoRA initialization and saliency =====")

    vanilla_is_lora = is_lora_adapter_dir(args.vanilla_dir)

    if os.path.exists(sal_path):
        accelerator.print("[Phase1] Saliency cache found. Skipping Phase 1.")
        cache = load_torch_dict(sal_path, accelerator)
        saliency_text = cache["saliency_text"]
        saliency_multi = cache["saliency_multi"]

    else:
        model_sal_base = LlavaForConditionalGeneration.from_pretrained(
            args.base_model_dir,
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model_sal_base.resize_token_embeddings(len(processor.tokenizer))

        if vanilla_is_lora:
            model_sal = load_vanilla_lora_weights(model_sal_base, args.vanilla_dir, accelerator)

        else:
            accelerator.print("[Phase1] Full-model reference detected. Running SVD initialization.")

            model_ft_cpu = LlavaForConditionalGeneration.from_pretrained(
                args.vanilla_dir,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            ft_snapshot = snapshot_ft_params(model_ft_cpu)
            accelerator.print(f"[Phase1] Snapshot collected for {len(ft_snapshot)} full parameters")
            del model_ft_cpu
            torch.cuda.empty_cache()

            need_keys = set(ft_snapshot.keys())
            base_params_cpu = load_base_params_from_safetensors_index(
                args.base_model_dir,
                need_keys=need_keys,
                accelerator=accelerator,
            )
            accelerator.print(f"[Phase1] Base parameters loaded: {len(base_params_cpu)}/{len(need_keys)}")
            torch.cuda.empty_cache()

            model_sal = get_peft_model(model_sal_base, lora_config)
            model_sal.enable_input_require_grads()
            model_sal.print_trainable_parameters()

            if os.path.exists(svd_lora_path):
                accelerator.print("[Phase1] SVD LoRA cache found. Loading cached LoRA tensors.")
                lora_state = load_torch_dict(svd_lora_path, accelerator)
                with torch.no_grad():
                    for name, p in model_sal.named_parameters():
                        if ("lora_A" in name or "lora_B" in name) and name in lora_state:
                            p.copy_(lora_state[name].to(device=p.device, dtype=p.dtype))
            else:
                accelerator.print("[Phase1] Computing SVD initialization from delta weights.")
                svd_init_lora_from_delta(
                    ft_snapshot=ft_snapshot,
                    base_params_cpu=base_params_cpu,
                    peft_model=model_sal,
                    lora_r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    accelerator=accelerator,
                )

                lora_state = {
                    name: p.detach().cpu()
                    for name, p in model_sal.named_parameters()
                    if "lora_A" in name or "lora_B" in name
                }
                save_torch_dict(svd_lora_path, lora_state, accelerator)

                if accelerator.is_main_process:
                    accelerator.print(f"[Phase1] Saving initialized PEFT checkpoint to {svd_peft_path}")
                    model_sal.save_pretrained(svd_peft_path)

            del ft_snapshot, base_params_cpu
            torch.cuda.empty_cache()
            accelerator.print("[Memory] Released SVD intermediate tensors")

        sal_loader_multi = DataLoader(
            forget_multi_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_multi,
        )
        sal_loader_uni = DataLoader(
            forget_uni_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_uni,
        )

        saliency_text, saliency_multi = compute_saliency_scores(
            model_sal,
            sal_loader_uni,
            sal_loader_multi,
            accelerator,
        )

        save_torch_dict(
            sal_path,
            {
                "saliency_text": saliency_text,
                "saliency_multi": saliency_multi,
            },
            accelerator,
        )

        del model_sal
        torch.cuda.empty_cache()
        accelerator.print("[Memory] Released saliency model")

    # =========================================================================
    # Phase 2
    # =========================================================================
    accelerator.print("\n===== Phase 2: Parameter partition =====")
    mask_shared, mask_text_preferred, mask_mm_preferred = partition_parameters(
        saliency_text,
        saliency_multi,
        top_k_ratio=args.top_k_ratio,
        modality_margin=args.modality_margin,
        accelerator=accelerator,
    )
    del saliency_text, saliency_multi
    torch.cuda.empty_cache()
    accelerator.print("[Memory] Released saliency tensors")

    # =========================================================================
    # Phase 3
    # =========================================================================
    accelerator.print("\n===== Phase 3: Training model setup =====")
    model, _ = load_model_and_processor(args)
    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(svd_lora_path):
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        accelerator.print("[Phase3] Loading SVD-initialized LoRA weights")
        lora_state = load_torch_dict(svd_lora_path, accelerator)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ("lora_A" in name or "lora_B" in name) and name in lora_state:
                    p.copy_(lora_state[name].to(device=p.device, dtype=p.dtype))
        accelerator.print("[Phase3] SVD LoRA weights loaded")

    elif vanilla_is_lora:
        accelerator.print("[Phase3] Loading LoRA adapter from vanilla_dir")
        model = PeftModel.from_pretrained(model, args.vanilla_dir, is_trainable=True)
        model.enable_input_require_grads()
        accelerator.print("[Phase3] LoRA adapter loaded")

    else:
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        accelerator.print("[Phase3] No SVD cache found. Using Gaussian LoRA initialization")

    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    train_lora_names = {
        n for n, p in model.named_parameters()
        if p.requires_grad and is_llm_lora_param(n)
    }

    for mask_name, mask_set in [
        ("mask_shared", mask_shared),
        ("mask_text_preferred", mask_text_preferred),
        ("mask_mm_preferred", mask_mm_preferred),
    ]:
        unknown = mask_set - train_lora_names
        assert not unknown, (
            f"[Assert] {mask_name} contains parameter names not found in the training model:\n"
            + "\n".join(sorted(unknown)[:5])
        )
    accelerator.print("[Assert] Partition names match training model")

    train_loader_multi = DataLoader(
        forget_multi_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multi,
    )
    train_loader_uni = DataLoader(
        forget_uni_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_uni,
    )
    retain_loader_multi = DataLoader(
        retain_multi_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multi,
    )
    retain_loader_uni = DataLoader(
        retain_uni_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_uni,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    n_update_steps = math.ceil(len(train_loader_multi) / args.grad_accum_steps) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_update_steps,
    )

    (
        model,
        optimizer,
        train_loader_multi,
        train_loader_uni,
        retain_loader_multi,
        retain_loader_uni,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader_multi,
        train_loader_uni,
        retain_loader_multi,
        retain_loader_uni,
        lr_scheduler,
    )

    retain_multi_iter = cycle(retain_loader_multi)
    retain_uni_iter = cycle(retain_loader_uni)

    accelerator.print(
        f"\n===== Phase 3: Training =====\n"
        f"[Train] grad_accum_steps={args.grad_accum_steps}, clip_grad_norm={args.clip_grad_norm}\n"
    )

    prev_avg_forget = None
    for epoch in range(args.num_epochs):
        model.train()
        total_forget = 0.0
        total_retain = 0.0
        n_optimizer_step = 0
        accum_grads = {}
        window_steps = 0
        total_mini = len(train_loader_multi)

        bar = tqdm(
            zip(train_loader_multi, train_loader_uni),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            total=total_mini,
            disable=not accelerator.is_main_process,
        )

        for mini_step, (multi_batch, uni_batch) in enumerate(bar):
            f_loss, step_grads_forget = csau_forget_step(
                model=model,
                multi_batch=multi_batch,
                uni_batch=uni_batch,
                mask_shared=mask_shared,
                mask_text_preferred=mask_text_preferred,
                mask_mm_preferred=mask_mm_preferred,
                args=args,
                accelerator=accelerator,
                grad_accum_steps=args.grad_accum_steps,
            )

            step_grads_retain = {}
            r_loss_accum = 0.0
            for _ in range(args.retain_steps_per_forget):
                r_loss_i, grads_i = csau_retain_step(
                    model=model,
                    retain_multi_batch=next(retain_multi_iter),
                    retain_uni_batch=next(retain_uni_iter),
                    active_params=mask_shared | mask_text_preferred | mask_mm_preferred,
                    grad_accum_steps=args.grad_accum_steps,
                    accelerator=accelerator,
                )
                r_loss_accum += r_loss_i
                for name, g in grads_i.items():
                    step_grads_retain[name] = step_grads_retain.get(name, 0) + g

            r_loss = r_loss_accum / args.retain_steps_per_forget
            for name in step_grads_retain:
                step_grads_retain[name] = step_grads_retain[name] / args.retain_steps_per_forget

            step_grads = {}
            all_param_names = set(step_grads_forget.keys()) | set(step_grads_retain.keys())
            for name in all_param_names:
                gf = step_grads_forget.get(name, torch.zeros(1))
                gr = step_grads_retain.get(name, torch.zeros(1))
                step_grads[name] = gf + gr

            for name, grad in step_grads.items():
                accum_grads[name] = accum_grads.get(name, 0) + grad

            window_steps += 1
            total_forget += f_loss
            total_retain += r_loss

            is_last_mini = (mini_step + 1 == total_mini)
            is_accum_step = (window_steps == args.grad_accum_steps or is_last_mini)

            if is_accum_step:
                if window_steps < args.grad_accum_steps:
                    correction = args.grad_accum_steps / window_steps
                    for name in accum_grads:
                        accum_grads[name] = accum_grads[name] * correction

                n_optimizer_step += 1
                _log_merged_grad_norms(
                    accum_grads,
                    mask_shared,
                    mask_text_preferred,
                    mask_mm_preferred,
                    accelerator,
                )

                apply_gradients(model, accum_grads, accelerator)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad],
                    max_norm=args.clip_grad_norm,
                )

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                accum_grads = {}
                window_steps = 0

            bar.set_postfix({
                "forget": f"{f_loss:.4f}",
                "retain": f"{r_loss:.4f}",
            })

        avg_forget = total_forget / total_mini
        avg_retain = total_retain / total_mini
        accelerator.print(
            f"[Train] Epoch {epoch + 1}: "
            f"avg_forget={avg_forget:.4f}, avg_retain={avg_retain:.4f}, optimizer_steps={n_optimizer_step}"
        )

        if prev_avg_forget is not None:
            trend = "up" if avg_forget >= prev_avg_forget else "down"
            accelerator.print(
                f"[Train] Forget loss trend: {prev_avg_forget:.4f} -> {avg_forget:.4f} ({trend})"
            )
        prev_avg_forget = avg_forget

        accelerator.wait_for_everyone()
        epoch_save_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_save_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(epoch_save_dir)
        accelerator.print(f"[Save] Checkpoint saved to {epoch_save_dir}")

        if avg_retain > args.retain_loss_threshold:
            accelerator.print(
                f"[Train] Early stop triggered: avg_retain={avg_retain:.4f} > threshold={args.retain_loss_threshold}"
            )
            break

    accelerator.print("[Train] Training complete")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSAU (LLaVA only)")

    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Base LLaVA model directory.",
    )
    parser.add_argument(
        "--vanilla_dir",
        type=str,
        required=True,
        help=(
            "Reference model directory used in Phase 1. "
            "If it is a full model, the script computes delta(base->full) and runs SVD init. "
            "If it is a LoRA adapter dir, the script loads it directly and computes saliency."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--data_split_dir",
        type=str,
        required=True,
        help="Directory containing forget_xx / retain_xx parquet splits.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory for SVD and saliency cache.",
    )

    parser.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        help="Forget ratio used to locate split folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Reserved sequence length argument.",
    )

    parser.add_argument(
        "--alpha_shared",
        type=float,
        default=2.0,
        help="Forget gradient scaling factor for shared parameters.",
    )
    parser.add_argument(
        "--beta_specific",
        type=float,
        default=1.0,
        help="Forget gradient scaling factor for preferred parameters.",
    )
    parser.add_argument(
        "--alpha_forget",
        type=float,
        default=1.0,
        help="Global coefficient for unimodal forget loss.",
    )
    parser.add_argument(
        "--gamma_sym",
        type=float,
        default=0.5,
        help="Symmetry strength between text-preferred and multimodal-preferred gradients.",
    )
    parser.add_argument(
        "--top_k_ratio",
        type=float,
        default=0.3,
        help="Top ratio of salient parameters used for partitioning.",
    )
    parser.add_argument(
        "--modality_margin",
        type=float,
        default=0.15,
        help="Margin used to split shared/text_preferred/mm_preferred parameters.",
    )
    parser.add_argument(
        "--retain_loss_threshold",
        type=float,
        default=30.0,
        help="Early-stop threshold on average retain loss.",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
    )
    parser.add_argument(
        "--retain_steps_per_forget",
        type=int,
        default=4,
        help="Number of retain steps after each forget step.",
    )

    args = parser.parse_args()
    main(args)

