import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset

from data_process.CLEAR_process import (
    CAPTION_MODE,
    TEXT_QA_MODE,
    CLEARDataset,
)


def _build_clear_batch(examples, processor, mode: str):
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


def _to_device_batch(batch, device):
    out = []
    for x in batch:
        if torch.is_tensor(x):
            out.append(x.to(device))
        else:
            out.append(x)
    return tuple(out)


def load_model_and_processor(args):
    print(f"[Model] Loading processor from: {args.base_model_dir}")
    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print(f"[Model] Loading base model from: {args.base_model_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = model.config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    vocab_size = len(processor.tokenizer)
    if vocab_size > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(vocab_size)

    print(f"[Model] Loading pre-unlearning LoRA from: {args.lora_dir}")
    model = PeftModel.from_pretrained(
        model,
        args.lora_dir,
        is_trainable=False,
        local_files_only=True,
    )

    print("[Model] Merging LoRA into base model before pruning")
    model = model.merge_and_unload()

    return model, processor


def build_dataloaders(args, processor):
    forget_path = os.path.join(args.data_dir, f"forget{args.forget_split_ratio}+tofu")
    retain_path = os.path.join(args.data_dir, f"retain{100 - args.forget_split_ratio}+tofu")

    print(f"[Data] forget path: {forget_path}")
    print(f"[Data] retain path: {retain_path}")

    forget_raw = load_dataset(forget_path, split="train")
    retain_raw = load_dataset(retain_path, split="train")

    forget_caption_ds = CLEARDataset(forget_raw, mode=CAPTION_MODE)
    forget_text_ds = CLEARDataset(forget_raw, mode=TEXT_QA_MODE)
    retain_caption_ds = CLEARDataset(retain_raw, mode=CAPTION_MODE)
    retain_text_ds = CLEARDataset(retain_raw, mode=TEXT_QA_MODE)

    print(f"[Data] forget  CAPTION={len(forget_caption_ds)}, TEXT_QA={len(forget_text_ds)}")
    print(f"[Data] retain  CAPTION={len(retain_caption_ds)}, TEXT_QA={len(retain_text_ds)}")

    if len(forget_caption_ds) == 0:
        raise ValueError("[Data] forget CAPTION is empty")
    if len(forget_text_ds) == 0:
        raise ValueError("[Data] forget TEXT_QA is empty")
    if len(retain_caption_ds) == 0:
        raise ValueError("[Data] retain CAPTION is empty")
    if len(retain_text_ds) == 0:
        raise ValueError("[Data] retain TEXT_QA is empty")

    col_caption = make_collate_fn(processor, mode="multimodal")
    col_text = make_collate_fn(processor, mode="unimodal")

    return {
        "forget_mm": DataLoader(forget_caption_ds, args.batch_size, shuffle=False, collate_fn=col_caption),
        "forget_um": DataLoader(forget_text_ds, args.batch_size, shuffle=False, collate_fn=col_text),
        "retain_mm": DataLoader(retain_caption_ds, args.batch_size, shuffle=False, collate_fn=col_caption),
        "retain_um": DataLoader(retain_text_ds, args.batch_size, shuffle=False, collate_fn=col_text),
    }


class ActivationCollector:
    def __init__(self):
        self.activations = {
            "multimodal": defaultdict(list),
            "unimodal": defaultdict(list),
        }
        self.handles = []

    def register_hook(self, module, layer_name: str, modality: str):
        def hook(_module, _inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs is None:
                return
            out = outputs.detach()
            if out.dim() == 2:
                tensor = out.float()
            elif out.dim() >= 3:
                tensor = out.float().reshape(-1, out.shape[-1])
            else:
                return
            self.activations[modality][layer_name].append(tensor.cpu())

        handle = module.register_forward_hook(hook)
        self.handles.append(handle)

    def clear(self):
        self.activations = {
            "multimodal": defaultdict(list),
            "unimodal": defaultdict(list),
        }

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def select_target_layers_llava(model, vision_last_n=3, text_last_n=3):
    targets = []

    vision_layers = model.vision_tower.vision_model.encoder.layers
    text_layers = model.language_model.model.layers

    v_start = max(0, len(vision_layers) - vision_last_n)
    t_start = max(0, len(text_layers) - text_last_n)

    for i in range(v_start, len(vision_layers)):
        layer = vision_layers[i]
        if hasattr(layer.mlp, "fc1"):
            targets.append(("multimodal", f"vision_fc1_{i}", layer.mlp.fc1))
        if hasattr(layer.mlp, "fc2"):
            targets.append(("multimodal", f"vision_fc2_{i}", layer.mlp.fc2))

    for i in range(t_start, len(text_layers)):
        layer = text_layers[i]
        if hasattr(layer.mlp, "gate_proj"):
            targets.append(("unimodal", f"lang_gate_proj_{i}", layer.mlp.gate_proj))
        if hasattr(layer.mlp, "up_proj"):
            targets.append(("unimodal", f"lang_up_proj_{i}", layer.mlp.up_proj))
        if hasattr(layer.mlp, "down_proj"):
            targets.append(("unimodal", f"lang_down_proj_{i}", layer.mlp.down_proj))

    return targets


def register_hooks_for_manu(model, collector, args):
    targets = select_target_layers_llava(
        model,
        vision_last_n=args.vision_last_n,
        text_last_n=args.text_last_n,
    )
    for modality, layer_name, module in targets:
        collector.register_hook(module, layer_name, modality)
    return [x[1] for x in targets]


ARGS_ACTIVATION_THRESHOLD = 1e-1


@torch.no_grad()
def collect_importance_scores(model, dataloader, collector, modality, device, max_batches=None):
    model.eval()
    collector.clear()

    metric_values = defaultdict(lambda: defaultdict(list))
    total_steps = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Collect {modality}", total=total_steps)):
        if max_batches is not None and step >= max_batches:
            break

        batch = _to_device_batch(batch, device)
        _ = _forward(batch, model, modality)

        for layer_name, chunks in collector.activations[modality].items():
            if not chunks:
                continue

            acts = torch.cat(chunks, dim=0).clamp(min=-1e3, max=1e3)
            metric_values["I_abs"][layer_name].append(acts.abs().mean(dim=0))
            metric_values["I_freq"][layer_name].append((acts.abs() > ARGS_ACTIVATION_THRESHOLD).float().mean(dim=0))
            metric_values["I_var"][layer_name].append(acts.std(dim=0))
            metric_values["I_rms"][layer_name].append(torch.sqrt((acts ** 2).mean(dim=0)))

        collector.activations[modality] = defaultdict(list)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_scores = defaultdict(dict)
    for metric, layer_dict in metric_values.items():
        for layer_name, values in layer_dict.items():
            final_scores[metric][layer_name] = torch.stack(values, dim=0).mean(dim=0)

    return final_scores


def combine_modal_scores(mm_scores, um_scores):
    merged = defaultdict(dict)
    for metric in set(mm_scores.keys()) | set(um_scores.keys()):
        if metric in mm_scores:
            merged[metric].update(mm_scores[metric])
        if metric in um_scores:
            merged[metric].update(um_scores[metric])
    return merged


def compute_combined_scores(forget_scores, retain_scores, weights=None, epsilon=1e-5):
    if weights is None:
        weights = {"I_abs": 2.0, "I_freq": 0.0, "I_var": 2.0, "I_rms": 2.0}

    combined_scores = {}
    for metric in forget_scores:
        if metric not in retain_scores:
            continue
        for layer_name in forget_scores[metric]:
            if layer_name not in retain_scores[metric]:
                continue

            score = weights.get(metric, 1.0) * (
                (forget_scores[metric][layer_name] / (retain_scores[metric][layer_name] + epsilon)) - 1.0
            )

            if layer_name not in combined_scores:
                combined_scores[layer_name] = score
            else:
                combined_scores[layer_name] = combined_scores[layer_name] + score

    return combined_scores


def compute_top_k_pruning_mask(combined_scores_dict, top_k_percent):
    all_scores = torch.cat([v.flatten() for v in combined_scores_dict.values()])
    k = max(1, int((top_k_percent / 100.0) * all_scores.numel()))
    topk_vals, _ = torch.topk(all_scores, k, largest=True)
    threshold = topk_vals[-1]

    masks = {}
    for layer_name, scores in combined_scores_dict.items():
        masks[layer_name] = (scores >= threshold).float()

    return masks, threshold.item(), k, all_scores.numel()


def apply_mask_to_linear(layer: torch.nn.Linear, mask_1d: torch.Tensor):
    mask_1d = mask_1d.to(layer.weight.device).float()

    if mask_1d.numel() == layer.weight.shape[0]:
        keep = (1.0 - mask_1d).view(-1, 1)
        layer.weight.data *= keep
        if layer.bias is not None and layer.bias.shape[0] == mask_1d.shape[0]:
            layer.bias.data *= (1.0 - mask_1d)
        return

    if mask_1d.numel() == layer.weight.shape[1]:
        layer.weight.data *= (1.0 - mask_1d).view(1, -1)
        return

    raise ValueError(
        f"Mask size {mask_1d.numel()} incompatible with layer weight shape {tuple(layer.weight.shape)}"
    )


def apply_structural_pruning_llava(model, pruning_masks):
    applied = []

    for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
        key = f"vision_fc1_{layer_idx}"
        if hasattr(layer.mlp, "fc1") and key in pruning_masks:
            apply_mask_to_linear(layer.mlp.fc1, pruning_masks[key])
            applied.append(key)

        key = f"vision_fc2_{layer_idx}"
        if hasattr(layer.mlp, "fc2") and key in pruning_masks:
            apply_mask_to_linear(layer.mlp.fc2, pruning_masks[key])
            applied.append(key)

    for layer_idx, layer in enumerate(model.language_model.model.layers):
        key = f"lang_gate_proj_{layer_idx}"
        if hasattr(layer.mlp, "gate_proj") and key in pruning_masks:
            apply_mask_to_linear(layer.mlp.gate_proj, pruning_masks[key])
            applied.append(key)

        key = f"lang_up_proj_{layer_idx}"
        if hasattr(layer.mlp, "up_proj") and key in pruning_masks:
            apply_mask_to_linear(layer.mlp.up_proj, pruning_masks[key])
            applied.append(key)

        key = f"lang_down_proj_{layer_idx}"
        if hasattr(layer.mlp, "down_proj") and key in pruning_masks:
            apply_mask_to_linear(layer.mlp.down_proj, pruning_masks[key])
            applied.append(key)

    return applied


def save_masks(save_dir: str, masks: Dict[str, torch.Tensor], scores: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "masks": {k: v.cpu() for k, v in masks.items()},
            "scores": {k: v.cpu() for k, v in scores.items()},
            "metadata": metadata,
        },
        os.path.join(save_dir, "manu_masks.pt"),
    )


def main(args):
    global ARGS_ACTIVATION_THRESHOLD
    ARGS_ACTIVATION_THRESHOLD = args.activation_threshold

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n[{datetime.now()}] MANU")
    print("Command: " + " ".join(sys.argv))
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    model, processor = load_model_and_processor(args)
    dataloaders = build_dataloaders(args, processor)

    device = next(model.parameters()).device
    print(f"[Device] main device: {device}")

    collector = ActivationCollector()
    selected_layers = register_hooks_for_manu(model, collector, args)
    print("[MANU] selected layers:")
    for name in selected_layers:
        print("  ", name)

    print("\n[1/4] Collect forget multimodal scores")
    forget_mm_scores = collect_importance_scores(
        model, dataloaders["forget_mm"], collector, "multimodal", device, args.max_batches
    )

    print("\n[2/4] Collect forget unimodal scores")
    forget_um_scores = collect_importance_scores(
        model, dataloaders["forget_um"], collector, "unimodal", device, args.max_batches
    )

    print("\n[3/4] Collect retain multimodal scores")
    retain_mm_scores = collect_importance_scores(
        model, dataloaders["retain_mm"], collector, "multimodal", device, args.max_batches
    )

    print("\n[4/4] Collect retain unimodal scores")
    retain_um_scores = collect_importance_scores(
        model, dataloaders["retain_um"], collector, "unimodal", device, args.max_batches
    )

    collector.remove()

    forget_scores = combine_modal_scores(forget_mm_scores, forget_um_scores)
    retain_scores = combine_modal_scores(retain_mm_scores, retain_um_scores)

    weights = {
        "I_abs": args.weight_abs,
        "I_freq": args.weight_freq,
        "I_var": args.weight_var,
        "I_rms": args.weight_rms,
    }

    combined_scores = compute_combined_scores(
        forget_scores=forget_scores,
        retain_scores=retain_scores,
        weights=weights,
        epsilon=args.epsilon,
    )

    if not combined_scores:
        raise RuntimeError("[MANU] no combined scores produced; please check hooks and data loading")

    pruning_masks, threshold, k, total = compute_top_k_pruning_mask(
        combined_scores, args.prune_percent
    )

    applied_layers = apply_structural_pruning_llava(model, pruning_masks)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)

    metadata = {
        "base_model_dir": args.base_model_dir,
        "lora_dir": args.lora_dir,
        "data_dir": args.data_dir,
        "forget_split_ratio": args.forget_split_ratio,
        "prune_percent": args.prune_percent,
        "vision_last_n": args.vision_last_n,
        "text_last_n": args.text_last_n,
        "activation_threshold": args.activation_threshold,
        "threshold": threshold,
        "num_pruned_neurons": k,
        "num_candidate_neurons": total,
        "applied_layers": applied_layers,
        "weights": weights,
    }
    save_masks(args.save_dir, pruning_masks, combined_scores, metadata)

    print("\n=== MANU finished ===")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser("MANU")

    p.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Base LLaVA model path",
    )
    p.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Pre-unlearning LoRA adapter directory",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output directory for pruned model",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="CLEAR data root containing forget{n}+tofu and retain{n}+tofu",
    )

    p.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        help="Forget ratio integer, e.g. 1 / 5 / 10",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Use only the first N batches of each split for faster debugging",
    )

    p.add_argument("--prune_percent", type=float, default=10.0)
    p.add_argument("--epsilon", type=float, default=1e-5)
    p.add_argument("--activation_threshold", type=float, default=1e-1)
    p.add_argument("--vision_last_n", type=int, default=3)
    p.add_argument("--text_last_n", type=int, default=3)

    p.add_argument("--weight_abs", type=float, default=2.0)
    p.add_argument("--weight_freq", type=float, default=0.0)
    p.add_argument("--weight_var", type=float, default=2.0)
    p.add_argument("--weight_rms", type=float, default=2.0)

    main(p.parse_args())
