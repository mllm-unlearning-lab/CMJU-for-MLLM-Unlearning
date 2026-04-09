import os
import ast
import json
import argparse
from io import BytesIO
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from datetime import datetime


# =============================================================================
# Utils
# =============================================================================

def json2token(obj: Any, sort_json_key: bool = True) -> str:
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        return "".join(f"<s_{k}>{json2token(obj[k], sort_json_key)}</s_{k}>" for k in keys)
    if isinstance(obj, list):
        return "<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def _safe_literal_eval(value):
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _supports_chat_template(processor) -> bool:
    return hasattr(processor, "apply_chat_template")


def _build_prompt_and_full_text(processor, question: str, answer: str, with_image: bool):
    if _supports_chat_template(processor):
        user_content = []
        if with_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})

        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        prompt_text = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
        ).strip()

        full_text = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
        ).strip()
        return prompt_text, full_text

    if with_image:
        prompt_text = f"USER: <image>\n{question}\nASSISTANT:"
        full_text = f"USER: <image>\n{question}\nASSISTANT: {answer}"
    else:
        prompt_text = f"USER: {question}\nASSISTANT:"
        full_text = f"USER: {question}\nASSISTANT: {answer}"
    return prompt_text, full_text


def _make_labels_from_prompt_lengths(
    input_ids: torch.Tensor,
    prompt_lengths,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    labels = input_ids.clone()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    for i, prompt_len in enumerate(prompt_lengths):
        prompt_len = min(prompt_len, labels.shape[1])
        labels[i, :prompt_len] = -100
    return labels


# =============================================================================
# Dataset
# =============================================================================

class BaseQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, qa_field: str, require_image: bool):
        self.samples = self._flatten(df, qa_field, require_image)

    def _flatten(self, df: pd.DataFrame, qa_field: str, require_image: bool):
        samples = []
        for idx, row in df.iterrows():
            qa = _safe_literal_eval(row[qa_field])
            if qa is None:
                continue

            image = None
            if require_image:
                try:
                    image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
                except Exception as e:
                    print(f"[Dataset] image load failed at row {idx}: {e}")
                    continue

            q_map = qa.get("question", {})
            a_map = qa.get("answer", {})
            for k in q_map:
                if k in a_map:
                    samples.append(
                        {
                            "image": image,
                            "question": json2token(q_map[k]),
                            "answer": json2token(a_map[k]),
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MultimodalDataset(BaseQADataset):
    def __init__(self, df):
        super().__init__(df=df, qa_field="MM_QA", require_image=True)


class UnimodalDataset(BaseQADataset):
    def __init__(self, df):
        super().__init__(df=df, qa_field="UM_QA", require_image=False)


# =============================================================================
# Collate
# =============================================================================

def collate_multimodal(examples, processor, mm_max_length=1024, log_lengths=False):
    images = []
    full_texts = []
    prompt_lengths = []

    for ex in examples:
        prompt_text, full_text = _build_prompt_and_full_text(
            processor=processor,
            question=ex["question"],
            answer=ex["answer"],
            with_image=True,
        )

        prompt_inputs = processor(
            text=[prompt_text],
            images=[ex["image"]],
            padding=False,
            truncation=True,
            max_length=mm_max_length,
            return_tensors="pt",
        )
        prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

        images.append(ex["image"])
        full_texts.append(full_text)

    batch = processor(
        text=full_texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=mm_max_length,
        return_tensors="pt",
    )

    if log_lengths:
        print(f"[collate_mm] batch={len(examples)} seq_len={batch['input_ids'].shape[1]}")

    batch["labels"] = _make_labels_from_prompt_lengths(
        batch["input_ids"],
        prompt_lengths,
        processor.tokenizer.pad_token_id,
    )
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def collate_unimodal(examples, processor, max_length=384, log_lengths=False):
    full_texts = []
    prompt_lengths = []

    for ex in examples:
        prompt_text, full_text = _build_prompt_and_full_text(
            processor=processor,
            question=ex["question"],
            answer=ex["answer"],
            with_image=False,
        )

        prompt_inputs = processor(
            text=[prompt_text],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_lengths.append(prompt_inputs["input_ids"].shape[1])
        full_texts.append(full_text)

    batch = processor(
        text=full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if log_lengths:
        print(f"[collate_um] batch={len(examples)} seq_len={batch['input_ids'].shape[1]}")

    batch["labels"] = _make_labels_from_prompt_lengths(
        batch["input_ids"],
        prompt_lengths,
        processor.tokenizer.pad_token_id,
    )
    return batch["input_ids"], batch["attention_mask"], None, batch["labels"]


# =============================================================================
# Model / Processor
# =============================================================================

def load_model_and_processor(args):
    print(f"[Init] Loading base model from: {args.base_model_dir}")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    print(f"[Init] Loading LoRA from: {args.lora_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_dir,
        is_trainable=False,
        local_files_only=True,
    )
    model = model.merge_and_unload()
    print("[Init] Merged base model and LoRA into a full model for pruning")

    print(f"[Init] Loading processor from: {args.base_model_dir}")
    processor = AutoProcessor.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
    )

    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = model.config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# =============================================================================
# Activation Collector
# =============================================================================

class ActivationCollector:
    def __init__(self, freq_threshold=1e-1):
        self.freq_threshold = freq_threshold
        self.activations = {
            "multimodal": {},
            "unimodal": {},
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
                acts = out.float()
            elif out.dim() >= 3:
                acts = out.float().reshape(-1, out.shape[-1])
            else:
                return

            acts = acts.clamp(min=-1e3, max=1e3)

            self.activations[modality][layer_name] = {
                "abs_sum": acts.abs().sum(dim=0).cpu(),
                "freq_sum": (acts.abs() > self.freq_threshold).float().sum(dim=0).cpu(),
                "sq_sum": (acts ** 2).sum(dim=0).cpu(),
                "sum": acts.sum(dim=0).cpu(),
                "sumsq": (acts ** 2).sum(dim=0).cpu(),
                "count": acts.shape[0],
            }

        self.handles.append(module.register_forward_hook(hook))

    def clear(self):
        self.activations = {
            "multimodal": {},
            "unimodal": {},
        }

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =============================================================================
# Layer selection / hooks
# =============================================================================

def select_target_layers_llava(model, vision_last_n=3, text_last_n=3):
    targets = []
    vision_layers = model.vision_tower.vision_model.encoder.layers
    text_layers = model.language_model.model.layers

    for i in range(max(0, len(vision_layers) - vision_last_n), len(vision_layers)):
        layer = vision_layers[i]
        if hasattr(layer.mlp, "fc1"):
            targets.append(("multimodal", f"vision_fc1_{i}", layer.mlp.fc1))
        if hasattr(layer.mlp, "fc2"):
            targets.append(("multimodal", f"vision_fc2_{i}", layer.mlp.fc2))

    for i in range(max(0, len(text_layers) - text_last_n), len(text_layers)):
        layer = text_layers[i]
        if hasattr(layer.mlp, "gate_proj"):
            targets.append(("unimodal", f"lang_gate_proj_{i}", layer.mlp.gate_proj))
        if hasattr(layer.mlp, "up_proj"):
            targets.append(("unimodal", f"lang_up_proj_{i}", layer.mlp.up_proj))
        if hasattr(layer.mlp, "down_proj"):
            targets.append(("unimodal", f"lang_down_proj_{i}", layer.mlp.down_proj))

    return targets


def register_hooks_for_manu(model, collector, args):
    targets = select_target_layers_llava(model, args.vision_last_n, args.text_last_n)
    for modality, layer_name, module in targets:
        collector.register_hook(module, layer_name, modality)
    return [x[1] for x in targets]


# =============================================================================
# Forward / device
# =============================================================================

def invoke(batch, model, mode):
    if mode == "multimodal":
        input_ids, attention_mask, pixel_values, labels = batch
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
    if mode == "unimodal":
        input_ids, attention_mask, _, labels = batch
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    raise ValueError(mode)


def _to_device_batch(batch, device):
    out = []
    for x in batch:
        if torch.is_tensor(x):
            out.append(x.to(device, non_blocking=True))
        else:
            out.append(x)
    return tuple(out)


# =============================================================================
# Score collection
# =============================================================================

@torch.no_grad()
def collect_importance_scores(
    model,
    dataloader,
    collector,
    modality,
    device,
    max_batches=None,
    log_interval=50,
):
    model.eval()
    collector.clear()

    stats = defaultdict(dict)
    total_steps = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Collect {modality}", total=total_steps)):
        if max_batches is not None and step >= max_batches:
            break

        batch = _to_device_batch(batch, device)
        _ = invoke(batch, model, modality)

        if log_interval > 0 and step % log_interval == 0:
            print(f"[{modality}] step={step} seq_len={batch[0].shape[1]}")

        for layer_name, cur in collector.activations[modality].items():
            if layer_name not in stats:
                stats[layer_name] = {
                    "abs_sum": cur["abs_sum"].clone(),
                    "freq_sum": cur["freq_sum"].clone(),
                    "sq_sum": cur["sq_sum"].clone(),
                    "sum": cur["sum"].clone(),
                    "sumsq": cur["sumsq"].clone(),
                    "count": cur["count"],
                }
            else:
                stats[layer_name]["abs_sum"] += cur["abs_sum"]
                stats[layer_name]["freq_sum"] += cur["freq_sum"]
                stats[layer_name]["sq_sum"] += cur["sq_sum"]
                stats[layer_name]["sum"] += cur["sum"]
                stats[layer_name]["sumsq"] += cur["sumsq"]
                stats[layer_name]["count"] += cur["count"]

        collector.activations[modality] = {}

    final_scores = defaultdict(dict)
    for layer_name, s in stats.items():
        n = max(s["count"], 1)
        mean = s["sum"] / n
        var = (s["sumsq"] / n) - mean.pow(2)
        var = torch.clamp(var, min=0.0)

        final_scores["I_abs"][layer_name] = s["abs_sum"] / n
        final_scores["I_freq"][layer_name] = s["freq_sum"] / n
        final_scores["I_var"][layer_name] = torch.sqrt(var)
        final_scores["I_rms"][layer_name] = torch.sqrt(s["sq_sum"] / n)

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


# =============================================================================
# Pruning
# =============================================================================

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
        f"Mask size {mask_1d.numel()} incompatible with weight shape {tuple(layer.weight.shape)}"
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


# =============================================================================
# Data
# =============================================================================

def build_dataloaders(args, processor):
    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")

    forget_file = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_file = os.path.join(retain_folder, "train-00000-of-00001.parquet")

    print(f"[Data] Forget file: {forget_file}")
    print(f"[Data] Retain file: {retain_file}")

    forget_df = pd.read_parquet(forget_file)
    retain_df = pd.read_parquet(retain_file)

    forget_mm = MultimodalDataset(forget_df)
    forget_um = UnimodalDataset(forget_df)
    retain_mm = MultimodalDataset(retain_df)
    retain_um = UnimodalDataset(retain_df)

    print(f"[Data] forget_mm samples: {len(forget_mm)}")
    print(f"[Data] forget_um samples: {len(forget_um)}")
    print(f"[Data] retain_mm samples: {len(retain_mm)}")
    print(f"[Data] retain_um samples: {len(retain_um)}")

    common_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available() and args.pin_memory,
    }

    if args.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = args.persistent_workers

    return {
        "forget_mm": DataLoader(
            forget_mm,
            collate_fn=lambda x: collate_multimodal(
                x, processor, args.mm_max_length, args.log_batch_lengths
            ),
            **common_loader_kwargs,
        ),
        "forget_um": DataLoader(
            forget_um,
            collate_fn=lambda x: collate_unimodal(
                x, processor, args.max_length, args.log_batch_lengths
            ),
            **common_loader_kwargs,
        ),
        "retain_mm": DataLoader(
            retain_mm,
            collate_fn=lambda x: collate_multimodal(
                x, processor, args.mm_max_length, args.log_batch_lengths
            ),
            **common_loader_kwargs,
        ),
        "retain_um": DataLoader(
            retain_um,
            collate_fn=lambda x: collate_unimodal(
                x, processor, args.max_length, args.log_batch_lengths
            ),
            **common_loader_kwargs,
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    print(datetime.now())
    print("\n===== Configuration =====")
    print("Command:")
    print(" ".join(os.sys.argv))
    print("\nArguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=========================\n")

    model, processor = load_model_and_processor(args)
    dataloaders = build_dataloaders(args, processor)

    device = next(model.parameters()).device
    print(f"[Init] Main device: {device}")

    collector = ActivationCollector(freq_threshold=args.freq_threshold)
    selected_layers = register_hooks_for_manu(model, collector, args)

    print("[Init] Selected layers:")
    for x in selected_layers:
        print(" ", x)

    print("\n[1/4] Collect forget multimodal scores")
    forget_mm_scores = collect_importance_scores(
        model,
        dataloaders["forget_mm"],
        collector,
        "multimodal",
        device,
        args.max_batches,
        args.log_interval,
    )

    print("\n[2/4] Collect forget unimodal scores")
    forget_um_scores = collect_importance_scores(
        model,
        dataloaders["forget_um"],
        collector,
        "unimodal",
        device,
        args.max_batches,
        args.log_interval,
    )

    print("\n[3/4] Collect retain multimodal scores")
    retain_mm_scores = collect_importance_scores(
        model,
        dataloaders["retain_mm"],
        collector,
        "multimodal",
        device,
        args.max_batches,
        args.log_interval,
    )

    print("\n[4/4] Collect retain unimodal scores")
    retain_um_scores = collect_importance_scores(
        model,
        dataloaders["retain_um"],
        collector,
        "unimodal",
        device,
        args.max_batches,
        args.log_interval,
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
        forget_scores,
        retain_scores,
        weights,
        args.epsilon,
    )
    if not combined_scores:
        raise RuntimeError("No combined scores produced. Check hooks and dataset loading.")

    pruning_masks, threshold, k, total = compute_top_k_pruning_mask(
        combined_scores,
        args.prune_percent,
    )
    applied_layers = apply_structural_pruning_llava(model, pruning_masks)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)

    metadata = {
        "base_model_dir": args.base_model_dir,
        "lora_dir": args.lora_dir,
        "forget_split_ratio": args.forget_split_ratio,
        "prune_percent": args.prune_percent,
        "vision_last_n": args.vision_last_n,
        "text_last_n": args.text_last_n,
        "threshold": threshold,
        "num_pruned_neurons": k,
        "num_candidate_neurons": total,
        "applied_layers": applied_layers,
        "weights": weights,
        "mm_max_length": args.mm_max_length,
        "um_max_length": args.max_length,
    }
    print(json.dumps(metadata, indent=2))
    print("=== MANU finished ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MANU pruning baseline (LLaVA only)")

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
        help="LoRA directory to merge into the base model before pruning.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the pruned model.",
    )

    parser.add_argument(
        "--data_split_dir",
        type=str,
        required=True,
        help="Directory containing forget_xx and retain_xx parquet splits.",
    )
    parser.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        help="Forget ratio used to locate the split folders.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for activation collection.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Unimodal max length.",
    )
    parser.add_argument(
        "--mm_max_length",
        type=int,
        default=1024,
        help="Multimodal max length.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to collect per dataloader. None means all batches.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pin_memory in dataloaders.",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Enable persistent_workers when num_workers > 0.",
    )

    parser.add_argument(
        "--prune_percent",
        type=float,
        default=10.0,
        help="Global top-k pruning percentage.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
        help="Small constant used in score normalization.",
    )
    parser.add_argument(
        "--freq_threshold",
        type=float,
        default=1e-1,
        help="Activation threshold used for frequency statistics.",
    )

    parser.add_argument(
        "--vision_last_n",
        type=int,
        default=3,
        help="Number of last vision layers to collect.",
    )
    parser.add_argument(
        "--text_last_n",
        type=int,
        default=3,
        help="Number of last language layers to collect.",
    )

    parser.add_argument(
        "--weight_abs",
        type=float,
        default=2.0,
        help="Weight for I_abs score.",
    )
    parser.add_argument(
        "--weight_freq",
        type=float,
        default=0.0,
        help="Weight for I_freq score.",
    )
    parser.add_argument(
        "--weight_var",
        type=float,
        default=2.0,
        help="Weight for I_var score.",
    )
    parser.add_argument(
        "--weight_rms",
        type=float,
        default=2.0,
        help="Weight for I_rms score.",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Logging interval during activation collection.",
    )
    parser.add_argument(
        "--log_batch_lengths",
        action="store_true",
        help="Log batch sequence lengths in collate.",
    )

    args = parser.parse_args()
    main(args)

