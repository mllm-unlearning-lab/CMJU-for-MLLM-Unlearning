import os
import sys
import math
import argparse
from itertools import cycle
from datetime import datetime

import torch
import torch.nn.functional as F
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


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    accelerator.print(f"\n[{datetime.now()}] NPO")
    accelerator.print("Command: " + " ".join(sys.argv))
    for k, v in sorted(vars(args).items()):
        accelerator.print(f"  {k}: {v}")

    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    vocab_size = len(processor.tokenizer)

    accelerator.print(f"[Model] Loading student = base + LoRA (trainable): {args.lora_dir}")
    student_base = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if vocab_size > student_base.get_input_embeddings().weight.shape[0]:
        student_base.resize_token_embeddings(vocab_size)

    student = PeftModel.from_pretrained(student_base, args.lora_dir, is_trainable=True)
    for n, p in student.named_parameters():
        p.requires_grad_(is_llm_lora_param(n))
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable()
    student.print_trainable_parameters()

    accelerator.print(f"[Model] Loading oracle = base + LoRA (frozen): {args.lora_dir}")
    oracle_base = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if vocab_size > oracle_base.get_input_embeddings().weight.shape[0]:
        oracle_base.resize_token_embeddings(vocab_size)

    oracle = PeftModel.from_pretrained(oracle_base, args.lora_dir, is_trainable=False)
    oracle.eval()
    for p in oracle.parameters():
        p.requires_grad_(False)

    retain_ratio = 100 - args.forget_split_ratio
    forget_path = os.path.join(args.data_dir, f"forget{args.forget_split_ratio}+tofu")
    retain_path = os.path.join(args.data_dir, f"retain{retain_ratio}+tofu")
    accelerator.print(f"[Data] forget path: {forget_path}")

    forget_raw = load_dataset(forget_path, split="train")
    accelerator.print(f"[Data] forget raw={len(forget_raw)}")

    forget_caption_ds = CLEARDataset(forget_raw, mode=CAPTION_MODE)
    forget_text_ds = CLEARDataset(forget_raw, mode=TEXT_QA_MODE)

    accelerator.print(
        f"[Data] forget  CAPTION={len(forget_caption_ds)}, TEXT_QA={len(forget_text_ds)}"
    )
    for name, ds in [
        ("forget CAPTION", forget_caption_ds),
        ("forget TEXT_QA", forget_text_ds),
    ]:
        if len(ds) == 0:
            raise ValueError(f"[Data] {name} is empty.")

    col_caption = make_collate_fn(processor, mode="multimodal")
    col_text = make_collate_fn(processor, mode="unimodal")

    dl_f_caption = DataLoader(
        forget_caption_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=col_caption,
    )
    dl_f_text = DataLoader(
        forget_text_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=col_text,
    )

    retain_caption_iter = None
    retain_text_iter = None
    dl_r_caption = None
    dl_r_text = None

    if args.use_retain:
        accelerator.print(f"[Data] retain path: {retain_path}")
        retain_raw = load_dataset(retain_path, split="train")
        accelerator.print(f"[Data] retain raw={len(retain_raw)}")

        retain_caption_ds = CLEARDataset(retain_raw, mode=CAPTION_MODE)
        retain_text_ds = CLEARDataset(retain_raw, mode=TEXT_QA_MODE)

        accelerator.print(
            f"[Data] retain  CAPTION={len(retain_caption_ds)}, TEXT_QA={len(retain_text_ds)}"
        )
        for name, ds in [
            ("retain CAPTION", retain_caption_ds),
            ("retain TEXT_QA", retain_text_ds),
        ]:
            if len(ds) == 0:
                raise ValueError(f"[Data] {name} is empty.")

        dl_r_caption = DataLoader(
            retain_caption_ds,
            args.batch_size,
            shuffle=True,
            collate_fn=col_caption,
        )
        dl_r_text = DataLoader(
            retain_text_ds,
            args.batch_size,
            shuffle=True,
            collate_fn=col_text,
        )

        accelerator.print(
            f"[Retain] Enabled. retain_steps_per_forget={args.retain_steps_per_forget}"
        )
    else:
        accelerator.print("[Retain] Disabled. Using forget set only.")

    optimizer = AdamW(student.parameters(), lr=args.lr)
    total_mini = len(dl_f_caption)
    n_steps = math.ceil(total_mini / args.grad_accum_steps) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_steps,
    )

    if args.use_retain:
        (
            student,
            oracle,
            optimizer,
            dl_f_caption,
            dl_f_text,
            dl_r_caption,
            dl_r_text,
            lr_scheduler,
        ) = accelerator.prepare(
            student,
            oracle,
            optimizer,
            dl_f_caption,
            dl_f_text,
            dl_r_caption,
            dl_r_text,
            lr_scheduler,
        )
        retain_caption_iter = cycle(dl_r_caption)
        retain_text_iter = cycle(dl_r_text)
    else:
        (
            student,
            oracle,
            optimizer,
            dl_f_caption,
            dl_f_text,
            lr_scheduler,
        ) = accelerator.prepare(
            student,
            oracle,
            optimizer,
            dl_f_caption,
            dl_f_text,
            lr_scheduler,
        )

    forget_text_iter = cycle(dl_f_text)

    accelerator.print(
        f"\n===== NPO Training (use_retain={args.use_retain}, "
        f"grad_accum_steps={args.grad_accum_steps}, "
        f"clip_norm={args.clip_grad_norm}) =====\n"
    )

    for epoch in range(args.num_epochs):
        student.train()
        total_log = 0.0
        n_opt = 0

        bar = tqdm(
            dl_f_caption,
            total=total_mini,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for mini_step, caption_f in enumerate(bar):
            text_f = next(forget_text_iter)
            is_last = (mini_step + 1 == total_mini)
            is_accum = ((mini_step + 1) % args.grad_accum_steps == 0) or is_last

            out_s_m = _forward(caption_f, student, "multimodal")
            loss_s_m = out_s_m.loss
            with torch.no_grad():
                out_o_m = _forward(caption_f, oracle, "multimodal")
                loss_o_m = out_o_m.loss
            neg_log_ratios_m = loss_s_m - loss_o_m
            loss_npo_m = -F.logsigmoid(args.beta * neg_log_ratios_m).mean() * 2.0 / args.beta
            accelerator.backward(loss_npo_m / args.grad_accum_steps)
            del out_s_m, out_o_m

            out_s_u = _forward(text_f, student, "unimodal")
            loss_s_u = out_s_u.loss
            with torch.no_grad():
                out_o_u = _forward(text_f, oracle, "unimodal")
                loss_o_u = out_o_u.loss
            neg_log_ratios_u = loss_s_u - loss_o_u
            loss_npo_u = -F.logsigmoid(args.beta * neg_log_ratios_u).mean() * 2.0 / args.beta
            accelerator.backward(loss_npo_u / args.grad_accum_steps)
            del out_s_u, out_o_u

            retain_log_m = 0.0
            retain_log_u = 0.0
            if args.use_retain:
                for _ in range(args.retain_steps_per_forget):
                    r_cap = next(retain_caption_iter)
                    out_rm = _forward(r_cap, student, "multimodal")
                    r_loss_m = out_rm.loss
                    accelerator.backward(
                        (r_loss_m / args.retain_steps_per_forget) / args.grad_accum_steps
                    )
                    retain_log_m += r_loss_m.item()
                    del out_rm

                for _ in range(args.retain_steps_per_forget):
                    r_txt = next(retain_text_iter)
                    out_ru = _forward(r_txt, student, "unimodal")
                    r_loss_u = out_ru.loss
                    accelerator.backward(
                        (r_loss_u / args.retain_steps_per_forget) / args.grad_accum_steps
                    )
                    retain_log_u += r_loss_u.item()
                    del out_ru

                retain_log_m /= max(1, args.retain_steps_per_forget)
                retain_log_u /= max(1, args.retain_steps_per_forget)

            step_log = loss_npo_m.item() + loss_npo_u.item()
            if args.use_retain:
                step_log += retain_log_m + retain_log_u
            total_log += step_log

            if is_accum:
                actual = (mini_step + 1) % args.grad_accum_steps
                if actual != 0:
                    corr = args.grad_accum_steps / actual
                    for p in accelerator.unwrap_model(student).parameters():
                        if p.grad is not None:
                            p.grad.mul_(corr)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in accelerator.unwrap_model(student).parameters() if p.requires_grad],
                    max_norm=args.clip_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                n_opt += 1

            postfix = {
                "avg": f"{total_log / (mini_step + 1):.4f}",
                "opt": n_opt,
                "npo_m": f"{loss_npo_m.item():.3f}",
                "npo_u": f"{loss_npo_u.item():.3f}",
            }
            if args.use_retain:
                postfix.update({
                    "r_m": f"{retain_log_m:.3f}",
                    "r_u": f"{retain_log_u:.3f}",
                })
            bar.set_postfix(postfix)

        accelerator.print(
            f"Epoch {epoch + 1} | avg_log={total_log / total_mini:.4f} | opt_steps={n_opt}"
        )

        accelerator.wait_for_everyone()
        save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        accelerator.unwrap_model(student).save_pretrained(save_path)
        accelerator.print(f"[Save] -> {save_path}")

    accelerator.print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser("NPO")

    p.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Base LLaVA model directory.",
    )
    p.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Pre-unlearning LoRA adapter directory used by both student and oracle.",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save unlearned checkpoints.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="CLEAR data root containing forget{n}+tofu and retain{100-n}+tofu.",
    )

    p.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        help="Forget split integer used to build split paths.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    p.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="NPO temperature parameter.",
    )

    p.add_argument(
        "--use_retain",
        type=int,
        default=0,
        help="1 to enable retain CE, 0 to disable.",
    )
    p.add_argument(
        "--retain_steps_per_forget",
        type=int,
        default=1,
        help="Number of retain CE steps after each forget step.",
    )

    main(p.parse_args())
