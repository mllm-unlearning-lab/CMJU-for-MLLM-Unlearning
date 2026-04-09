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


def kl_loss(prob_p, prob_q):
    return -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    accelerator.print(f"\n[{datetime.now()}] KL")
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
    accelerator.print(f"[Data] retain path: {retain_path}")

    forget_raw = load_dataset(forget_path, split="train")
    retain_raw = load_dataset(retain_path, split="train")
    accelerator.print(f"[Data] forget raw={len(forget_raw)} | retain raw={len(retain_raw)}")

    forget_caption_ds = CLEARDataset(forget_raw, mode=CAPTION_MODE)
    forget_text_ds = CLEARDataset(forget_raw, mode=TEXT_QA_MODE)
    retain_caption_ds = CLEARDataset(retain_raw, mode=CAPTION_MODE)
    retain_text_ds = CLEARDataset(retain_raw, mode=TEXT_QA_MODE)

    accelerator.print(
        f"[Data] forget  CAPTION={len(forget_caption_ds)}, TEXT_QA={len(forget_text_ds)}\n"
        f"[Data] retain  CAPTION={len(retain_caption_ds)}, TEXT_QA={len(retain_text_ds)}"
    )

    for name, ds in [
        ("forget CAPTION", forget_caption_ds),
        ("forget TEXT_QA", forget_text_ds),
        ("retain CAPTION", retain_caption_ds),
        ("retain TEXT_QA", retain_text_ds),
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

    optimizer = AdamW(student.parameters(), lr=args.lr)
    total_mini = len(dl_f_caption)
    n_steps = math.ceil(total_mini / args.grad_accum_steps) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_steps,
    )

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

    forget_text_iter = cycle(dl_f_text)
    retain_caption_iter = cycle(dl_r_caption)
    retain_text_iter = cycle(dl_r_text)

    accelerator.print(
        f"\n===== KL Training "
        f"(retain_steps_per_forget={args.retain_steps_per_forget}, "
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

            out_f_m = _forward(caption_f, student, "multimodal")
            forget_m = out_f_m.loss
            accelerator.backward((-forget_m) / args.grad_accum_steps)
            del out_f_m

            kl_m_log = 0.0
            for _ in range(args.retain_steps_per_forget):
                r_cap = next(retain_caption_iter)
                out_s = _forward(r_cap, student, "multimodal")
                with torch.no_grad():
                    out_t = _forward(r_cap, oracle, "multimodal")
                p = torch.softmax(out_s.logits, dim=-1)
                q = torch.softmax(out_t.logits, dim=-1)
                kl_m = kl_loss(p, q)
                accelerator.backward(
                    (kl_m / args.retain_steps_per_forget) / args.grad_accum_steps
                )
                kl_m_log += kl_m.item()
                del out_s, out_t, p, q
            kl_m_log /= max(1, args.retain_steps_per_forget)

            out_f_u = _forward(text_f, student, "unimodal")
            forget_u = out_f_u.loss
            accelerator.backward((-forget_u) / args.grad_accum_steps)
            del out_f_u

            kl_u_log = 0.0
            for _ in range(args.retain_steps_per_forget):
                r_txt = next(retain_text_iter)
                out_s = _forward(r_txt, student, "unimodal")
                with torch.no_grad():
                    out_t = _forward(r_txt, oracle, "unimodal")
                p = torch.softmax(out_s.logits, dim=-1)
                q = torch.softmax(out_t.logits, dim=-1)
                kl_u = kl_loss(p, q)
                accelerator.backward(
                    (kl_u / args.retain_steps_per_forget) / args.grad_accum_steps
                )
                kl_u_log += kl_u.item()
                del out_s, out_t, p, q
            kl_u_log /= max(1, args.retain_steps_per_forget)

            step_log = forget_m.item() + forget_u.item() + kl_m_log + kl_u_log
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

            bar.set_postfix(
                avg=f"{total_log / (mini_step + 1):.4f}",
                opt=n_opt,
                kl_m=f"{kl_m_log:.3f}",
                kl_u=f"{kl_u_log:.3f}",
                f_m=f"{forget_m.item():.3f}",
                f_u=f"{forget_u.item():.3f}",
            )

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
    p = argparse.ArgumentParser("KL")

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
    p.add_argument(
        "--retain_steps_per_forget",
        type=int,
        default=1,
        help="Number of retain KL steps after each forget step.",
    )
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    main(p.parse_args())
