import os
import sys
import math
import argparse
from itertools import cycle
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    get_scheduler,
    LlavaForConditionalGeneration,
)
from peft import PeftModel

from unlearn_dataset import (
    Multimodal_Dataset,
    Unimodal_Dataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
)

sys.path.append("../")
sys.path.append("../../")


def load_model_and_processor(args):
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    print(f"[Init] Loading finetuned LoRA from: {args.lora_dir}")
    model = PeftModel.from_pretrained(
        model,
        args.lora_dir,
        is_trainable=True,
        local_files_only=True,
    )
    print("[Init] Loaded base model + finetuned LoRA as the pre-unlearning model")

    processor = AutoProcessor.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
    )
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


def main(args):
    print(datetime.now())

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)

    accelerator.print("\n===== Configuration =====")
    accelerator.print("Command:")
    accelerator.print(" ".join(sys.argv))
    accelerator.print("\nArguments:")
    for k, v in sorted(vars(args).items()):
        accelerator.print(f"  {k}: {v}")
    accelerator.print("=========================\n")

    model, processor = load_model_and_processor(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
    )

    accelerator.print(f"[Init] Tokenizer length: {len(tokenizer)}")

    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        accelerator.print("[Init] Resizing embeddings to match tokenizer size")
        model.resize_token_embeddings(len(tokenizer))

    model.print_trainable_parameters()
    accelerator.print(f"[Init] Model type: {'PEFT' if isinstance(model, PeftModel) else 'Base'}")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")

    forget_parquet_file = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, "train-00000-of-00001.parquet")

    accelerator.print(f"[Data] Loading forget split from: {forget_parquet_file}")
    accelerator.print(f"[Data] Loading retain split from: {retain_parquet_file}")

    df_forget = pd.read_parquet(forget_parquet_file)
    df_retain = pd.read_parquet(retain_parquet_file)

    accelerator.print(f"[Data] Forget samples: {len(df_forget)}")
    accelerator.print(f"[Data] Retain samples: {len(df_retain)}")

    forget_multimodal_dataset = Multimodal_Dataset(df=df_forget)
    forget_unimodal_dataset = Unimodal_Dataset(df=df_forget)
    retain_multimodal_dataset = Multimodal_Dataset(df=df_retain)
    retain_unimodal_dataset = Unimodal_Dataset(df=df_retain)

    train_dataloader_forget_multimodal = DataLoader(
        forget_multimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args),
    )
    train_dataloader_forget_unimodal = DataLoader(
        forget_unimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args),
    )
    train_dataloader_retain_multimodal = DataLoader(
        retain_multimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args),
    )
    train_dataloader_retain_unimodal = DataLoader(
        retain_unimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    n_update_steps = (
        math.ceil(len(train_dataloader_forget_multimodal) / args.grad_accum_steps)
        * args.num_epochs
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_update_steps,
    )

    (
        model,
        optimizer,
        train_dataloader_forget_multimodal,
        train_dataloader_forget_unimodal,
        train_dataloader_retain_multimodal,
        train_dataloader_retain_unimodal,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader_forget_multimodal,
        train_dataloader_forget_unimodal,
        train_dataloader_retain_multimodal,
        train_dataloader_retain_unimodal,
        lr_scheduler,
    )

    retain_multimodal_iter = cycle(train_dataloader_retain_multimodal)
    retain_unimodal_iter = cycle(train_dataloader_retain_unimodal)

    accelerator.print(
        f"\n===== GD Training =====\n"
        f"[Train] retain_steps_per_forget={args.retain_steps_per_forget}, "
        f"grad_accum_steps={args.grad_accum_steps}, "
        f"clip_grad_norm={args.clip_grad_norm}\n"
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        n_optimizer_step = 0
        total_mini = len(train_dataloader_forget_multimodal)

        progress_bar = tqdm(
            zip(train_dataloader_forget_multimodal, train_dataloader_forget_unimodal),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            total=total_mini,
            disable=not accelerator.is_main_process,
        )

        for mini_step, (forget_multi_batch, forget_uni_batch) in enumerate(progress_bar):
            is_last_mini = (mini_step + 1 == total_mini)
            is_accum_step = ((mini_step + 1) % args.grad_accum_steps == 0) or is_last_mini

            outputs_f_m = invoke(forget_multi_batch, model, "multimodal")
            loss_f_m = outputs_f_m.loss
            accelerator.backward(-loss_f_m / args.grad_accum_steps)

            retain_loss_m = 0.0
            for _ in range(args.retain_steps_per_forget):
                outputs_r_m = invoke(next(retain_multimodal_iter), model, "multimodal")
                loss_r_m = outputs_r_m.loss
                accelerator.backward(
                    (loss_r_m / args.retain_steps_per_forget) / args.grad_accum_steps
                )
                retain_loss_m += loss_r_m.item()

            retain_loss_m /= max(1, args.retain_steps_per_forget)

            outputs_f_u = invoke(forget_uni_batch, model, "unimodal")
            loss_f_u = outputs_f_u.loss
            accelerator.backward(-loss_f_u / args.grad_accum_steps)

            retain_loss_u = 0.0
            for _ in range(args.retain_steps_per_forget):
                outputs_r_u = invoke(next(retain_unimodal_iter), model, "unimodal")
                loss_r_u = outputs_r_u.loss
                accelerator.backward(
                    (loss_r_u / args.retain_steps_per_forget) / args.grad_accum_steps
                )
                retain_loss_u += loss_r_u.item()

            retain_loss_u /= max(1, args.retain_steps_per_forget)

            step_loss = (
                loss_f_m.item()
                + loss_f_u.item()
                + retain_loss_m
                + retain_loss_u
            )
            total_loss += step_loss

            if is_accum_step:
                actual_steps = (mini_step + 1) % args.grad_accum_steps
                if actual_steps != 0:
                    correction = args.grad_accum_steps / actual_steps
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
                n_optimizer_step += 1

                progress_bar.set_postfix(
                    {
                        "forget_m": f"{loss_f_m.item():.4f}",
                        "forget_u": f"{loss_f_u.item():.4f}",
                        "retain_m": f"{retain_loss_m:.4f}",
                        "retain_u": f"{retain_loss_u:.4f}",
                        "avg": f"{total_loss / (mini_step + 1):.4f}",
                        "opt": n_optimizer_step,
                    }
                )

        avg_loss = total_loss / total_mini
        accelerator.print(
            f"[Train] Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, optimizer_steps={n_optimizer_step}"
        )

        accelerator.wait_for_everyone()
        epoch_save_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_save_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(epoch_save_dir)
        accelerator.print(f"[Save] Checkpoint saved to {epoch_save_dir}")

    accelerator.print("[Train] Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GD unlearning baseline (LLaVA only)")

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
        help=(
            "Finetuned LoRA directory. It is mounted on top of base_model_dir "
            "to form the pre-unlearning model."
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
        help="Training batch size.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
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
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Reserved sequence length argument. Kept for compatibility.",
    )
    parser.add_argument(
        "--retain_steps_per_forget",
        type=int,
        default=1,
        help="Number of retain steps after each forget step.",
    )

    args = parser.parse_args()
    main(args)

