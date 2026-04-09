import os
import sys
import math
import argparse
from itertools import cycle
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
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


def load_llava_base(model_path: str):
    return LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


def attach_lora(model, lora_dir: str, is_trainable: bool):
    return PeftModel.from_pretrained(
        model,
        lora_dir,
        is_trainable=is_trainable,
        local_files_only=True,
    )


def load_processor(base_model_dir: str):
    processor = AutoProcessor.from_pretrained(
        base_model_dir,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return processor


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


def npo_loss(student_loss, oracle_loss, beta):
    neg_log_ratio = student_loss - oracle_loss
    return -F.logsigmoid(beta * neg_log_ratio).mean() * 2.0 / beta


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

    processor = load_processor(args.base_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
    )
    accelerator.print(f"[Init] Tokenizer length: {len(tokenizer)}")

    accelerator.print(f"[Init] Loading STUDENT = base + LoRA from: {args.lora_dir}")
    student_base = load_llava_base(args.base_model_dir)
    student = attach_lora(student_base, args.lora_dir, is_trainable=True)

    accelerator.print(f"[Init] Loading ORACLE = base + LoRA from: {args.lora_dir}")
    oracle_base = load_llava_base(args.base_model_dir)
    oracle = attach_lora(oracle_base, args.lora_dir, is_trainable=False)
    oracle.eval()
    for p in oracle.parameters():
        p.requires_grad_(False)

    student.resize_token_embeddings(len(processor.tokenizer))
    oracle.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > student.get_input_embeddings().weight.shape[0]:
        accelerator.print("[Init] Resizing student embedding matrix to match tokenizer size")
        student.resize_token_embeddings(len(tokenizer))

    try:
        student.enable_input_require_grads()
    except Exception:
        pass

    try:
        student.gradient_checkpointing_enable()
        accelerator.print("[Init] Student gradient checkpointing enabled")
    except Exception:
        accelerator.print("[Init] Student gradient checkpointing not enabled")

    student.print_trainable_parameters()
    accelerator.print(
        f"[Init] Student model type: {'PEFT' if isinstance(student, PeftModel) else 'Base'}"
    )

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")

    forget_parquet_file = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, "train-00000-of-00001.parquet")

    accelerator.print(f"[Data] Loading forget split from: {forget_parquet_file}")
    df_forget = pd.read_parquet(forget_parquet_file)
    accelerator.print(f"[Data] Forget samples: {len(df_forget)}")

    forget_multimodal_dataset = Multimodal_Dataset(df=df_forget)
    forget_unimodal_dataset = Unimodal_Dataset(df=df_forget)

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

    train_dataloader_retain_multimodal = None
    train_dataloader_retain_unimodal = None

    if args.use_retain:
        accelerator.print(f"[Data] Loading retain split from: {retain_parquet_file}")
        df_retain = pd.read_parquet(retain_parquet_file)
        accelerator.print(f"[Data] Retain samples: {len(df_retain)}")

        retain_multimodal_dataset = Multimodal_Dataset(df=df_retain)
        retain_unimodal_dataset = Unimodal_Dataset(df=df_retain)

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

        accelerator.print(
            f"[Retain] Enabled: retain_steps_per_forget={args.retain_steps_per_forget}, "
            f"lambda_retain={args.lambda_retain}, "
            f"lambda_retain_uni={args.lambda_retain_uni}"
        )
    else:
        accelerator.print("[Retain] Disabled: original NPO behavior (forget-only)")

    optimizer = AdamW(student.parameters(), lr=args.lr)

    total_mini = len(train_dataloader_forget_multimodal)
    n_update_steps = math.ceil(total_mini / args.grad_accum_steps) * args.num_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_update_steps,
    )

    if args.use_retain:
        (
            student,
            oracle,
            optimizer,
            train_dataloader_forget_multimodal,
            train_dataloader_forget_unimodal,
            train_dataloader_retain_multimodal,
            train_dataloader_retain_unimodal,
            lr_scheduler,
        ) = accelerator.prepare(
            student,
            oracle,
            optimizer,
            train_dataloader_forget_multimodal,
            train_dataloader_forget_unimodal,
            train_dataloader_retain_multimodal,
            train_dataloader_retain_unimodal,
            lr_scheduler,
        )
        retain_multimodal_iter = cycle(train_dataloader_retain_multimodal)
        retain_unimodal_iter = cycle(train_dataloader_retain_unimodal)
    else:
        (
            student,
            oracle,
            optimizer,
            train_dataloader_forget_multimodal,
            train_dataloader_forget_unimodal,
            lr_scheduler,
        ) = accelerator.prepare(
            student,
            oracle,
            optimizer,
            train_dataloader_forget_multimodal,
            train_dataloader_forget_unimodal,
            lr_scheduler,
        )
        retain_multimodal_iter = None
        retain_unimodal_iter = None

    accelerator.print(
        f"\n===== NPO Training =====\n"
        f"[Train] use_retain={args.use_retain}, "
        f"beta={args.beta}, "
        f"grad_accum_steps={args.grad_accum_steps}, "
        f"clip_grad_norm={args.clip_grad_norm}\n"
    )

    for epoch in range(args.num_epochs):
        student.train()
        total_log = 0.0
        n_optimizer_step = 0

        progress_bar = tqdm(
            zip(train_dataloader_forget_multimodal, train_dataloader_forget_unimodal),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            total=total_mini,
            disable=not accelerator.is_main_process,
        )

        for mini_step, (forget_multi_batch, forget_uni_batch) in enumerate(progress_bar):
            is_last_mini = (mini_step + 1 == total_mini)
            is_accum_step = ((mini_step + 1) % args.grad_accum_steps == 0) or is_last_mini

            # Multimodal NPO forget loss
            outputs_s_m = invoke(forget_multi_batch, student, "multimodal")
            student_loss_m = outputs_s_m.loss
            with torch.no_grad():
                outputs_o_m = invoke(forget_multi_batch, oracle, "multimodal")
                oracle_loss_m = outputs_o_m.loss

            loss_multi = npo_loss(student_loss_m, oracle_loss_m, args.beta)
            accelerator.backward(loss_multi / args.grad_accum_steps)
            del outputs_s_m, outputs_o_m

            # Unimodal NPO forget loss
            outputs_s_u = invoke(forget_uni_batch, student, "unimodal")
            student_loss_u = outputs_s_u.loss
            with torch.no_grad():
                outputs_o_u = invoke(forget_uni_batch, oracle, "unimodal")
                oracle_loss_u = outputs_o_u.loss

            loss_uni = npo_loss(student_loss_u, oracle_loss_u, args.beta)
            accelerator.backward(loss_uni / args.grad_accum_steps)
            del outputs_s_u, outputs_o_u

            retain_log_m = 0.0
            retain_log_u = 0.0

            if args.use_retain:
                for _ in range(args.retain_steps_per_forget):
                    retain_multi_batch = next(retain_multimodal_iter)
                    outputs_r_m = invoke(retain_multi_batch, student, "multimodal")
                    retain_loss_m = outputs_r_m.loss
                    accelerator.backward(
                        (args.lambda_retain * retain_loss_m / args.retain_steps_per_forget)
                        / args.grad_accum_steps
                    )
                    retain_log_m += retain_loss_m.item()
                    del outputs_r_m

                for _ in range(args.retain_steps_per_forget):
                    retain_uni_batch = next(retain_unimodal_iter)
                    outputs_r_u = invoke(retain_uni_batch, student, "unimodal")
                    retain_loss_u = outputs_r_u.loss
                    accelerator.backward(
                        (args.lambda_retain_uni * retain_loss_u / args.retain_steps_per_forget)
                        / args.grad_accum_steps
                    )
                    retain_log_u += retain_loss_u.item()
                    del outputs_r_u

                retain_log_m /= max(1, args.retain_steps_per_forget)
                retain_log_u /= max(1, args.retain_steps_per_forget)

            step_log = loss_multi.item() + loss_uni.item()
            if args.use_retain:
                step_log += args.lambda_retain * retain_log_m + args.lambda_retain_uni * retain_log_u
            total_log += step_log

            if is_accum_step:
                actual_steps = (mini_step + 1) % args.grad_accum_steps
                if actual_steps != 0:
                    correction = args.grad_accum_steps / actual_steps
                    for p in accelerator.unwrap_model(student).parameters():
                        if p.grad is not None:
                            p.grad.mul_(correction)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in accelerator.unwrap_model(student).parameters() if p.requires_grad],
                    max_norm=args.clip_grad_norm,
                )

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                n_optimizer_step += 1

                postfix = {
                    "forget_m": f"{loss_multi.item():.4f}",
                    "forget_u": f"{loss_uni.item():.4f}",
                    "avg": f"{total_log / (mini_step + 1):.4f}",
                    "opt": n_optimizer_step,
                }
                if args.use_retain:
                    postfix.update(
                        {
                            "retain_m": f"{retain_log_m:.4f}",
                            "retain_u": f"{retain_log_u:.4f}",
                        }
                    )
                progress_bar.set_postfix(postfix)

        accelerator.print(
            f"[Train] Epoch {epoch + 1}: avg_log={total_log / total_mini:.4f}, "
            f"optimizer_steps={n_optimizer_step}"
        )

        accelerator.wait_for_everyone()
        epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        accelerator.unwrap_model(student).save_pretrained(epoch_dir)
        accelerator.print(f"[Save] Checkpoint saved to {epoch_dir}")

    accelerator.print("[Train] Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPO unlearning baseline (LLaVA only)")

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
            "Pre-unlearning LoRA directory. It is loaded onto the base model for both "
            "the trainable student and the frozen oracle."
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
        default=15,
        help="Forget ratio used to locate the split folders.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Training batch size.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="NPO beta coefficient.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=6.2e-6,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Reserved sequence length argument. Kept for compatibility.",
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
        "--use_retain",
        type=int,
        default=0,
        help="1 to enable retain CE training, 0 to disable it.",
    )
    parser.add_argument(
        "--retain_steps_per_forget",
        type=int,
        default=1,
        help="Number of retain steps after each forget step when retain is enabled.",
    )
    parser.add_argument(
        "--lambda_retain",
        type=float,
        default=1.0,
        help="Multimodal retain CE weight.",
    )
    parser.add_argument(
        "--lambda_retain_uni",
        type=float,
        default=1.0,
        help="Unimodal retain CE weight.",
    )

    args = parser.parse_args()
    main(args)

