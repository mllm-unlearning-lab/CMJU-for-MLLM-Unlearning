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
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    get_scheduler,
    LlavaForConditionalGeneration,
)
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel

from ft_dataset import (
    Multimodal_Dataset,
    Unimodal_Dataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
)

sys.path.append("../")
sys.path.append("../../")


LORA_TARGET_MODULES = (
    r"^language_model\.model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def load_model_and_processor(args):
    print(f"[Init] Loading base model from: {args.base_model_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    print(f"[Init] Loading processor from: {args.base_model_dir}")
    processor = AutoProcessor.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    return model, processor


def build_lora_model(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    return model


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
    accelerator.print(f"[Init] Processor tokenizer length: {len(processor.tokenizer)}")
    accelerator.print(f"[Init] Tokenizer length: {len(tokenizer)}")

    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        accelerator.print("[Init] Resizing embeddings to match tokenizer size")
        model.resize_token_embeddings(len(tokenizer))

    os.makedirs(args.save_dir, exist_ok=True)

    accelerator.print("[Init] Building PEFT model")
    model = build_lora_model(model)
    model.print_trainable_parameters()

    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    try:
        model.gradient_checkpointing_enable()
        accelerator.print("[Init] Gradient checkpointing enabled")
    except Exception:
        accelerator.print("[Init] Gradient checkpointing not enabled")

    accelerator.print(
        f"[Init] Model type: {'PEFT' if isinstance(model, PeftModel) else 'Base'}"
    )

    accelerator.print(f"[Data] Loading parquet from: {args.data_dir}")
    df = pd.read_parquet(args.data_dir)
    accelerator.print(f"[Data] Samples: {len(df)}")

    multimodal_dataset = Multimodal_Dataset(df=df)
    unimodal_dataset = Unimodal_Dataset(df=df)

    accelerator.print(f"[Data] Multimodal samples: {len(multimodal_dataset)}")
    accelerator.print(f"[Data] Unimodal samples: {len(unimodal_dataset)}")

    train_dataloader_multimodal = DataLoader(
        multimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args),
    )
    train_dataloader_unimodal = DataLoader(
        unimodal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_mini = len(train_dataloader_multimodal)
    n_update_steps = math.ceil(total_mini / args.grad_accum_steps) * args.num_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_update_steps,
    )

    (
        model,
        optimizer,
        train_dataloader_multimodal,
        train_dataloader_unimodal,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader_multimodal,
        train_dataloader_unimodal,
        lr_scheduler,
    )

    accelerator.print(
        f"\n===== Finetuning =====\n"
        f"[Train] batch_size={args.batch_size}, "
        f"grad_accum_steps={args.grad_accum_steps}, "
        f"lr={args.lr}, "
        f"num_epochs={args.num_epochs}\n"
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_m_loss = 0.0
        total_u_loss = 0.0

        progress_bar = tqdm(
            zip(train_dataloader_multimodal, train_dataloader_unimodal),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            total=total_mini,
            disable=not accelerator.is_main_process,
        )

        for step, (batch_m, batch_u) in enumerate(progress_bar):
            is_last_mini = (step + 1 == total_mini)
            is_accum_step = ((step + 1) % args.grad_accum_steps == 0) or is_last_mini

            outputs_m = invoke(batch_m, model, "multimodal")
            loss_m = outputs_m.loss
            accelerator.backward(loss_m / args.grad_accum_steps)
            total_m_loss += loss_m.item()
            del outputs_m

            outputs_u = invoke(batch_u, model, "unimodal")
            loss_u = outputs_u.loss
            accelerator.backward(loss_u / args.grad_accum_steps)
            total_u_loss += loss_u.item()
            del outputs_u

            step_loss = loss_m.item() + loss_u.item()
            total_loss += step_loss

            if is_accum_step:
                actual_steps = (step + 1) % args.grad_accum_steps
                if actual_steps != 0:
                    correction = args.grad_accum_steps / actual_steps
                    for p in accelerator.unwrap_model(model).parameters():
                        if p.grad is not None:
                            p.grad.mul_(correction)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            avg_loss = total_loss / (step + 1)
            avg_m_loss = total_m_loss / (step + 1)
            avg_u_loss = total_u_loss / (step + 1)

            progress_bar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "mm": f"{avg_m_loss:.4f}",
                    "um": f"{avg_u_loss:.4f}",
                }
            )

        epoch_loss = total_loss / max(1, total_mini)
        epoch_m_loss = total_m_loss / max(1, total_mini)
        epoch_u_loss = total_u_loss / max(1, total_mini)

        accelerator.print(
            f"[Train] Epoch {epoch + 1}: "
            f"avg_loss={epoch_loss:.4f}, "
            f"avg_mm_loss={epoch_m_loss:.4f}, "
            f"avg_um_loss={epoch_u_loss:.4f}"
        )

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.save_dir)
    accelerator.print(f"[Save] LoRA adapter saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a pre-unlearning LoRA on LLaVA"
    )

    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Base LLaVA model directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the finetuned LoRA adapter.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Training parquet file path.",
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
        "--lr",
        type=float,
        default=2e-5,
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
        default=384,
        help="Reserved sequence length argument.",
    )

    args = parser.parse_args()
    main(args)

