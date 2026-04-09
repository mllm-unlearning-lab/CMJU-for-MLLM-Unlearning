import argparse
import json
import os
import random
import re
from collections import deque
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, get_scheduler

from data_process.CLEAR_process import (
    CAPTION_MODE,
    IMAGE_QA_MODE,
    TEXT_QA_MODE,
    ClearCollator,
    CLEARDataset,
)

LLM_TARGET_MODULE_PATTERN = re.compile(
    r"^language_model\.model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank0_print(accelerator: Accelerator, msg: str) -> None:
    if accelerator.is_main_process:
        print(msg, flush=True)


def print_run_info(accelerator: Accelerator, args: argparse.Namespace) -> None:
    rank0_print(accelerator, "=" * 100)
    rank0_print(accelerator, f"Run start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rank0_print(accelerator, "Arguments:")
    for key, value in vars(args).items():
        rank0_print(accelerator, f"  {key}: {value}")
    rank0_print(accelerator, "=" * 100)


def get_lora_target_modules(model: torch.nn.Module, accelerator: Accelerator) -> list[str]:
    target_modules = []

    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if LLM_TARGET_MODULE_PATTERN.fullmatch(module_name):
            target_modules.append(module_name)

    if not target_modules:
        raise ValueError(
            "No LoRA target modules matched. "
            "Check model architecture and target regex."
        )

    rank0_print(accelerator, f"Matched {len(target_modules)} LoRA target modules.")
    preview_count = min(12, len(target_modules))
    rank0_print(accelerator, "Preview of matched modules:")
    for name in target_modules[:preview_count]:
        rank0_print(accelerator, f"  {name}")

    return target_modules


def verify_lora_trainability(model: torch.nn.Module, accelerator: Accelerator) -> None:
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable_params)
    total_count = sum(p.numel() for _, p in model.named_parameters())

    rank0_print(
        accelerator,
        f"Trainable params: {trainable_count:,} / {total_count:,} "
        f"({100 * trainable_count / total_count:.4f}%)"
    )

    if trainable_count == 0:
        raise ValueError("No trainable parameters found after applying LoRA.")

    preview_count = min(10, len(trainable_params))
    rank0_print(accelerator, "Preview of trainable parameter names:")
    for name, _ in trainable_params[:preview_count]:
        rank0_print(accelerator, f"  {name}")


def load_model_and_processor(args: argparse.Namespace, accelerator: Accelerator):
    rank0_print(accelerator, f"Loading model from: {args.base_model_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(args.base_model_dir)
    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    vocab_size = len(processor.tokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if vocab_size > embedding_size:
        rank0_print(accelerator, f"Resizing token embeddings: {embedding_size} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)
    else:
        rank0_print(
            accelerator,
            f"Keep token embeddings unchanged: tokenizer_size={vocab_size}, embedding_size={embedding_size}"
        )

    model.config.use_cache = False

    if args.lora_dir is not None:
        rank0_print(accelerator, f"Loading existing LoRA adapter from: {args.lora_dir}")
        model = PeftModel.from_pretrained(
            model,
            args.lora_dir,
            is_trainable=True,
        )
        verify_lora_trainability(model, accelerator)
        return model, processor

    target_modules = get_lora_target_modules(model, accelerator)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, lora_config)
    verify_lora_trainability(model, accelerator)

    return model, processor


def build_train_dataloader(args: argparse.Namespace, processor, accelerator: Accelerator):
    raw_dataset = load_dataset(args.dataset_name, split="train")

    requested_modes = set(x.strip().lower() for x in args.train_modes.split(",") if x.strip())
    valid_modes = {"text_qa", "image_qa", "caption"}
    unknown_modes = requested_modes - valid_modes
    if unknown_modes:
        raise ValueError(f"Unknown train_modes: {unknown_modes}")

    datasets_to_concat = []

    if "text_qa" in requested_modes:
        text_qa_dataset = CLEARDataset(raw_dataset, mode=TEXT_QA_MODE)
        rank0_print(accelerator, f"TEXT_QA dataset size : {len(text_qa_dataset)}")
        datasets_to_concat.append(text_qa_dataset)
    else:
        rank0_print(accelerator, "TEXT_QA dataset skipped.")

    if "image_qa" in requested_modes:
        image_qa_dataset = CLEARDataset(raw_dataset, mode=IMAGE_QA_MODE)
        rank0_print(accelerator, f"IMAGE_QA dataset size: {len(image_qa_dataset)}")
        datasets_to_concat.append(image_qa_dataset)
    else:
        rank0_print(accelerator, "IMAGE_QA dataset skipped.")

    if "caption" in requested_modes:
        caption_dataset = CLEARDataset(raw_dataset, mode=CAPTION_MODE)
        rank0_print(accelerator, f"CAPTION dataset size : {len(caption_dataset)}")
        datasets_to_concat.append(caption_dataset)
    else:
        rank0_print(accelerator, "CAPTION dataset skipped.")

    if len(datasets_to_concat) == 0:
        raise ValueError("No dataset selected. Please check --train_modes.")

    train_dataset = datasets_to_concat[0] if len(datasets_to_concat) == 1 else ConcatDataset(datasets_to_concat)
    rank0_print(accelerator, f"TOTAL train size     : {len(train_dataset)}")

    num_workers = 0 if args.debug else args.num_workers
    if args.debug and args.num_workers != 0:
        rank0_print(accelerator, "Debug mode enabled: forcing num_workers=0.")

    collator = ClearCollator(
        processor=processor,
        ans_only=args.ans_only,
        debug=args.debug and accelerator.is_main_process,
        debug_max_prints=args.debug_max_prints,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collator,
    )
    return train_loader


def run_grad_check(model, train_dataloader, accelerator: Accelerator) -> None:
    model.train()
    batch = next(iter(train_dataloader))
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss

    rank0_print(
        accelerator,
        f"Sanity check - loss.requires_grad={loss.requires_grad}, "
        f"loss.dtype={loss.dtype}, loss={loss.detach().float().item():.6f}"
    )

    if not loss.requires_grad:
        raise RuntimeError(
            "Sanity check failed: loss.requires_grad is False. "
            "This usually means LoRA modules were not properly attached, "
            "or no trainable parameters participated in the forward pass."
        )


def save_checkpoint(
    accelerator: Accelerator,
    model,
    processor,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    checkpoint_dir = os.path.join(args.save_dir, f"checkpoint-epoch-{epoch}")
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    trainer_state_dir = os.path.join(checkpoint_dir, "trainer_state")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)

        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "base_model_dir": args.base_model_dir,
            "lora_dir": args.lora_dir,
            "ans_only": args.ans_only,
        }
        with open(os.path.join(checkpoint_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    accelerator.wait_for_everyone()
    accelerator.save_state(trainer_state_dir)
    accelerator.wait_for_everyone()
    rank0_print(accelerator, f"Checkpoint saved to: {checkpoint_dir}")


def save_final_adapter(accelerator: Accelerator, model, processor, args: argparse.Namespace) -> None:
    final_dir = os.path.join(args.save_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)

    accelerator.wait_for_everyone()
    rank0_print(accelerator, f"Final adapter saved to: {final_dir}")


def load_training_state_if_needed(accelerator: Accelerator, args: argparse.Namespace):
    if not args.resume_from_checkpoint:
        return 0, 0

    checkpoint_dir = args.resume_from_checkpoint
    trainer_state_dir = os.path.join(checkpoint_dir, "trainer_state")
    meta_path = os.path.join(checkpoint_dir, "meta.json")

    if not os.path.isdir(trainer_state_dir):
        raise ValueError(f"trainer_state directory not found: {trainer_state_dir}")
    if not os.path.isfile(meta_path):
        raise ValueError(f"meta.json not found: {meta_path}")

    rank0_print(accelerator, f"Loading training state from: {checkpoint_dir}")
    accelerator.load_state(trainer_state_dir)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    start_epoch = int(meta["epoch"])
    global_step = int(meta["global_step"])

    rank0_print(
        accelerator,
        f"Resume success: next epoch starts from {start_epoch + 1}, "
        f"loaded global_step={global_step}"
    )
    return start_epoch, global_step


def main(args: argparse.Namespace):
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    print_run_info(accelerator, args)

    model, processor = load_model_and_processor(args, accelerator)

    if args.gradient_checkpointing:
        rank0_print(accelerator, "Gradient checkpointing enabled.")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        rank0_print(accelerator, "Gradient checkpointing disabled.")

    train_dataloader = build_train_dataloader(args, processor, accelerator)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = max(
        1,
        (len(train_dataloader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    run_grad_check(model, train_dataloader, accelerator)

    start_epoch, global_step = load_training_state_if_needed(accelerator, args)

    for epoch_idx in range(start_epoch, args.num_epochs):
        epoch_num = epoch_idx + 1
        model.train()

        running_losses = deque(maxlen=args.loss_window_size)
        epoch_loss_sum = 0.0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch_num}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                if not loss.requires_grad:
                    raise RuntimeError(
                        f"loss.requires_grad=False at epoch={epoch_num}, step={step}. "
                        "Likely LoRA target modules were not correctly attached."
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            reduced_loss = accelerator.gather(loss.detach().float()).mean().item()
            epoch_loss_sum += reduced_loss
            running_losses.append(reduced_loss)

            if accelerator.sync_gradients:
                global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                step_loss=f"{reduced_loss:.4f}",
                avg_loss=f"{sum(running_losses) / len(running_losses):.4f}",
                lr=f"{current_lr:.2e}",
                update_step=global_step,
            )

        epoch_avg_loss = epoch_loss_sum / len(train_dataloader)
        rank0_print(
            accelerator,
            f"[Epoch {epoch_num}] epoch_avg_loss={epoch_avg_loss:.6f}, global_step={global_step}"
        )

        should_save = (
            args.save_every_n_epochs > 0
            and (epoch_num % args.save_every_n_epochs == 0)
        ) or (epoch_num == args.num_epochs)

        if should_save:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                processor=processor,
                args=args,
                epoch=epoch_num,
                global_step=global_step,
            )

    save_final_adapter(accelerator, model, processor, args)
    accelerator.wait_for_everyone()
    rank0_print(accelerator, "Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA LoRA fine-tuning on CLEAR")

    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--dataset_name", type=str, default="data/CLEAR/full+tofu")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--ans_only", action="store_true")
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--loss_window_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_max_prints", type=int, default=2)

    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument(
        "--train_modes",
        type=str,
        default="text_qa,image_qa,caption",
        help="Comma-separated modes from: text_qa,image_qa,caption",
    )

    args = parser.parse_args()
    main(args)
