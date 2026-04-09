# CMJU on CLEAR

This directory contains the implementation of **CMJU** and baseline unlearning methods on the **CLEAR** benchmark.

## Dataset

- CLEAR: https://huggingface.co/datasets/therem/CLEAR

Please prepare the CLEAR dataset under your target data directory.

## Files

- `CLEAR_finetune.py`: fine-tuning script for obtaining the **pre-unlearning LoRA**
- `CLEAR_eval.py`: evaluation script
- `GA.py`: Gradient Ascent
- `GD.py`: Gradient Difference
- `KL.py`: KL-based unlearning
- `NPO.py`: Negative Preference Optimization
- `MANU.py`: pruning-based unlearning
- `CSAU.py`: CMJU method on CLEAR

## Workflow

1. Prepare the dataset
2. Fine-tune the base model to obtain the **pre-unlearning LoRA**
3. Run one unlearning method
4. Evaluate the unlearned model

For most LoRA-based methods, the pre-unlearning model is:

    base_model + pre-unlearning LoRA

For `MANU.py`, the LoRA is first merged into the base model and pruning is then applied to the merged model.

## Fine-tuning

Example:

    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    accelerate launch --num_processes 4 CLEAR_finetune.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --save_dir /path/to/save/pre_unlearning_model \
      --dataset_name data/CLEAR/full+tofu \
      --batch_size 4 \
      --num_epochs 6 \
      --lr 5e-5 \
      --gradient_accumulation_steps 1 \
      --ans_only \
      --gradient_checkpointing \
      --pin_memory \
      --save_every_n_epochs 2 \
      --mixed_precision fp16 \
      --num_workers 0

The final LoRA adapter is typically saved under:

    /save_dir/final_adapter

## Unlearning

### GA

    python GA.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_dir data/CLEAR \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --lr 3e-6 \
      --num_epochs 5 \
      --clip_grad_norm 1.0

### GD

    python GD.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_dir data/CLEAR \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --retain_steps_per_forget 4 \
      --lr 1e-5 \
      --num_epochs 5 \
      --clip_grad_norm 1.0

### KL

    python KL.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_dir data/CLEAR \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --retain_steps_per_forget 1 \
      --lr 1e-5 \
      --num_epochs 5 \
      --clip_grad_norm 1.0

### NPO

    python NPO.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_dir data/CLEAR \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --lr 1e-5 \
      --num_epochs 5 \
      --beta 0.4 \
      --clip_grad_norm 1.0 \
      --use_retain 1 \
      --retain_steps_per_forget 1

### MANU

    python MANU.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/pruned_model \
      --data_dir data/CLEAR \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --prune_percent 10 \
      --vision_last_n 3 \
      --text_last_n 3

### CSAU

    python CSAU.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_dir data/CLEAR \
      --cache_dir /path/to/cache_dir \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --lr 5e-6 \
      --num_epochs 10 \
      --alpha_shared 1.0 \
      --beta_specific 2.0 \
      --alpha_forget 1.0 \
      --retain_steps_per_forget 4 \
      --gamma_sym 0.5 \
      --top_k_ratio 0.3 \
      --modality_margin 0.05 \
      --retain_loss_threshold 30.0 \
      --clip_grad_norm 1.0

## Evaluation

Example for LoRA-based methods:

    python CLEAR_eval.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --model_path /path/to/model_or_lora \
      --model_type lora \
      --eval_list "forget retain realface realworld" \
      --output_file clear_result \
      --output_folder /path/to/output_dir \
      --data_folder data/CLEAR \
      --forget_cls_folder forget5_perturbed \
      --forget_gen_folder forget5+tofu \
      --retain_cls_folder retain_perturbed \
      --retain_gen_folder retain95+tofu \
      --realface_folder real_faces \
      --realworld_folder real_world

Example for full-model methods such as `MANU`:

    python CLEAR_eval.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --model_path /path/to/full_model \
      --model_type full \
      --eval_list "forget retain realface realworld" \
      --output_file clear_result \
      --output_folder /path/to/output_dir \
      --data_folder data/CLEAR \
      --forget_cls_folder forget5_perturbed \
      --forget_gen_folder forget5+tofu \
      --retain_cls_folder retain_perturbed \
      --retain_gen_folder retain95+tofu \
      --realface_folder real_faces \
      --realworld_folder real_world

## Notes

- LoRA-based methods: `GA`, `GD`, `KL`, `NPO`, `CSAU`
- Full-model method: `MANU`
- One adapter checkpoint is usually saved per epoch for LoRA-based unlearning methods
- `CLEAR_eval.py` uses:
  - `model_type=lora` for LoRA checkpoints
  - `model_type=full` for merged/pruned full models
