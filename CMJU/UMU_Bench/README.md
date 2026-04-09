# CMJU on UMU-Bench

This directory contains the implementation of **CMJU** and baseline unlearning methods on the **UMU-Bench** benchmark.

## Dataset

- UMU-Bench: https://huggingface.co/datasets/chengyewang/UMU-bench

Please prepare the processed data splits under your target data directory.

## Files

- `finetune.py`: fine-tuning script for obtaining the **pre-unlearning LoRA**
- `eval.py`: evaluation script
- `GA.py`: Gradient Ascent
- `GD.py`: Gradient Difference
- `KL.py`: KL-based unlearning
- `NPO.py`: Negative Preference Optimization
- `MANU.py`: pruning-based unlearning
- `CSAU.py`: CMJU method on UMU-Bench

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

    CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --save_dir /path/to/save/pre_unlearning_model \
      --data_dir /path/to/umu_bench_train.parquet \
      --batch_size 4 \
      --lr 2e-5 \
      --num_epochs 5 \
      --max_length 384

The final LoRA adapter is typically saved under:

    /save_dir/final_adapter

## Unlearning

### GA

    python GA.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_split_dir /path/to/data_split \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --lr 5e-5 \
      --num_epochs 5 \
      --clip_grad_norm 1.0

### GD

    python GD.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_split_dir /path/to/data_split \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --retain_steps_per_forget 2 \
      --lr 5e-5 \
      --num_epochs 5 \
      --clip_grad_norm 1.0

### KL

    python KL.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_split_dir /path/to/data_split \
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
      --data_split_dir /path/to/data_split \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --grad_accum_steps 2 \
      --beta 0.4 \
      --lr 1e-5 \
      --num_epochs 5 \
      --clip_grad_norm 1.0 \
      --use_retain 1 \
      --retain_steps_per_forget 1

### MANU

    python MANU.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/pruned_model \
      --data_split_dir /path/to/data_split \
      --forget_split_ratio 5 \
      --batch_size 4 \
      --max_batches 100 \
      --prune_percent 10 \
      --vision_last_n 3 \
      --text_last_n 3

### CSAU

    python CSAU.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --lora_dir /path/to/pre_unlearning_lora \
      --save_dir /path/to/save/checkpoints \
      --data_split_dir /path/to/data_split \
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

    python eval.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --model_path /path/to/model_or_lora \
      --model_type lora \
      --forget_split_ratio 5 \
      --data_split_dir /path/to/data_split \
      --output_path /path/to/output_dir \
      --output_file result.json

Example for full-model methods such as `MANU`:

    python eval.py \
      --base_model_dir /path/to/llava-1.5-7b-hf \
      --model_path /path/to/full_model \
      --model_type full \
      --forget_split_ratio 5 \
      --data_split_dir /path/to/data_split \
      --output_path /path/to/output_dir \
      --output_file result.json

## Notes

- LoRA-based methods: `GA`, `GD`, `KL`, `NPO`, `CSAU`
- Full-model method: `MANU`
- One adapter checkpoint is usually saved per epoch for LoRA-based unlearning methods
- `eval.py` uses:
  - `model_type=lora` for LoRA checkpoints
  - `model_type=full` for merged/pruned full models
