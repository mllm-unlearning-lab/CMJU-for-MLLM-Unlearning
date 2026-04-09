# ✨Baselines 

## GA

This section describes how to run the GA (Gradient Ascent) baseline.
GA performs forgetting by maximizing the training loss on the forget set, starting from a pre-unlearning model constructed as base model + LoRA.

### ✅ Usage

To execute the GA baseline, run the following command:

```bash
python GA.py \
    --base_model_dir <path_to_base_model> \
    --lora_dir <path_to_pre_unlearning_lora> \
    --save_dir <path_to_save_checkpoint> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

📁 **Output**

The script saves one adapter checkpoint per epoch under save_dir/epoch_x.

## GD

This section describes how to run the GD (Gradient Difference) baseline.
GD maximizes the loss on the forget set while minimizing the loss on the retain set, starting from a pre-unlearning model constructed as base model + LoRA.

To execute the GD baseline, run the following command:

```bash
python GD.py \
    --vanilla_dir <path_to_vanilla_model> \
    --save_dir <path_to_save_forget_model> \
    --data_split_dir <path_to_data_split> \
    --gamma <gamma_value> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --alpha <alpha_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>

python GD.py \
    --base_model_dir <path_to_base_model> \
    --lora_dir <path_to_pre_unlearning_lora> \
    --save_dir <path_to_save_checkpoint> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --retain_steps_per_forget <retain_steps_per_forget> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

## KL

This section describes how to run the KL baseline.
KL maximizes the task loss on the forget set while preserving retained knowledge by minimizing KL divergence between the current model (student) and a frozen pre-unlearning model (oracle) on the retain set. Both student and oracle are initialized from base model + LoRA.

### ✅ Usage

To execute the KL baseline, run the following command:

```bash
python KL.py \
    --base_model_dir <path_to_base_model> \
    --lora_dir <path_to_pre_unlearning_lora> \
    --save_dir <path_to_save_checkpoint> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --retain_steps_per_forget <retain_steps_per_forget> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

## NPO

This section describes how to run the NPO (Negative Preference Optimization) baseline.
NPO performs forgetting by comparing the current model against a frozen pre-unlearning reference model on the forget set. In this implementation, both the trainable student and the frozen oracle are initialized from base model + LoRA. An optional retain loss can also be enabled.

### ✅ Usage

To execute the NPO baseline, run the following command:

```bash
python NPO.py \
    --base_model_dir <path_to_base_model> \
    --lora_dir <path_to_pre_unlearning_lora> \
    --save_dir <path_to_save_checkpoint> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --beta <beta_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

## MANU

This section describes how to run the MANU baseline.
MANU is a neuron-pruning baseline. It first constructs a full model by merging base model + LoRA, then collects activation statistics on forget and retain data, computes neuron importance scores, and finally prunes the top-ranked neurons globally.

### ✅ Usage

To execute the MANY baseline, run the following command:

```bash
python MANU.py \
    --base_model_dir <path_to_base_model> \
    --lora_dir <path_to_lora> \
    --save_dir <path_to_save_pruned_model> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --max_length <unimodal_max_length> \
    --mm_max_length <multimodal_max_length> \
    --num_workers <num_workers> \
    --prune_percent <prune_percent> \
    --vision_last_n <vision_last_n> \
    --text_last_n <text_last_n> \
    --weight_abs <weight_abs> \
    --weight_freq <weight_freq> \
    --weight_var <weight_var> \
    --weight_rms <weight_rms>
```
