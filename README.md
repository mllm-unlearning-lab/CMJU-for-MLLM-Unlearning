# CMJU for MLLM Unlearning

Official implementation of **CMJU**: **Cross-Modal Joint Unlearning for Balanced Forgetting in Multimodal Large Language Models**.

CMJU studies balanced cross-modal unlearning in multimodal large language models, aiming to remove target knowledge consistently across both **multimodal** and **text-only** inputs.

## Overview

CMJU is designed for the modality-complete supervision setting and consists of two stages:

- **Stage 1:** Dual-Path Saliency Estimation and LoRA Grouping
- **Stage 2:** Differentiated Consistency Unlearning

## Repository Structure

    CMJU-for-MLLM-Unlearning/
    ├── README.md
    └── CMJU/
        ├── CLEAR/
        └── UMU_Bench/

## Datasets

- **UMU-Bench**: https://huggingface.co/datasets/chengyewang/UMU-bench
- **CLEAR**: https://huggingface.co/datasets/therem/CLEAR

## Code

- `CMJU/UMU_Bench/`: code for experiments on **UMU-Bench**
- `CMJU/CLEAR/`: code for experiments on **CLEAR**

Each directory contains:

- fine-tuning scripts
- unlearning methods:
  - `GA.py`
  - `GD.py`
  - `KL.py`
  - `NPO.py`
  - `MANU.py`
  - `CSAU.py`
- evaluation scripts

## Workflow

1. Prepare the dataset
2. Fine-tune the base model
3. Run an unlearning method
4. Evaluate the unlearned model

For most LoRA-based methods:

    base_model + pre-unlearning LoRA

For `MANU`, the LoRA is first merged into the base model and then pruning is applied.

## Usage

See the dataset-specific instructions:

- [UMU-Bench README](./CMJU/UMU_Bench/README.md)
- [CLEAR README](./CMJU/CLEAR/README.md)
