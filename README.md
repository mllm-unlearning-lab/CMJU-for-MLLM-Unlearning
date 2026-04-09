````markdown
# CMJU for MLLM Unlearning

Official implementation of **CMJU**: **Cross-Modal Joint Unlearning for Balanced Forgetting in Multimodal Large Language Models**.

CMJU studies **balanced cross-modal unlearning** in multimodal large language models. It aims to remove target knowledge in a coordinated way across both **multimodal** and **text-only** inputs.

## Overview

CMJU is designed for the **modality-complete supervision** setting, where paired multimodal and text-only forget samples are both available.

The framework contains two stages:

- **Stage 1:** Dual-Path Saliency Estimation and LoRA Grouping
- **Stage 2:** Differentiated Consistency Unlearning

CMJU explicitly coordinates multimodal and text-only forgetting signals to improve cross-modal forgetting balance while preserving retain-set performance.

## Repository Structure

```text
CMJU-for-MLLM-Unlearning/
├── README.md
└── CMJU/
    ├── CLEAR/
    └── UMU_Bench/
````

* `CMJU/CLEAR/`: code for experiments on the **CLEAR** benchmark
* `CMJU/UMU_Bench/`: code for experiments on the **UMU-Bench** benchmark

## Supported Datasets

* **UMU-Bench**: [https://huggingface.co/datasets/chengyewang/UMU-bench](https://huggingface.co/datasets/chengyewang/UMU-bench)
* **CLEAR**: [https://huggingface.co/datasets/therem/CLEAR](https://huggingface.co/datasets/therem/CLEAR)

## Main Components

Each dataset directory includes:

* fine-tuning scripts for obtaining the **pre-unlearning model**
* unlearning methods:

  * `GA.py`
  * `GD.py`
  * `KL.py`
  * `NPO.py`
  * `MANU.py`
  * `CSAU.py`
* evaluation scripts

## Workflow

1. Prepare the dataset
2. Fine-tune the base MLLM
3. Run an unlearning method
4. Evaluate the unlearned model

For most LoRA-based methods, the pre-unlearning model is:

```text
base_model + pre-unlearning LoRA
```

For pruning-based methods such as `MANU`, the LoRA is first merged into the base model and pruning is then applied to the merged model.

## Results

Experiments on **UMU-Bench** and **CLEAR** show that CMJU achieves better **cross-modal forgetting balance** than strong baselines while maintaining competitive retain-set performance.

## Usage

Please refer to the dataset-specific directories for detailed instructions:

* `CMJU/UMU_Bench/`
* `CMJU/CLEAR/`

## Acknowledgements

We thank the authors and maintainers of:

* UMU-Bench
* CLEAR
* Hugging Face Transformers
* PEFT
* Accelerate
