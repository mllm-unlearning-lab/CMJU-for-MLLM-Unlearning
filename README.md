# CMJU for MLLM Unlearning

Official implementation of **CMJU**: **Cross-Modal Joint Unlearning for Balanced Forgetting in Multimodal Large Language Models**.

CMJU studies **balanced cross-modal unlearning** in multimodal large language models (MLLMs). Unlike conventional unlearning settings that mainly focus on a single input form, CMJU targets a more challenging setting where the same knowledge may remain retrievable through **both multimodal inputs and text-only inputs**. To address this, CMJU explicitly coordinates forgetting signals from the two paths and aims to remove target knowledge in a more balanced way while preserving non-target knowledge.

---

## Highlights

- A unified framework for **balanced cross-modal forgetting** in MLLMs
- Built under the **modality-complete supervision** setting
- Supports experiments on two multimodal unlearning benchmarks:
  - **UMU-Bench**
  - **CLEAR**
- Includes the implementation of:
  - **CMJU**
  - **GA**
  - **GD**
  - **KL**
  - **NPO**
  - **MANU**
- Provides dataset-specific training and evaluation pipelines

---

## Method Overview

CMJU is designed for the scenario where target knowledge may be activated from both:

- **multimodal inputs**
- **text-only inputs**

The framework contains two stages:

### Stage 1: Dual-Path Saliency Estimation and LoRA Grouping

CMJU estimates parameter saliency separately along:

- the **text-only path**
- the **multimodal path**

Based on these saliency signals, CMJU:

1. selects active LoRA parameters
2. groups them into:
   - **text-preferred**
   - **multimodal-preferred**
   - **shared**

### Stage 2: Differentiated Consistency Unlearning

CMJU then performs joint unlearning on multimodal and text-only forget samples with coordinated update mechanisms, including:

- gradient balancing
- gradient merging
- shared-group scaling
- symmetry constraint

This design encourages forgetting to proceed in a more coordinated way across modalities, while maintaining retain-set performance.

---

## Repository Structure

```text
CMJU-for-MLLM-Unlearning/
├── README.md
└── CMJU/
    ├── CLEAR/
    │   ├── CLEAR_finetune.py
    │   ├── CLEAR_eval.py
    │   ├── GA.py
    │   ├── GD.py
    │   ├── KL.py
    │   ├── NPO.py
    │   ├── MANU.py
    │   ├── CSAU.py
    │   └── ...
    └── UMU_Bench/
        ├── finetune.py
        ├── eval.py
        ├── GA.py
        ├── GD.py
        ├── KL.py
        ├── NPO.py
        ├── MANU.py
        ├── CSAU.py
        └── ...
