# Efficient Fine-Tuning of TinyLlama (1.1B) for Edge Deployment

## Project Overview
This project investigates the feasibility of deploying specialized Large Language Models (LLMs) in extreme low-resource environments (e.g., rural Nepal) by fine-tuning the **TinyLlama-1.1B** model. 

Building upon the findings of the **"LoRA Land" Technical Report (Zhao et al., Predibase)**, which benchmarked LoRA adapters for 2B-7B models, this project extends the analysis to the **1B parameter scale**. The goal was to validate whether algorithmic efficiency techniques (4-bit Quantization + LoRA) could enable real-time inference on commodity hardware with limited VRAM.

**Context:** Developed as part of a research initiative to democratize AI access for offline-first applications (e.g., travel assistance, agricultural support) where cloud connectivity is unreliable.

## 📊 Key Experimental Results

I benchmarked the system using a specific "Travel Intent" dataset. The results demonstrate that domain-specific agents can run efficiently on consumer-grade hardware.

| Metric | TinyLlama-1.1B (Ours) | Research Implication |
| :--- | :--- | :--- |
| **Inference VRAM** | **1.03 GB** | **Critical Finding:** The model fits entirely within the RAM of a Raspberry Pi 4/5 or a low-end laptop, enabling true edge deployment. |
| **Training VRAM** | **1.62 GB** | Fine-tuning can be performed on free-tier GPUs (e.g., Colab T4) or standard consumer GPUs, lowering the barrier to entry for researchers in developing nations. |
| **Throughput** | **~12.0 tokens/sec** | Exceeds the average human reading speed (~5-8 tokens/sec), ensuring a fluid user experience despite the low-compute environment. |
| **Model Size** | 1.1 Billion | Proves that sub-2B models do not suffer from "capacity collapse" when fine-tuned with a higher LoRA rank (r=16). |

## 🛠️ Methodology & Tech Stack

*   **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v0.1`
*   **Technique:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation).
*   **Optimization:** 4-bit Normal Float (NF4) Quantization via `bitsandbytes`.
*   **Libraries:** Hugging Face `transformers`, `peft`, `trl` (SFTTrainer).
*   **Task:** Travel Intent Recognition and Response Generation.

## 📄 References
*   **LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4** (Zhao et al., 2024). Used as a baseline for experimental design and efficiency comparison.
*   **TinyLlama:** An Open Source Small Language Model (Zhang et al., 2024).

---
*Author: Ankit Singh*
