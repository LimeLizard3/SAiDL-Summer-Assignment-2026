# 🏰 SAiDL Transformer: The Modular Skyscraper

A high-performance, modular Transformer Language Model built for the **SAiDL Summer Induction Assignment**. This repository focuses on modern attention efficiency, long-context positional stability, and hybrid convolutional architectures.

---

## 🚀 Progress & Completion Status

All Core-ML assignment tasks and bonus challenges have been **fully completed, compiled, and benchmarked**:

*   **Task 1: Baseline Architecture:** [COMPLETED] Autoregressive Transformer with Absolute Position Embeddings trained on WikiText-2.
*   **Task 2: Attention Variants:** [COMPLETED] Implemented Multi-Query Attention (MQA), Sliding Window Attention, and Causal Linear Attention (with prefix sums).
*   **Task 3: Positional Logic:** [COMPLETED] Integrated Rotary Positional Embeddings (RoPE) and Attention with Linear Biases (ALiBi).
*   **Task 4: Hybrid Convolutional Designs:** [COMPLETED] Implemented Pre-Attention 1D Causal Convolutions (Design A) and Interleaved Depthwise Separable layers (Design B).
*   **Bonus Tasks (AFT variants):** [COMPLETED] Integrated Attention-Free Transformer (AFT-Simple, AFT-Local, AFT-Full, AFT-Conv) with relative distance-based biases.
*   **Global Evaluation & Diagnostics:** [COMPLETED] Automated script running diagnostic benchmarks across all 12+ architectural combinations.

---

## 🏗️ Project Vision
The goal of this project is to transform a rigid "Absolute" Transformer into a flexible, state-of-the-art architecture. We implemented and verified **4 Attention Mechanisms**, **3 Positional Encoding Strategies**, and **2 Hybrid Convolutional Designs** to prove how modern AI models scale.

### 🌟 Key Features
*   **Modular Attention Engines (Task 2)**:
    *   **Standard**: Full $O(N^2)$ global context.
    *   **Multi-Query (MQA)**: Ultra-efficient key-value sharing.
    *   **Linear**: $O(N)$ complexity for high-speed inference.
    *   **Sliding Window**: Focused attention for long-range efficiency.
*   **Convolutional Hybrids (Task 4)**:
    *   **Pre-Attention**: 1D filtering before global attention for best perplexity.
    *   **Interleaved Blocks**: Alternating floors for maximum speed/memory efficiency.
*   **Dynamic Positioning (Task 3)**:
    *   **Absolute**: The Task 1 baseline.
    *   **RoPE (Rotary)**: Geometric vector rotation for relative distance.
    *   **ALiBi**: Linear distance-based penalty for extreme context.

---

## 📂 Documentation Map (Beginner's Guides)
We have built a comprehensive library of **Line-by-Line breakdowns** for every single file:

| Component | Detailed Line-by-Line Guide | Covered Files | Description |
| :--- | :--- | :--- | :--- |
| **The Blueprint** | [`config_explanation.md`](./config_explanation.md) | [`config.py`](../config.py) | Configuration and hyperparameters framework. |
| **The Data** | [`data_explanation.md`](./data_explanation.md) | [`data.py`](../data.py) | WikiText-2 vocabulary, tokenization, and batch loading. |
| **The Brain** | [`model_explanation.md`](./model_explanation.md) | [`model.py`](../model.py) | Core modular Transformer architecture. |
| **The Engines** | [`attention_variants_explanation.md`](./attention_variants_explanation.md) | [`attention_variants.py`](../attention_variants.py), [`benchmark_aft.py`](../benchmark_aft.py) | Multi-Query, Sliding Window, Linear, and AFT attention variants. |
| **The Hybrids** | [`conv_hybrid_explanation.md`](./conv_hybrid_explanation.md) | [`conv_logic.py`](../conv_logic.py) | Pre-Attention Causal and Interleaved Convolution layers. |
| **The Integrity** | [`causality_explanation.md`](./causality_explanation.md) | [`causality_test.py`](../causality_test.py) | "Lie detector" test checking temporal autoregressive properties. |
| **The Positioning**| [`positional_explanation.md`](./positional_explanation.md) | [`positional_logic.py`](../positional_logic.py) | RoPE (rotary) and ALiBi positional encoding mechanisms. |
| **The Training** | [`train_explanation.md`](./train_explanation.md) | [`train.py`](../train.py) | Language modeling training loop, logging, and evaluation. |
| **The Diagnostic** | [`evaluate_all_explanation.md`](./evaluate_all_explanation.md) | [`evaluate_all.py`](../evaluate_all.py) | Automatic profiling of all 12+ model combinations. |
| **The Stability** | [`extrapolation_explanation.md`](./extrapolation_explanation.md) | [`extrapolation_test.py`](../extrapolation_test.py) | Evaluates context length generalization (Task 3). |

---

## 🛠️ Quick Start

### 1. Installation
```powershell
pip install torch datasets tiktoken
```

### 2. Run Global Diagnostic (Task 4)
Run this script to automatically verify all 14+ model combinations and generate a performance leaderboard:
```powershell
python evaluate_all.py
```

---

## 📊 Final Benchmarks
All performance metrics (Speed, Memory, Stability) are captured in the official **[`benchmarking_results.md`](./benchmarking_results.md)**.

**Built by [LimeLizard3] for SAiDL 2026.** 🟢 🏹 📊
