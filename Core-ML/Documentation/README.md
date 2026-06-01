# SAiDL Transformer: Modular Architecture

A modular Transformer Language Model implemented in PyTorch for the SAiDL Summer Induction Assignment. This project evaluates alternative attention mechanisms, positional encoding strategies, and hybrid convolutional configurations on the WikiText-2 dataset.

---

## Progress & Completion Status

All track assignments and bonus items have been implemented and benchmarked:
* **Task 1: Baseline Architecture:** Autoregressive Transformer with absolute position embeddings.
* **Task 2: Attention Variants:** Multi-Query Attention (MQA), Sliding Window Attention, and Causal Linear Attention (via prefix sums).
* **Task 3: Positional Logic:** Rotary Positional Embeddings (RoPE) and Attention with Linear Biases (ALiBi).
* **Task 4: Hybrid Convolutional Designs:** Pre-Attention 1D Causal Convolutions (Design A) and Interleaved Depthwise Separable layers (Design B).
* **Bonus Tasks (AFT variants):** Attention-Free Transformer (AFT-Simple, AFT-Local, AFT-Full, AFT-Conv) with relative distance-based biases.
* **Global Evaluation & Diagnostics:** Automated diagnostic benchmarks running across all architectural combinations.

---

## Project Configuration & Features

We implemented and verified **4 Attention Mechanisms**, **3 Positional Encoding Strategies**, and **2 Hybrid Convolutional Designs**:

### Attention Variants (Task 2)
* **Standard:** Standard global self-attention with quadratic complexity.
* **Multi-Query (MQA):** Key-value head sharing to optimize memory footprint.
* **Linear:** Linear attention complexity for context length scaling.
* **Sliding Window:** Localized attention window to reduce compute.

### Positional Encodings (Task 3)
* **Absolute:** Default absolute position embeddings.
* **RoPE (Rotary):** Rotary positional embeddings for relative distance.
* **ALiBi:** Linear biases applied directly to attention scores.

### Convolutional Hybrids (Task 4)
* **Pre-Attention:** 1D convolutions before global self-attention.
* **Interleaved Blocks:** Alternating convolutional and self-attention layers.

---

## Documentation Map

Detailed descriptions and line-by-line code reviews for all components:

| Component | Detailed Line-by-Line Guide | Covered Files | Description |
| :--- | :--- | :--- | :--- |
| **Configuration** | [`config_explanation.md`](./config_explanation.md) | [`config.py`](../config.py) | Configuration and hyperparameters framework. |
| **Data Loading** | [`data_explanation.md`](./data_explanation.md) | [`data.py`](../data.py) | WikiText-2 vocabulary, tokenization, and batch loading. |
| **Model Core** | [`model_explanation.md`](./model_explanation.md) | [`model.py`](../model.py) | Core modular Transformer architecture. |
| **Attention Variants** | [`attention_variants_explanation.md`](./attention_variants_explanation.md) | [`attention_variants.py`](../attention_variants.py), [`benchmark_aft.py`](../benchmark_aft.py) | Multi-Query, Sliding Window, Linear, and AFT attention variants. |
| **Convolution Hybrids** | [`conv_hybrid_explanation.md`](./conv_hybrid_explanation.md) | [`conv_logic.py`](../conv_logic.py) | Pre-Attention Causal and Interleaved Convolution layers. |
| **Causality Verification** | [`causality_explanation.md`](./causality_explanation.md) | [`causality_test.py`](../causality_test.py) | Test script checking temporal autoregressive properties. |
| **Positional Encodings** | [`positional_explanation.md`](./positional_explanation.md) | [`positional_logic.py`](../positional_logic.py) | RoPE (rotary) and ALiBi positional encoding mechanisms. |
| **Training Loop** | [`train_explanation.md`](./train_explanation.md) | [`train.py`](../train.py) | Language modeling training loop, logging, and evaluation. |
| **Global Evaluation** | [`evaluate_all_explanation.md`](./evaluate_all_explanation.md) | [`evaluate_all.py`](../evaluate_all.py) | Automatic profiling of all 12+ model combinations. |
| **Context Extrapolation** | [`extrapolation_explanation.md`](./extrapolation_explanation.md) | [`extrapolation_test.py`](../extrapolation_test.py) | Evaluates context length generalization (Task 3). |

---

## Quick Start

### 1. Installation
```bash
pip install torch datasets tiktoken
```

### 2. Run Global Diagnostics
Run the script to verify all model combinations and profile performance:
```bash
python evaluate_all.py
```
