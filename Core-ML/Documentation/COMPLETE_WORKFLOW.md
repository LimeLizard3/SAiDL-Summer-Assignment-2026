# 🛠️ Core-ML: Execution Workflow and File Requirements

This document details the exact Python files required for each phase of the **Core-ML** project and the order in which they should be executed to reproduce the results.

---

## 🏛️ Central Architecture (Core dependencies)
These files are the engine of all tasks. Most scripts depend on them:
*   `model.py`: The main Transformer neural network.
*   `attention_variants.py`: Housing for MQA, Linear, and AFT attention logic.
*   `positional_logic.py`: Logic for RoPE and ALiBi embeddings.
*   `conv_logic.py`: Causal 1D Convolutional layers.
*   `config.py`: The global dashboard to toggle features.
*   `data.py`: Automated tokenizer and WikiText-2 loader.

---

## 📈 Task-by-Task Workflow

### Task 1: Building the Baseline
*   **Objective**: Train a standard Transformer on WikiText-2.
*   **Required Files**: `config.py`, `data.py`, `model.py`, `train.py`.
*   **Execution Order**:
    1.  **Configure**: Ensure `config.py` has `attention_type = "standard"`.
    2.  **Execute**: Run `python Core-ML/train.py`.

### Task 2: Advanced Attention Benchmarking
*   **Objective**: Compare speed/VRAM of MQA, Linear, and Sliding Window attention.
*   **Required Files**: `config.py`, `data.py`, `model.py`, `attention_variants.py`, `evaluate_all.py`.
*   **Execution Order**:
    1.  **Configure**: Edit `config.py` to select the variant (e.g., `attention_type = "mqa"`).
    2.  **Execute**: Run `python Core-ML/evaluate_all.py`.

### Task 3: Context Extrapolation (RoPE/ALiBi)
*   **Objective**: Verify model stability at 2048+ tokens.
*   **Required Files**: `config.py`, `model.py`, `positional_logic.py`, `extrapolation_test.py`.
*   **Execution Order**:
    1.  **Configure**: Edit `config.py` to set `pos_type = "alibi"` or `"rope"`.
    2.  **Execute**: Run `python Core-ML/extrapolation_test.py`.

### Task 4: Convolutional Hybrids
*   **Objective**: Filter local features using convolutions before or between attention.
*   **Required Files**: `config.py`, `model.py`, `conv_logic.py`, `evaluate_all.py`.
*   **Execution Order**:
    1.  **Configure**: Set `use_conv = True` and `conv_type` in `config.py`.
    2.  **Execute**: Run `python Core-ML/evaluate_all.py`.

---

## 🎁 BONUS TASK: Attention-Free Transformer (AFT)
*   **Objective**: Complete elimination of the attention matrix.
*   **Required Files**: `model.py`, `attention_variants.py`, `benchmark_aft.py`.
*   **Execution Order**:
    1.  **Execute**: Run `python Core-ML/benchmark_aft.py`.

---

## ✅ Global Verification
*   **Objective**: Confirm causal integrity (no info leaks from the future).
*   **Required Files**: `model.py`, `causality_test.py`.
*   **Execution Order**:
    1.  **Execute**: Run `python Core-ML/causality_test.py`.

---
**Focus: Core-ML Track Finalized.** 🟢
