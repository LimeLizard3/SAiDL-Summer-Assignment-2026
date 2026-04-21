# 🚀 The Complete Engineering Journey: From Baseline to AFT Bonus

This document outlines the step-by-step workflow for the **SAiDL Summer Induction Assignment**. It serves as a guide for anyone starting from scratch to reproduce the entire project.

---

## 🛠️ Phase 0: Environment Setup
Before starting any tasks, ensure all dependencies are installed.
```bash
pip install -r requirements.txt
```
This installs core components: **PyTorch**, **Datasets** (HuggingFace), **Tiktoken** (OpenAI BPE), and **Gymnasium** (for the RL track).

---

## 📈 Track 1: Core Machine Learning (The Transformer Skyscraper)

### Task 1: Building the Foundation (The Baseline)
*   **Goal**: Create a standard Transformer and train it on **WikiText-2**.
*   **Execution**:
    1.  Set `attention_type = "standard"` in `config.py`.
    2.  Run `python Core-ML/train.py`.
*   **Outcome**: A functional language model baseline with $\sim 35,000$ Perplexity.

### Task 2: Attention Evolution (MQA & Linear)
*   **Goal**: Optimize the attention mechanism for speed and memory.
*   **Execution**:
    1.  Switch `attention_type` to `mqa`, `linear`, or `sliding_window` in `config.py`.
    2.  Run `python Core-ML/evaluate_all.py` to benchmark speed.
*   **Outcome**: Verified 2.2x speedup using **Multi-Query Attention (MQA)**.

### Task 3: Breaking the Context Barrier (Extrapolation)
*   **Goal**: Enable the model to handle tokens beyond its training limit.
*   **Execution**:
    1.  Set `pos_encoding` to `rope` or `alibi` in `config.py`.
    2.  Run `python Core-ML/extrapolation_test.py`.
*   **Outcome**: Stable inference up to 2048 tokens (doubling the training window).

### Task 4: The Hybrid Core (Causal Convolutions)
*   **Goal**: Combine global attention with local convolutional filters.
*   **Execution**:
    1.  Choose a hybrid config in `config.py`.
    2.  Run `python Core-ML/evaluate_all.py`.
*   **Outcome**: **The Accuracy Winner** (at the time) with 27,515 PPL.

---

## 🎁 The Bonus Task: Attention-Free Transformer (AFT)

*   **Goal**: Remove the dot-product attention entirely.
*   **Execution**: 
    1.  Run `python Core-ML/benchmark_aft.py`.
*   **Logic**: AFT uses element-wise operations to achieve $O(T)$ memory complexity.
*   **Outcome**: **AFT-Local** became the new Quality Champion (PPL 17,177), and **AFT-Simple** became the new Speed Demon (30k tokens/sec).

---

## 🤖 Track 2: Reinforcement Learning (Optional Track)

> [!NOTE]
> The **Reinforcement Learning/** folder contains a separate track where we applied Transformers as agents. **RLHF** (Reinforcement Learning from Human Feedback) scripts are also located there for future research, though not used in the core Task 1-4 benchmarks.

*   **Execution**: Run `python "Reinforcement Learning/train_transformer.py"`.
*   **Logic**: Uses a Transformer-Actor with **TD3** to solve the `Hopper-v5` environment with history-based memory.

---

## ✅ Final Verification Checklist
To confirm the repo is fully functional from scratch:
1.  [ ] `python Core-ML/causality_test.py` (Must pass "Causal Integrity Passed")
2.  [ ] `python Core-ML/evaluate_all.py` (Must generate a benchmark table)
3.  [ ] `python Core-ML/extrapolation_test.py` (Check for ALiBi stability)

**Project Status: COMPLETE.** 🟢
