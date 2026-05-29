# 🏰 SAiDL Summer Induction Assignment 2026

Welcome to the official repository for the **SAiDL Summer Induction Assignment 2026**. This codebase demonstrates the implementation, optimization, and validation of state-of-the-art sequence models applied across two distinct tracks: **Core Machine Learning (Transformer Architectures)** and **Reinforcement Learning (Continuous Control & Sequence Policy Optimization)**.

---

## 🚀 Key Highlights & Architecture

### 💎 [Track 1: Core Machine Learning](./Core-ML)
We designed and implemented a **highly modular Transformer Language Model** from scratch, trained and evaluated on the **WikiText-2** dataset. The design decouples attention, positional logic, and local filters to evaluate architectural trade-offs systematically.
* **4 Attention Variants:** Standard Dot-Product, Multi-Query Attention (MQA), Sliding Window Attention, and Linear Attention.
* **3 Positional Encodings:** Absolute Positional Embeddings, Rotary Positional Embedding (RoPE), and Attention with Linear Biases (ALiBi).
* **Convolution Hybrids:** Integrating Causal 1D Convolutions either as a Pre-Attention filter (Design A) or Interleaved Depthwise Separable layers (Design B).
* **Attention-Free Transformer (AFT):** Implemented AFT-Simple, AFT-Local, and AFT-Full to entirely bypass the quadratic $O(N^2)$ dot-product attention bottleneck.
* **Academic Report:** A complete LaTeX publication detailing the methodology, benchmarks, and extrapolation analysis is available at [SAiDL_Core-ML_Report.tex](./Core-ML/documentation/SAiDL_Core-ML_Report.tex).

### 🤖 [Track 2: Reinforcement Learning](./Reinforcement%20Learning)
We extended continuous deep reinforcement learning on **Gymnasium's Hopper-v5** to handle partial observability and delayed rewards using advanced sequence models.
* **TD3 Baseline:** Implemented a robust Twin Delayed DDPG (TD3) baseline with MLP policy networks trained over 1,000,000 steps.
* **Causal Transformer Policy:** Replaced the actor's MLP with a Causal Transformer policy to retain past trajectory history.
* **xLSTM Integration:** Integrated Extended LSTM (sLSTM and mLSTM) architectures within the TD3 policy network to solve long-horizon memory tasks.
* **Challenge Benchmarks:** Robust comparison under two hostile configurations:
  * **POMDP Challenge:** Masking velocity coordinates to force the agent to infer velocity from state history.
  * **Delayed Reward Challenge:** Delaying reward feedback by $K=10$ steps to test sparse credit assignment.
* **Algorithm Distillation (AD):** Pre-trained sequence-conditioned Transformers on trajectories of active policy learning to achieve in-context reinforcement learning.
* **RLHF (Reinforcement Learning from Human Feedback):** Trained a preference-based reward model from simulated query pairs to align TD3 policies with preferences.
* **Academic Report:** A complete LaTeX publication draft detailing the RL methodology, benchmarks, and sequence policy evaluation is available at [SAiDL_RL_Report.tex](./Reinforcement%20Learning/documentation/SAiDL_RL_Report.tex).
* **Mathematical & Engineering Deep-Dive:** See the details in [Optimizations.md](./Reinforcement%20Learning/documentation/Optimizations.md).

---

## 📂 Repository Layout

```
SAiDL-Summer-Assignment-2026/
├── Core-ML/
│   ├── model.py                 # Core Modular Transformer architecture
│   ├── positional_logic.py      # RoPE and ALiBi embedding logic
│   ├── attention_variants.py    # MQA, Sliding Window, and Linear Attention layers
│   ├── conv_logic.py            # Causal 1D convolutions (Design A & B)
│   ├── config.py                # Hyperparameters for Core-ML
│   ├── data.py                  # WikiText-2 data loading and preprocessing
│   ├── train.py                 # Base training loop for Core-ML
│   ├── evaluate_all.py          # Auto-run diagnostics across all 12 combinations
│   ├── extrapolation_test.py    # Tests sequence length extrapolation
│   ├── benchmark_aft.py         # Benchmarking for AFT variants
│   └── documentation/           # Academic reports and task-specific writeups
│       └── SAiDL_Core-ML_Report.tex # Complete, formatted LaTeX report
├── Reinforcement Learning/
│   ├── td3.py                   # TD3 agent with MLP policy
│   ├── train.py                 # Baseline TD3 training script
│   ├── train_transformer.py     # TD3 with Causal Transformer policy
│   ├── train_xlstm.py           # TD3 with mLSTM/sLSTM policies
│   ├── train_ad.py              # Algorithm Distillation training script
│   ├── train_rlhf.py            # RLHF policy tuning script
│   ├── xlstm_model.py           # Implementation of xLSTM cells & policies
│   ├── ad_model.py              # AD Transformer architecture
│   ├── ad_dataset.py            # AD sequence-conditioned dataset loading
│   ├── reward_model.py          # RLHF preference reward model
│   ├── rlhf_trainer.py          # RLHF trainer logic
│   ├── generate_ad_dataset.py   # Generates training history dataset for AD
│   ├── eval_ad.py               # Evaluation script for Algorithm Distillation
│   ├── run_benchmarks.py        # Automated benchmarks for POMDP and delayed rewards
│   └── documentation/           # Technical explanations of RL components
```

---

## ⚙️ Quick Start & Setup

### 1. Installation
Clone the repository and install the comprehensive dependencies:
```bash
git clone https://github.com/LimeLizard3/SAiDL-Summer-Assignment-2026.git
cd SAiDL-Summer-Assignment-2026
pip install -r requirements.txt
```

### 2. Running Core-ML
To replicate the evaluations and benchmarking in Track 1:
```bash
cd Core-ML

# Run all 12 core modular combinations automatically
python evaluate_all.py

# Benchmark Attention-Free Transformer (AFT) variants
python benchmark_aft.py

# Test sequence length extrapolation stability
python extrapolation_test.py
```

### 3. Running Reinforcement Learning
To train or evaluate continuous control agents in Track 2:
```bash
cd "Reinforcement Learning"

# Run automated POMDP and Delayed Reward benchmark comparisons
python run_benchmarks.py

# Generate history data for Algorithm Distillation
python generate_ad_dataset.py

# Train Algorithm Distillation model
python train_ad.py

# Train reward model and policy under RLHF
python train_rlhf.py
```

---

## 📊 Core-ML Performance Summary

The modular architectures were benchmarked on Perplexity (PPL), Inference Speed, and VRAM footprint:

| Modular Variant | Perplexity (PPL) $\downarrow$ | Tokens/Sec $\uparrow$ | Peak VRAM (MB) $\downarrow$ |
| :--- | :---: | :---: | :---: |
| **Standard + Absolute (Baseline)** | 34,940.81 | 12,560.3 | 525.4 |
| **Standard + RoPE** | 29,708.72 | 17,744.2 | 525.4 |
| **Standard + ALiBi** | 29,508.13 | 29,206.4 | 525.4 |
| **MQA + Absolute** | 32,572.44 | 30,447.3 | 524.8 |
| **MQA + RoPE** | 33,442.34 | 26,203.8 | 524.9 |
| **MQA + ALiBi** | 30,014.46 | 27,564.7 | 524.8 |
| **Sliding Window + Absolute** | 31,787.91 | 29,841.4 | 525.4 |
| **Sliding Window + RoPE** | 29,939.30 | 27,420.6 | 525.4 |
| **Sliding Window + ALiBi** | 30,421.29 | 28,666.8 | 525.4 |
| **Linear + RoPE** | 30,210.36 | 16,543.6 | 540.1 |
| **AFT-Simple (Bonus)** | 19,298.07 | **29,779.9** | **522.2** |
| **AFT-Local (Bonus)** | **17,176.86** | 16,803.3 | 590.1 |
| **AFT-Full (Bonus)** | 19,405.19 | 16,847.6 | 596.2 |
| **HYBRID (Pre-Attn) MQA+ALiBi** | 27,515.80 | 12,075.8 | 526.5 |
| **HYBRID (Interleaved) MQA+ALiBi** | 28,808.97 | 24,248.7 | 521.3 |

*For in-depth analysis on why specific architectures fail/succeed under context window scaling, see the full [SAiDL_Core-ML_Report.tex](./Core-ML/documentation/SAiDL_Core-ML_Report.tex).*

---

## 📊 Reinforcement Learning Performance Summary

The sequence policy networks (Causal Transformer vs. xLSTM) were benchmarked on Hopper-v5 under two stressors: Partial Observability (POMDP / velocity masking) and Delayed Rewards ($K=10$):

| Stressor Challenge | Architecture | Peak Reward $\uparrow$ | Final Reward | Overall Mean Reward | Post-Exploration Mean (Step $\ge$ 25k) | Peak Step |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **POMDP (Partial Obs)** | Causal Transformer | 82.42 | 73.53 | 62.34 $\pm$ 8.33 | **62.89 $\pm$ 8.98** | 110,000 |
| | **xLSTM** | **83.71** | **75.12** | 52.65 $\pm$ 22.98 | 61.95 $\pm$ 10.72 | **75,000** |
| **Delayed Reward ($k=10$)**| Causal Transformer | **399.71** | **399.53** | **137.29 $\pm$ 138.94**| **155.51 $\pm$ 145.47** | 125,000 |
| | **xLSTM** | 320.02 | 156.71 | 90.23 $\pm$ 85.16 | 106.97 $\pm$ 83.80 | 130,000 |

*For in-depth discussion on memory bottlenecks, state estimation capabilities, and sequence-based policies, see the full [SAiDL_RL_Report.tex](./Reinforcement%20Learning/documentation/SAiDL_RL_Report.tex).*

---

## 🛠️ Advanced Engineering & Algorithmic Optimizations

We implemented critical optimizations to scale sequence training and enhance generalization stability:

1. **Scheduled Action Masking & Jitter:** Solves causal confusion and copycat shortcuts in sequence-conditioned imitation learning. The action masking rate is dynamically scheduled: $p_{mask} = \min(0.2 + 0.1 \times \text{epoch}, 0.8)$, forcing the policy to leverage state-history over self-copying.
2. **Automatic Mixed Precision (AMP):** Utilizes `torch.amp.autocast` and `GradScaler` to double execution speed and decrease GPU memory footprints during sequence training.
3. **Gradient Accumulation:** Normalizes micro-batch gradients to simulate large batch sizes (e.g. 512) on memory-constrained consumer GPUs.
4. **Recurrent Loop Projection Offloading:** Moves parallelizable input-to-hidden linear projections outside recurrent loops in xLSTM layers, bypassing kernel launch overheads.

---

**Developed by [LimeLizard3] for the SAiDL 2026 Induction Program.** 🟢 🏹 📊
