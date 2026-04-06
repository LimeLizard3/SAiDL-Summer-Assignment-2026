# 🏰 SAiDL Transformer: The Modular Skyscraper

A high-performance, modular Transformer Language Model built for the **SAiDL Summer Induction Assignment**. This repository focuses on modern attention efficiency, long-context positional stability, and hybrid convolutional architectures.

---

## 🚀 Project Vision
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

| Component | Detailed Line-by-Line Guide |
| :--- | :--- |
| **The Blueprint** | [`config_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/config_explanation.md) |
| **The Data** | [`data_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/data_explanation.md) |
| **The Brain** | [`model_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/model_explanation.md) |
| **The Engines** | [`attention_variants_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/attention_variants_explanation.md) |
| **The Hybrids** | [`conv_hybrid_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/conv_hybrid_explanation.md) |
| **The Integrity** | [`causality_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/causality_explanation.md) |
| **The Positioning**| [`positional_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/positional_explanation.md) |
| **The Training** | [`train_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/train_explanation.md) |
| **The Diagnostic** | [`evaluate_all_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/evaluate_all_explanation.md) |
| **The Stability** | [`extrapolation_explanation.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/extrapolation_explanation.md) |

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
All performance metrics (Speed, Memory, Stability) are captured in the official **[`benchmarking_results.md`](file:///c:/Users/gauth/.gemini/antigravity/scratch/SaiDL%20Project/Core%20ML/benchmarking_results.md)**.

**Built by [LimeLizard3] for SAiDL 2026.** 🟢 🏹 📊
