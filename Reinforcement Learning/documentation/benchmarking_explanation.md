# Benchmarking & Diagnostics

This document details the benchmarking, evaluation, and diagnostic files used to profile policy performance under stressors, compute attention attribution, and render plots.

---

## 📂 File Summaries

### 1) [`attention_diagnostics.py`](../attention_diagnostics.py)
* **Purpose:** Runs online evaluation rollouts of trained policies to log raw self-attention weight maps and trace attention entropy ($H(A_t)$) across sequence timesteps.

### 2) [`attribution_analysis.py`](../attribution_analysis.py)
* **Purpose:** Implements the **Chefer Protocol** (relevancy propagation) and the **Absolute Saliency Engine** to map information flow in Causal Transformers, capturing inhibitory and positive historical attention spikes.

### 3) [`benchmark_pomdp.py`](../benchmark_pomdp.py)
* **Purpose:** Profiles and logs MLP vs. Transformer policies under sensory mask limits, noise levels, and delayed credit assignment conditions.

### 4) [`benchmark_pos_encodings.py`](../benchmark_pos_encodings.py)
* **Purpose:** Profiles learning efficiency when combining Causal Transformers with Learned, Sinusoidal, or Rotary (RoPE) positional encodings in hidden-velocity tasks.

### 5) [`delayed_reward_wrapper.py`](../delayed_reward_wrapper.py)
* **Purpose:** Implements a Gym wrapper to store rewards in a FIFO queue, delaying feedback by $K$ steps.

### 6) [`plot_historical.py`](../plot_historical.py)
* **Purpose:** Parses training runs logs (stored in JSON/CSV formats under `results/`) and renders clean performance curves.

### 7) [`robustness_analysis.py`](../robustness_analysis.py)
* **Purpose:** Evaluates agent behaviors under individual sensor failure coordinates (e.g. joint velocities only) and varying scale noise vectors.

### 8) [`run_benchmarks.py`](../run_benchmarks.py)
* **Purpose:** Orchestrator script to sequence run benchmark suites across multiple hyperparameters.
