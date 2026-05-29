# In-Context Reinforcement Learning via Algorithm Distillation

This document details the implementation of **Algorithm Distillation (AD)**, which treats reinforcement learning as a sequence modeling task to achieve in-context adaptation without gradient updates at inference time.

---

## 📂 File Summaries

### 1) [`generate_ad_dataset.py`](../generate_ad_dataset.py)
* **Purpose:** Collects learning history datasets by running training sessions with an online TD3 policy and saving intermediate checkpoints of state, action, and reward sequences.
* **Key Components:**
  * Rolls out episodes in `Hopper-v5` environment using policies of varying expertise (from random exploration to expert gaits).
  * Structures learning histories into sequences of $(s_t, a_t, r_t)$ transitions.

### 2) [`ad_dataset.py`](../ad_dataset.py)
* **Purpose:** Defines the PyTorch `Dataset` class for loading and preprocessing sequence learning histories.
* **Key Components:**
  * **Scheduled Action Masking:** Automatically masks historical actions during training with a dynamic probability $p_{\text{mask}}$ to prevent the causal sequence model from copycatting previous actions and force it to learn true state-reward transitions.
  * **History Jitter:** Injects small Gaussian noise into input contexts (states/actions) to act as regularizers and prevent overfitting.

### 3) [`ad_model.py`](../ad_model.py)
* **Purpose:** Defines the Causal Transformer architecture used to distill the reinforcement learning process.
* **Key Components:**
  * Tokenizes or projects continuous state-action-reward histories.
  * Employs causal self-attention masking to predict the next expert action given the sequence context.

### 4) [`train_ad.py`](../train_ad.py)
* **Purpose:** Runs the offline sequence imitation learning loop to train the AD model.
* **Key Components:**
  * Integrates **Automatic Mixed Precision (AMP)** (`torch.amp.autocast`) to double training throughput.
  * Uses **Gradient Accumulation** to simulate large sequence batch sizes within VRAM limits.

### 5) [`eval_ad.py`](../eval_ad.py)
* **Purpose:** Evaluates the trained AD agent online in MuJoCo.
* **Key Components:**
  * Deploys the model in `Hopper-v5` environment.
  * Evaluates policy adaptation in-context by appending new interactions directly to the self-attention history buffer at each step, requiring no gradient calculation.
