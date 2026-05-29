# Reinforcement Learning from Human Feedback (RLHF)

This document details the implementation of preference-based reward learning and policy alignment, removing reliance on environment-defined rewards.

---

## 📂 File Summaries

### 1) [`reward_model.py`](../reward_model.py)
* **Purpose:** Implements the Preference Reward Model (Jury Ensemble).
* **Key Components:**
  * **`RewardModel` MLP:** Maps state-action pairs to scalar rewards.
  * **Jury Ensemble:** Implements a pessimistic minimum consensus over three independent reward networks to prevent reward hacking:
    $$R_{\text{Jury}}(s,a) = \min_{k \in \{1, 2, 3\}} R_{\psi_k}(s, a)$$

### 2) [`rlhf_pretraining.py`](../rlhf_pretraining.py)
* **Purpose:** Pre-trains the Jury Ensemble on a database of simulated preferences.
* **Key Components:**
  * Uses the Bradley-Terry preference model and binary cross-entropy loss to optimize reward weights.

### 3) [`rlhf_trainer.py`](../rlhf_trainer.py)
* **Purpose:** Implements the active online preference alignment logic.
* **Key Components:**
  * **Eternal Textbook Protocol:** Trains the Jury on a mixture of pre-training expert buffers (`old_buffer`) and the student's live trajectory buffers (`new_buffer`) to prevent catastrophic forgetting.

### 4) [`train_rlhf.py`](../train_rlhf.py)
* **Purpose:** Runs the aligned policy tuning loop.
* **Key Components:**
  * Coordinates policy updates under the **Reward Governor** (min-value Jury minimum, scaling, and clipping constraints).
