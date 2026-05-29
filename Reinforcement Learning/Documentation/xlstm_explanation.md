# xLSTM Policy Backbone: Recurrent Locomotion

This directory details the integration of the **Extended LSTM (xLSTM)** architecture as the recurrent neural network backbone for the TD3 locomotion policy, replacing standard Feedforward MLPs and Causal Transformers.

---

## 📂 File Summaries

### 1) [`xlstm_model.py`](../xlstm_model.py)
* **Purpose:** Implements the core xLSTM architecture cells and the sequence-conditioned TD3 Actor/Critic policy networks.
* **Key Components:**
  * **`sLSTMCell` & `mLSTMCell`:** Implements the sLSTM cell (scalar memory, exponential gating, and normalizer state) and mLSTM cell (matrix memory with covariance updates).
  * **`xLSTMStack`:** Chains sLSTM/mLSTM layers together with feedforward blocks and layer normalization.
  * **`xLSTMActor` & `xLSTMCritic`:** Constructs sequence-aware policy networks using the recurrent xLSTM stack.
  * **Recurrent Loop Projection Offloading:** To optimize GPU performance, the model pre-computes linear projections (Queries, Keys, Values, and gates) in parallel *outside* the sequential recurrence loop. The cell loop then simply slices these tensors at each step, eliminating sequential kernel launch bottlenecks and accelerating execution.

### 2) [`train_xlstm.py`](../train_xlstm.py)
* **Purpose:** Entry point for training the xLSTM-based TD3 policy.
* **Key Components:**
  * Orchestrates online interaction with MuJoCo's `Hopper-v5` environment.
  * Evaluates xLSTM policies under **POMDP (hidden-velocity)** or **Delayed Reward** settings.
  * Manages sequence-based target policy updates, action smoothing, and episodic logging.
