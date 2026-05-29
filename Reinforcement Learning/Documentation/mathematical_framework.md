# Mathematical Framework: The Physics of Robust Transformer RL

This document provides a comprehensive breakdown of every mathematical formula and conceptual algorithm used in the SAiDL Transformer Assignment.

---

## 1. The Transformer Architecture (The Brain)

The Transformer behaves as a sequence-to-sequence map that assigns "importance" to historical data using attention weights.

### A. Scaled Dot-Product Attention
This formula determines how much focus a query ($Q$) should give to a specific memory key ($K$).
$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$
*   **$Q, K, V$**: Queries, Keys, and Values (linear projections of state/action history).
*   **$\sqrt{d_k}$**: Scaling factor to prevent large dot-product values from "saturating" the softmax (preventing vanishing gradients).
*   **$M$**: Causal Mask (Temporal Mask).

### B. Causal (Temporal) Masking
Ensures the Transformer cannot "cheat" by looking at future frames.
$$ M_{i,j} = 
\begin{cases} 
0 & \text{if } j \le i \\
-\infty & \text{if } j > i 
\end{cases} 
$$
*   **Located in**: `model.py` -> `TransformerActor.forward` using `torch.tril`.

### C. Layer Normalization
Stabilizes the distribution of activations between layers.
$$ y = \frac{x - \text{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \cdot \gamma + \beta $$
*   **Located in**: `model.py` -> `TransformerBlock_RL`.

### D. Rotary Positional Embedding (RoPE)
Encodes temporal order by rotating Query ($Q$) and Key ($K$) vectors in 2D space.

#### 1. The 2D Rotation Matrix
To rotate a point $(x, y)$ by an angle $\theta$, we apply the standard rotation matrix:
$$ \begin{pmatrix} x_{new} \\ y_{new} \end{pmatrix} = \begin{pmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} $$
Expanding this gives two fundamental equations for neural feature rotation:
*   $x_{new} = x \cos(\theta) - y \sin(\theta)$
*   $y_{new} = y \cos(\theta) + x \sin(\theta)$

#### 2. The Computational "rotate_half" Trick
To avoid slow matrix multiplication, we implement the **rotate_half** optimization:
If $q = (x, y)$, then $\text{rotate\_half}(q) = (-y, x)$.
The final RoPE implementation is calculated via element-wise multiplication:
$$ q_{\text{rope}} = (q \cdot \cos) + (\text{rotate\_half}(q) \cdot \sin) $$
$$ q_{\text{rope}} = (x \cos, y \cos) + (-y \sin, x \sin) = (x \cos - y \sin, y \cos + x \sin) $$
This perfectly recreates the rotation matrix using only basic GPU addition and multiplication.
*   **Located in**: `model.py` -> `StandardAttention_RL.apply_rope`.

---

## 2. Twin-Delayed DDPG (The Optimizer)

TD3 is an actor-critic algorithm specifically designed to solve the "Overestimation Bias" inherent in Reinforcement Learning.

### A. Clipped Double-Q Learning
The "Twin" critics calculate two independent predictions of success, and we take the most conservative one to avoid over-confidence.
$$ y = r + \gamma \min(Q_{\text{target,1}}(s', a'), Q_{\text{target,2}}(s', a')) $$
*   **Located in**: `td3.py` -> `TD3.train` (Clipped Double-Q).

### B. Target Policy Smoothing
We add clipped noise to the next action to ensure the agent doesn't overfit to specific high-value spikes in the Q-function.
$$ a_{target} = \text{Clip}(\pi_{\text{target}}(s') + \text{Clip}(\epsilon, -c, c), \text{low}, \text{high}) $$
*   **$\epsilon \sim \mathcal{N}(0, \sigma)$**: Gaussian noise.
*   **Located in**: `td3.py` -> `TD3.train` (Policy Noise).

### C. Soft Target Updates (Polyak Updates)
Instead of copying weights directly, we slowly "melt" the current weights into the target brain.
$$ \theta_{target} \leftarrow \tau \theta + (1 - \tau) \theta_{target} $$
*   **Located in**: `td3.py` -> `TD3.train` (Soft Updates).

---

## 3. RLHF: Reinforcement Learning from Human Feedback

The RLHF pipeline uses preference math to teach the AI human values.

### A. Bradley-Terry Preference Model
Maps the neural reward scores of two segments ($\sigma_1, \sigma_2$) to a probability that a human prefers one over the other.
$$ P(\sigma_1 \succ \sigma_2) = \frac{\exp(\sum r_{\sigma_1})}{\exp(\sum r_{\sigma_1}) + \exp(\sum r_{\sigma_2})} = \text{Sigmoid}\left(\sum r_{\sigma_1} - \sum r_{\sigma_2}\right) $$
*   **Located in**: `rlhf_trainer.py` -> `RLHFTrainer.train_step`.

### B. Preference Cross-Entropy Loss
Trains the Reward Model to match human choices ($y=1$ for preference).
$$ \mathcal{L}(\psi) = - \mathbb{E}_{(\sigma_1, \sigma_2, y)} \left[ y \log P(\sigma_1 \succ \sigma_2) + (1-y) \log P(\sigma_2 \succ \sigma_1) \right] $$
*   **Located in**: `rlhf_trainer.py` -> `RLHFTrainer.train_step` (BCE with Logits).

---

## 4. Normalization & Diagnostics (The Lab)

### A. Z-Score Standardization
Centers environmental data around 0 with a standard deviation of 1.
$$ z = \frac{x - \mu}{\sigma + \epsilon} $$
*   **Located in**: `model.py` -> `Normalizer`.

### B. Shannon Attention Entropy
Measures the "Focus" or "Sharpness" of the Transformer's brain.
$$ H(A_t) = - \sum_{i=1}^{L} P(A_{t,i}) \log P(A_{t,i}) $$
*   **High $H$**: Broad attention (scanning history).
*   **Low $H$**: Sharp attention (calculating a specific detail).
*   **Located in**: `attention_diagnostics.py` -> `calculate_entropy_per_step`.

### C. Welford's Algorithm (Chan et al.)
Used for the Parallel update of running mean and variance without storing the entire dataset.
$$ \mu_{new} = \mu_{old} + \frac{n_b}{n_{tot}}(\mu_{batch} - \mu_{old}) $$
*   **Located in**: `model.py` -> `update_mean_var_count_from_moments`.

---

## 5. Hyperparameter Summary

| Parameter | Symbol | Value | Reason |
| :--- | :--- | :--- | :--- |
| **Discount Factor** | $\gamma$ | $0.99$ | Balances long-term goals vs. instant rewards. |
| **Update Rate** | $\tau$ | $0.005$ | Ensures the "Frozen Brain" updates slowly and stably. |
| **Sequence Length** | $L$ | $32$ | Allows a 1.6-second "Memory Buffer." |
| **Regularization** | $L2$ | $1e-4$ | Prevents the Reward Model from "Over-judging." |
| **Precision** | $AMP$ | $16-bit$ | Turbocharges GPU math using GradScaler multipliers. |
