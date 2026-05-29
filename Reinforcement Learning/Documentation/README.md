# 🤖 SAiDL Reinforcement Learning: Hardened Control under Partial Observability

This directory is dedicated to the Reinforcement Learning (RL) component of the **SAiDL Summer Induction Assignment**. Here, we focus on engineering robust sequence-conditioned policies capable of continuous locomotion under partial observability, noisy inputs, and reward delays in the MuJoCo `Hopper-v5` environment.

---

## 🚀 Status & Progress

All phases, subtasks, and bonus objectives have been **fully implemented, stabilized, and evaluated**:

*   **Phase 1 (TD3 Baseline):** [COMPLETED] 
    *   Twin Delayed DDPG (TD3) baseline implemented using Multi-Layer Perceptron (MLP) networks. 
    *   Trained over 3 seeds for 1,000,000 steps, achieving stable hopping.
*   **Phase 2 (Causal Transformer):** [COMPLETED]
    *   Replaced reactive MLP with a sequence-conditioned Causal Transformer policy to retain past historical states and actions.
    *   Stabilized training via online normalizer correction (independent tracking), Polyak smoothing ($\tau=0.0005$), and gradient clipping ($0.5$).
*   **Stressor Challenges (POMDP & Credit Delay):** [COMPLETED]
    *   Tested policy resilience under velocity sensor masking, Gaussian static noise ($\sigma=0.1$), and delayed rewards ($K=10$).
    *   Proven that context-aware Causal Transformers outperform reactive MLPs by performing implicit temporal calculus on position history.
*   **Jury-Aligned RLHF:** [COMPLETED]
    *   Trained a Pessimistic Consensus Jury of 3 reward models using simulated preferences.
    *   Stabilized online alignment using the Eternal Textbook protocol to prevent catastrophic forgetting.
*   **In-Context Algorithm Distillation (Bonus C):** [COMPLETED]
    *   Offline distilled online learning histories using Scheduled Action Masking ($20\% \to 80\%$ dropout) and Context Jitter.
    *   Successfully evaluated in-context adaptation online in MuJoCo, achieving a peak reward $>600$ without backpropagation.
*   **Recurrent xLSTM Policies (Bonus D):** [COMPLETED]
    *   Integrated mLSTM/sLSTM stacks as the recurrent policy backbone.
    *   Optimized GPU performance via Recurrent Loop Projection Offloading.

---

## 📂 Documentation & Code Explanations

We have compiled detailed, line-by-line guides and theoretical breakdowns for all files in this module:

| Document | Covered Files | Description |
| :--- | :--- | :--- |
| **[Report.tex](./SAiDL_RL_Report.tex)** | All Files | Complete LaTeX academic paper detailing findings. |
| **[Optimizations.md](./Optimizations.md)** | All Files | Mathematical and engineering optimizations log. |
| **[TD3 Architecture](./td3_explanation.md)** | [`td3.py`](../td3.py) | Clipped double-Q, target smoothing, and delay updates. |
| **[MLP Training](./train_explanation.md)** | [`train.py`](../train.py) | Baseline MLP control loop and execution. |
| **[Transformer Actor](./model_explanation.md)** | [`model.py`](../model.py) | Causal attention block and RoPE/ALiBi embeddings. |
| **[Replay Buffer](./replay_buffer_explanation.md)** | [`replay_buffer.py`](../replay_buffer.py) | Consecutive trajectory sequence sampling. |
| **[Transformer Training](./transformer_explanation.md)** | [`train_transformer.py`](../train_transformer.py) | Transformer training control loop and args. |
| **[xLSTM Backbone](./xlstm_explanation.md)** | [`xlstm_model.py`](../xlstm_model.py), [`train_xlstm.py`](../train_xlstm.py) | Recurrent stack and loop projection offloading. |
| **[Algorithm Distillation](./algorithm_distillation_explanation.md)** | [`generate_ad_dataset.py`](../generate_ad_dataset.py), [`ad_dataset.py`](../ad_dataset.py), [`ad_model.py`](../ad_model.py), [`train_ad.py`](../train_ad.py), [`eval_ad.py`](../eval_ad.py) | In-context adaptation, masking, and VRAM scaling. |
| **[Jury RLHF](./rlhf_explanation.md)** | [`reward_model.py`](../reward_model.py), [`rlhf_pretraining.py`](../rlhf_pretraining.py), [`rlhf_trainer.py`](../rlhf_trainer.py), [`train_rlhf.py`](../train_rlhf.py) | Bradley-Terry ensemble and Eternal Textbook mixture. |
| **[Benchmarking](./benchmarking_explanation.md)** | [`attention_diagnostics.py`](../attention_diagnostics.py), [`attribution_analysis.py`](../attribution_analysis.py), [`benchmark_pomdp.py`](../benchmark_pomdp.py), [`benchmark_pos_encodings.py`](../benchmark_pos_encodings.py), [`delayed_reward_wrapper.py`](../delayed_reward_wrapper.py), [`plot_historical.py`](../plot_historical.py), [`robustness_analysis.py`](../robustness_analysis.py), [`run_benchmarks.py`](../run_benchmarks.py) | Relevancy propagation, diagnostics, and plots. |
| **[Math Framework](./mathematical_framework.md)** | - | Theoretical POMDP formulation and credit assignment. |