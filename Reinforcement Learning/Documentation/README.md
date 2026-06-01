# SAiDL Reinforcement Learning: Continuous Control under Partial Observability

This directory contains the Reinforcement Learning (RL) component of the SAiDL Summer Induction Assignment. The focus is on implementing robust sequence-conditioned policies for continuous locomotion under partial observability, noisy inputs, and reward delays in the MuJoCo `Hopper-v5` environment.

---

## Status & Progress

All phases, tasks, and bonus objectives have been implemented and evaluated:

* **Phase 1 (TD3 Baseline):** Twin Delayed DDPG (TD3) baseline policy using MLP networks trained over 1,000,000 steps.
* **Phase 2 (Causal Transformer):** Replaced the reactive MLP with a sequence-conditioned Causal Transformer policy. Stabilized training via independent normalizer tracking, Polyak smoothing ($\tau=0.0005$), and gradient clipping ($0.5$).
* **Stressor Challenges (POMDP & Credit Delay):** Evaluated policy resilience under velocity sensor masking, Gaussian static noise ($\sigma=0.1$), and delayed rewards ($K=10$).
* **Jury-Aligned RLHF:** Trained an ensemble of 3 reward models using simulated preferences, aligned via the Eternal Textbook replay mixture.
* **In-Context Algorithm Distillation (Bonus C):** Distilled online learning histories using Causal Transformers with Scheduled Action Masking ($20\% \to 80\%$ dropout) and context jitter.
* **Recurrent xLSTM Policies (Bonus D):** Integrated mLSTM/sLSTM stacks as the recurrent policy backbone, optimized via loop projection offloading.

---

## Documentation & Code Walkthroughs

Detailed reviews and walkthroughs for all files in this module:

| Document | Covered Files | Description |
| :--- | :--- | :--- |
| **[Report.pdf](./SAiDL_RL_Report.pdf)** | All Files | Complete PDF academic paper detailing findings. |
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