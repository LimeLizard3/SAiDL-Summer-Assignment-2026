# 🤖 Reinforcement Learning Assignment

This directory is dedicated to the Reinforcement Learning (RL) component of the **SAiDL Summer Induction Assignment**.

## 🚀 Status
- **Phase 1 (Baseline)**: [COMPLETED] - TD3 on Hopper-v5 (1M steps, 3 seeds).
- **Phase 2 (Transformer)**: [UPCOMING] - Integrating Causal Transformer.
- **Goal**: Implement and verify RL algorithms based on the assignment criteria.

---

## 📈 Phase 1: TD3 Baseline
We successfully implemented a robust **Twin Delayed DDPG (TD3)** baseline for the **Hopper-v5** environment. 
- **Algorithm**: TD3 with Twin Critics and Policy Smoothing.
- **Training**: 1,000,000 steps per seed across 3 random seeds.
- **Results**: The agent successfully learned to hop, reaching average rewards of ~1000.
- **Documentation**: Detailed line-by-line breakdowns of the code are available in the `documentation/` folder.

---

## 🏗️ Next Steps: Phase 2
The next phase involve replacing the MLP Actor with a **Causal Transformer** and analyzing its performance under partial observability and noise.