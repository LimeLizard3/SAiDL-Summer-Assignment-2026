import numpy as np
import matplotlib.pyplot as plt
import os

def plot_baseline():
    plt.figure(figsize=(10, 6))
    seeds = [0, 1, 2]
    all_rewards = []
    
    for seed in seeds:
        path = f"results/TD3_Hopper-v5_{seed}.npy"
        if os.path.exists(path):
            res = np.load(path)
            plt.plot(res, alpha=0.3, label=f"MLP Seed {seed}")
            all_rewards.append(res)
    
    if all_rewards:
        mean_rewards = np.mean(all_rewards, axis=0)
        plt.plot(mean_rewards, color='black', linewidth=2, label="MLP Mean")
        
    plt.title("Baseline: Standard TD3 (MLP) on Hopper-v5 (Original Unoptimized)")
    plt.xlabel("Steps (x10000)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Old Graphs/TD3_Baseline_Results_Old.png")
    plt.close()

def plot_transformer_comparison():
    plt.figure(figsize=(10, 6))
    lengths = [4, 8, 16, 32]
    
    # Plot MLP baseline mean for reference
    mlp_rewards = []
    for seed in [0, 1, 2]:
        path = f"results/TD3_Hopper-v5_{seed}.npy"
        if os.path.exists(path):
            mlp_rewards.append(np.load(path))
    if mlp_rewards:
        plt.plot(np.mean(mlp_rewards, axis=0), color='black', linestyle='--', alpha=0.5, label="Standard TD3 (MLP)")

    for L in lengths:
        path = f"results_transformer/TD3_Transformer_L{L}_S0.npy"
        if os.path.exists(path):
            res = np.load(path)
            plt.plot(res, label=f"Transformer (L={L})")
            
    plt.title("Hopper-v5: Transformer vs. MLP Baseline (Original 1M Steps)")
    plt.xlabel("Training Steps (x10000)")
    plt.ylabel("Average Evaluation Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Old Graphs/Transformer_Comparison_Old.png")
    plt.close()

def plot_rlhf():
    plt.figure(figsize=(10, 6))
    
    # Plot MLP baseline for reference (Seed 0)
    mlp_path = "results/TD3_Hopper-v5_0.npy"
    if os.path.exists(mlp_path):
        mlp_res = np.load(mlp_path)
        plt.plot(mlp_res, color='grey', linestyle='--', alpha=0.5, label="Standard TD3 (MLP)")

    # Plot RLHF performance
    rlhf_path = "results_rlhf/TD3_RLHF_S0.npy"
    if os.path.exists(rlhf_path):
        rlhf_res = np.load(rlhf_path)
        plt.plot(rlhf_res, color='purple', linewidth=2, label="Transformer (RLHF - Learned Rewards)")
        
        # Calculate Trendline (using a simple polynomial fit like the user's graph)
        if len(rlhf_res) > 5:
            z = np.polyfit(np.arange(len(rlhf_res)), rlhf_res, 3)
            p = np.poly1d(z)
            plt.plot(p(np.arange(len(rlhf_res))), color='red', linestyle='--', alpha=0.5, label="RLHF Trend")

    plt.title("Task 2d: RLHF Ground-Truth Performance (Original Seed 0)")
    plt.xlabel("Training Steps (x10000)")
    plt.ylabel("Reward (Environment)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Old Graphs/RLHF_Performance_Old.png")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists("Old Graphs"):
        os.makedirs("Old Graphs")
    plot_baseline()
    plot_transformer_comparison()
    plot_rlhf()
    print("Historical graphs regenerated from original .npy files.")
