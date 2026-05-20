# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import os

def plot_final_rlhf():
    plt.figure(figsize=(12, 7))
    
    # 1. Plot the CORRECT Optimized MLP Baseline
    mlp_path = "results/TD3_Hopper-v5_0_stable.npy"
    if os.path.exists(mlp_path):
        mlp_res = np.load(mlp_path)
        # Resample to match RLHF evaluation frequency if needed
        plt.plot(mlp_res, color='grey', linestyle='--', alpha=0.6, label="Optimized TD3 (MLP) - Stable Baseline")
    
    # 2. Plot the Transformer RLHF Performance
    rlhf_path = "results_rlhf/TD3_RLHF_S0.npy"
    rlhf_res = None
    if os.path.exists(rlhf_path):
        rlhf_res = np.load(rlhf_path)
        # Plot the actual points
        plt.plot(rlhf_res, color='purple', linewidth=2, label="Transformer (RLHF - Learned Rewards)")
        
        # 3. Calculate a proper Moving Average Trendline (instead of a risky polynomial)
        if len(rlhf_res) > 5:
            window = 5
            trend = np.convolve(rlhf_res, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(rlhf_res)), trend, color='red', linestyle='-', linewidth=2, label="RLHF Performance Trend")

    plt.title("Task 2d: Transformer RLHF vs. Optimized MLP Baseline", fontsize=14)
    plt.xlabel("Evaluation Milestones (Every 5,000 steps)", fontsize=12)
    plt.ylabel("Reward (Environment Ground Truth)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure the axis matches our actual training duration
    if rlhf_res is not None:
        plt.xlim(0, max(len(rlhf_res), 60))
        
    plt.savefig("Graphs/RLHF_Final_Comparison.png", dpi=300)
    print("Corrected Victory Graph saved to Graphs/RLHF_Final_Comparison.png")
    plt.close()

if __name__ == "__main__":
    plot_final_rlhf()
