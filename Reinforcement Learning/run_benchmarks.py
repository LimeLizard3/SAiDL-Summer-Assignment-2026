# pyrefly: ignore [missing-import]
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    # Run synchronously and stream stdout/stderr
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] Command failed with return code {process.returncode}")
    else:
        print(f"[SUCCESS] Command completed successfully")

def main():
    max_steps = 150000
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results_xlstm")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(script_dir, "Graphs"), exist_ok=True)

    # 1. RUN TRAINING RUNS
    # We run side-by-side comparisons of xLSTM vs. Transformer under POMDP (Partial) and Delayed-Reward configurations
    runs = [
        # POMDP Challenge (masked velocities)
        {"policy": "transformer", "mode": "partial", "steps": max_steps},
        {"policy": "xlstm", "mode": "partial", "steps": max_steps},
        # Delayed Reward Challenge (K=10 step delay)
        {"policy": "transformer", "mode": "delayed", "steps": max_steps},
        {"policy": "xlstm", "mode": "delayed", "steps": max_steps},
    ]

    for run in runs:
        file_name = f"TD3_{run['policy']}_L16_M_{run['mode']}_S0.npy"
        results_path = os.path.join(save_dir, file_name)
        
        # Skip if already exists and is fully completed to allow resuming/caching
        if os.path.exists(results_path):
            try:
                data = np.load(results_path)
                expected_len = run['steps'] // 5000
                if len(data) >= expected_len:
                    print(f"Results for {run['policy']} ({run['mode']}) already exist and are complete ({len(data)}/{expected_len} evals). Skipping training.")
                    continue
                else:
                    print(f"Results for {run['policy']} ({run['mode']}) exist but are incomplete ({len(data)}/{expected_len} evals). Restarting training...")
            except Exception as e:
                print(f"Could not load existing results for {run['policy']} ({run['mode']}): {e}. Restarting training...")
            
        cmd = f"python -u \"{os.path.join(script_dir, 'train_xlstm.py')}\" --policy {run['policy']} --mode {run['mode']} --max_timesteps {run['steps']} --seq_len 16 --seed 0 --save_dir \"{save_dir}\""
        run_cmd(cmd)

    # 2. PLOT RESULTS
    print("\n>>> Generating Comparison Plots...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Define colors and styles
    colors = {"xlstm": "#1f77b4", "transformer": "#ff7f0e"} # Sleek blue and orange
    
    modes = ["partial", "delayed"]
    mode_titles = {
        "partial": "POMDP: Hidden-Velocity Challenge (Hopper-v5)",
        "delayed": "Sparse Credit Assignment: Delayed-Reward Challenge (Hopper-v5)"
    }
    
    for mode in modes:
        plt.figure(figsize=(10, 6), dpi=300)
        
        for policy in ["transformer", "xlstm"]:
            file_name = f"TD3_{policy}_L16_M_{mode}_S0.npy"
            results_path = os.path.join(save_dir, file_name)
            
            if os.path.exists(results_path):
                data = np.load(results_path)
                x = np.arange(len(data)) * 5000  # Evaluated every 5k steps
                
                # Apply moving average smoothing for cleaner visualization
                window_size = 3
                if len(data) >= window_size:
                    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                    smoothed_x = x[window_size-1:]
                    plt.plot(smoothed_x, smoothed_data, label=f"{policy.upper()}-TD3 (Smoothed)", color=colors[policy], linewidth=2.5)
                    plt.plot(x, data, color=colors[policy], alpha=0.15, linewidth=1.0) # Raw curve in background
                else:
                    plt.plot(x, data, label=f"{policy.upper()}-TD3", color=colors[policy], linewidth=2.0)
            else:
                print(f"[WARNING] Missing results file for {policy} in {mode} mode.")

        plt.title(mode_titles[mode], fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Average Return (10 evaluation episodes)", fontsize=12)
        plt.legend(frameon=True, fontsize=11, facecolor='white', edgecolor='none', shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        mode_suffix = "POMDP" if mode == "partial" else "DelayedReward"
        plot_path = os.path.join(script_dir, "Graphs", f"xLSTM_vs_TD3_{mode_suffix}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
