import torch
import numpy as np
import os
from attention_diagnostics import run_diagnostics

def print_raw_weights():
    model_path = "./models/TD3_Transformer_L32_S0"
    robust_path = "./models/TD3_Transformer_Robust"
    seq_len = 32
    
    if not os.path.exists(model_path + "_actor") or not os.path.exists(robust_path + "_actor"):
        print("Models not found.")
        return

    # Run rollouts
    print("Collecting diagnostics (this may take a moment)...")
    clean_weights, _ = run_diagnostics(model_path, mode="clean", seq_len=seq_len, episodes=1)
    partial_weights, _ = run_diagnostics(robust_path, mode="partial", seq_len=seq_len, episodes=1)

    print("\n" + "="*50)
    print("MEAN ATTENTION WEIGHTS COMPARISON (L=32)")
    print("="*50)
    print(f"{'Step':<10} | {'Clean':<15} | {'Hidden Vel':<15}")
    print("-" * 50)
    
    # We look at the top 10 most attended frames
    indices = np.argsort(clean_weights)[::-1]
    
    for i in range(seq_len):
        step = i - 31
        print(f"{step:<10} | {clean_weights[i]:<15.6f} | {partial_weights[i]:<15.6f}")

    print("\nTOTAL ATTENTION (PAST 10 FRAMES):")
    clean_past = np.sum(clean_weights[:-10])
    partial_past = np.sum(partial_weights[:-10])
    print(f"Clean:   {clean_past:.4f}")
    print(f"Partial: {partial_past:.4f}")

if __name__ == "__main__":
    print_raw_weights()
