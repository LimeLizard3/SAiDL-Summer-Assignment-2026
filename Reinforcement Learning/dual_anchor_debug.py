import torch
import numpy as np
import os
from attention_diagnostics import run_diagnostics

def analyze_spikes():
    model_path = "./models/TD3_Transformer_L32_S0"
    robust_path = "./models/TD3_Transformer_Robust"
    seq_len = 32
    
    if not os.path.exists(model_path + "_actor") or not os.path.exists(robust_path + "_actor"):
        print("Models not found.")
        return

    # Run rollouts
    print("Capturing brain waves...")
    clean_weights, _ = run_diagnostics(model_path, mode="clean", seq_len=seq_len, episodes=1)
    partial_weights, _ = run_diagnostics(robust_path, mode="partial", seq_len=seq_len, episodes=1)

    print("\n" + "="*50)
    print("THE DUAL-ANCHOR DISCOVERY (L=32)")
    print("="*50)
    
    print("\n[VITAL COMPARISON: THE CALCULUS SPIKE]")
    print(f"Step -1 (Just happened): Clean={clean_weights[-2]:.4f} | Robust={partial_weights[-2]:.4f}")
    print(f"Step -2 (Recent history): Clean={clean_weights[-3]:.4f} | Robust={partial_weights[-3]:.4f}")
    
    print("\n[VITAL COMPARISON: THE DEEP MEMORY]")
    print(f"Step -31 (Oldest): Clean={clean_weights[0]:.4f} | Robust={partial_weights[0]:.4f}")
    
    print("\n[VITAL COMPARISON: THE MIDDLE NOISE]")
    print(f"Step -15 (Middle): Clean={clean_weights[16]:.4f}  | Robust={partial_weights[16]:.4f}")

    diff = (partial_weights[-2] + partial_weights[-3]) - (clean_weights[-2] + clean_weights[-3])
    print(f"\nCONCLUSION: The Robust model is paying {diff*100:.1f}% MORE attention to recent 'Calculus' frames than the clean model.")

if __name__ == "__main__":
    analyze_spikes()
