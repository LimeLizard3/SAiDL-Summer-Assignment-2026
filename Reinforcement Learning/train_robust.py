import gymnasium as gym
import torch
import numpy as np
import os
from td3 import TD3
from model import Normalizer
from replay_buffer import ReplayBuffer

# THE ROBUSTNESS SPRINT: THE SCIENTIFIC EXTREME
# This script is designed for "Generalization Testing." 
# We take a high-performing Champion and force it into an environment where 
# it can NO LONGER see its own velocity (Partial Observability).
# This forces the Transformer to calculate speed from its internal memory buffer.

def train_robust(seed=0, steps=20000):
    # 1. Environment Setup: Using the standard Hopper benchmark
    env_name = "Hopper-v5"
    env = gym.make(env_name)
    
    # 2. Reproducibility: Locking the seeds ensures the "Survivor" can be recreated
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # 3. Hardware Acceleration: Use GPU (CUDA) if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. The Architecture: We use L=32 to give the robot a long 1.6-second memory window
    seq_len = 32
    policy = TD3(state_dim, action_dim, max_action, device, use_transformer=True, seq_len=seq_len)
    
    # 5. Transfer Learning (VITAL): We don't start from scratch.
    # We load our best-performing L=32 Champion to "re-train" its brain.
    model_path = "./models/TD3_Transformer_L32_S0"
    policy.actor.load_state_dict(torch.load(model_path + "_actor", map_location=device))
    print(f"Loaded Champion: {model_path}")
    
    # 6. Sensory Calibration: We load the Expert's eyes (Normalizer).
    # This ensures the robot still understands the "scale" of the world.
    normalizer = Normalizer(shape=(state_dim,))
    normalizer.load(model_path)
    
    # 7. Experience Management: Setting up the buffer for the "Sprint"
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # 8. THE BLINDFOLD (The core of Task 3):
    # This nested function "destroys" the velocity sensors (index 6 onwards)
    # forcing the agent to rely entirely on history to navigate.
    def mask_velocity(s):
        s_masked = s.copy()
        s_masked[6:] = 0 # Blinding the velocity sensors
        return s_masked

    # 9. Initialization: Start the simulation with a clean slate
    state, _ = env.reset(seed=seed)
    state_history = []
    action_history = []
    
    # Pre-fill memory with "blank" masked frames to initialize the Transformer
    for _ in range(seq_len):
        state_history.append(mask_velocity(np.zeros(state_dim)))
        action_history.append(np.zeros(action_dim))

    print(f"Starting Robustness Sprint ({steps} steps)...")
    
    # 10. THE TRAINING LOOP: The "Challenge" phase
    for t in range(steps):
        # A. Apply the Blindfold to the current observation
        obs_masked = mask_velocity(state)
        
        # B. Normalize the blinded input using the Expert's scales
        state_norm = normalizer(obs_masked, update=False)
        
        # C. Update Memory Buffer (Sliding Window of 32 steps)
        state_history.append(state_norm)
        if len(state_history) > seq_len:
            state_history.pop(0)
            
        # D. Brain Execution: Transformer looks at memory to "Guess" its speed
        action = policy.select_action(state_norm, state_history, action_history)
        
        # E. Physics Simulation: Apply the action to the world
        next_state, reward, terminated, truncated, _ = env.step(action)
        done_bool = float(terminated or truncated)
        
        # F. Blind the next state as well
        next_obs_masked = mask_velocity(next_state)
        next_state_norm = normalizer(next_obs_masked, update=False)
        
        # G. Replay Buffer storage (Teaching the Judge how to survive being blind)
        replay_buffer.add(state_norm, action, next_state_norm, reward, done_bool)
        
        state = next_state
        action_history.append(action)
        if len(action_history) > seq_len:
            action_history.pop(0)

        # 11. Policy Update: Re-calculating the neural weights
        # We start training after 2000 steps of exploration
        if t > 2000:
            policy.train(replay_buffer, batch_size=64)

        # 12. Episode Resets: Handle collisions or success/timeouts
        if terminated or truncated:
            state, _ = env.reset()
            state_history = []
            action_history = []
            for _ in range(seq_len):
                state_history.append(mask_velocity(np.zeros(state_dim)))
                action_history.append(np.zeros(action_dim))

        # 13. Progress Monitoring: Log every 5000 steps
        if (t + 1) % 5000 == 0:
            print(f"Step {t+1}/{steps} complete.")

    # 14. DATA PRESERVATION: Save the newly developed "Survivor" brain.
    # This model now contains the "Dual-Anchor" attention strategy discovered in Task 3.
    save_path = "./models/TD3_Transformer_Robust"
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(policy.actor.state_dict(), save_path + "_actor")
    normalizer.save(save_path)
    print(f"[SUCCESS] Robust Specialist brain saved to {save_path}")

if __name__ == "__main__":
    train_robust()
