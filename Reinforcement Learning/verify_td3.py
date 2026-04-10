from train import train_seed
import os

if __name__ == "__main__":
    # Short test to verify logic: 10,000 steps, start training after 500 steps
    # We only run 1 seed for verification.
    print("Starting verification test...")
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # We override the parameters for a quick test
    train_seed(seed=0, env_name="Hopper-v5", max_timesteps=2000, start_timesteps=1000, batch_size=10)
    print("Verification successful!")
