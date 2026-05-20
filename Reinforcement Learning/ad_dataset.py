import os
import pickle
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader

class ADDataset(Dataset): #Passing Dataset tells PyTorch to make the class a specialized version of std Dataset
    def __init__(self, data_dir, seq_len=100):
        self.seq_len = seq_len
        data = []
        
        # Load all pkl files in order
        files = sorted([f for f in os.listdir(data_dir) if f.startswith("history_chunk_") and f.endswith(".pkl")], 
                       key=lambda x: int(x.split('_')[2].split('.')[0]) if x.split('_')[2].split('.')[0].isdigit() else 999999)
        #f for f creates a list by looking at every file f inside data_dir, then it filters
        #sorted and key basically ensure that it's ordered properly, and the whole line with x is basically grabbing the number

        
        print(f"Loading {len(files)} data chunks...")
        
        for i, file in enumerate(files):
            if i % 10 == 0:
                print(f"Loading chunk {i}/{len(files)}...")
            with open(os.path.join(data_dir, file), 'rb') as f:
                chunk = pickle.load(f)
                data.extend(chunk) #Adds data 
        
        print("Pre-converting dataset to flat arrays...")
        self.states = np.array([step['obs'] for step in data], dtype=np.float32)
        self.actions = np.array([step['action'] for step in data], dtype=np.float32)
        self.rewards = np.array([step['reward'] for step in data], dtype=np.float32).reshape(-1, 1)
        
        self.total_steps = len(self.states)
        print(f"Dataset loaded: {self.total_steps} total transitions.")

    def __len__(self): #Required function for custom datasets
        # We want to be able to sample any sequence of length seq_len
        return self.total_steps - self.seq_len

    def __getitem__(self, idx):
        # Slice the pre-converted arrays instantly
        states = self.states[idx : idx + self.seq_len]
        actions = self.actions[idx : idx + self.seq_len]
        rewards = self.rewards[idx : idx + self.seq_len]
        
        return {
            'states': torch.from_numpy(states),
            'actions': torch.from_numpy(actions),
            'rewards': torch.from_numpy(rewards)
        }

if __name__ == "__main__":
    # Test the dataset
    ds = ADDataset(data_dir="ad_dataset", seq_len=50)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    
    batch = next(iter(dl)) #next gives the very next set and iter gets the next iteration. This replaces a for loop
    print("Batch shapes:")
    print(f"States:  {batch['states'].shape}")  # [B, K, state_dim]
    print(f"Actions: {batch['actions'].shape}") # [B, K, action_dim]
    print(f"Rewards: {batch['rewards'].shape}") # [B, K, 1]
