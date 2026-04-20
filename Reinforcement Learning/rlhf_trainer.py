import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RLHFTrainer:
    """
    Handles the preference-based training of the RewardModel.
    Implements the Bradley-Terry model for preference learning.
    """
    def __init__(self, reward_model, lr=3e-4, device="cuda"):
        self.reward_model = reward_model
        # REPAIR LINE: Added weight_decay (L2 regularization).
        # This prevents the Judge from 'over-committing' to noisy student data,
        # keeping his opinions smooth and stable.
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=lr, weight_decay=1e-4) 
        self.device = device

    def train_step(self, replay_buffer, batch_size=64, segment_len=50):
        """Perform one step of preference learning."""
        # 1. Sample pairs of segments
        batch = replay_buffer.sample_segment_pairs(batch_size, segment_len)
        if batch is None:
            return 0.0
            
        s1, a1, r1_true, s2, a2, r2_true = batch #In replay_buffer, it O/Ps values in this order

        # 2. Generate labels (the 'Human' choice)
        # Based on Christiano et al: 1 if r1 > r2, 0.5 if tied, 0 if r2 > r1
        # We use a epsilon margin to decide ties if we want, but usually straight comparison works.
        labels = (r1_true > r2_true).float()
        
        # 3. Calculate predicted segment rewards
        # Shape: (Batch, 1)
        sum_r1 = self.reward_model.get_segment_reward(s1, a1)
        sum_r2 = self.reward_model.get_segment_reward(s2, a2)
        
        # 4. Bradley-Terry Loss (Log-likelihood of the human preference)
        # P(sigma1 > sigma2) = exp(sum_r1) / (exp(sum_r1) + exp(sum_r2))
        # This is equivalent to sigmoid(sum_r1 - sum_r2)
        logits = sum_r1 - sum_r2
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        #Pushes logits through softmax and gets a %, then compares it to labels to get loss
        
        # 5. Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train_epoch(self, replay_buffer, iterations=100, batch_size=64, segment_len=50):
        """Trains for multiple iterations."""
        losses = []
        for _ in range(iterations):
            loss = self.train_step(replay_buffer, batch_size, segment_len)
            losses.append(loss)
        return np.mean(losses)
