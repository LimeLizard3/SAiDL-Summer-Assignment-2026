import numpy as np
import torch

class ReplayBuffer:
    """A standard Replay Buffer to store experiences."""
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def add(self, state, action, next_state, reward, done):
        """Add a new transition to the buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample_sequence(self, batch_size, seq_len):
        """Sample a batch of sequences for Transformer actors."""
        if self.size - 1 <= seq_len:
            # Not enough data yet to sample a full sequence + 1 next step
            return None
            
        ind = np.random.randint(seq_len, self.size - 1, size=batch_size) # -1 to ensure next_state exists
        
        s_seq = np.zeros((batch_size, seq_len, self.state.shape[1]))
        a_seq = np.zeros((batch_size, seq_len, self.action.shape[1]))
        ns_seq = np.zeros((batch_size, seq_len, self.state.shape[1]))
        na_seq = np.zeros((batch_size, seq_len, self.action.shape[1]))
        
        for i, idx in enumerate(ind):
            # 1. Current Step Sequence
            s_win = self.state[idx-seq_len+1 : idx+1]
            a_win = self.action[idx-seq_len+1 : idx] # Length L-1
            n_done_win = self.not_done[idx-seq_len+1 : idx+1]
            
            boundary = np.where(n_done_win[:-1] == 0)[0]
            curr_start = boundary[-1] + 1 if len(boundary) > 0 else 0
            
            s_seq[i, curr_start:] = s_win[curr_start:]
            # Action history ends 1 step before the current state
            if curr_start < seq_len - 1:
                a_seq[i, curr_start : seq_len-1] = a_win[curr_start:]
            
            # 2. Next Step Sequence
            ns_win = np.concatenate([self.state[idx-seq_len+2 : idx+1], self.next_state[idx:idx+1]], axis=0)
            na_win = self.action[idx-seq_len+2 : idx+1] # Length L-1
            
            ns_boundary = np.where(n_done_win[1:] == 0)[0]
            next_start = ns_boundary[-1] + 1 if len(ns_boundary) > 0 else 0
            
            ns_seq[i, next_start:] = ns_win[next_start:]
            if next_start < seq_len - 1:
                na_seq[i, next_start : seq_len-1] = na_win[next_start:]

        return (
            torch.FloatTensor(s_seq).to(self.device),
            torch.FloatTensor(a_seq).to(self.device),
            torch.FloatTensor(ns_seq).to(self.device),
            torch.FloatTensor(na_seq).to(self.device),
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device), # Added current action
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
