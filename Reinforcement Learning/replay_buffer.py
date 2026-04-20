import numpy as np
import torch

class ReplayBuffer:
    """A standard Replay Buffer to store experiences."""
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0 #Keeps track of where in the array the next experience should be written (it's basically an index)
        self.size = 0 #Keeps track of how many experiences are CURRENTLY in the buffer (Grows to max_size)

        self.state = np.zeros((max_size, state_dim)) #state_dim is a 1D array full of values for that specific state. self.state stacks them ontop of eachother to form a 2D array
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim)) #Game engine passes next_state to our replay buffer
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        #For index 99 say, we have 3 things: self.state[99] (Before picture), self.action[99] (Decision), self.next_state[99] (Future state {100th state})

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
        self.not_done[self.ptr] = 1. - done #if 0, it tells the algorithm that it's done and that that was the last state

        self.ptr = (self.ptr + 1) % self.max_size #Once we reach max memory, % ensures that we go back to the beginning and start overwritng old data
        self.size = min(self.size + 1, self.max_size) #Increases count of how many items are in buffer and caps it at max_size

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        ind = np.random.randint(0, self.size, size=batch_size) #Why self.size and not self.max_size? Imagine if we haven't reached max_size yet!
                                                               #batch_size just tells NumPy how many random numbers to generate

        return (
            torch.FloatTensor(self.state[ind]).to(self.device), #.to device moves data from comp's normal RAM into the GPU's memory 
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device) #PyTorch NNs need it to be in tensors for it to understand and do math on it
        ) #NumPy instantly goes to the giant self.state grid and yanks out the exact rows given by ind and bundles them into a smaller 2D grid all at once

    def sample_sequence(self, batch_size, seq_len):
        """Sample a batch of sequences for Transformer actors (it gets a memory now)."""
        if self.size - 1 <= seq_len:
            # Not enough data yet to sample a full sequence + 1 next step
            return None
            
        ind = np.random.randint(seq_len, self.size - 1, size=batch_size) #If we're at state 99 of 0 to 99, we need to find next_state but that wont exist, so we -1
        
        s_seq = np.zeros((batch_size, seq_len, self.state.shape[1]))
        a_seq = np.zeros((batch_size, seq_len, self.action.shape[1]))
        ns_seq = np.zeros((batch_size, seq_len, self.state.shape[1]))
        na_seq = np.zeros((batch_size, seq_len, self.action.shape[1]))
        #We're padding with 0s because we're respecting the "done" boundary. Imagine if it assumed that once you die you teleport so it's a great
        #way to travel, so, we pad everything with 0s (these are the "blank books" we're starting out with)
        for i, idx in enumerate(ind): #Loop through ind, but also provide a counter on which loop I'm on 
             #Sadly we require a for loop here because dealing with sequences and boundaries are complex enough to warrant it

            # 1. Current Step Sequence (The windows)
            s_win = self.state[idx-seq_len+1 : idx+1]
            a_win = self.action[idx-seq_len+1 : idx] #Length L-1 as transformer is supposed to look at current state, but not the future
            n_done_win = self.not_done[idx-seq_len+1 : idx+1] #Our 1D array of flags
            
            boundary = np.where(n_done_win[:-1] == 0)[0] #Checks the past for deaths
            #[:-1] slices off the last index. If Mario dies on the last action, it doesn't matter as it's a "perfect life cycle"
            #.where(...)[0] is just formatting so that we don't get a weird tuple
            curr_start = boundary[-1] + 1 if len(boundary) > 0 else 0
            #First half of the code is if there's a death, and if there IS a death, we only grab the latest one [:-1]
            #Once we grab the frame of latest death, we add +1 to that, as that's where the next starting position is!  
                      
            s_seq[i, curr_start:] = s_win[curr_start:] #Replaces the 0s with safe rows for notebook/loop i (anything before is 0 padded)
            # Action history ends 1 step before the current state
            if curr_start < seq_len - 1:
                a_seq[i, curr_start : seq_len-1] = a_win[curr_start:]
            #IMPT: If you let the AI see the action that it's supposed to take in its current state, it'd cheat. That's why s_seq has one more data point than a_seq

            # 2. Next Step Sequence
            ns_win = np.concatenate([self.state[idx-seq_len+2 : idx+1], self.next_state[idx:idx+1]], axis=0)
            #+2 as we're looking into the future, so we drop the oldest state
            #self.next_state grabs the future state (We write with that type of indexing instead of [idx] to preserve 2D structure for concatenation)
            #We then concatenate vertically to form 1 row
            #Why don't we just write self.state[5:10]? 
            #What if we're taking the newest step ever taken, so the next state doesn't exist yet. We solve this by gluing
            #Say we don't know the future state yet, self.next_state[] does have it though and that's why gluing works but slicing doesn't as the next state is padded with 0s


            na_win = self.action[idx-seq_len+2 : idx+1] #No glue as we're not allowed to see the final action in the sequence
            #Critic uses this array (and ns_win) to evaluate what action is needed for state 100 if we're at state 99
            #ns_win contains the future state in next_state and na_win contains the action done at the state we're in

            #a_seq contains everything we did PRIOR to the state we're in and s_seq contains the state of what we're in right now. It's the PRESENT
            
            ns_boundary = np.where(n_done_win[1:] == 0)[0] #Checks the future for deaths
            next_start = ns_boundary[-1] + 1 if len(ns_boundary) > 0 else 0 #If he died next_start tells the code to start reading right after that else just restart from beginning
            
            ns_seq[i, next_start:] = ns_win[next_start:]
            if next_start < seq_len - 1:
                na_seq[i, next_start : seq_len-1] = na_win[next_start:]


            #So essentially, if we detect a death in the past or the future, we put a 0 there
            #The reason we have next_start is so that we cover both directions at the same time and so that when we look at future state we don't get confused 
            #If we didn't, the AI would think that it could die and just "teleport"
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

    def sample_segment_pairs(self, batch_size, segment_len):
        """Sample pairs of continuous segments for RLHF preference training."""
        if self.size < segment_len + 1:
            return None

        # Sample starting indices for segment 1 and segment 2
        # We ensure they have enough room and don't overlap for better diversity
        ind1 = np.random.randint(0, self.size - segment_len, size=batch_size)
        ind2 = np.random.randint(0, self.size - segment_len, size=batch_size)

        s1 = np.zeros((batch_size, segment_len, self.state.shape[1]))
        a1 = np.zeros((batch_size, segment_len, self.action.shape[1]))
        r1 = np.zeros((batch_size, 1))

        s2 = np.zeros((batch_size, segment_len, self.state.shape[1]))
        a2 = np.zeros((batch_size, segment_len, self.action.shape[1]))
        r2 = np.zeros((batch_size, 1))

        for i in range(batch_size):
            # Extract segment 1
            idx1 = ind1[i]
            s1[i] = self.state[idx1 : idx1 + segment_len]
            a1[i] = self.action[idx1 : idx1 + segment_len]
            # Accumulate ground truth rewards for labeling
            r1[i] = np.sum(self.reward[idx1 : idx1 + segment_len])

            # Extract segment 2
            idx2 = ind2[i]
            s2[i] = self.state[idx2 : idx2 + segment_len]
            a2[i] = self.action[idx2 : idx2 + segment_len]
            r2[i] = np.sum(self.reward[idx2 : idx2 + segment_len])

        return (
            torch.FloatTensor(s1).to(self.device),
            torch.FloatTensor(a1).to(self.device),
            torch.FloatTensor(r1).to(self.device),
            torch.FloatTensor(s2).to(self.device),
            torch.FloatTensor(a2).to(self.device),
            torch.FloatTensor(r2).to(self.device)
        )

    def save(self, filename):
        """Save the replay buffer to a file."""
        np.savez_compressed(
            filename + "_buffer.npz",
            state=self.state,
            action=self.action,
            reward=self.reward,
            next_state=self.next_state,
            not_done=self.not_done,
            ptr=np.array([self.ptr]),
            size=np.array([self.size])
        )

    def load(self, filename):
        """Load the replay buffer from a file."""
        data = np.load(filename + "_buffer.npz")
        self.state = data["state"]
        self.action = data["action"]
        self.reward = data["reward"]
        self.next_state = data["next_state"]
        self.not_done = data["not_done"]
        self.ptr = int(data["ptr"][0])
        self.size = int(data["size"][0])
