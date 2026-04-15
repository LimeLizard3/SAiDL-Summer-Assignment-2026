import copy
import torch
import numpy as np
import torch.nn.functional as F
from model import Actor, Critic

class TD3:
    """The Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm."""
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        use_transformer=False,
        seq_len=32
    ):
        self.use_transformer = use_transformer
        self.seq_len = seq_len
        
        if use_transformer:
            from model import TransformerActor
            self.actor = TransformerActor(state_dim, action_dim, max_action).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device) #Remember, we're copying the NN here from the blueprint (the class from model.py)
            
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        #We use a concept called BOOTSTRAPPING here during copy.deepcopy() by which the model makes its own "Y-labels" and predicts from there; if we didn't have that, the base would
        #keep changing and the model wouldn't learn at all due to an unstable base.

        # Setup Automatic Mixed Precision (AMP) to use rapid 16-bit Tensor Core math
        self.scaler = torch.amp.GradScaler('cuda')

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device

        self.total_it = 0

    def select_action(self, state, state_history=None, action_history=None):
        """Actor selects an action for a given state."""
        if self.use_transformer:
            # Transformer needs (B, L, Dim)
            state_seq = torch.FloatTensor(np.array(state_history)).unsqueeze(0).to(self.device)
            action_seq = torch.FloatTensor(np.array(action_history)).unsqueeze(0).to(self.device)
            #Remember when we converted deques to lists? This is why, so that we could convert them to tensors
            #IMPORTANT DEBUGGING: we use np.array to prevent a CPU bottleneck
            return self.actor(state_seq, action_seq).cpu().data.numpy().flatten()
            #.cpu() brings it to the physics engine. GPU is where the AI lives but the game lives on the CPU as that's where eqns are processed for it
            #data.numpy() strips PyTorch baggage away and converts raw Nos into a NumPY array (also gymnasium doesn't know what a Tensor is)
            #flatten() squashes away extra brackets so that the physics engine doesn't crash
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device) #NOTE: We could also just write .unsqueeze(0) and get literally the exact same thing, but hey
            #1 means add 1 row, -1 means "adjust with me" RE columns cos the No. of actions could change
            return self.actor(state).cpu().data.numpy().flatten() #.actor() shoves state data into the internal forward function

    def train(self, replay_buffer, batch_size=256):
        """Perform a single step of the TD3 training loop."""
        self.total_it += 1

        # 1. Sample from the Replay Buffer
        if self.use_transformer:
            # Sequence sampling for the Transformer
            sample = replay_buffer.sample_sequence(batch_size, self.seq_len)
            if sample is None:
                return # Not enough data to train yet
            state_seq, action_seq, next_state_seq, next_action_seq, state, action, next_state, reward, not_done = sample
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) #replay_buffer is defined in train.py, it's not here as this isn't the "pantry"

        with torch.no_grad():
            # 2. Select next action with Target Policy Smoothing
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.autocast(device_type=device_type):
                if self.use_transformer:
                    # Target actor uses the shifted history for the next step
                    next_action_prediction = self.actor_target(next_state_seq, next_action_seq)
                else:
                    next_action_prediction = self.actor_target(next_state)

            noise = (torch.randn_like(next_action_prediction) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) 
            #_like() makes it so that it has dimensions of whatever's passed
            #.clamp(min,max): if smaller than min, forced upto min and bigger than max, forced down to max
            next_action = (next_action_prediction + noise).clamp(-self.max_action, self.max_action)

            # 3. Compute Target Q-value using Clipped Double-Q
            with torch.autocast(device_type=device_type):
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q #Reward is the cold hard truth that just happened, and the other terms are to look towards the future to   
                                                                 #PREDICT what will happen and how many more points the AI will earn before not_done=0
        # 4. Update Critics
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.autocast(device_type=device_type):
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #mse = Mean Squared Error

        self.critic_optimizer.zero_grad() #Cleaning out the gradients so that the AI doesn't get confused from previous training cycles
        self.scaler.scale(critic_loss).backward() #Backprop using AMP scaler
        self.scaler.step(self.critic_optimizer) #This is Gradient Descent
        self.scaler.update()

        # 5. Delayed Policy Updates
        if self.total_it % self.policy_freq == 0:
            # Update Actor
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.autocast(device_type=device_type):
                if self.use_transformer:
                    # Transformer uses sequence context
                    actor_loss = -self.critic.Q1(state, self.actor(state_seq, action_seq)).mean()
                else:
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean() #We only need one critic here, .mean() because we're doing this over 256 cases, negative is there to
            self.actor_optimizer.zero_grad()                                  #trick GD into increasing the actor loss as much as it can so that it becomes more favourable
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer) #This is Gradient Descent (It auto takes the blueprint given by backprop)
            self.scaler.update()

            #When we optimize, we aren't changing our Q score at all, we're just working around the optimization algos to ensure that they
            #understand that a more -ve loss translates into a higher Q values which translates into better actions

            # Soft Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()): #zip pairs data together, .parameters() gets the list of parameters for each model
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) #Tau particularly tells us how much of each we're taking and then we merge them to update the frozen brain

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            #Note, the _ at the end of copy is what's telling the for loop to overwrite the contents with what we're passing automatically. It's updating it in place
            
    def save(self, filename):
        """Save the model weights."""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """Load the model weights."""
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
