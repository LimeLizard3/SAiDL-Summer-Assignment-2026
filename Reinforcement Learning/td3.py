import copy
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch.nn.functional as F
import os
from model import Actor, Critic

class TD3:
    """The Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm."""
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        actor_lr=1e-4, #If the actor is faster than the critic, it wont learn properly, so we slow it down 
        critic_lr=3e-4,#by giving them two different lrs
        gamma=0.99,
        tau=0.0005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        use_transformer=False,
        use_xlstm=False,
        seq_len=32,
        pos_encoding_type='learned' #You can easily change this later
    ):
        self.use_transformer = use_transformer
        self.use_xlstm = use_xlstm
        self.seq_len = seq_len
        self.pos_encoding_type = pos_encoding_type
        
        self.actor: torch.nn.Module
        if use_xlstm:
            from xlstm_model import xLSTMActor
            self.actor = xLSTMActor(state_dim, action_dim, max_action, max_len=seq_len).to(device)
        elif use_transformer:
            from model import TransformerActor
            self.actor = TransformerActor(state_dim, action_dim, max_action, pos_encoding_type=pos_encoding_type).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device) #Remember, we're copying the NN here from the blueprint (the class from model.py)
            
        self.actor_target: torch.nn.Module = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # 5. Stricter Discipline: Schedulers for long-term decay
        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=0.999998) # Very slow decay
        self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=0.999998)
        #We use a concept called BOOTSTRAPPING here during copy.deepcopy() by which the model makes its own "Y-labels" and predicts from there; if we didn't have that, the base would
        #keep changing and the model wouldn't learn at all due to an unstable base.

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        
        # [AMP] Initializes the Gradient Scaler. This acts as a digital shock-absorber that multiplies the AI's error
        # before backward propagation, preventing the tiny 16-bit decimals from crashing into 0.0 (underflowing) or reaching the limit and overflowing
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')

        self.total_it = 0

    def select_action(self, state, state_history=None, action_history=None, return_attn=False):
        """Actor selects an action for a given state."""
        if self.use_xlstm:
            return self.actor.select_action(state, state_history, action_history, return_attn=return_attn)
        elif self.use_transformer:
            # We use the actor's internal select_action to handle history-to-tensor conversion
            return self.actor.select_action(state, state_history, action_history, return_attn=return_attn)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device) 
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, step_schedulers=True):
        """Perform a single step of the TD3 training loop."""
        self.total_it += 1

        # Initialize local variables to avoid unbound-name warnings
        state_seq = None
        action_seq = None
        next_state_seq = None
        next_action_seq = None

        # 1. Sample from the Replay Buffer
        if self.use_transformer or self.use_xlstm:
            # Sequence sampling for the Transformer/xLSTM
            sample = replay_buffer.sample_sequence(batch_size, self.seq_len)
            if sample is None:
                return # Not enough data to train yet
            state_seq, action_seq, next_state_seq, next_action_seq, state, action, next_state, reward, not_done = sample
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) #replay_buffer is defined in train.py, it's not here as this isn't the "pantry"

        with torch.no_grad():
            # [AMP] Intercepts heavy mathematical formulas here and dynamically shrinks the data
            # down to 16-bit to turbocharge computation speed on the GPU Tensor Cores.
            with torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                # 2. Select next action with Target Policy Smoothing
                if self.use_transformer or self.use_xlstm:
                    # Target actor uses the shifted history for the next step
                    next_action_prediction = self.actor_target(next_state_seq, next_action_seq)
                else:
                    next_action_prediction = self.actor_target(next_state)
    
                noise = (torch.randn_like(next_action_prediction) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) 
                #_like() makes it so that it has dimensions of whatever's passed
                #.clamp(min,max): if smaller than min, forced upto min and bigger than max, forced down to max
                next_action = (next_action_prediction + noise).clamp(-self.max_action, self.max_action)
    
                # 3. Compute Target Q-value using Clipped Double-Q
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.gamma * target_Q #Reward is the cold hard truth that just happened, and the other terms are to look towards the future to   
                                                                     #PREDICT what will happen and how many more points the AI will earn before not_done=0
        
        # 4. Update Critics
        # [AMP] Automatically utilizes super-fast 16-bit math for calculating the Critic's error (loss).
        with torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #mse = Mean Squared Error

        self.critic_optimizer.zero_grad() #Cleaning out the gradients so that the AI doesn't get confused from previous training cycles
        
        # [AMP] Scales the loss up by a massive factor safely so it doesn't break the float limits during backprop.
        critic_loss_scaled = self.scaler.scale(critic_loss)
        assert isinstance(critic_loss_scaled, torch.Tensor)
        critic_loss_scaled.backward() #Backprop, scaled for AMP
        
        # [STABILITY] Unscale gradients before clipping to prevent spikes
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        
        # [AMP] Shrinks the math back down to normal and applies the Gradient Descent updates to the Critics.
        self.scaler.step(self.critic_optimizer) #This is Gradient Descent
        
        # [AMP] Readjusts the multiplier shield up or down so future runs stay mathematically stable.
        self.scaler.update() #Update scaler multipliers

        if step_schedulers:
            self.critic_scheduler.step() #LR decay basically

        # 5. Delayed Policy Updates
        if self.total_it % self.policy_freq == 0:
            # Update Actor
            # [AMP] Supercharges the Actor's sequence math using 16-bit precision.
            with torch.amp.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                if self.use_transformer or self.use_xlstm:
                    # Transformer/xLSTM uses sequence context
                    actor_loss = -self.critic.Q1(state, self.actor(state_seq, action_seq)).mean()
                else:
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean() #We only need one critic here, .mean() because we're doing this over 256 cases, negative is there to
            self.actor_optimizer.zero_grad()                                      #trick GD into increasing the actor loss as much as it can so that it becomes more favourable
            
            # [AMP] Expands the Actor loss to stop it from underflowing (hitting zero).
            actor_loss_scaled = self.scaler.scale(actor_loss)
            assert isinstance(actor_loss_scaled, torch.Tensor)
            actor_loss_scaled.backward()
            
            # [STABILITY] Unscale gradients before clipping to prevent spikes
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            
            # [AMP] Un-scales the matrix and tweaks the Actor's neural weights safely.
            self.scaler.step(self.actor_optimizer) #This is Gradient Descent (It auto takes the blueprint given by backprop)
            
            # [AMP] Automatically manages the internal 16-bit overflow thresholds for the Actor.
            self.scaler.update()

            if step_schedulers:
                self.actor_scheduler.step()

            #When we optimize, we aren't changing our Q score at all, we're just working around the optimization algos to ensure that they
            #understand that a more -ve loss translates into a higher Q values which translates into better actions

            # Soft Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()): #zip pairs data together, .parameters() gets the list of parameters for each model
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) #Tau particularly tells us how much of each we're taking and then we merge them to update the frozen brain

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            #Note, the _ at the end of copy is what's telling the for loop to overwrite the contents with what we're passing automatically. It's updating it in place
            
    def save(self, filename):
        """Save the model weights and training state."""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.scaler.state_dict(), filename + "_scaler")
        # Save iteration count
        np.save(filename + "_total_it.npy", np.array([self.total_it]))

    def load(self, filename):
        """Load the model weights and training state."""
        self.critic.load_state_dict(torch.load(filename + "_critic"), strict=False)
        try:
            self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        except:
            print(f"Warning: Critic optimizer for {filename} could not be loaded (likely architectural shift). Skipping.")
            
        self.actor.load_state_dict(torch.load(filename + "_actor"), strict=False)
        try:
            self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        except:
            print(f"Warning: Actor optimizer for {filename} could not be loaded (likely architectural shift). Skipping.")
        
        if os.path.exists(filename + "_scaler"): #We save GradScaler to save the scale factor to prevent math spikes
            self.scaler.load_state_dict(torch.load(filename + "_scaler"))
        
        if os.path.exists(filename + "_total_it.npy"): #Saving which interation we're on so that we know when to update Actor
            self.total_it = int(np.load(filename + "_total_it.npy")[0])
