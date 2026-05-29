# Phase 2 Documentation: The Transformer Brain

This document provides a line-by-line breakdown of the new components implemented in Phase 2 to transition from MLP to Transformer.

## 1. `model.py`: The Transformer Actor

### `TransformerActor`
- **`self.state_emb = nn.Linear(state_dim, d_model)`**: Unlike text, we don't have tokens. We use a linear projection to map the raw sensor numbers into the Transformer's workspace.
- **`self.action_emb = nn.Linear(action_dim, d_model)`**: To keep the model "Causal," it needs to know what actions it just took. This maps the motor torques from the last step into the same workspace.
- **`x = self.state_emb(state_seq) + self.action_emb(action_seq)`**: We combine the state and action information. This tells the Transformer, "I was at this position ($s_t$) after doing this movement ($a_{t-1}$)."
- **`mask = torch.tril(...)`**: This is the **Causal Mask**. It ensures the Hopper cannot "cheat" by looking into the future. It's the same mechanism used in GPT-4.
- **`last_token = x[:, -1, :]`**: Even though we process a whole window, we only use the information from the **very last moment** to decide what to do right now.

## 2. `replay_buffer.py`: Sequence Sampling

### `sample_sequence`
- **`ind = np.random.randint(seq_len, self.size - 1, ...)`**: We pick a random moment in history, making sure there's enough room behind it ($L$) and one step ahead ($+1$ for the target actor).
- **`boundary = np.where(n_done_win[:-1] == 0)[0]`**: This is critical! If the Hopper fell 5 steps ago and the environment reset, we shouldn't show the Transformer the "pre-death" history. We cut the sequence at that boundary.
- **`na_win = self.action[idx-seq_len+2 : idx+1]`**: This "shifts" the window by one step so the **Target Actor** can see the history leading up to the *next* state.

## 3. `td3.py`: The Training Update

### `train` (Transformer Path)
- **`sample = replay_buffer.sample_sequence(...)`**: Instead of random frames, we now fetch full video clips (histories).
- **`next_action_prediction = self.actor_target(next_state_seq, next_action_seq)`**: The target actor predicts the next smooth action while looking at the history. This is much more stable than just looking at the single next frame.
- **`actor_loss = -self.critic.Q1(state, self.actor(state_seq, action_seq)).mean()`**: This is the "Brain Training." The Actor learns to adjust its weights by asking the Critic, "If I look at this history, would this action make me richer?"

## 4. `train_transformer.py`: The Orchestrator
- **`state_history = deque(..., maxlen=seq_len) `**: While the agent is running live, we use a `deque` (sliding window) to keep its short-term memory fresh. Every time a new frame comes in, the oldest one is "forgotten."
