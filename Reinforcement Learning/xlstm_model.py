# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import numpy as np
import math

class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer to prevent information leakage from the future.
    """
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size - 1, 
            groups=d_model
        )
        #Lout = L + 2P -(K-1)
        #Lout = L + 2(K-1) - (K-1) = L+K-1
        
    def forward(self, x):
        # Input shape: (B, L, d_model) -> Transpose to (B, d_model, L) for Conv1d
        x = x.transpose(1, 2)
        out = self.conv(x)
        # Slice off the extra padded steps at the end to maintain causality
        out = out[..., :x.size(-1)] 
        return out.transpose(1, 2)

class sLSTMCell(nn.Module):
    """
    sLSTM Cell (Scalar LSTM) with exponential gating and stabilizer state.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gates: Forget (f), Input (i), Output (o), and Cell Candidate (z)
        self.W = nn.Linear(d_model, 4 * hidden_size)
        self.R = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        
    def forward(self, x_t, h_prev, c_prev, n_prev, m_prev):
        # Project input and recurrent states
        pre_act = self.W(x_t) + self.R(h_prev) # Shape: (B, 4 * hidden_size), pre_act = Wxt + bw + Rht-1
        f_tilde, i_tilde, z_tilde, o_tilde = torch.chunk(pre_act, 4, dim=-1)
        #forget, input, candidate, output

        # Update stabilizer: Prevents overflowing due to e^X exponential 
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        
        # Compute stabilized exponential gates
        f_gate = torch.exp(f_tilde + m_prev - m_t)
        i_gate = torch.exp(i_tilde - m_t)
        
        # Update cell and normalizer states
        c_t = f_gate * c_prev + i_gate * torch.tanh(z_tilde)
        #Start with old memory, erase unwated old memory (f_gate * c_prev)
        #Draft new details, scale new details (i_gate * tanh(z_tilde))
        #Write to notepad (c_t)
        n_t = f_gate * n_prev + i_gate #Normalizer
        
        # Output gate
        o_gate = torch.sigmoid(o_tilde)
        
        # Normalized cell state hidden output (Decide what to read)
        h_t = o_gate * (c_t / (n_t + 1e-8))
        
        return h_t, c_t, n_t, m_t

class MultiHeadmLSTMCell(nn.Module):
    """
    Multi-Head mLSTM Cell (Matrix LSTM) with covariance matrix memory.
    """
    def __init__(self, d_model, hidden_size, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # QKV linear projections
        self.q_proj = nn.Linear(d_model, hidden_size)
        self.k_proj = nn.Linear(d_model, hidden_size)
        self.v_proj = nn.Linear(d_model, hidden_size)
        
        # Gating projections: 1 forget, 1 input, 1 output per head (That's why *3)
        self.W_gates = nn.Linear(d_model, 3 * num_heads)
        
    def forward(self, x_t, C_prev, n_prev, m_prev):
        B = x_t.size(0)
        H = self.num_heads
        d_k = self.head_dim
        
        # Project and reshape to head dims
        q = self.q_proj(x_t).view(B, H, d_k)
        k = (self.k_proj(x_t) / math.sqrt(d_k)).view(B, H, d_k) #Scale to prevent dot products from growing too large
        v = self.v_proj(x_t).view(B, H, d_k)
        
        # Gates projection
        gates = self.W_gates(x_t) # Shape: (B, 3 * H)
        f_tilde, i_tilde, o_tilde = torch.chunk(gates, 3, dim=-1) # Each has shape (B, H)
        
        f_tilde = f_tilde.unsqueeze(2) # Shape: (B, H, 1)
        i_tilde = i_tilde.unsqueeze(2) # Shape: (B, H, 1)
        o_tilde = o_tilde.unsqueeze(2) # Shape: (B, H, 1)
        
        # Stabilizer: 
        m_t = torch.max(f_tilde + m_prev, i_tilde) # Shape: (B, H, 1)
        
        # Stabilized exponential gates
        f_gate = torch.exp(f_tilde + m_prev - m_t) # Shape: (B, H, 1)
        i_gate = torch.exp(i_tilde - m_t) # Shape: (B, H, 1)
        
        # Outer product for covariance update: v_t k_t^T
        vk = torch.matmul(v.unsqueeze(3), k.unsqueeze(2)) # Shape: (B, H, d_k, d_k)
        
        # Update matrix state C_t and normalizer vector n_t
        C_t = f_gate.unsqueeze(3) * C_prev + i_gate.unsqueeze(3) * vk # Shape: (B, H, d_k, d_k)
        n_t = f_gate * n_prev + i_gate * k # Shape: (B, H, d_k)
        
        # Query memory
        C_q = torch.matmul(C_t, q.unsqueeze(3)).squeeze(3) # Shape: (B, H, d_k)
        
        # Normalization factor: n_t^T q_t
        nq = torch.sum(n_t * q, dim=-1, keepdim=True) # Shape: (B, H, 1) d_k becomes 1
        h_tilde = C_q / torch.max(torch.abs(nq), torch.ones_like(nq)) # Shape: (B, H, d_k)
        
        # Output gating
        o_gate = torch.sigmoid(o_tilde) # Shape: (B, H, 1)
        h_t = (o_gate * h_tilde).view(B, H * d_k) # Shape: (B, hidden_size)
        
        return h_t, C_t, n_t, m_t

class mLSTMLayer(nn.Module):
    """
    mLSTM Layer block containing LayerNorm, Causal Conv1D, mLSTM Cell, and projection.
    """
    def __init__(self, d_model, hidden_size, num_heads=4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel_size=4)
        self.cell = MultiHeadmLSTMCell(d_model, hidden_size, num_heads=num_heads)
        self.proj = nn.Linear(hidden_size, d_model)
        
    def forward(self, x):
        # 1. Pre-LayerNorm & Causal Conv
        x_norm = self.ln(x)
        x_conv = self.conv(x_norm)
        
        B, L, _ = x_conv.size()
        H = self.cell.num_heads
        d_k = self.cell.head_dim
        device = x.device
        
        # 2. Sequential processing loop
        C = torch.zeros(B, H, d_k, d_k, device=device)
        n = torch.zeros(B, H, d_k, device=device)
        m = torch.zeros(B, H, 1, device=device)
        
        outputs = []
        for t in range(L):
            x_t = x_conv[:, t, :]
            h_t, C, n, m = self.cell(x_t, C, n, m)
            outputs.append(h_t)
            
        h = torch.stack(outputs, dim=1) # Shape: (B, L, hidden_size)
        
        # 3. Residual connection
        return x + self.proj(h)

class sLSTMLayer(nn.Module):
    """
    sLSTM Layer block containing LayerNorm, sLSTM Cell, and projection.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.cell = sLSTMCell(d_model, hidden_size)
        self.proj = nn.Linear(hidden_size, d_model)
        
    def forward(self, x):
        # 1. Pre-LayerNorm
        x_norm = self.ln(x)
        
        B, L, _ = x_norm.size()
        device = x.device
        
        # 2. Sequential processing loop
        h_t = torch.zeros(B, self.cell.hidden_size, device=device)
        c_t = torch.zeros(B, self.cell.hidden_size, device=device)
        n_t = torch.zeros(B, self.cell.hidden_size, device=device)
        m_t = torch.zeros(B, self.cell.hidden_size, device=device)
        
        outputs = []
        for t in range(L):
            x_t = x_norm[:, t, :]
            h_t, c_t, n_t, m_t = self.cell(x_t, h_t, c_t, n_t, m_t)
            outputs.append(h_t)
            
        h = torch.stack(outputs, dim=1) # Shape: (B, L, hidden_size)
        
        # 3. Residual connection
        return x + self.proj(h)

class xLSTMActor(nn.Module):
    """
    xLSTM-based Policy Actor Backbone replacing the Causal Transformer.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action, 
        d_model=128, 
        num_layers=2, 
        num_heads=4, 
        dropout=0.1, 
        max_len=32
    ):
        super().__init__()
        self.max_action = max_action
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        
        # Input state and action embeddings
        self.state_emb = nn.Linear(state_dim, d_model)
        self.action_emb = nn.Linear(action_dim, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Alternating stacked xLSTM blocks (mLSTM -> sLSTM)
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx % 2 == 0:
                self.layers.append(mLSTMLayer(d_model, d_model, num_heads=num_heads))
            else:
                self.layers.append(sLSTMLayer(d_model, d_model))
                
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)
        
    @property
    def device(self):
        return next(self.parameters()).device #Checks which device the parameters are on
        
    def select_action(self, state_norm, state_history, action_history, return_attn=False):
        """
        Drop-in inference action selection method matching TransformerActor.
        """
        state_seq = torch.FloatTensor(np.array(state_history)).unsqueeze(0).to(self.device)
        action_seq = torch.FloatTensor(np.array(action_history)).unsqueeze(0).to(self.device)
        
        action = self.forward(state_seq, action_seq)
        return action.cpu().data.numpy().flatten()
        
    def forward(self, state_seq, action_seq):
        # Input projection & fusion
        x = self.state_emb(state_seq) + self.action_emb(action_seq)
        x = self.drop(x)
        
        # Process through alternating layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        
        # Predict the action using the final sequence step hidden representation
        last_token = x[:, -1, :]
        action = self.max_action * torch.tanh(self.head(last_token))
        return action
