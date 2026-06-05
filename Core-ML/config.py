from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 50257  # Default for tiktoken gpt2
    max_seq_len: int = 1024
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    
    # Architecture Variants
    attention_type: str = "standard" # Options: "standard", "mqa", "linear", "sliding_window", "aft"
    window_size: int = 50 # Only used if attention_type == "sliding_window"
    
    # AFT Options (Bonus Task)
    aft_mode: str = "full" # Options: "full", "simple", "local", "conv"
    aft_window_size: int = 50 # Used if aft_mode == "local"
    
    # Positional Variants (Task 3)
    pos_type: str = "absolute" # Options: "absolute", "rope", "alibi"
    rope_use_interpolation: bool = False
    rope_train_seq_len: int = 512
    
    # Convolution Hybrids (Task 4)
    use_conv: bool = False
    conv_type: str = "pre_attention" # "pre_attention" or "interleaved"
    conv_kernel_size: int = 3
    
    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    epochs: int = 1
    weight_decay: float = 0.01
