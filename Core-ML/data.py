import torch  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore
from datasets import load_dataset  # type: ignore
import tiktoken  # type: ignore

class WikiTextDataset(Dataset):
    """
    Custom Dataset class to chunk tokenized text into input-target sequences.
    """
    def __init__(self, data, seq_len):
        # Store raw tokenized integer IDs
        self.data = data
        # Maximum sequence length for the model input
        self.seq_len = seq_len

    def __len__(self):
        # Number of complete sequence chunks that can be extracted
        return len(self.data) // self.seq_len

    def __getitem__(self, index):
        # Calculate indices to slice a chunk of size seq_len + 1
        start = index * self.seq_len
        end = start + self.seq_len
        # Slice raw token list to get sequence plus next-token target
        chunk = self.data[start:end+1]
        # x is the input sequence: shape [seq_len] (contains tokens 0 to seq_len-1)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # y is the target sequence: shape [seq_len] (contains tokens 1 to seq_len, shifted by 1)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloaders(seq_len=1024, batch_size=16):
    """
    Loads WikiText-2 from Hugging Face, tokenizes it using GPT-2 BPE, and returns DataLoaders.
    """
    print("Loading WikiText-2 dataset...")
    # Load raw dataset splits (train, validation, test) from HF datasets
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize the standard GPT-2 Byte-Pair Encoding (BPE) tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    def encode_split(split):
        # Helper function to flat-map all text entries in a dataset split to tokens
        print(f"Tokenizing {split} split...")
        tokens = []
        for text in dataset[split]['text']:
            if text.strip():
                # Convert string to list of integer IDs
                tokens.extend(enc.encode(text, allowed_special={"<|endoftext|>"}) )
        return tokens

    # Tokenize each individual split to create flat lists of integer tokens
    train_tokens = encode_split('train')
    val_tokens = encode_split('validation')
    test_tokens = encode_split('test')

    # Wrap the token sequences into WikiTextDataset instances
    train_ds = WikiTextDataset(train_tokens, seq_len)
    val_ds = WikiTextDataset(val_tokens, seq_len)

    # Create PyTorch DataLoaders; yields batches of shape [batch_size, seq_len]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Get the vocabulary size from GPT-2 tokenizer (typically 50257)
    vocab_size = enc.n_vocab
    return train_loader, val_loader, vocab_size
