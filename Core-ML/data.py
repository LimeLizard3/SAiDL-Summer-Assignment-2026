import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tiktoken

class WikiTextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.data[start:end+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloaders(seq_len=1024, batch_size=16):
    print("Loading WikiText-2 dataset...")
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer (using gpt2 tokenizer for basic BPE)
    enc = tiktoken.get_encoding("gpt2")
    
    def encode_split(split):
        print(f"Tokenizing {split} split...")
        tokens = []
        for text in dataset[split]['text']:
            if text.strip():
                tokens.extend(enc.encode(text, allowed_special={"<|endoftext|>"}) )
        return tokens

    train_tokens = encode_split('train')
    val_tokens = encode_split('validation')
    test_tokens = encode_split('test')

    train_ds = WikiTextDataset(train_tokens, seq_len)
    val_ds = WikiTextDataset(val_tokens, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    vocab_size = enc.n_vocab
    return train_loader, val_loader, vocab_size
