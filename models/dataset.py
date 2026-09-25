import urllib.request
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(path: str = "data/shakespeare.txt") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading TinyShakespeare -> {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, vocab: dict = None):
        self.seq_len = seq_len

        if vocab is None:
            chars      = sorted(set(text))
            self.vocab = {c: i for i, c in enumerate(chars)}
        else:
            self.vocab = vocab

        self.itos = {i: c for c, i in self.vocab.items()}
        self.data = torch.tensor([self.vocab[c] for c in text if c in self.vocab], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_dataloaders(seq_len: int, batch_size: int, val_frac: float = 0.1, data_path: str = "data/shakespeare.txt"):
    path = download_shakespeare(data_path)
    text = open(path, encoding="utf-8").read()

    full_ds  = CharDataset(text, seq_len)
    n_val    = int(len(full_ds) * val_frac)
    n_train  = len(full_ds) - n_val

    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    return train_loader, val_loader, full_ds.vocab_size, full_ds.vocab
