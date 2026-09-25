import torch
import torch.nn as nn
import torch.nn.functional as F
from gtah.attention_torch import GabrielAttention, DenseAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, attn_module: nn.Module, dropout: float):
        super().__init__()
        self.attn  = attn_module
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), causal=causal))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_mult: int,
        dropout: float,
        attn_type: str,
        window_size: int = 32,
        fan_out: int = 4,
        q_decay: float = 0.5,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.attn_type = attn_type

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

        blocks = []
        for _ in range(num_layers):
            if attn_type == "gabriel":
                attn = GabrielAttention(d_model, num_heads, window_size, fan_out, q_decay, epsilon)
            else:
                attn = DenseAttention(d_model, num_heads)
            blocks.append(TransformerBlock(d_model, num_heads, ffn_mult, attn, dropout))

        self.blocks  = nn.ModuleList(blocks)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, causal=True)
        x    = self.norm(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict, attn_type: str) -> MiniLM:
    return MiniLM(
        vocab_size   = cfg["vocab_size"],
        seq_len      = cfg["seq_len"],
        d_model      = cfg["d_model"],
        num_heads    = cfg["num_heads"],
        num_layers   = cfg["num_layers"],
        ffn_mult     = cfg["ffn_mult"],
        dropout      = cfg["dropout"],
        attn_type    = attn_type,
        window_size  = cfg["window_size"],
        fan_out      = cfg["fan_out"],
        q_decay      = cfg["q_decay"],
        epsilon      = cfg["epsilon"],
    )
