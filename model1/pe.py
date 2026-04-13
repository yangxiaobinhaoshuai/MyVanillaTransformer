import torch
import math
from typing import cast
from torch import Tensor
from torch.nn import Module, Embedding, Dropout


class TokenEmbedding(Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.emb(x) * math.sqrt(self.d_model)


class PosEmbedding(Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:

        # x shape: bs, seq_len, d_model
        x = x + cast(Tensor, self.pe)[:, : x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    torch.manual_seed(1)

    bs, seq_len, d_model = 2, 64, 128
    vocab_size = 512

    x = torch.randint(0, vocab_size, (bs, seq_len))

    token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

    token_out = token_emb(x)

    assert token_out.shape == (2, 64, 128), " token shape err"

    pos_emb = PosEmbedding(d_model=d_model)

    pos_out = pos_emb(token_out)

    assert pos_out.shape == (2, 64, 128), " pos shape err"

    print("pe test all passed")
