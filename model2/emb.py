import torch
from torch import nn, Tensor
import math
from typing import cast


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_length: int, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_length, d_model)

        pos = torch.arange(0, max_length).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin( pos * div_term)
        pe[:, 1::2] = torch.cos( pos * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)



    def forword(self, x: Tensor) -> Tensor:
        x = x + cast(Tensor, self.pe)[:, : x.size(1)]
        return self.dropout(x)



class PositionalEmbedding2(nn.Module):

    def __init__(self, max_len: int, d_model: int, dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, d_model).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.pe)[:, : x.size(1)]
        return self.dropout(x)
    


            
