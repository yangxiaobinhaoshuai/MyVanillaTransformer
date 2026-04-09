import torch
from torch.nn import Module
from torch import Tensor
from model.mha import MultiHeadAttention


class EncoderLayer(Module):
    """
    input:
    x: bs, seq_len, d_model
    mask: bs, 1, seq_len, seq_len

    outout:
    x: bs, seq_len, d_model
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Encoder 用 self-attn
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 同一个 x 作为 q, k, v
        # TODO ffh here
        # self.ffn = 

        # 每个子层后接一层 Add & Norm
        # norm1 



        pass

    def forward(self, x: Tensor) -> Tensor: ...


class Encoder(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor: ...


if __name__ == "__main__":
    torch.manual_seed(0)
    pass
