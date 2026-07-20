from torch import nn, Tensor
from model2.mha import MultiHeadAttnModule
from model2.ffn import PositionWiseFFN
from torch.nn import LayerNorm, Dropout


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttnModule(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        self.ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor: ...
