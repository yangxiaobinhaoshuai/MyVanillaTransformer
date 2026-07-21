from torch import nn, Tensor
from model2.mha import MultiHeadAttnModule
from model2.ffn import PositionWiseFFN
from torch.nn import LayerNorm, Dropout, ModuleList
from typing import Optional


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

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:

        attn_out, attn_w = self.self_attn(x, x, x, mask)

        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)

        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn_w


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, list[Tensor]]:

        all_attn_w = []

        for layer in self.layers:
            x, attn_w = layer(x, mask)
            all_attn_w.append(attn_w)

        return x, all_attn_w
