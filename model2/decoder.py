import torch
from torch import Tensor
from torch import nn
from model2.mha import MultiHeadAttnModule
from model2.encoder import Encoder
from model2.ffn import PositionWiseFFN
from torch.nn import LayerNorm, Dropout, ModuleList, Module
from typing import Optional


class DecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttnModule(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        self.cross_attn = MultiHeadAttnModule(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )

        self.ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        tgt_mask: Optional[Tensor],
        src_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:

        self_attn_out, self_attn_w = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out, cross_attn_w = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn_w, cross_attn_w


class Decoder(Module):
    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layers = ModuleList(
            [
                DecoderLayer(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        all_self_attn_w = []
        all_cross_attn_w = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, enc_out, tgt_mask, src_mask)
            all_self_attn_w.append(self_attn_w)
            all_cross_attn_w.append(cross_attn_w)

        return x, all_self_attn_w, all_cross_attn_w
    


