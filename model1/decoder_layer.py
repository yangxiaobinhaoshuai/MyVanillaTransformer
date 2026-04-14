import torch
from torch.nn import Module

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model1.mha import MultiHeadAttn
from model1.ffn import PoswiseFFN

from torch.nn import LayerNorm, Dropout, ModuleList
from torch import Tensor


class DecoderLayer(Module):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttn(n_heads, d_model, dropout)
        self.cross_attn = MultiHeadAttn(n_heads, d_model, dropout)

        self.ffn = PoswiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.droput1 = Dropout(dropout)
        self.droput2 = Dropout(dropout)
        self.droput3 = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self_attn_out, self_attn_w = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.droput1(self_attn_out))

        cross_attn_out, cross_attn_w = self.cross_attn(x, enc_out, enc_out, src_mask)

        x = self.norm2(x + self.droput2(cross_attn_out))

        ffn_out = self.ffn(cross_attn_out)

        x = self.norm3(x + self.droput3(ffn_out))

        return x, self_attn_w, cross_attn_w


class Decoder(Module):
    def __init__(
        self, n_layers: int, n_heads: int, d_model: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()

        self.layers = ModuleList(
            [
                DecoderLayer(
                    n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:

        all_self_attn_w = []
        all_cross_attn_w = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, enc_out, src_mask, tgt_mask)
            all_self_attn_w.append(self_attn_w)
            all_cross_attn_w.append(cross_attn_w)

        return x, all_self_attn_w, all_cross_attn_w


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, d_model, d_ff, n_layers = 2, 128, 256, 4
    seq_len_src = 64
    seq_len_tgt = 96
    n_heads = 4

    decode_input = torch.randn(bs, seq_len_tgt, d_model)
    encode_out = torch.randn(bs, seq_len_src, d_model)

    tgt_mask = (
        torch.tril(torch.ones(seq_len_tgt, seq_len_tgt))
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(bs, -1, -1, -1)
    )

    src_mask = torch.ones(bs, 1, 1, seq_len_src)

    decoder_layer = DecoderLayer(
        n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=0.0
    )

    (
        layer_out,
        layer_self_attn_w,
        layer_cross_attn_w,
    ) = decoder_layer(decode_input, encode_out, src_mask, tgt_mask)

    print("layer_out shape: ", layer_out.shape)
    print("layer self attn w shape: ", layer_self_attn_w.shape)
    print("layer cross attn w shape: ", layer_cross_attn_w.shape)

    assert layer_out.shape == (bs, seq_len_tgt, d_model), " layer out shape err"
    assert layer_self_attn_w.shape == (bs, n_heads, seq_len_tgt, seq_len_tgt), (
        "layer attn w shape err"
    )

    decoder = Decoder(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff)

    dec_out, dec_self_all_attn_w, dec_cross_all_attn_w = decoder(
        decode_input, encode_out, src_mask, tgt_mask
    )

    print("dec_out shape: ", dec_out.shape)
    print("dec attn w shape size: ", len(dec_self_all_attn_w))
    print("dec attn w 0 shape: ", dec_self_all_attn_w[0].shape)

    assert dec_out.shape == (bs, seq_len_tgt, d_model), " dec out shape err"
    assert dec_self_all_attn_w[0].shape == (bs, n_heads, seq_len_tgt, seq_len_tgt), (
        "dec all attn w 0 shape err"
    )

    print("decoder test all passed.")
