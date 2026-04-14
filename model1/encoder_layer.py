import torch
from torch.nn import Module, LayerNorm, Dropout, ModuleList
from torch import Tensor
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model1.mha import MultiHeadAttn
from model1.ffn import PoswiseFFN


class EncoderLayer(Module):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttn(n_heads, d_model, dropout)

        self.ffn = PoswiseFFN(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:

        attn_out, attn_w = self.self_attn(x, x, x, mask)

        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)

        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn_w


class Encoder(Module):
    def __init__(
        self, n_layers: int, n_heads: int, d_model: int, d_ff: int, droput: float = 0.1
    ):
        super().__init__()

        self.layers = ModuleList(
            [
                EncoderLayer(
                    n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=droput
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:

        # 保存每一层的 attn_w 堆叠
        all_attn_w = []

        for layer in self.layers:
            x, attn_w = layer(x, mask)
            all_attn_w.append(attn_w)

        return x, all_attn_w


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, seq_len, d_model = 2, 64, 256
    n_heads = 4
    n_layers = 6
    d_ff = 512

    x = torch.randn(bs, seq_len, d_model)

    # bs, 1, seq_len, seq_len
    mask = torch.ones(bs, 1, seq_len, seq_len)

    encoder_layer = EncoderLayer(n_heads=n_heads, d_model=d_model, d_ff=d_ff)

    layer_out, layer_attn_w = encoder_layer(x)

    print("layer output shape: ", layer_out.shape)
    print("layer attn_w shape: ", layer_attn_w.shape)

    assert layer_out.shape == (bs, seq_len, d_model), "layer out shape err"
    assert layer_attn_w.shape == (bs, n_heads, seq_len, seq_len), (
        "layer attn w shape err"
    )

    encoder = Encoder(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff)

    enc_out, all_attn_w = encoder(x, mask)

    print("enc out shape: ", enc_out.shape)
    print("all attn w len: ", len(all_attn_w))

    assert enc_out.shape == (bs, seq_len, d_model), "enc out shape err"
    assert len(all_attn_w) == n_layers, " all_attn_w len err"
    assert all_attn_w[0].shape == (bs, n_heads, seq_len, seq_len), (
        "all attn w 0 shape err"
    )

    print(" encoder all test passed")
