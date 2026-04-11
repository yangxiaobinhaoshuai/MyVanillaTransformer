from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, ReLU

# Allow running this file directly via `uv run python model/encoder_example.py`.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.mha import MultiHeadAttention
from model.ffn import PositionWiseFFN


class EncoderLayer(Module):
    """
    input:
        x:    bs, seq_len, d_model
        mask: bs, 1, seq_len, seq_len

    output:
        x:    bs, seq_len, d_model
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Encoder 这里用的是 self-attention：
        # 同一个 x 同时作为 query / key / value。
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # 每个子层后面都接一组 Add & Norm。
        # norm1 对应 attention 子层，norm2 对应 FFN 子层。
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # 1. self-attention
        # 输出仍然保持 (bs, seq_len, d_model)。
        attn_out, attn_w = self.self_attn(x, x, x, mask)

        # 2. attention 子层的 Add & Norm
        # 残差连接把原输入直接加回来，帮助保留信息和稳定梯度。
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)

        # 3. FFN 子层的 Add & Norm
        # 这一层结束后，shape 仍然不变，方便继续堆叠下一层 EncoderLayer。
        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn_w


class Encoder(Module):
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
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        # 保存每一层的注意力权重，调试或可视化时会有用。
        all_attn_w = []

        for layer in self.layers:
            # 上一层输出直接作为下一层输入，形成 N 层堆叠。
            x, attn_w = layer(x, mask)
            all_attn_w.append(attn_w)

        return x, all_attn_w


if __name__ == "__main__":
    torch.manual_seed(1)

    bs, seq_len = 2, 6
    d_model, n_heads, d_ff = 128, 4, 256
    n_layers = 2

    x = torch.randn(bs, seq_len, d_model)
    mask = torch.ones(bs, 1, seq_len, seq_len)

    print("=== input check ===")
    print("input x shape:", x.shape)
    print("mask shape:", mask.shape)

    encoder_layer = EncoderLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
    )
    layer_out, layer_attn_w = encoder_layer(x, mask)

    print("=== encoder layer check ===")
    print("encoder layer out shape:", layer_out.shape)
    print("encoder layer attn_w shape:", layer_attn_w.shape)
    print("encoder layer keep same shape:", layer_out.shape == x.shape)

    assert layer_out.shape == (bs, seq_len, d_model)
    assert layer_attn_w.shape == (bs, n_heads, seq_len, seq_len)

    encoder = Encoder(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
    )
    encoder_out, all_attn_w = encoder(x, mask)

    print("=== encoder stack check ===")
    print("encoder out shape:", encoder_out.shape)
    print("num layers:", len(all_attn_w))
    print("layer 0 attn_w shape:", all_attn_w[0].shape)
    print("layer 1 attn_w shape:", all_attn_w[1].shape)

    assert encoder_out.shape == (bs, seq_len, d_model)
    assert len(all_attn_w) == n_layers
    assert all_attn_w[0].shape == (bs, n_heads, seq_len, seq_len)
    assert all_attn_w[1].shape == (bs, n_heads, seq_len, seq_len)
    print("Encoder example test passed")
