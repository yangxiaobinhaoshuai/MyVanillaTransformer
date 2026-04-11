import torch
from torch.nn import Module, LayerNorm, Dropout, ModuleList
from torch import Tensor
from model.mha import MultiHeadAttention
from model.ffn import PositionWiseFFN


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
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Encoder 用 self-attn
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 同一个 x 作为 q, k, v

        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # 每个子层后接一层 Add & Norm
        # norm1 是 attn 子层， norm2 对应 fnn 子层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:

        # self-attn
        # output shape: bs, seq_len, d_model
        attn_out, attn_w = self.self_attn(x, x, x, mask)

        # attn 子层 Add & Norm
        # residual keep info & stabilze weights
        x = self.norm1(x + self.dropout1(attn_out))

        ffn = self.ffn(x)

        # ffn 子层的 Add & Norm
        # 这一层结束后 shape 不变，方便下一层堆叠
        x = self.norm2(x + self.dropout2(ffn))
        return x, attn_w


class Encoder(Module):
    def __init__(
        self, n_layers: int, d_models: int, n_heads: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.layers = ModuleList(
            [EncoderLayer(d_models, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        # 保存每一层的 attn weight,调试可视化会有用
        all_attn_w = []
        for layer in self.layers:
            # 上一层的输出直接作为下一层的输入，形成 N 层堆叠
            x, attn_w = layer(x, mask)
            all_attn_w.append(attn_w)
        return x, all_attn_w


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, seq_len = 2, 6
    d_model, n_heads, d_ff = 128, 4, 256
    n_layers = 2

    x = torch.randn(bs, seq_len, d_model)
    mask = torch.ones(bs, 1, seq_len, seq_len)

    print("intput x shape:", x.shape)
    print("mask shape:", mask.shape)

    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, 0.0)
    layer_out, layer_attn_w = encoder_layer(x, mask)

    print("encoder layer out shape:", layer_out.shape)
    print("encoder layer attn_w shape:", layer_attn_w.shape)
    print("encoder layer keep same shape:", layer_out.shape == x.shape)

    assert layer_out.shape == (bs, seq_len, d_model)
    assert layer_attn_w.shape == (bs, n_heads, seq_len)

    encoder = Encoder(n_layers, d_model, n_heads, d_ff, 0.0)

    encoder_out, all_attn_w = encoder(x, mask)

    print("encoder out shape:", encoder_out.shape)
    print("num shape:", len(all_attn_w))
    print("layer 0 attn_w shape:", all_attn_w[0].shape)
    print("layer 1 attn_w shape[1].shape:", all_attn_w[1].shape)
