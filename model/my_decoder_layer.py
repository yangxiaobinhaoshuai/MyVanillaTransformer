import torch
from torch import Tensor
from torch.nn import Module, LayerNorm, Dropout, ModuleList
from model.mha import MultiHeadAttention
from model.ffn import PositionWiseFFN


class DecoderLayer(Module):
    """
    input :
    x : bs, seq_len_tgt, d_model #  decoder 自己的输入 target
    enc_out:  bs, seq_len_src, d_model # encoder 最终输出 memory
    tgt_mask : bs,1, seq_len_tg, seq_len_tgt # casual + padding mask
    src mask : bs, 1, 1, seq_len_src # encoder padding mask
    ourput :
    x: bs, seq_len_tgt, d_model
    self_attn_w: bs, n_heads, seq_len_tgt, seq_len_tgt
    cross_attn_w: bs, n_heads, seq_len_tgt, seq_len_src
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. Masked self-attn
        # target seq 内部的 self-attn 必须加 causal mask 防止看到未来
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 2. cross-attn
        # query 来自 decoder 自己的 x ,key/value 来自 encoder 和输出 enc_out
        # 这是 decoder 读取 encoder 的信息的地方
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # pos-wise FFN, 和 encoder 里完全一样
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # 每个子层后面都跟一组 Add & Norm decoder 有三个子层，所以有三组
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
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # 1. maksed self-attn
        # qkv 都来自 x, 用 tag_mask 屏蔽未来的位置 + padding
        self_attn_out, self_attn_w = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # cross-attn
        # q 来自 decoder 当前的 x， k/v 来自 encoder 和输出 enc_out
        # src_masked 用来屏蔽 encoder 侧的 padding
        cross_attn_out, cross_attn_w = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # 3.ffn
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn_w, cross_attn_w


class Decoder(Module):
    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        enc_out: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        # 收集每一层的两种注意力权重，方便调试
        all_self_attn_w = []
        all_cross_attn_w = []
        for layer in self.layers:
            # 上一层的输出作为下一层的输入，enc_out 每层都复用同一个
            x, self_attn_w, cross_attn_w = layer(x, enc_out, tgt_mask, src_mask)
            all_self_attn_w.append(self_attn_w)
            all_cross_attn_w.append(cross_attn_w)

        return x, all_self_attn_w, all_cross_attn_w


if __name__ == "__main__":
    torch.manual_seed(9)

    bs = 2
    seq_len_src, seq_len_tgt = 6, 5
    d_model, n_heads, d_ff = 128, 4, 256
    n_layers = 2

    # decoder 的输入 (target 侧， 比如训练时 shift 过的 target)
    x = torch.randn(bs, seq_len_tgt, d_model)
    # encoder 的最终输出 (memory)， cross-attn 的 kv 来源
    enc_out = torch.randn(bs, seq_len_src, d_model)

    # target 侧的 causal mask： 下三角矩阵，屏蔽未来的位置
    # shape 要和 attn 分类对齐: bs ,1 ,seq_len_tgt, seq_len_tgt
    causal = torch.tril(torch.ones(seq_len_tgt, seq_len_tgt))
    tgt_mask = causal.unsqueeze(0).unsqueeze(0).expand(bs, 1, seq_len_tgt, seq_len_tgt)

    # encoder 侧 padding mask 这里假设没有 pading ， 全 1
    # shape: bs,1,1,seq_len_src
    src_mask = torch.ones(bs, 1, 1, seq_len_src)

    print("x shape:", x.shape)
    print("enc_out shape: ", enc_out.shape)
    print("tgt_mask shape: ", tgt_mask.shape)
    print("src_mask shape:", src_mask.shape)

    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0.0)

    layer_out, layer_self_w, layer_cross_w = decoler_layer(
        x, enc_out, tgt_mask, src_mask
    )

    print(" decoer layer out shape: ", layer_out.shape)
    print(" self attn_w shape: ", layer_self_w.shape)
    print(" cross attn_w shape: ", layer_cross_w.shape)

    assert layer_out.shape == (bs, seq_len_tgt, d_model)
    assert layer_self_w.shape == (bs, n_heads, seq_len_tgt, seq_len_tgt)
    assert layer_cross_w.shape == (bs, n_heads, seq_len_tgt, seq_len_src)

    decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout=0.0)
    dec_out, all_self_w, all_cross_w = decoder(x, enc_out, tgt_mask, src_mask)

    print("decoder out shape :", dec_out.shape)
    print(" num layers: ", len(all_self_w))
    print(" layer 0 self attn_w shape: ", all_self_w.shape)
    print(" layer 0 cross attn_w shape :", all_cross_w.shape)

    assert dec_out.shape == (bs, seq_len_tgt, d_model)
    assert len(all_self_w) == n_layers
    assert len(all_cross_w) == n_layers
    assert all_self_w[0].shape == (bs, n_heads, seq_len_tgt, seq_len_tgt)
    assert all_self_w[1].shape == (bs, n_heads, seq_len_tgt, seq_len_src)

    # 验证 causal mask 生效： 第 0 人位置只能看自己，后面的位置的注意力必须为 0
    assert all_self_w[0][0, 0, 0, 1:].sum().item() < 1e-6, "causal mask failed"
    print("Decoder example test passed")
