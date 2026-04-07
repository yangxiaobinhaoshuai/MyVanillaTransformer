import math
import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    # vocab_size: 词表大小（有多少个 token）
    # d_model:    每个 token 映射成多少维的向量
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        # x: (bs, seq_len)  —— token id
        # 论文里对 embedding 权重乘了 sqrt(d_model)，防止 PE 的幅度相对太大
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # d_model:  向量维度，需要和 TokenEmbedding 一致
    # max_len:  支持的最大序列长度
    # dropout:  加完 PE 后做一次 dropout
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预先算好所有位置的 PE，存成 buffer（不参与训练）
        # pe shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # pos shape: (max_len, 1)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term shape: (d_model/2,)
        # 公式里的 10000^(2i/d_model)，用 exp+log 数值更稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数列用 sin
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数列用 cos

        # 加一个 batch 维，方便后面直接加到输入上
        # pe shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer：pe 会随模型保存/加载，但不会被当作参数更新
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (bs, seq_len, d_model)
        # self.pe[:, :seq_len] 取出对应长度的 PE，广播加到整个 batch
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size, d_model, max_len = 1000, 128, 50
    bs, seq_len = 2, 20

    token_emb = TokenEmbedding(vocab_size, d_model)
    pos_enc = PositionalEncoding(d_model, max_len)

    x = torch.randint(0, vocab_size, (bs, seq_len))  # 随机 token id
    out = token_emb(x)
    print("TokenEmbedding output:", out.shape)  # 期望 (2, 20, 128)

    out = pos_enc(out)
    print("PositionalEncoding output:", out.shape)  # 期望 (2, 20, 128)

    assert out.shape == (bs, seq_len, d_model)
    print("PE test passed")
