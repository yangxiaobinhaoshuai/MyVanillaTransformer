import torch
from torch.nn import Module
from torch.nn import Embedding, Dropout
from torch import Tensor
import math
from typing import cast


class TokenEmbedding(Module):
    """
    vocab_size : token num
    d_model : 每个 token 映射成的向量
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        x: bs, seq_len ---- token id
        paper : embedding * sqrt ，防止 PE 幅度相对太大
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(Module):
    # d_model
    # max_len
    # dropout 加完 PE 做一次 dropout
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0):
        super().__init__()

        self.dropout = Dropout(dropout)

        # 预先算好所有位置的 PE，存成 buffer 不参与训练
        # pe shape ( max_len, d_model )
        pe = torch.zeros(max_len, d_model)

        # pos shape: max_len 1
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term shape: (d_model/2,)
        # 公式里的 10000^(2i/d_model),用 exp + log 数值更稳定
        # 10000
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)  # 偶数用 sin
        pe[:, 1::2] = torch.cos(pos * div_term)  # 奇数用 cos

        # 加一个 batch 维度，方便后面加到输入上
        # pe shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # pe 会随模型保存/加载，但不会被当作参数更新
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: bs, seq_len, d_model
        # self.pe[:,:seq_len] 取出对就长度的 pe 广播加到整个 batch
        x = x + cast(Tensor, self.pe)[:, : x.size(1)]
        return self.dropout(x)
