import torch
from torch.nn import Module, Linear, ReLU, Dropout
from torch import Tensor


class PositionWiseFNN(Module):
    """
    input:
    x: bs, seq_len, d_model

    output:
    out: bs, seq_len, d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # 每个 token 的位置过同一个两层 ffn
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == "__name__":
    torch.manual_seed(0)

    bs, seq_len, d_model, d_ff = 2, 6, 128, 512

    x = torch.randn(bs, seq_len, d_model)

    ffn = PositionWiseFNN(d_model, d_ff, dropout=0.0)

    out = ffn(x)

    print("x shape: ", x.shape)
    print("out shape: ", out.shape)

    assert out.shape == (bs, seq_len, d_model), "ffn shape test err"
    print("FFN test passed")
