import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Dropout


class PositionWiseFFN(Module):
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
        # 每个 token 位置独立过同一个两层 MLP
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, seq_len, d_model, d_ff = 2, 6, 128, 512
    x = torch.randn(bs, seq_len, d_model)

    ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, dropout=0.0)
    out = ffn(x)

    print("input shape :", x.shape)
    print("output shape:", out.shape)

    assert out.shape == (bs, seq_len, d_model), "ffn output shape err"
    print("PositionWiseFFN test passed")
