from torch import nn, Tensor
from torch.nn import Linear, ReLU, Dropout


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.relu1 = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
