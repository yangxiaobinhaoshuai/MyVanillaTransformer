from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout
from torch import Tensor


class MyMultiHeadAttn(Module):
    """ """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self.dropout = Dropout(dropout)

    
    def split_heads(self,x:Tensor) -> Tensor:
        bs, seq_len = x.shape
        
        pass

    pass
