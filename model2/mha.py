from torch import nn, Tensor
from torch.nn import Linear, Dropout
from typing import Optional

from model2.attn import scaled_dot_prod_att


class MultiHeadAttnModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisble by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self.dropout = Dropout(dropout)

    def split_heads(self, x: Tensor) -> Tensor:
        bs, seq_len, _ = x.shape

        x = x.view(bs, seq_len, self.n_heads, self.d_k)

        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x: Tensor) -> Tensor:

        bs, _, seq_len, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(bs, seq_len, self.d_model)

        return x

    def forward(
        self,
        x: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output, attn_w = scaled_dot_prod_att(Q, K, V, mask, self.dropout)

        output = self.combine_heads(attn_output)

        output = self.W_o(output)

        return output, attn_w
