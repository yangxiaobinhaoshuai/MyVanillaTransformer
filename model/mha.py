from torch import Tensor
from torch.nn import Dropout, Linear, Module
import torch

from model.my_attn import scale_dot_production_attn


class MultiHeadAttention(Module):
    """
    input:
        query: bs, seq_q, d_model
        key:   bs, seq_k, d_model
        value: bs, seq_k, d_model

    output:
        output: bs, seq_q, d_model
        attn_w: bs, n_heads, seq_q, seq_k
    """

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

    def split_heads(self, x: Tensor) -> Tensor:
        bs, seq_len, _ = x.shape

        # bs, seq_len, d_model -> bs, seq_len, n_heads, d_k
        x = x.view(bs, seq_len, self.n_heads, self.d_k)

        # bs, seq_len, n_heads, d_k -> bs, n_heads, seq_len, d_k
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x: Tensor) -> Tensor:
        bs, _, seq_len, _ = x.shape

        # bs, n_heads, seq_len, d_k -> bs, seq_len, n_heads, d_k
        x = x.transpose(1, 2).contiguous()

        # bs, seq_len, n_heads, d_k -> bs, seq_len, d_model
        x = x.view(bs, seq_len, self.d_model)
        return x

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # 1. linear projection
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. per-head attention
        attn_output, attn_w = scale_dot_production_attn(
            Q,
            K,
            V,
            mask=mask,
            dropout=self.dropout,
        )

        # 4. combine heads
        output = self.combine_heads(attn_output)

        # 5. final linear
        output = self.W_o(output)

        return output, attn_w


if __name__ == "__main__":
    torch.manual_seed(1)

    bs, seq_len, d_model, n_heads = 2, 6, 128, 4

    query = torch.randn(bs, seq_len, d_model)
    key = torch.randn(bs, seq_len, d_model)
    value = torch.randn(bs, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

    output, attn_w = mha(query, key, value)

    print("output shape:", output.shape)
    print("attn_w shape:", attn_w.shape)

    assert output.shape == (bs, seq_len, d_model), "output shape error"
    assert attn_w.shape == (bs, n_heads, seq_len, seq_len), "attn_w shape error"

    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output_masked, attn_w_masked = mha(query, key, value, mask=causal_mask)

    print("masked output shape:", output_masked.shape)

    assert attn_w_masked[0, 0, 0, 1:].sum().item() < 1e-6, "causal mask failed"
    print("MultiHeadAttention test passed")
