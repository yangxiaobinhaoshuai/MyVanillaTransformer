import torch
from torch import Tensor
from torch.nn import Module, Linear, Dropout
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model1.scaled_dot_prod_attn import scaled_dot_prod_attn


class MultiHeadAttn(Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be diviable by n_heads"

        d_k = d_model // n_heads

        self.d_k = d_k
        self.n_heads = n_heads
        self.d_model = d_model

        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.o_linear = Linear(d_model, d_model)

        self.dropout = Dropout(dropout)

    def split_heads(self, x: Tensor) -> Tensor:

        bs, seq_len, _ = x.shape

        # bs, seq_len, d_model

        # b n h d
        x = x.view(bs, seq_len, self.n_heads, self.d_k)

        # b h n d
        x = x.transpose(2, 1)

        return x

    def combine_heads(self, x: Tensor) -> Tensor:

        bs, _, seq_len, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(bs, seq_len, self.d_model)

        return x

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        # bs, seq_len, d_model
        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)

        # bs, n_heads,seq_len, d_k
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output, attn_w = scaled_dot_prod_attn(Q, K, V, mask, self.dropout)

        # bs, seq_len, d_model
        output = self.combine_heads(output)

        output = self.o_linear(output)

        return output, attn_w


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, n_heads, seq_len, d_model = 2, 4, 64, 128

    Q = torch.randn(bs, seq_len, d_model)
    K = torch.randn(bs, seq_len, d_model)
    V = torch.randn(bs, seq_len, d_model)

    mha = MultiHeadAttn(n_heads=n_heads, d_model=d_model, dropout=0.0)

    output, attn_w = mha(Q, K, V)

    print("output shape: ", output.shape)
    print("attn_w shape: ", attn_w.shape)

    assert output.shape == (bs, seq_len, d_model), (
        f"Expected {(bs, seq_len, d_model)}, got {output.shape}"
    )

    assert attn_w.shape == (bs, n_heads, seq_len, seq_len), "attn_w shape err"

    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    masked_output, masked_attn_w = mha(Q, K, V, mask)

    print("masked output shape: ", masked_output.shape)
    print("masked attn_w shape: ", masked_attn_w.shape)

    assert masked_attn_w[0, 0, 0, 1:].sum().item() < 1e-6, "causal mask failed"

    print("mha test all passed")
