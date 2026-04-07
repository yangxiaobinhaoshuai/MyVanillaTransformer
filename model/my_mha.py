import torch
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout
from torch import Tensor
from model.my_attn import scale_dot_production_attn


class MyMultiHeadAttn(Module):
    """
    input :
    q: bs,seq_len,d_model
    k: bs,seq_len,d_model
    v: bs,seq_len,d_model
    output:
    attn_output: bs,seq_len,d_model
    attn_w: bs,n_heads,seq_q,seq_k
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
        # bs, seq_len, d_model
        bs, seq_len, _ = x.shape
        # bs, seq_len, n_heads,d_k
        x = x.view(bs, seq_len, self.n_heads, self.d_k)
        # bs, n_heads, seq_len, d_k
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x: Tensor) -> Tensor:
        bs, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(bs, seq_len, self.d_model)
        return x

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output, attn_w = scale_dot_production_attn(Q, K, V, mask, self.dropout)

        output = self.combine_heads(attn_output)

        output = self.W_o(output)

        return output, attn_w


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, seq_len, d_model, n_heads = 2, 6, 128, 4

    query = torch.randn(bs, seq_len, d_model)
    key = torch.randn(bs, seq_len, d_model)
    value = torch.randn(bs, seq_len, d_model)

    mha = MyMultiHeadAttn(d_model, n_heads=n_heads, dropout=0.0)

    attn_output, attn_w = mha(query, key, value)

    print("output shape ", attn_output.shape)
    print("attn_w shape ", attn_w.shape)

    assert attn_output.shape == (bs, seq_len, d_model), "output shape err"
    assert attn_w.shape == (bs, n_heads, seq_len, seq_len), "attn_w shape err"

    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output_masked, attn_w_masked = mha(query, key, value, causal_mask)

    print("output mask shape ", output_masked.shape)

    assert attn_w_masked[0, 0, 0, 1:].sum().item() < 1e-6, "causal mask failed"

    print("MyMultiHeadAttn test passed")
