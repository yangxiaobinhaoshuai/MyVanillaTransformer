from torch import Tensor
from torch.nn import Dropout
import math
from torch.nn import functional as F
import torch


def scale_dot_production_attn(
    Q: Tensor,  # bs, n_heads, seq_q, d_k
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
    dropout: Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Attn( Q,K,V ) =. Softmax( QK^T / sqrt(d_k) ) * V

    1. why / sqrt(d_k) 防 QK 点积 随 d_k 增大而增大

    2. output :
     1. output  bs, n_heads, seq_q, d_v
     2. attn_w  bs, n_heads, seq_q, seq_k

    """

    d_k = Q.size(-1)

    # 1. attn score

    scores = Q.matmul(K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. mask

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3.
    attn_w = F.softmax(scores, dim=-1)

    # 4. dropout
    if dropout is not None:
        attn_w = dropout(attn_w)

    output = attn_w.matmul(V)

    return output, attn_w


if __name__ == "__main__":
    torch.manual_seed(1)

    bs, n_heads, seq_len, d_k, d_v = 2, 4, 10, 64, 64

    Q = torch.randn(bs, n_heads, seq_len, d_k)
    K = torch.randn(bs, n_heads, seq_len, d_k)
    V = torch.randn(bs, n_heads, seq_len, d_v)

    output, attn_w = scale_dot_production_attn(Q, K, V)

    print("output: ", output)
    print("attn_w: ", attn_w)

    assert torch.allclose(attn_w.sum(dim=1), torch.ones(bs, n_heads, seq_len)), (
        "weight sum != 1"
    )

    causal_mask: Tensor = (
        torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    )

    output_masked, attn_w_masked = scale_dot_production_attn(Q, K, V, causal_mask)
