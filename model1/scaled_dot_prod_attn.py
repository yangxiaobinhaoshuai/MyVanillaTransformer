import torch

import math
from torch import Tensor
from torch.nn import Dropout
import torch.nn.functional as F


def scaled_dot_prod_attn(
    Q: Tensor,  # b h n d
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
    dropout: Dropout | None = None,
) -> tuple[Tensor, Tensor]:

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = torch.masked_fill(scores, mask == 0, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_w = dropout(attn_w)

    output = torch.matmul(attn_w, V)

    return output, attn_w


if __name__ == "__main__":
    torch.manual_seed(0)

    bs, n_heads, seq_len, d_k = 2, 4, 64, 128

    Q = torch.randn(bs, n_heads, seq_len, d_k)
    K = torch.randn(bs, n_heads, seq_len, d_k)
    V = torch.randn(bs, n_heads, seq_len, d_k)

    output, attn_w = scaled_dot_prod_attn(Q, K, V)

    print("output shape:", output.shape)
    print("attn_w shape:", attn_w.shape)

    assert torch.allclose(
        torch.sum(attn_w, dim=-1), torch.ones(bs, n_heads, seq_len)
    ), "output weight is illegal"

    mask = torch.ones(seq_len, seq_len)

    # 1,1,seq_len,seq_len
    mask = torch.tril(mask).unsqueeze(0).unsqueeze(0)
    masked_output, masked_attn_w = scaled_dot_prod_attn(Q, K, V, mask)

    print("masked output shaep:", masked_output.shape)
    print("maked attn_w sahpe:", masked_attn_w.shape)

    assert torch.sum(masked_attn_w[0, 0, 0, 1:]) < 1e-6, "masked output is illegal"
    # assert masked_attn_w[0,0,0,1:].sum().item() < 1e-6, "masked output is illegal"

    print("attn test all passed")
