import torch
from torch import Tensor
from typing import Optional
from torch.nn import Dropout
import torch.nn.functional as F
import math


def scaled_dot_prod_att(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[Dropout] = None,
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
