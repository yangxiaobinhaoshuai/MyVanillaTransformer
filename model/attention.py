import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,  # (batch, n_heads, seq_q, d_k)
    K: torch.Tensor,  # (batch, n_heads, seq_k, d_k)
    V: torch.Tensor,  # (batch, n_heads, seq_k, d_v)
    mask: torch.Tensor | None = None,  # (batch, 1, seq_q, seq_k) 或 None
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    论文公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    为什么要除以 sqrt(d_k)？
        Q 和 K 的点积随 d_k 增大而增大，导致 softmax 进入梯度极小的饱和区。
        除以 sqrt(d_k) 把方差拉回到 1，让梯度流动更稳定。

    返回：
        output  : (batch, n_heads, seq_q, d_v)  — 加权求和后的值
        attn_w  : (batch, n_heads, seq_q, seq_k) — 注意力权重（可视化用）
    """
    d_k = Q.size(-1)

    # Step 1: 计算相似度分数，shape → (batch, n_heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: 应用 mask（Decoder 的 masked self-attention 用到）
    # mask 为 True 的位置填充 -inf，softmax 后趋近于 0（即"忽略"该位置）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 3: softmax 归一化，得到注意力权重
    attn_w = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_q, seq_k)

    # Step 4: dropout（训练时随机丢弃部分注意力连接，防止过拟合）
    if dropout is not None:
        attn_w = dropout(attn_w)

    # Step 5: 用注意力权重对 V 加权求和
    output = torch.matmul(attn_w, V)  # (batch, n_heads, seq_q, d_v)

    return output, attn_w


# ---------------------------------------------------------------------------
# 单元测试：直接运行此文件时执行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch, n_heads, seq_len, d_k, d_v = 2, 4, 10, 64, 64

    Q = torch.randn(batch, n_heads, seq_len, d_k)
    K = torch.randn(batch, n_heads, seq_len, d_k)
    V = torch.randn(batch, n_heads, seq_len, d_v)

    output, attn_w = scaled_dot_product_attention(Q, K, V)

    print("✓ output shape :", output.shape)   # 期望 (2, 4, 10, 64)
    print("✓ attn_w shape :", attn_w.shape)   # 期望 (2, 4, 10, 10)

    # 验证注意力权重每行之和为 1
    assert torch.allclose(attn_w.sum(dim=-1), torch.ones(batch, n_heads, seq_len)), \
        "注意力权重行和不为 1！"
    print("✓ 注意力权重行和 = 1，验证通过")

    # 测试带 mask 的情况（模拟 decoder：seq_q=3 时只能看到前 i 个 token）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output_masked, attn_w_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print("✓ masked output shape:", output_masked.shape)

    # 验证 causal mask 有效：第 0 行只有位置 0 有权重，其余应为 0
    assert attn_w_masked[0, 0, 0, 1:].sum().item() < 1e-6, \
        "Causal mask 失效！第 0 行不应该 attend 到后续位置"
    print("✓ Causal mask 验证通过")
