# Vanilla Transformer 实现计划

## 文件结构

```
myVanillaTransformer/
├── main.py                   # 入口：跑一个演示/训练
├── model/
│   ├── __init__.py
│   ├── embedding.py          # Token Embedding + Positional Encoding
│   ├── attention.py          # Scaled Dot-Product + Multi-Head Attention
│   ├── feed_forward.py       # Position-wise Feed-Forward Network
│   ├── encoder.py            # EncoderLayer + Encoder (N 层堆叠)
│   ├── decoder.py            # DecoderLayer + Decoder (N 层堆叠)
│   └── transformer.py        # 完整 Transformer 模型
├── data/
│   └── dataset.py            # 玩具数据集（Copy Task）
└── train.py                  # 训练循环
```

---

## 每个文件的职责

| 文件 | 核心概念 | 对应论文章节 |
|------|---------|------------|
| `embedding.py` | 词向量 + sin/cos 位置编码 | §3.4, §3.5 |
| `attention.py` | Q/K/V 点积注意力 + 多头注意力 | §3.2 |
| `feed_forward.py` | 两层线性 + ReLU | §3.3 |
| `encoder.py` | Add & Norm + 自注意力 + FFN | §3.1 |
| `decoder.py` | Masked 自注意力 + 交叉注意力 + FFN | §3.1 |
| `transformer.py` | 组装完整模型 + 输出层 | §3 |
| `dataset.py` | Copy Task（输出=输入，方便验证） | - |
| `train.py` | 训练循环、损失、优化器 | - |

---

## 推荐实现顺序（从底层到顶层）

### Step 1 — `model/embedding.py`
最简单，无依赖。两个类：
- `TokenEmbedding`：nn.Embedding 包装
- `PositionalEncoding`：用 sin/cos 公式手算

**学习重点**：为什么 RNN 不需要位置编码，但 Transformer 需要？

### Step 2 — `model/attention.py`
核心中的核心。两个函数/类：
- `scaled_dot_product_attention(Q, K, V, mask)`：手写 softmax(QK^T/√d_k)V
- `MultiHeadAttention`：把 Q/K/V 切分成 h 个头，各自算注意力再拼接

**学习重点**：Q/K/V 分别是什么，为什么要除以 √d_k，mask 在 Decoder 里起什么作用

### Step 3 — `model/feed_forward.py`
- `PositionWiseFFN`：Linear → ReLU → Linear（每个位置独立计算）

### Step 4 — `model/encoder.py`
- `EncoderLayer`：自注意力 + Add&Norm + FFN + Add&Norm
- `Encoder`：N 个 EncoderLayer 堆叠

**学习重点**：残差连接（Add）和 Layer Norm 的作用

### Step 5 — `model/decoder.py`
- `DecoderLayer`：Masked 自注意力 + Add&Norm + 交叉注意力 + Add&Norm + FFN + Add&Norm
- `Decoder`：N 个 DecoderLayer 堆叠

**学习重点**：Masked 注意力防止看到未来 token；交叉注意力的 K/V 来自 Encoder 输出

### Step 6 — `model/transformer.py`
组装：Encoder + Decoder + 最终线性层（映射到词表大小）

### Step 7 — `data/dataset.py`
**Copy Task**：输入一个随机整数序列，目标输出 = 输入（最简单的 seq2seq 任务，验证模型能否学习）

### Step 8 — `train.py` + `main.py`
训练循环、Adam 优化器、交叉熵损失

---

## 学习建议

1. **每写完一个模块就测试 shape**：用 `torch.randn` 造假输入，检查输出维度是否符合预期，远比写完所有再调试容易
2. **在代码里标注 shape**：形状通常是 `(batch, seq_len, d_model)`，注意力权重是 `(batch, n_heads, seq_len, seq_len)`
3. **Copy Task 是最好的验证**：模型能学会"输出=输入"，说明整个 Encoder-Decoder 信息流是通的
4. **推荐参考**：Andrej Karpathy 的 "Let's build GPT from scratch"，注意力机制讲得非常清楚

---

## 验证方式
1. 每个模块写完后：`python -c "from model.attention import MultiHeadAttention; ..."` 检查 shape
2. 全部完成后：`python main.py` 跑 Copy Task，loss 应该能从 ~2.3 降到 ~0.1
