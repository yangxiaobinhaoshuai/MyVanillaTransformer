import torch
from torch.nn import Module, Linear
from torch import Tensor
from model.pe import TokenEmbedding, PositionalEncoding
from model.encoder_layer import Encoder
from model.decoder_layer import Decoder


class MyTransformer(Module):
    """
    transformer:
    assemble:
    1. src / tgt token embedding
    2. pe
    3. encoder stack
    4. decoder stack
    5. final linear projection

    train, mask 构造， loss 计算可以继续在外层补
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.src_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_emb = TokenEmbedding(tgt_vocab_size, d_model)

        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        self.output_proj = Linear(d_model, tgt_vocab_size)

    def encode(
        self, src_tokens: Tensor, src_mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        src_x = self.src_emb(src_tokens)
        src_x = self.src_pos_encoding(src_x)

        enc_out, enc_attn_w = self.encoder(src_x, src_mask)

        return enc_out, enc_attn_w

    def decode(
        self,
        tgt_tokens: Tensor,
        enc_out: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:

        tgt_x = self.tgt_emb(tgt_tokens)
        tgt_x = self.tgt_pos_encoding(tgt_x)

        dec_out, self_attn_w, cross_attn_w = self.decoder(
            tgt_x, enc_out, tgt_mask, src_mask
        )

        return dec_out, self_attn_w, cross_attn_w

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, list[Tensor]]]:

        enc_out, enc_attn_w = self.encode(src_tokens, src_mask)

        dec_out, dec_self_attn_w, dec_cross_attn_w = self.decode(
            tgt_tokens, enc_out, tgt_mask, src_mask
        )

        logits = self.output_proj(dec_out)

        attn_info = {
            "encoder_self_attn": enc_attn_w,
            "decoder_self_attn": dec_self_attn_w,
            "decoder_cross_attn": dec_cross_attn_w,
        }

        return logits, attn_info


if __name__ == "__main__":
    torch.manual_seed(0)

    bs = 2
    src_seq_len = 6
    tgt_seq_len = 5
    src_vocab_size = 100
    tgt_vocab_size = 120
    d_model = 128
    n_heads = 4
    d_ff = 256
    n_encoder_layers = 2
    n_decoder_layers = 2

    src_tokens = torch.randint(
        0,
        src_vocab_size,
        (
            bs,
            src_seq_len,
        ),
    )
    tgt_tokens = torch.randint(0, tgt_vocab_size, (bs, tgt_seq_len))

    src_mask = torch.ones(bs, 1, 1, src_seq_len)
    causal = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len))
    tgt_mask = causal.unsqueeze(0).unsqueeze(0).expand(bs, 1, tgt_seq_len, tgt_seq_len)

    model = MyTransformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_encoder_layers,
        n_decoder_layers,
        dropout=0.0,
    )

    logits, attn_info = model(src_tokens, tgt_tokens, src_mask, tgt_mask)

    print("src_tokens shape:", src_tokens.shape)
    print("tgt_tokens shape:", tgt_tokens.shape)
    print("logits shape: ", logits.shape)
    print("encoder attn layers: ", len(attn_info["encoder_self_attn"]))
    print("decoder self attn layers: ", len(attn_info["decoder_self_attn"]))

    assert logits.shape == (bs, tgt_seq_len, tgt_vocab_size)
    assert len(attn_info["encoder_self_attn"]) == n_encoder_layers
    assert len(attn_info["decoder_cross_attn"]) == n_decoder_layers
    print("Transformer asseble test passed")
