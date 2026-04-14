import torch
from torch.nn import Module, Linear
from torch import Tensor

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model1.pe import TokenEmbedding, PosEmbedding
from model1.encoder_layer import Encoder
from model1.decoder_layer import Decoder

from rich.traceback import install

install(show_locals=False)


class Transformer(Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        src_n_layers: int,
        tgt_n_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.src_token_emb = TokenEmbedding(vocab_size=src_vocab_size, d_model=d_model)
        self.tgt_token_emb = TokenEmbedding(vocab_size=tgt_vocab_size, d_model=d_model)

        self.src_pos_emb = PosEmbedding(
            d_model=d_model, max_len=max_len, dropout=dropout
        )
        self.tgt_pos_emb = PosEmbedding(
            d_model=d_model, max_len=max_len, dropout=dropout
        )

        self.encoder = Encoder(
            n_layers=src_n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            droput=dropout,
        )

        self.decoder = Decoder(
            n_layers=tgt_n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.output_proj = Linear(d_model, tgt_vocab_size)

    def encode(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:

        token_emb_out = self.src_token_emb(x)
        pe_emb_out = self.src_pos_emb(token_emb_out)

        encoder_out, encoder_attn_w = self.encoder(pe_emb_out, mask)

        return encoder_out, encoder_attn_w

    def decode(
        self,
        x: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:

        token_emb_out = self.tgt_token_emb(x)
        pe_emb_out = self.tgt_pos_emb(token_emb_out)

        decoder_out, self_attn_w, cross_attn_w = self.decoder(
            pe_emb_out, enc_out, src_mask, tgt_mask
        )

        return decoder_out, self_attn_w, cross_attn_w

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, list[Tensor]]]:

        enc_out, enc_attn_w = self.encode(src_tokens, src_mask)

        dec_out, dec_self_attn_w, dec_cross_attn_w = self.decode(
            tgt_tokens, enc_out, src_mask, tgt_mask
        )

        logits = self.output_proj(dec_out)

        attn_info = {
            "enc_attn_w": enc_attn_w,
            "dec_self_attn_w": dec_self_attn_w,
            "dec_cross_attn_w": dec_cross_attn_w,
        }

        return logits, attn_info


if __name__ == "__main__":
    torch.manual_seed(0)

    bs = 2

    src_seq_len = 5
    tgt_seq_len = 6
    src_vocab_size = 100
    tgt_vocab_size = 120
    src_n_layers = 4
    tgt_n_layers = 8

    d_model = 128
    d_ff = 256

    n_heads = 4

    max_length = 5000

    src_tokens = torch.randint(0, src_vocab_size, (bs, src_seq_len))
    tgt_tokens = torch.randint(0, tgt_vocab_size, (bs, tgt_seq_len))

    src_mask = torch.ones(bs, 1, 1, src_seq_len)
    casual = torch.ones(tgt_seq_len, tgt_seq_len)
    tgt_mask = casual.unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        src_n_layers=src_n_layers,
        tgt_n_layers=tgt_n_layers,
        max_len=max_length,
    )

    logits, attn_info = model(src_tokens, tgt_tokens, src_mask, tgt_mask)

    print(" logits.shape: ", logits.shape)

    assert logits.shape == (bs, tgt_seq_len, tgt_vocab_size), " logits shape err"

    print(" enc attn w shape: ", attn_info["enc_attn_w"][0].shape)

    assert attn_info["enc_attn_w"][0].shape == (
        bs,
        n_heads,
        src_seq_len,
        src_seq_len,
    ), "enc attn w shape err"

    assert attn_info["dec_self_attn_w"][0].shape == (
        bs,
        n_heads,
        tgt_seq_len,
        tgt_seq_len,
    ), "dec self attn shape err"

    print(" dec cross attn w shape: ", attn_info["dec_cross_attn_w"][0].shape)

    assert attn_info["dec_cross_attn_w"][0].shape == (
        bs,
        n_heads,
        tgt_seq_len,
        src_seq_len,
    ), "dec cross attn w shape err"

    print("transformer test all passed")
