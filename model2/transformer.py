import torch
from torch.nn import Module, Linear
from torch import Tensor

from model2.encoder import Encoder
from model2.decoder import Decoder
from model2.emb import TokenEmbedding, PositionalEmbedding
from typing import Optional


class Transformer(Module):
    def __int__(
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

        self.src_pe = PositionalEmbedding(d_model, max_len, dropout)
        self.tgt_pe = PositionalEmbedding(d_model, max_len, dropout)

        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        self.output_proj = Linear(d_model, tgt_vocab_size)

    def encode(
        self,
        src_tokens: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        src_x = self.src_emb(src_tokens)
        src_x = self.src_pe(src_x)
        enc_out, enc_attn_w = self.encode(src_x, src_mask)
        return enc_out, enc_attn_w

    def decode(
        self,
        tgt_tokens: Tensor,
        enc_out: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:

        tgt_x = self.tgt_emb(tgt_tokens)
        tgt_x = self.tgt_pe(tgt_x)
        dec_out, self_attn_w, cross_attn_w = self.decoder(
            tgt_x,
            enc_out,
            tgt_mask,
            src_mask,
        )
        return dec_out, self_attn_w, cross_attn_w

    def forward(self, x: Tensor) -> Tensor: ...
