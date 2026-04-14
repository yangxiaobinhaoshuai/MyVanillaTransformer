import torch
from torch.utils.data import Dataset

from torch import Tensor


class CopyTaskDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        vocab_size: int,
        seq_len: int,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()

        self.nums = num_samples
        self.vocab_size = vocab_size

        self.seq_len = seq_len

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.samples = [self._build_sample() for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.nums

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.samples[idx]

    def _build_sample(self) -> dict[str, Tensor]:
        content_tokens = self._build_sequence()
        src_tokens = content_tokens.clone()
        tgt_tokens = torch.cat(
            [
                torch.tensor([self.bos_idx], dtype=content_tokens.dtype),
                content_tokens,
                torch.tensor([self.eos_idx], dtype=content_tokens.dtype),
            ]
        )

        return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

    def _build_sequence(self) -> Tensor:
        return torch.randint(3, self.vocab_size, (self.seq_len,))
