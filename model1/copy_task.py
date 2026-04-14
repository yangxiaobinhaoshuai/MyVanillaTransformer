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

    def __len__(self) -> int:
        return self.nums

    def __getitem__(self,idx:int) -> dict[str, Tensor]:
        del idx
        
        src_tokens = self._build_sequence()
        tgt_tokens = src_tokens.clone()
        return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

    def _build_sequence(self) -> Tensor:
        return torch.randint(3, self.vocab_size, (self.seq_len,))
