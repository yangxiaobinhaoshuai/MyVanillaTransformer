import torch
from torch import Module, Tensor


class MyCopyTaskDataset(Module):
    """
    最小版 copy task dataset

    objective:
    1. 随机生成 source token 序列
    2. target 与 source 保持一致
    3. 预留 special token, 方便后面接训练拓展脚本

    - padding
    - bos / eos
    - mask 构造
    - tgt_input / tgt_ouput 切分
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idex: int = 2,
    ):
        super().__init__()

        assert vocab_size > 3, "vocab size at least > 3"
        assert seq_len >= 1, "seq_len at least >=1 "

        self.num_smaples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idex

    def __len__(self) -> int:
        return self.num_smaples

    def _build_sequence(self) -> Tensor:
        """
        生成一个不含 special token 的随机 token 序列
        token 范围从 3 开始， 避开 pad / bos / eos
        """
        return torch.randint(3, self.vocab_size, (self.seq_len,))

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        del idx

        src_tokens = self._build_sequence()
        tgt_tokens = src_tokens.clone()

        return {
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = MyCopyTaskDataset(num_samples=4, seq_len=6, vocab_size=20)

    sample = dataset[0]

    print("dataset len: ", len(dataset))
    print("src_tokens:", sample["src_tokens"])
    print("tgt_tokens:", sample["tgt_tokens"])
    print("src shape:", sample["src_tokens"].shape)
    print("tgt shape:", sample["tgt_tokens"].shape)

    assert len(dataset) == 4
    assert sample["src_tokens"].shape == (6,)
    assert sample["tgt_tokens"].shape == (6,)
    assert torch.equal(sample["src_tokens"], sample["tgt_tokens"])

    print("CopyTaskDataset test passed")
