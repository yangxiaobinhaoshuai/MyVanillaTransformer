import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model1.transformer import Transformer

from torch.utils.data import DataLoader

from model1.copy_task import CopyTaskDataset

from torch.nn import CrossEntropyLoss

from torch.optim import Adam

from rich.traceback import install

install(show_locals=False)


def train_one_epoch(
    model: Transformer,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
    dataloader: DataLoader,
    device: torch.device,
):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        src_tokens = batch["src_tokens"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)

        bs, src_seq_len = src_tokens.shape
        _, tgt_seq_len = tgt_tokens.shape

        src_mask = torch.ones(bs, 1, 1, src_seq_len).to(device)
        tgt_mask = (
            torch.tril(torch.ones(tgt_seq_len, tgt_seq_len))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, -1, -1, -1)
            .to(device)
        )

        logits, _ = model(src_tokens, tgt_tokens, src_mask, tgt_mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def _build_dataloader(
    bs: int, num_examples: int, vocab_size: int, seq_len: int
) -> DataLoader:

    dataset = CopyTaskDataset(
        num_samples=num_examples, vocab_size=vocab_size, seq_len=seq_len
    )
    return DataLoader(dataset=dataset, batch_size=bs, shuffle=True)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 200
    d_model = 128
    d_ff = 256
    n_heads = 4
    src_n_layers = 6
    tgt_n_layers = 7

    seq_len = 64
    batch_size = 24
    epochs = 20
    num_examples = 100

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        src_n_layers=src_n_layers,
        tgt_n_layers=tgt_n_layers,
    ).to(device)

    dataloader = _build_dataloader(
        bs=batch_size,
        num_examples=num_examples,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    criterion = CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, criterion, optimizer, dataloader, device)
        print(f"epoch: {epoch + 1}, loss:{avg_loss:.4f}")


if __name__ == "__main__":
    main()
