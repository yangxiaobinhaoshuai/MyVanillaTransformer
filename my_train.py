import torch
from torch.utils.data import DataLoader
from model.my_transformer import MyTransformer
from data.my_dataset import MyCopyTaskDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def build_dataloader(
    num_samples: int, seq_len: int, vocab_size: int, batch_size: int
) -> DataLoader:
    dataset = MyCopyTaskDataset(num_samples, seq_len, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_causal_mask(batch_size: int, seq_len: int, device: torch.device) -> Tensor:
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def train_one_epoch(
    model: MyTransformer,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: Adam,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        src_tokens = batch["src_tokens"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)

         batch_size, src_seq_len = src_tokens.shape
         _,tgt_seq_len = tgt_tokens.shape


        src_mask = torch.ones(batch_size, 1, 1, src_seq_len, device=device)
        tgt_mask = build_causal_mask(batch_size,tgt_seq_len,device=)

        logits, _ = model(
            src_tokens,
            tgt_tokens,
            src_mask,
            tgt_mask
        )

        # 当前模板直接把 copy task 当作逐位置分类练通整个前向流程
        loss = criterion(logits.reshape(-1,logits.size(-1)),tgt_tokens.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 20
    seq_len = 6
    num_samples = 128
    batch_size = 16
    num_epochs = 3

    model = MyTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.0,
    ).to(device)

    dataloader = build_dataloader(num_samples, seq_len, vocab_size, batch_size)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            criterion,
            optimizer,
            device
        )
        print(f"epoch {epoch +1 }: loss = {avg_loss:.4f}")


if __name__ == "__main__":
    main()
