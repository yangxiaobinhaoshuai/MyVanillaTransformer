import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model1.copy_task import CopyTaskDataset
from model1.train import CHECKPOINT_PATH
from model1.transformer import Transformer


def _load_checkpoint(checkpoint_path: Path = CHECKPOINT_PATH) -> tuple[Transformer, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = Transformer(
        src_vocab_size=config["vocab_size"],
        tgt_vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        n_heads=config["n_heads"],
        src_n_layers=config["src_n_layers"],
        tgt_n_layers=config["tgt_n_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def render_tokens(tokens: torch.Tensor, bos_idx: int, eos_idx: int, pad_idx: int) -> str:
    pieces = []
    for token in tokens.tolist():
        if token == bos_idx:
            pieces.append("<bos>")
        elif token == eos_idx:
            pieces.append("<eos>")
        elif token == pad_idx:
            pieces.append("<pad>")
        else:
            pieces.append(f"tok{token}")
    return " ".join(pieces)


def greedy_decode(
    model: Transformer,
    src_tokens: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    max_new_tokens: int,
) -> torch.Tensor:
    device = src_tokens.device
    bs, src_seq_len = src_tokens.shape

    src_mask = torch.ones(bs, 1, 1, src_seq_len, device=device)
    generated = torch.full((bs, 1), bos_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            tgt_seq_len = generated.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1)

            logits, _ = model(src_tokens, generated, src_mask, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token == eos_idx):
                break

    return generated


def main():
    model, config = _load_checkpoint()

    device = next(model.parameters()).device
    dataset = CopyTaskDataset(
        num_samples=1,
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        pad_idx=config["pad_idx"],
        bos_idx=config["bos_idx"],
        eos_idx=config["eos_idx"],
    )
    sample = dataset[0]

    src_tokens = sample["src_tokens"].unsqueeze(0).to(device)
    target_tokens = sample["tgt_tokens"][1:]
    pred_tokens = greedy_decode(
        model,
        src_tokens,
        bos_idx=config["bos_idx"],
        eos_idx=config["eos_idx"],
        max_new_tokens=config["seq_len"] + 1,
    )[0, 1:].cpu()

    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"src:    {src_tokens[0].cpu().tolist()}")
    print(f"target: {target_tokens.tolist()}")
    print(f"pred:   {pred_tokens.tolist()}")
    print(f"src_text:    {render_tokens(src_tokens[0].cpu(), config['bos_idx'], config['eos_idx'], config['pad_idx'])}")
    print(f"target_text: {render_tokens(target_tokens.cpu(), config['bos_idx'], config['eos_idx'], config['pad_idx'])}")
    print(f"pred_text:   {render_tokens(pred_tokens, config['bos_idx'], config['eos_idx'], config['pad_idx'])}")


if __name__ == "__main__":
    main()
