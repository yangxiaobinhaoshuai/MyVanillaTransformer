# model1

Run training:

```bash
uv run python model1/train.py
```

After training, the checkpoint is saved to:

```text
model1/checkpoints/copy_task.pt
```

Run inference:

```bash
uv run python model1/infer.py
```

`infer.py` loads the checkpoint above and prints:

- `src`: source token ids
- `target`: expected target token ids, ending with `<eos>`
- `pred`: greedy-decoded token ids
- `src_text` / `target_text` / `pred_text`: readable token strings such as `tok93` and `<eos>`

Note: the rendered `*_text` output is only a readable view of token ids like `tok93`; it is not natural-language text.
