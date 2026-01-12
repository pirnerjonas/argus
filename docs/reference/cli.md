# CLI reference

## Global

```bash
argus --help
```

Argus uses subcommands: `list`, `stats`, `view`, and `split`.

## list

Scan a directory tree and report detected datasets.

```bash
argus-cv list --path . --max-depth 3
```

Options:

- `--path`, `-p`: root directory to search
- `--max-depth`, `-d`: maximum depth to search (1-10)

## stats

Show instance counts per class and per split.

```bash
argus-cv stats --dataset-path /datasets/retail
```

Options:

- `--dataset-path`, `-d`: dataset root path

## view

Launch an interactive annotation viewer.

```bash
argus-cv view --dataset-path /datasets/retail --split val
```

Options:

- `--dataset-path`, `-d`: dataset root path
- `--split`, `-s`: split to view (train, val, test)

## split

Create train/val/test splits from an unsplit dataset.

```bash
argus-cv split --dataset-path /datasets/animals \
  --output-path /datasets/animals_splits \
  --ratio 0.8,0.1,0.1 \
  --stratify \
  --seed 42
```

Options:

- `--dataset-path`, `-d`: dataset root path
- `--output-path`, `-o`: output directory (default: "splits" inside dataset path)
- `--ratio`, `-r`: train/val/test ratio (default: 0.8,0.1,0.1)
- `--stratify/--no-stratify`: enable or disable stratified splitting (default: enabled)
- `--seed`: random seed (default: 42)
