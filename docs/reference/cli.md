# CLI reference

## Global

```bash
argus-cv --help
```

Argus uses subcommands: `list`, `stats`, `view`, `split`, and `convert`.

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
- `--opacity`, `-o`: mask overlay opacity (mask datasets only)

## split

Create train/val/test splits from an unsplit dataset.

```bash
argus-cv split --dataset-path /datasets/animals \
  --output-path /datasets/animals_splits \
  --ratio 0.8,0.1,0.1 \
  --seed 42
```

Options:

- `--dataset-path`, `-d`: dataset root path
- `--output-path`, `-o`: output directory (default: "splits" inside dataset path)
- `--ratio`, `-r`: train/val/test ratio (default: 0.8,0.1,0.1)
- `--seed`: random seed (default: 42)

## convert

Convert a dataset from one format to another. Currently supports converting
MaskDataset to YOLO segmentation format.

```bash
argus-cv convert --input-path /path/to/masks \
  --output-path /path/to/output \
  --to yolo-seg
```

Options:

- `--input-path`, `-i`: source dataset path (default: ".")
- `--output-path`, `-o`: output directory (default: "converted" next to the input path)
- `--to`: target format (currently only `yolo-seg`)
- `--epsilon-factor`, `-e`: polygon simplification factor (default: 0.005)
- `--min-area`, `-a`: minimum contour area in pixels (default: 100.0)
