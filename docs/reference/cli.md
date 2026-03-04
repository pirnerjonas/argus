# CLI reference

## Global

```bash
argus-cv --help
```

Commands:

- `list`: discover datasets under a directory tree
- `stats`: print class/split statistics
- `view`: open interactive annotation viewer
- `split`: create train/val/test splits from an unsplit dataset
- `filter`: keep only selected classes in a copied dataset
- `convert`: convert between supported segmentation formats

## list

```bash
argus-cv list --path /datasets --max-depth 3
```

- `--path`, `-p` (default: `.`): root directory to scan
- `--max-depth`, `-d` (default: `3`, range: `1..10`): scan depth

## stats

```bash
argus-cv stats -d /datasets/retail
```

- `--dataset-path`, `-d` (default: `.`): dataset root path

## view

```bash
argus-cv view -d /datasets/retail --split val --opacity 0.5
```

- `--dataset-path`, `-d` (default: `.`): dataset root path
- `--split`, `-s`: split to view (`train`, `val`, `test`)
- `--max-classes`, `-m`: cap classes in classification grid viewer
- `--opacity`, `-o` (default: `0.5`): mask overlay opacity (`0.0..1.0`)

## split

```bash
argus-cv split -d /datasets/animals -o /datasets/animals_splits -r 0.8,0.1,0.1 --seed 42
```

- `--dataset-path`, `-d` (default: `.`): unsplit dataset root
- `--output-path`, `-o` (default: `splits`): output directory
- `--ratio`, `-r` (default: `0.8,0.1,0.1`): train/val/test ratio
- `--seed` (default: `42`): random seed

## filter

```bash
argus-cv filter -d /datasets/coco -o /datasets/coco_filtered --classes person,car --no-background
```

- `--dataset-path`, `-d` (default: `.`): source dataset root
- `--output`, `-o` (default: `filtered`): output directory
- `--classes`, `-c` (required): comma-separated classes to keep
- `--no-background`: drop images with no labels after filtering
- `--symlinks`: symlink images instead of copying

## convert

```bash
argus-cv convert -i /datasets/masks -o /datasets/yolo_seg --to yolo-seg
argus-cv convert -i /datasets/yolo_seg -o /datasets/coco --to coco
argus-cv convert -i /datasets/yolo_seg -o /datasets/rf_coco --to roboflow-coco
```

- `--input-path`, `-i` (default: `.`): source dataset path
- `--output-path`, `-o` (default: `converted`): output directory
- `--to` (default: `yolo-seg`): `yolo-seg`, `coco`, or `roboflow-coco`
- `--epsilon-factor`, `-e` (default: `0.005`): polygon simplification for mask -> YOLO
- `--min-area`, `-a` (default: `100.0`): minimum contour area for mask -> YOLO
