# Quickstart

Point Argus at a dataset root. It will detect YOLO or COCO automatically.

## 1. List datasets under a directory

```bash
argus-cv list --path /datasets
```

You will get a table with format, task, classes, and splits.

## 2. Inspect class balance and background images

```bash
argus-cv stats -d /datasets/traffic
```

This prints per-class counts per split and a summary line with image totals.

## 3. Visual inspection

```bash
argus-cv view -d /datasets/traffic --split val
```

Controls inside the viewer:

- `N` or right arrow: next image
- `P` or left arrow: previous image
- Mouse wheel: zoom
- Drag: pan when zoomed
- `R`: reset zoom
- `Q` or `Esc`: quit

## 4. Split an unsplit dataset

```bash
argus-cv split -d /datasets/traffic -o /datasets/traffic_splits -r 0.8,0.1,0.1
```

This writes the split dataset to the output path and prints counts for each
split.
