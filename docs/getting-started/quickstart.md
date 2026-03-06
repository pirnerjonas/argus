# Quickstart

Point Argus at a dataset root. Argus detects YOLO, COCO, and mask layouts automatically.

## 1. List datasets under a directory

```bash
argus-cv list --path /datasets
```

You will get a table with format, task, classes, and splits.

## 2. Inspect class balance and background images

```bash
argus-cv stats /datasets/traffic
```

This prints per-class counts per split and a summary line with image totals.

## 3. Visual inspection

```bash
argus-cv view /datasets/traffic --split val
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
argus-cv split /datasets/traffic -o /datasets/traffic_splits -r 0.8,0.1,0.1
```

This writes the split dataset to the output path and prints counts for each
split.

## 5. Merge split dataset back to unsplit (optional)

```bash
argus-cv unsplit /datasets/traffic_splits -o /datasets/traffic_unsplit
```

## 6. Convert formats (optional)

```bash
# mask -> YOLO segmentation
argus-cv convert -i /datasets/traffic_masks -o /datasets/traffic_yolo --to yolo-seg

# YOLO segmentation -> COCO
argus-cv convert -i /datasets/traffic_yolo -o /datasets/traffic_coco --to coco
```
