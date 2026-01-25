# argus-cv

Vision AI dataset toolkit for working with YOLO, COCO, and semantic mask datasets.

**[Documentation](https://pirnerjonas.github.io/argus/)**

## Installation

```bash
uvx argus-cv
```

## Highlights

- Detects YOLO, COCO, and folder-based semantic mask datasets.
- Reports per-class stats (pixel coverage for mask datasets).
- Interactive viewer with bounding boxes, polygons, or mask overlays.
- Split unsplit YOLO/COCO datasets into train/val/test.

## Usage

```bash
# List datasets in current directory
uvx argus-cv list

# List datasets in specific path
uvx argus-cv list --path /path/to/datasets

# Limit search depth
uvx argus-cv list --path . --max-depth 2

# Show instance statistics for a dataset
uvx argus-cv stats --dataset-path /path/to/dataset

# Short form
uvx argus-cv stats -d /path/to/dataset

# View annotations interactively
uvx argus-cv view -d /path/to/dataset --split val

# Split an unsplit dataset into train/val/test
uvx argus-cv split -d /path/to/dataset -o /path/to/output -r 0.8,0.1,0.1
```
