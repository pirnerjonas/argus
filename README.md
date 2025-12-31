# argus

Vision AI dataset toolkit for working with YOLO and COCO datasets.

## Installation

```bash
uvx argus
```

## Usage

```bash
# List datasets in current directory
uvx argus list

# List datasets in specific path
uvx argus list --path /path/to/datasets

# Limit search depth
uvx argus list --path . --max-depth 2

# Show instance statistics for a dataset
uvx argus stats --dataset-path /path/to/dataset

# Short form
uvx argus stats -d /path/to/dataset
```
