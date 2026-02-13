# Python API reference

This page documents the supported library API for programmatic use.

Canonical imports use `argus.core`:

```python
from argus.core import (
    COCODataset,
    MaskDataset,
    YOLODataset,
    convert_mask_to_yolo_labels,
    convert_mask_to_yolo_seg,
    filter_coco_dataset,
    filter_mask_dataset,
    filter_yolo_dataset,
    mask_to_polygons,
    split_coco_dataset,
    split_yolo_dataset,
)
```

## Dataset classes

### `YOLODataset`

::: argus.core.YOLODataset

### `COCODataset`

::: argus.core.COCODataset

### `MaskDataset`

::: argus.core.MaskDataset

## Split operations

### `split_yolo_dataset`

::: argus.core.split_yolo_dataset

### `split_coco_dataset`

::: argus.core.split_coco_dataset

## Filter operations

### `filter_yolo_dataset`

::: argus.core.filter_yolo_dataset

### `filter_coco_dataset`

::: argus.core.filter_coco_dataset

### `filter_mask_dataset`

::: argus.core.filter_mask_dataset

## Convert operations

### `mask_to_polygons`

::: argus.core.mask_to_polygons

### `convert_mask_to_yolo_labels`

::: argus.core.convert_mask_to_yolo_labels

### `convert_mask_to_yolo_seg`

::: argus.core.convert_mask_to_yolo_seg
