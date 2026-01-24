"""Core dataset detection and handling."""

from argus.core.base import Dataset
from argus.core.coco import COCODataset
from argus.core.mask import ConfigurationError, MaskDataset
from argus.core.split import split_coco_dataset, split_yolo_dataset
from argus.core.yolo import YOLODataset

__all__ = [
    "Dataset",
    "YOLODataset",
    "COCODataset",
    "MaskDataset",
    "ConfigurationError",
    "split_coco_dataset",
    "split_yolo_dataset",
]
