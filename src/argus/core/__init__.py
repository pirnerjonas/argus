"""Core dataset detection and handling."""

from argus.core.background import add_background_to_coco, add_background_to_yolo
from argus.core.base import Dataset
from argus.core.coco import COCODataset
from argus.core.split import split_coco_dataset, split_yolo_dataset
from argus.core.yolo import YOLODataset

__all__ = [
    "Dataset",
    "YOLODataset",
    "COCODataset",
    "add_background_to_coco",
    "add_background_to_yolo",
    "split_coco_dataset",
    "split_yolo_dataset",
]
