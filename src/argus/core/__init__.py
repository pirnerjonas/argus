"""Core dataset detection and handling."""

from argus.core.base import Dataset
from argus.core.coco import COCODataset
from argus.core.yolo import YOLODataset

__all__ = ["Dataset", "YOLODataset", "COCODataset"]
