"""Core dataset detection and handling."""

from argus.core.base import Dataset
from argus.core.yolo import YOLODataset
from argus.core.coco import COCODataset

__all__ = ["Dataset", "YOLODataset", "COCODataset"]
