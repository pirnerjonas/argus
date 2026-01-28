"""Core dataset detection and handling."""

from argus.core.base import Dataset
from argus.core.coco import COCODataset
from argus.core.convert import (
    ConversionParams,
    Polygon,
    convert_mask_to_yolo_labels,
    convert_mask_to_yolo_seg,
    mask_to_polygons,
)
from argus.core.filter import (
    filter_coco_dataset,
    filter_mask_dataset,
    filter_yolo_dataset,
)
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
    "filter_yolo_dataset",
    "filter_coco_dataset",
    "filter_mask_dataset",
    "ConversionParams",
    "Polygon",
    "mask_to_polygons",
    "convert_mask_to_yolo_labels",
    "convert_mask_to_yolo_seg",
]
