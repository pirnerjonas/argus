"""Core dataset detection and handling."""

from argus.core.base import Dataset, Partitioning
from argus.core.coco import COCODataset, COCOLayout
from argus.core.convert import (
    ConversionParams,
    Polygon,
    convert_mask_to_yolo_labels,
    convert_mask_to_yolo_seg,
    convert_yolo_seg_to_coco,
    convert_yolo_seg_to_roboflow_coco,
    convert_yolo_seg_to_roboflow_coco_rle,
    mask_to_polygons,
)
from argus.core.filter import (
    filter_coco_dataset,
    filter_mask_dataset,
    filter_yolo_dataset,
)
from argus.core.mask import ConfigurationError, MaskDataset
from argus.core.split import (
    split_coco_dataset,
    split_mask_dataset,
    split_yolo_dataset,
    unsplit_coco_dataset,
    unsplit_mask_dataset,
    unsplit_yolo_dataset,
)
from argus.core.validation import ValidationIssue, ValidationReport, validate_dataset
from argus.core.yolo import YOLODataset

__all__ = [
    "Dataset",
    "Partitioning",
    "YOLODataset",
    "COCODataset",
    "COCOLayout",
    "MaskDataset",
    "ConfigurationError",
    "split_coco_dataset",
    "split_mask_dataset",
    "split_yolo_dataset",
    "unsplit_coco_dataset",
    "unsplit_mask_dataset",
    "unsplit_yolo_dataset",
    "filter_yolo_dataset",
    "filter_coco_dataset",
    "filter_mask_dataset",
    "ConversionParams",
    "Polygon",
    "mask_to_polygons",
    "convert_mask_to_yolo_labels",
    "convert_mask_to_yolo_seg",
    "convert_yolo_seg_to_coco",
    "convert_yolo_seg_to_roboflow_coco",
    "convert_yolo_seg_to_roboflow_coco_rle",
    "validate_dataset",
    "ValidationIssue",
    "ValidationReport",
]
