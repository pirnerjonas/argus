"""Ultralytics ground-truth helpers for cross-validating argus YOLO stats.

This module is NOT a test file. It wraps ultralytics to extract reference
stats from a YOLO dataset so that argus results can be compared against
the canonical implementation.
"""

import logging
from pathlib import Path

from ultralytics.data.dataset import YOLODataset as UltralyticsYOLODataset
from ultralytics.data.utils import check_det_dataset


def get_ground_truth_stats(yaml_path: Path, split: str) -> dict:
    """Load a YOLO split via ultralytics and return ground-truth stats.

    Args:
        yaml_path: Absolute path to the data.yaml file.
        split: Split key exactly as it appears in data.yaml (e.g. "train", "val").

    Returns:
        Dictionary with keys:
            total       - number of images ultralytics found for this split
            background  - images with zero bounding boxes
            instance_counts - {int_class_id: count} across all images
    """
    # Suppress ultralytics logging noise
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # check_det_dataset resolves paths and validates the YAML
    data_dict = check_det_dataset(str(yaml_path))

    # Build the dataset object for the requested split
    img_path = data_dict[split]
    dataset = UltralyticsYOLODataset(img_path=img_path, data=data_dict)

    total = len(dataset)
    background = 0
    instance_counts: dict[int, int] = {}

    for label_info in dataset.labels:
        bboxes = label_info["bboxes"]
        cls = label_info["cls"]

        if len(bboxes) == 0:
            background += 1
            continue

        for c in cls.flatten().tolist():
            class_id = int(c)
            instance_counts[class_id] = instance_counts.get(class_id, 0) + 1

    return {
        "total": total,
        "background": background,
        "instance_counts": instance_counts,
    }
