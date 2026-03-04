"""Conversion functions for dataset format transformation."""

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from argus.core.mask import MaskDataset

logger = logging.getLogger(__name__)


@dataclass
class ConversionParams:
    """Parameters for mask-to-polygon conversion.

    Attributes:
        class_id: Class ID for the resulting polygon.
        epsilon_factor: Douglas-Peucker simplification factor (relative to perimeter).
        min_area: Minimum contour area in pixels to include.
    """

    class_id: int = 0
    epsilon_factor: float = 0.005
    min_area: float = 100.0


@dataclass
class Polygon:
    """A polygon annotation with class ID and normalized points.

    Attributes:
        class_id: Class ID for this polygon.
        points: List of (x, y) points normalized to [0, 1].
    """

    class_id: int
    points: list[tuple[float, float]]

    def to_yolo(self) -> str:
        """Convert to YOLO segmentation format string.

        Returns:
            String in format: "class_id x1 y1 x2 y2 ... xn yn"
        """
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {coords}"


def mask_to_polygons(
    mask: NDArray[np.uint8],
    params: ConversionParams | None = None,
) -> list[Polygon]:
    """Convert a binary mask to simplified polygons.

    Args:
        mask: Binary mask (255 for foreground, 0 for background).
        params: Conversion parameters. Uses defaults if None.

    Returns:
        List of Polygon objects with normalized coordinates.
    """
    if params is None:
        params = ConversionParams()

    h, w = mask.shape[:2]
    polygons: list[Polygon] = []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params.min_area:
            continue

        # Simplify polygon using Douglas-Peucker
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = params.epsilon_factor * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Need at least 3 points for a valid polygon
        if len(simplified) < 3:
            continue

        # Normalize coordinates to [0, 1]
        points: list[tuple[float, float]] = []
        for point in simplified:
            x, y = point[0]
            points.append((x / w, y / h))

        polygons.append(Polygon(class_id=params.class_id, points=points))

    return polygons


def convert_mask_to_yolo_labels(
    mask: np.ndarray,
    class_ids: list[int],
    epsilon_factor: float = 0.005,
    min_area: float = 100.0,
) -> list[str]:
    """Convert a multi-class mask to YOLO label lines.

    Args:
        mask: Grayscale mask where pixel values represent class IDs.
        class_ids: List of class IDs to extract (excluding ignore index).
        epsilon_factor: Douglas-Peucker simplification factor.
        min_area: Minimum contour area in pixels.

    Returns:
        List of YOLO format label strings.
    """
    lines: list[str] = []

    for class_id in class_ids:
        # Create binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8) * 255

        params = ConversionParams(
            class_id=class_id,
            epsilon_factor=epsilon_factor,
            min_area=min_area,
        )
        polygons = mask_to_polygons(binary_mask, params)
        lines.extend(poly.to_yolo() for poly in polygons)

    return lines


def convert_mask_to_yolo_seg(
    dataset: MaskDataset,
    output_path: Path,
    epsilon_factor: float = 0.005,
    min_area: float = 100.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Convert a MaskDataset to YOLO segmentation format.

    Args:
        dataset: Source MaskDataset to convert.
        output_path: Output directory for YOLO dataset.
        epsilon_factor: Douglas-Peucker simplification factor.
        min_area: Minimum contour area in pixels.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        Dictionary with conversion statistics:
        - "images": Total images processed
        - "labels": Total label files created
        - "polygons": Total polygons extracted
        - "skipped": Images skipped (no mask or empty)
        - "warnings": Number of warnings (dimension mismatch, etc.)
    """
    stats = {
        "images": 0,
        "labels": 0,
        "polygons": 0,
        "skipped": 0,
        "warnings": 0,
    }

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)

    # Get class mapping and build id-to-name for data.yaml
    class_mapping = dataset.get_class_mapping()
    class_ids = sorted(class_mapping.keys())

    # Build data.yaml content
    data_yaml: dict = {
        "path": ".",
        "names": {i: class_mapping[i] for i in class_ids},
    }

    # Determine splits to process
    splits = dataset.splits if dataset.splits else [None]

    # Count total images for progress
    total_images = 0
    for split in splits:
        total_images += len(dataset.get_image_paths(split))

    current_image = 0

    for split in splits:
        split_name = split if split else "train"  # Default to train if unsplit

        # Create directories
        images_dir = output_path / "images" / split_name
        labels_dir = output_path / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Add split to data.yaml
        data_yaml[split_name] = f"images/{split_name}"

        # Process images in this split
        image_paths = dataset.get_image_paths(split)

        for image_path in image_paths:
            current_image += 1
            if progress_callback:
                progress_callback(current_image, total_images)

            stats["images"] += 1

            # Load mask
            mask = dataset.load_mask(image_path)
            if mask is None:
                logger.warning(f"No mask found for {image_path.name}, skipping")
                stats["skipped"] += 1
                continue

            # Load image to check dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Could not load image {image_path.name}, skipping")
                stats["skipped"] += 1
                continue

            # Check dimension match
            if img.shape[:2] != mask.shape[:2]:
                logger.warning(
                    f"Dimension mismatch for {image_path.name}: "
                    f"image={img.shape[:2]}, mask={mask.shape[:2]}"
                )
                stats["warnings"] += 1
                # Continue anyway - mask might still be usable

            # Get unique class IDs present in this mask (excluding ignore index)
            unique_ids = [
                int(v)
                for v in np.unique(mask)
                if v != dataset.ignore_index and v in class_ids
            ]

            if not unique_ids:
                # Empty mask (only background/ignored)
                logger.debug(f"Empty mask for {image_path.name}")
                stats["skipped"] += 1
                continue

            # Convert mask to YOLO labels
            label_lines = convert_mask_to_yolo_labels(
                mask, unique_ids, epsilon_factor, min_area
            )

            if not label_lines:
                # No polygons extracted (all contours too small)
                logger.debug(f"No valid polygons for {image_path.name}")
                stats["skipped"] += 1
                continue

            # Copy image to output
            dest_image = images_dir / image_path.name
            shutil.copy2(image_path, dest_image)

            # Write label file
            label_file = labels_dir / f"{image_path.stem}.txt"
            label_file.write_text("\n".join(label_lines) + "\n")

            stats["labels"] += 1
            stats["polygons"] += len(label_lines)

    # Write data.yaml
    data_yaml_path = output_path / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    return stats


def _get_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    """Read image and return (width, height).

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (width, height), or None if the image cannot be read.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def _parse_yolo_label_file(
    label_path: Path,
) -> list[tuple[int, list[tuple[float, float]]]]:
    """Parse a YOLO segmentation label file.

    Args:
        label_path: Path to the .txt label file.

    Returns:
        List of (class_id, points) tuples where points is a list of (x, y)
        normalized coordinate pairs.
    """
    annotations: list[tuple[int, list[tuple[float, float]]]] = []

    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError:
        return annotations

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:  # class_id + at least 3 coordinate pairs
            continue

        try:
            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            annotations.append((class_id, points))
        except (ValueError, IndexError):
            continue

    return annotations


def _yolo_polygon_to_coco_segmentation(
    points: list[tuple[float, float]],
    img_width: int,
    img_height: int,
) -> tuple[list[list[float]], list[float], float]:
    """Convert a YOLO polygon (possibly with donut bridge) to COCO segmentation.

    Rasterizes the polygon to a binary mask, then extracts contours with hierarchy
    to recover separate rings (outer boundary + holes).

    Args:
        points: Normalized (x, y) coordinate pairs from YOLO annotation.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple of (segmentation, bbox, area) where:
        - segmentation: list of polygon rings as flat coordinate lists
        - bbox: [x, y, w, h] bounding box
        - area: pixel area from mask
    """
    # Denormalize coordinates
    abs_points = np.array(
        [(x * img_width, y * img_height) for x, y in points], dtype=np.float32
    )

    # Rasterize polygon to binary mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    int_points = np.round(abs_points).astype(np.int32)
    cv2.fillPoly(mask, [int_points], 255)

    # Extract contours with hierarchy to recover outer/hole structure
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )

    segmentation: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Flatten contour to [x1, y1, x2, y2, ...] format
        ring = contour.reshape(-1).tolist()
        if len(ring) >= 6:  # At least 3 points (6 coordinates)
            segmentation.append(ring)

    # Compute bbox and area from the mask
    area = float(np.count_nonzero(mask))

    # Get bounding box from mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return segmentation, [0.0, 0.0, 0.0, 0.0], 0.0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox = [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]

    return segmentation, bbox, area


def _convert_yolo_seg_to_coco_layout(
    dataset: "YOLODataset",  # noqa: F821
    output_path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    roboflow_layout: bool = False,
) -> dict[str, int]:
    """Convert a YOLO segmentation dataset to COCO-compatible JSON outputs.

    Args:
        dataset: Source YOLODataset (must be segmentation task).
        output_path: Output directory for converted dataset.
        progress_callback: Optional callback(current, total) for progress updates.
        roboflow_layout: If True, write Roboflow COCO layout.

    Returns:
        Dictionary with conversion statistics:
        - "images": Total images processed
        - "annotations": Total annotations created
        - "skipped": Images skipped (could not read, etc.)
        - "warnings": Number of warnings
    """
    stats = {
        "images": 0,
        "annotations": 0,
        "skipped": 0,
        "warnings": 0,
    }

    output_path.mkdir(parents=True, exist_ok=True)

    # Build COCO categories from class names (YOLO 0-indexed → COCO 1-indexed)
    categories = [
        {"id": i + 1, "name": name, "supercategory": ""}
        for i, name in enumerate(dataset.class_names)
    ]

    # Determine splits to process
    splits = dataset.splits if dataset.splits else [None]

    # Count total images for progress
    total_images = 0
    for split in splits:
        total_images += len(dataset.get_image_paths(split))

    current_image = 0

    # Standard COCO keeps annotations in a dedicated folder.
    annotations_dir = output_path / "annotations"
    if not roboflow_layout:
        annotations_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_name = split if split else "train"

        if roboflow_layout:
            # Roboflow uses "valid" folder name for validation split.
            output_split_name = "valid" if split_name == "val" else split_name
            split_out_dir = output_path / output_split_name
            split_out_dir.mkdir(parents=True, exist_ok=True)
            images_out_dir = split_out_dir
            annotation_file = split_out_dir / "_annotations.coco.json"
        else:
            images_out_dir = output_path / "images" / split_name
            images_out_dir.mkdir(parents=True, exist_ok=True)
            annotation_file = annotations_dir / f"instances_{split_name}.json"

        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        image_id = 0
        annotation_id = 0

        image_paths = dataset.get_image_paths(split)

        for image_path in image_paths:
            current_image += 1
            if progress_callback:
                progress_callback(current_image, total_images)

            # Get image dimensions
            dims = _get_image_dimensions(image_path)
            if dims is None:
                logger.warning(f"Could not read image {image_path.name}, skipping")
                stats["skipped"] += 1
                continue

            img_width, img_height = dims
            image_id += 1
            stats["images"] += 1

            # Add COCO image entry
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": img_width,
                    "height": img_height,
                }
            )

            # Find corresponding label file
            if split:
                _, labels_dir = dataset.get_split_dirs(split)
            else:
                # Unsplit dataset: labels are directly in labels/
                labels_dir = dataset.path / "labels"
            label_path = labels_dir / f"{image_path.stem}.txt"

            if label_path.exists():
                annotations_data = _parse_yolo_label_file(label_path)

                for class_id, points in annotations_data:
                    segmentation, bbox, area = _yolo_polygon_to_coco_segmentation(
                        points, img_width, img_height
                    )

                    if not segmentation:
                        stats["warnings"] += 1
                        continue

                    annotation_id += 1
                    # YOLO class 0 → COCO category 1
                    coco_category_id = class_id + 1

                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": coco_category_id,
                            "segmentation": segmentation,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                    stats["annotations"] += 1

            # Copy image to output
            dest_image = images_out_dir / image_path.name
            shutil.copy2(image_path, dest_image)

        # Write annotation JSON for this split
        coco_data = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        with open(annotation_file, "w") as f:
            json.dump(coco_data, f)

    return stats


def convert_yolo_seg_to_coco(
    dataset: "YOLODataset",  # noqa: F821
    output_path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Convert a YOLO segmentation dataset to standard COCO layout.

    Output structure:
        output/
        ├── annotations/instances_{split}.json
        └── images/{split}/*.jpg
    """
    return _convert_yolo_seg_to_coco_layout(
        dataset=dataset,
        output_path=output_path,
        progress_callback=progress_callback,
        roboflow_layout=False,
    )


def convert_yolo_seg_to_roboflow_coco(
    dataset: "YOLODataset",  # noqa: F821
    output_path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Convert a YOLO segmentation dataset to Roboflow COCO layout.

    Output structure:
        output/
        ├── train/_annotations.coco.json
        ├── valid/_annotations.coco.json
        └── test/_annotations.coco.json
    """
    return _convert_yolo_seg_to_coco_layout(
        dataset=dataset,
        output_path=output_path,
        progress_callback=progress_callback,
        roboflow_layout=True,
    )
