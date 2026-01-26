"""Conversion functions for dataset format transformation."""

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
