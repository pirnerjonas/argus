"""Utilities for adding background-only images to datasets."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2

from argus.core.coco import COCODataset
from argus.core.split import _compute_split_sizes
from argus.core.yolo import YOLODataset

_SPLITS = ("train", "val", "test")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_source_images(source: Path) -> list[Path]:
    """Collect all image files from a source path.

    Args:
        source: Path to a single image file or directory containing images.

    Returns:
        List of image file paths.

    Raises:
        ValueError: If source doesn't exist or contains no valid images.
    """
    if not source.exists():
        raise ValueError(f"Source path does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Source file is not a supported image format: {source}")
        return [source]

    # Source is a directory
    images = [
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    ]

    if not images:
        raise ValueError(f"No valid image files found in: {source}")

    return sorted(images, key=lambda p: p.name)


def _assign_images_to_splits(
    images: list[Path],
    available_splits: list[str],
    ratios: tuple[float, float, float],
    seed: int | None,
) -> dict[str, list[Path]]:
    """Assign images to splits based on ratios.

    Args:
        images: List of image paths to distribute.
        available_splits: Splits available in the dataset (e.g., ["train", "val"]).
        ratios: Target ratios for train/val/test.
        seed: Random seed for shuffling. None for random.

    Returns:
        Dictionary mapping split name to list of image paths.
    """
    if not available_splits:
        # Unsplit dataset - all images go to a single bucket
        return {"unsplit": list(images)}

    # Re-normalize ratios for available splits only
    split_indices = {s: i for i, s in enumerate(_SPLITS)}
    available_ratios = [ratios[split_indices[s]] for s in available_splits]
    total_ratio = sum(available_ratios)

    if total_ratio == 0:
        # All available splits have 0 ratio - distribute evenly
        normalized = tuple(1.0 / len(available_splits) for _ in available_splits)
    else:
        normalized = tuple(r / total_ratio for r in available_ratios)

    # Build a full 3-tuple for _compute_split_sizes, zeroing out unavailable splits
    full_ratios = [0.0, 0.0, 0.0]
    for i, split in enumerate(available_splits):
        full_ratios[split_indices[split]] = normalized[i]

    sizes = _compute_split_sizes(len(images), tuple(full_ratios))

    # Shuffle images
    shuffled = list(images)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Assign to splits
    assignments: dict[str, list[Path]] = {}
    start = 0
    for split in _SPLITS:
        count = sizes[split]
        if count > 0:
            assignments[split] = shuffled[start : start + count]
            start += count

    return assignments


def add_background_to_yolo(
    dataset: YOLODataset,
    source: Path,
    ratios: tuple[float, float, float],
    seed: int | None = None,
) -> dict[str, int]:
    """Add background-only images to a YOLO dataset.

    Copies images to the appropriate images/{split}/ directory and creates
    empty .txt files in labels/{split}/.

    Args:
        dataset: Target YOLO dataset.
        source: Path to image file or directory of images.
        ratios: Train/val/test distribution ratios.
        seed: Random seed for distribution.

    Returns:
        Dictionary mapping split name to count of images added.

    Raises:
        ValueError: If source is invalid or images already exist.
    """
    images = _collect_source_images(source)
    available_splits = dataset.splits if dataset.splits else []
    assignments = _assign_images_to_splits(images, available_splits, ratios, seed)

    counts: dict[str, int] = {}
    duplicates: list[str] = []

    for split, split_images in assignments.items():
        if split == "unsplit":
            image_dir = dataset.path / "images"
            label_dir = dataset.path / "labels"
        else:
            image_dir = dataset.path / "images" / split
            label_dir = dataset.path / "labels" / split

        # Ensure directories exist
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        for img_path in split_images:
            dest_image = image_dir / img_path.name
            dest_label = label_dir / f"{img_path.stem}.txt"

            # Check for duplicates
            if dest_image.exists():
                duplicates.append(str(img_path.name))
                continue

            # Copy image
            shutil.copy2(img_path, dest_image)

            # Create empty label file (background = no annotations)
            dest_label.write_text("")

            added += 1

        counts[split] = added

    if duplicates:
        raise ValueError(
            f"Skipped {len(duplicates)} duplicate(s): {', '.join(duplicates[:5])}"
            + ("..." if len(duplicates) > 5 else "")
        )

    return counts


def add_background_to_coco(
    dataset: COCODataset,
    source: Path,
    ratios: tuple[float, float, float],
    seed: int | None = None,
) -> dict[str, int]:
    """Add background-only images to a COCO dataset.

    Copies images to the appropriate images/{split}/ directory and updates
    the corresponding annotation JSON file with new image entries (no annotations).

    Args:
        dataset: Target COCO dataset.
        source: Path to image file or directory of images.
        ratios: Train/val/test distribution ratios.
        seed: Random seed for distribution.

    Returns:
        Dictionary mapping split name to count of images added.

    Raises:
        ValueError: If source is invalid or images already exist.
    """
    images = _collect_source_images(source)
    available_splits = dataset.splits if dataset.splits else []
    assignments = _assign_images_to_splits(images, available_splits, ratios, seed)

    # Build mapping of split -> annotation file
    split_to_ann_file: dict[str, Path] = {}
    for ann_file in dataset.annotation_files:
        split = _get_coco_split_from_filename(ann_file.stem)
        split_to_ann_file[split] = ann_file

    counts: dict[str, int] = {}
    duplicates: list[str] = []

    for split, split_images in assignments.items():
        if not split_images:
            continue

        # Find annotation file for this split
        ann_file = split_to_ann_file.get(split)
        if not ann_file:
            # For unsplit, use first annotation file
            ann_file = dataset.annotation_files[0] if dataset.annotation_files else None

        if not ann_file or not ann_file.exists():
            raise ValueError(f"No annotation file found for split '{split}'")

        # Determine image directory
        if split == "unsplit" or split not in dataset.splits:
            image_dir = dataset.path / "images"
        else:
            image_dir = dataset.path / "images" / split
            if not image_dir.exists():
                # Fallback to flat images/
                image_dir = dataset.path / "images"

        image_dir.mkdir(parents=True, exist_ok=True)

        # Load annotation data
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        existing_filenames = {img.get("file_name") for img in data.get("images", [])}
        max_image_id = max(
            (img.get("id", 0) for img in data.get("images", [])), default=0
        )

        added = 0
        new_images = []

        for img_path in split_images:
            # Check for duplicates
            if img_path.name in existing_filenames:
                duplicates.append(str(img_path.name))
                continue

            dest_image = image_dir / img_path.name
            if dest_image.exists():
                duplicates.append(str(img_path.name))
                continue

            # Copy image
            shutil.copy2(img_path, dest_image)

            # Get image dimensions
            img = cv2.imread(str(dest_image))
            if img is None:
                # Fallback dimensions if image can't be read
                height, width = 0, 0
            else:
                height, width = img.shape[:2]

            # Create new image entry
            max_image_id += 1
            new_images.append({
                "id": max_image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            })

            added += 1

        # Update annotation file
        if new_images:
            data["images"] = data.get("images", []) + new_images
            with open(ann_file, "w", encoding="utf-8") as f:
                json.dump(data, f)

        counts[split] = added

    if duplicates:
        raise ValueError(
            f"Skipped {len(duplicates)} duplicate(s): {', '.join(duplicates[:5])}"
            + ("..." if len(duplicates) > 5 else "")
        )

    return counts


def _get_coco_split_from_filename(filename: str) -> str:
    """Extract split name from annotation filename."""
    name_lower = filename.lower()
    if "train" in name_lower:
        return "train"
    elif "val" in name_lower:
        return "val"
    elif "test" in name_lower:
        return "test"
    return "unsplit"
