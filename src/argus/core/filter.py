"""Dataset filtering utilities."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml

from argus.core.base import TaskType
from argus.core.coco import COCODataset
from argus.core.mask import MaskDataset
from argus.core.yolo import YOLODataset


def filter_yolo_dataset(
    dataset: YOLODataset,
    output_path: Path,
    classes: list[str],
    no_background: bool = False,
    use_symlinks: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Filter a YOLO dataset by class names.

    Args:
        dataset: Source YOLODataset to filter.
        output_path: Directory to write filtered dataset.
        classes: List of class names to keep.
        no_background: If True, exclude images with no annotations after filtering.
        use_symlinks: If True, create symlinks instead of copying images.
        progress_callback: Optional callback for progress updates (current, total).

    Returns:
        Dictionary with statistics: images, labels, skipped.
    """
    if dataset.task == TaskType.CLASSIFICATION:
        return _filter_yolo_classification(
            dataset, output_path, classes, use_symlinks, progress_callback
        )
    else:
        return _filter_yolo_detection_segmentation(
            dataset,
            output_path,
            classes,
            no_background,
            use_symlinks,
            progress_callback,
        )


def _filter_yolo_detection_segmentation(
    dataset: YOLODataset,
    output_path: Path,
    classes: list[str],
    no_background: bool,
    use_symlinks: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> dict[str, int]:
    """Filter YOLO detection/segmentation dataset."""
    # Build class ID mapping: old_id -> new_id
    # New IDs are sequential starting from 0
    old_to_new: dict[int, int] = {}
    new_class_names: list[str] = []

    for i, name in enumerate(dataset.class_names):
        if name in classes:
            new_id = len(new_class_names)
            old_to_new[i] = new_id
            new_class_names.append(name)

    if not new_class_names:
        raise ValueError(f"No matching classes found. Available: {dataset.class_names}")

    # Create output structure
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine splits
    splits = dataset.splits if dataset.splits else [""]
    has_splits = bool(dataset.splits)

    stats = {"images": 0, "labels": 0, "skipped": 0}

    # Collect all image/label pairs
    all_pairs: list[tuple[Path, Path, str]] = []
    labels_root = dataset.path / "labels"

    for split in splits:
        if has_splits:
            images_dir = dataset.path / "images" / split
            labels_dir = labels_root / split
        else:
            images_dir = dataset.path / "images"
            labels_dir = labels_root

        if not images_dir.is_dir():
            continue

        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".webp",
            }:
                continue

            label_file = labels_dir / f"{img_file.stem}.txt"
            all_pairs.append((img_file, label_file, split))

    total = len(all_pairs)

    for idx, (img_file, label_file, split) in enumerate(all_pairs):
        if progress_callback:
            progress_callback(idx, total)

        # Read and filter label file
        filtered_lines: list[str] = []

        if label_file.exists():
            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    try:
                        old_class_id = int(parts[0])
                    except ValueError:
                        continue

                    if old_class_id in old_to_new:
                        new_class_id = old_to_new[old_class_id]
                        parts[0] = str(new_class_id)
                        filtered_lines.append(" ".join(parts))

        # Skip if no annotations and no_background is True
        if no_background and not filtered_lines:
            stats["skipped"] += 1
            continue

        # Create output directories
        if has_splits:
            out_images_dir = output_path / "images" / split
            out_labels_dir = output_path / "labels" / split
        else:
            out_images_dir = output_path / "images"
            out_labels_dir = output_path / "labels"

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        # Copy/symlink image
        out_img = out_images_dir / img_file.name
        if use_symlinks:
            if not out_img.exists():
                out_img.symlink_to(img_file.resolve())
        else:
            if not out_img.exists():
                shutil.copy2(img_file, out_img)

        # Write filtered label
        out_label = out_labels_dir / f"{img_file.stem}.txt"
        with open(out_label, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_lines))
            if filtered_lines:
                f.write("\n")

        stats["images"] += 1
        stats["labels"] += 1

    if progress_callback:
        progress_callback(total, total)

    # Create data.yaml
    _create_yolo_yaml(output_path, new_class_names, splits if has_splits else [])

    return stats


def _filter_yolo_classification(
    dataset: YOLODataset,
    output_path: Path,
    classes: list[str],
    use_symlinks: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> dict[str, int]:
    """Filter YOLO classification dataset."""
    # Filter to only requested classes that exist
    new_class_names = [name for name in dataset.class_names if name in classes]

    if not new_class_names:
        raise ValueError(f"No matching classes found. Available: {dataset.class_names}")

    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "labels": 0, "skipped": 0}

    # Count total images for progress
    total = 0
    if dataset.splits:
        for split in dataset.splits:
            for class_name in new_class_names:
                class_dir = dataset.path / "images" / split / class_name
                if class_dir.is_dir():
                    total += sum(
                        1
                        for f in class_dir.iterdir()
                        if f.suffix.lower()
                        in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                    )
    else:
        # Flat structure
        for class_name in new_class_names:
            class_dir = dataset.path / class_name
            if class_dir.is_dir():
                total += sum(
                    1
                    for f in class_dir.iterdir()
                    if f.suffix.lower()
                    in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                )

    current = 0

    if dataset.splits:
        for split in dataset.splits:
            for class_name in new_class_names:
                src_dir = dataset.path / "images" / split / class_name
                dst_dir = output_path / "images" / split / class_name

                if not src_dir.is_dir():
                    continue

                dst_dir.mkdir(parents=True, exist_ok=True)

                for img_file in src_dir.iterdir():
                    if img_file.suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff",
                        ".webp",
                    }:
                        continue

                    if progress_callback:
                        progress_callback(current, total)
                    current += 1

                    dst_file = dst_dir / img_file.name
                    if use_symlinks:
                        if not dst_file.exists():
                            dst_file.symlink_to(img_file.resolve())
                    else:
                        if not dst_file.exists():
                            shutil.copy2(img_file, dst_file)

                    stats["images"] += 1
    else:
        # Flat structure
        for class_name in new_class_names:
            src_dir = dataset.path / class_name
            dst_dir = output_path / class_name

            if not src_dir.is_dir():
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)

            for img_file in src_dir.iterdir():
                if img_file.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tiff",
                    ".webp",
                }:
                    continue

                if progress_callback:
                    progress_callback(current, total)
                current += 1

                dst_file = dst_dir / img_file.name
                if use_symlinks:
                    if not dst_file.exists():
                        dst_file.symlink_to(img_file.resolve())
                else:
                    if not dst_file.exists():
                        shutil.copy2(img_file, dst_file)

                stats["images"] += 1

    if progress_callback:
        progress_callback(total, total)

    return stats


def _create_yolo_yaml(
    output_path: Path, class_names: list[str], splits: list[str]
) -> None:
    """Create data.yaml for YOLO dataset."""
    config: dict = {
        "path": ".",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if splits:
        if "train" in splits:
            config["train"] = "images/train"
        if "val" in splits:
            config["val"] = "images/val"
        if "test" in splits:
            config["test"] = "images/test"

    with open(output_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def filter_coco_dataset(
    dataset: COCODataset,
    output_path: Path,
    classes: list[str],
    no_background: bool = False,
    use_symlinks: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Filter a COCO dataset by class names.

    Args:
        dataset: Source COCODataset to filter.
        output_path: Directory to write filtered dataset.
        classes: List of class names to keep.
        no_background: If True, exclude images with no annotations after filtering.
        use_symlinks: If True, create symlinks instead of copying images.
        progress_callback: Optional callback for progress updates (current, total).

    Returns:
        Dictionary with statistics: images, annotations, skipped.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "annotations": 0, "skipped": 0}

    # Process each annotation file
    for ann_file in dataset.annotation_files:
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        # Build category mappings
        old_categories = data.get("categories", [])
        old_id_to_name: dict[int, str] = {}
        for cat in old_categories:
            if isinstance(cat, dict) and "id" in cat and "name" in cat:
                old_id_to_name[cat["id"]] = cat["name"]

        # Create new category list with remapped IDs
        old_to_new: dict[int, int] = {}
        new_categories: list[dict] = []
        new_id = 1  # COCO IDs typically start at 1

        for cat in old_categories:
            if isinstance(cat, dict) and "name" in cat and cat["name"] in classes:
                old_id = cat["id"]
                old_to_new[old_id] = new_id
                new_cat = cat.copy()
                new_cat["id"] = new_id
                new_categories.append(new_cat)
                new_id += 1

        if not new_categories:
            raise ValueError(
                f"No matching classes found. Available: {list(old_id_to_name.values())}"
            )

        # Filter annotations
        old_annotations = data.get("annotations", [])
        new_annotations: list[dict] = []
        images_with_annotations: set[int] = set()
        new_ann_id = 1

        for ann in old_annotations:
            if not isinstance(ann, dict):
                continue

            old_cat_id = ann.get("category_id")
            if old_cat_id not in old_to_new:
                continue

            new_ann = ann.copy()
            new_ann["id"] = new_ann_id
            new_ann["category_id"] = old_to_new[old_cat_id]
            new_annotations.append(new_ann)
            new_ann_id += 1
            stats["annotations"] += 1

            image_id = ann.get("image_id")
            if image_id is not None:
                images_with_annotations.add(image_id)

        # Filter images
        old_images = data.get("images", [])
        new_images: list[dict] = []
        included_image_ids: set[int] = set()
        new_img_id = 1

        # Build image ID mapping for annotation update
        old_to_new_img_id: dict[int, int] = {}

        for img in old_images:
            if not isinstance(img, dict) or "id" not in img:
                continue

            old_img_id = img["id"]

            # Skip if no_background and no annotations
            if no_background and old_img_id not in images_with_annotations:
                stats["skipped"] += 1
                continue

            old_to_new_img_id[old_img_id] = new_img_id
            new_img = img.copy()
            new_img["id"] = new_img_id
            new_images.append(new_img)
            included_image_ids.add(old_img_id)
            new_img_id += 1
            stats["images"] += 1

        # Update annotation image IDs and filter out annotations for excluded images
        final_annotations: list[dict] = []
        for ann in new_annotations:
            old_img_id = ann.get("image_id")
            if old_img_id in old_to_new_img_id:
                ann["image_id"] = old_to_new_img_id[old_img_id]
                final_annotations.append(ann)

        # Determine split from annotation file
        split = COCODataset._get_split_from_filename(
            ann_file.stem, ann_file.parent.name
        )

        # Check if this is Roboflow format (annotation in split directory)
        is_roboflow = ann_file.parent.name.lower() in ("train", "valid", "val", "test")

        # Create output annotation
        new_data = data.copy()
        new_data["categories"] = new_categories
        new_data["annotations"] = final_annotations
        new_data["images"] = new_images

        # Write annotation file
        if is_roboflow:
            # Roboflow format: annotations in split directories
            out_ann_dir = output_path / split
            out_ann_dir.mkdir(parents=True, exist_ok=True)
            out_ann_file = out_ann_dir / ann_file.name
        else:
            # Standard format: annotations in annotations/ directory
            out_ann_dir = output_path / "annotations"
            out_ann_dir.mkdir(parents=True, exist_ok=True)
            out_ann_file = out_ann_dir / ann_file.name

        with open(out_ann_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2)

        # Copy/symlink images
        for img in old_images:
            if not isinstance(img, dict) or "id" not in img:
                continue

            if img["id"] not in included_image_ids:
                continue

            file_name = img.get("file_name")
            if not file_name:
                continue

            # Find source image
            possible_paths = [
                dataset.path / "images" / split / file_name,
                dataset.path / "images" / file_name,
                dataset.path / split / file_name,
                dataset.path / file_name,
                ann_file.parent / file_name,  # Roboflow format
            ]

            src_path = None
            for p in possible_paths:
                if p.exists():
                    src_path = p
                    break

            if src_path is None:
                continue

            # Determine output directory
            if is_roboflow:
                out_img_dir = output_path / split
            else:
                out_img_dir = output_path / "images" / split
            out_img_dir.mkdir(parents=True, exist_ok=True)

            out_img = out_img_dir / file_name
            if use_symlinks:
                if not out_img.exists():
                    out_img.symlink_to(src_path.resolve())
            else:
                if not out_img.exists():
                    shutil.copy2(src_path, out_img)

    return stats


def filter_mask_dataset(
    dataset: MaskDataset,
    output_path: Path,
    classes: list[str],
    no_background: bool = False,
    use_symlinks: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Filter a mask dataset by class names.

    Args:
        dataset: Source MaskDataset to filter.
        output_path: Directory to write filtered dataset.
        classes: List of class names to keep.
        no_background: If True, exclude images with no annotations after filtering.
        use_symlinks: If True, create symlinks instead of copying images.
        progress_callback: Optional callback for progress updates (current, total).

    Returns:
        Dictionary with statistics: images, masks, skipped.
    """
    # Build class ID mapping
    old_mapping = dataset.get_class_mapping()
    old_name_to_id: dict[str, int] = {name: id for id, name in old_mapping.items()}

    # Create new mapping: old_id -> new_id
    old_to_new: dict[int, int] = {}
    new_class_names: list[str] = []

    # Start from 0 for background, then 1, 2, ... for other classes
    # If "background" is in classes, include it
    new_id = 0
    for name in classes:
        if name in old_name_to_id:
            old_id = old_name_to_id[name]
            old_to_new[old_id] = new_id
            new_class_names.append(name)
            new_id += 1

    if not new_class_names:
        raise ValueError(
            f"No matching classes found. Available: {list(old_mapping.values())}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "masks": 0, "skipped": 0}

    # Get all image paths
    image_paths = dataset.get_image_paths()
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        if progress_callback:
            progress_callback(idx, total)

        # Load mask
        mask = dataset.load_mask(img_path)
        if mask is None:
            stats["skipped"] += 1
            continue

        # Create filtered mask
        # Set all pixels to ignore_index first, then fill in kept classes
        new_ignore_index = 255
        new_mask = np.full(mask.shape, new_ignore_index, dtype=np.uint8)

        has_annotations = False
        for old_id, new_id in old_to_new.items():
            mask_pixels = mask == old_id
            if np.any(mask_pixels):
                has_annotations = True
                new_mask[mask_pixels] = new_id

        # Skip if no_background and no kept annotations
        if no_background and not has_annotations:
            stats["skipped"] += 1
            continue

        # Determine split from image path
        img_parts = img_path.parts
        images_dir_idx = None
        for i, part in enumerate(img_parts):
            if part == dataset.images_dir:
                images_dir_idx = i
                break

        if images_dir_idx is not None and images_dir_idx + 1 < len(img_parts) - 1:
            split = img_parts[images_dir_idx + 1]
            if split not in dataset.splits:
                split = None
        else:
            split = None

        # Create output directories
        if split:
            out_images_dir = output_path / dataset.images_dir / split
            out_masks_dir = output_path / dataset.masks_dir / split
        else:
            out_images_dir = output_path / dataset.images_dir
            out_masks_dir = output_path / dataset.masks_dir

        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_masks_dir.mkdir(parents=True, exist_ok=True)

        # Copy/symlink image
        out_img = out_images_dir / img_path.name
        if use_symlinks:
            if not out_img.exists():
                out_img.symlink_to(img_path.resolve())
        else:
            if not out_img.exists():
                shutil.copy2(img_path, out_img)

        # Write filtered mask
        mask_path = dataset.get_mask_path(img_path)
        if mask_path:
            out_mask = out_masks_dir / mask_path.name
            cv2.imwrite(str(out_mask), new_mask)

        stats["images"] += 1
        stats["masks"] += 1

    if progress_callback:
        progress_callback(total, total)

    # Create classes.yaml
    _create_mask_classes_yaml(output_path, new_class_names, dataset.ignore_index)

    return stats


def _create_mask_classes_yaml(
    output_path: Path, class_names: list[str], ignore_index: int | None
) -> None:
    """Create classes.yaml for mask dataset."""
    config: dict = {
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if ignore_index is not None:
        config["ignore_index"] = 255  # Use standard ignore index

    with open(output_path / "classes.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
