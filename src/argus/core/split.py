"""Dataset split and unsplit utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from argus.core.base import IMAGE_EXTENSIONS, TaskType
from argus.core.coco import COCODataset
from argus.core.mask import MaskDataset
from argus.core.yolo import YOLODataset

_SPLITS = ("train", "val", "test")
_COCO_SPLIT_DIRS = {"train", "val", "valid", "test"}
CollisionPolicy = Literal["error", "prefix-split", "hash"]


def parse_ratio(ratio: str) -> tuple[float, float, float]:
    """Parse a ratio string into train/val/test fractions."""
    parts = [p.strip() for p in ratio.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Ratio must have three comma-separated values.")

    values = [float(part) for part in parts]
    total = sum(values)

    if math.isclose(total, 0.0):
        raise ValueError("Ratio values must sum to a positive number.")

    if total > 1.0 + 1e-6:
        if math.isclose(total, 100.0, rel_tol=1e-3, abs_tol=1e-3):
            values = [val / 100.0 for val in values]
        else:
            raise ValueError("Ratio values must sum to 1.0 (or 100).")

    normalized = [val / sum(values) for val in values]
    return normalized[0], normalized[1], normalized[2]


def _compute_split_sizes(
    total: int, ratios: tuple[float, float, float]
) -> dict[str, int]:
    if total < 0:
        raise ValueError("Total must be non-negative.")

    raw = [total * ratio for ratio in ratios]
    base = [int(math.floor(val)) for val in raw]
    remainder = total - sum(base)

    fractional = [val - math.floor(val) for val in raw]
    order = sorted(range(len(fractional)), key=lambda i: fractional[i], reverse=True)
    for idx in order[:remainder]:
        base[idx] += 1

    # For small datasets, largest-remainder can still allocate 0 samples to a
    # non-zero split (e.g., 6 items with 0.8/0.1/0.1 -> 5/1/0). If there are
    # enough samples to cover each requested split, enforce a minimum of 1.
    nonzero_indices = [i for i, ratio in enumerate(ratios) if ratio > 0.0]
    if total >= len(nonzero_indices):
        for idx in nonzero_indices:
            if base[idx] > 0:
                continue

            # Take one sample from the split with the most samples.
            donor_candidates = [
                j
                for j in range(len(base))
                if j != idx and base[j] > (1 if ratios[j] > 0.0 else 0)
            ]
            if not donor_candidates:
                donor_candidates = [
                    j for j in range(len(base)) if j != idx and base[j] > 0
                ]
            if not donor_candidates:
                continue

            donor = max(donor_candidates, key=lambda j: base[j])
            base[donor] -= 1
            base[idx] += 1

    return dict(zip(_SPLITS, base, strict=True))


def _build_stratified_split(
    items: list[str],
    labels: dict[str, set[int]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    split_sizes = _compute_split_sizes(len(items), ratios)
    rng = random.Random(seed)

    class_counts: dict[int, int] = {}
    for item in items:
        for label in labels.get(item, set()):
            class_counts[label] = class_counts.get(label, 0) + 1

    remaining_class = {
        split: {cls: count * ratio for cls, count in class_counts.items()}
        for split, ratio in zip(_SPLITS, ratios, strict=True)
    }
    remaining_items = split_sizes.copy()
    assignments = {split: [] for split in _SPLITS}

    def sort_key(item: str) -> tuple[int, float]:
        return (-len(labels.get(item, set())), rng.random())

    for item in sorted(items, key=sort_key):
        candidates = [split for split in _SPLITS if remaining_items[split] > 0]
        if not candidates:
            break

        item_labels = labels.get(item, set())
        if item_labels:
            scores = {
                split: sum(
                    remaining_class[split].get(label, 0.0) for label in item_labels
                )
                for split in candidates
            }
            best_score = max(scores.values())
            best_splits = [split for split in candidates if scores[split] == best_score]
        else:
            best_splits = candidates

        if len(best_splits) > 1:
            max_remaining = max(remaining_items[split] for split in best_splits)
            best_splits = [
                split
                for split in best_splits
                if remaining_items[split] == max_remaining
            ]

        chosen = rng.choice(best_splits)
        assignments[chosen].append(item)
        remaining_items[chosen] -= 1
        for label in item_labels:
            remaining_class[chosen][label] = remaining_class[chosen].get(label, 0.0) - 1

    return assignments


def _build_random_split(
    items: list[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[str]]:
    split_sizes = _compute_split_sizes(len(items), ratios)
    rng = random.Random(seed)
    rng.shuffle(items)

    assignments = {}
    start = 0
    for split in _SPLITS:
        size = split_sizes[split]
        assignments[split] = items[start : start + size]
        start += size

    return assignments


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_image_path(base_path: Path, file_name: str) -> Path | None:
    candidates = [
        base_path / "images" / file_name,
        base_path / file_name,
        base_path / "images" / "train" / file_name,
        base_path / "images" / "val" / file_name,
        base_path / "images" / "test" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _hash_for_path(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]


def _resolve_collision_name(
    file_name: str,
    source_path: Path,
    split: str | None,
    output_dir: Path,
    collision_policy: CollisionPolicy,
) -> str:
    if collision_policy not in ("error", "prefix-split", "hash"):
        raise ValueError(f"Unsupported collision policy: {collision_policy}")

    candidate_name = Path(file_name).name
    candidate_path = output_dir / candidate_name
    if not candidate_path.exists():
        return candidate_name

    if collision_policy == "error":
        raise ValueError(f"File collision detected for {candidate_name}")

    stem = Path(candidate_name).stem
    suffix = Path(candidate_name).suffix
    if collision_policy == "prefix-split":
        prefix = split if split else "unsplit"
        candidate_name = f"{prefix}_{candidate_name}"
    else:
        candidate_name = f"{stem}_{_hash_for_path(source_path)}{suffix}"

    candidate_path = output_dir / candidate_name
    if not candidate_path.exists():
        return candidate_name

    # Final deterministic fallback for repeated collisions.
    base_stem = Path(candidate_name).stem
    suffix = Path(candidate_name).suffix
    index = 2
    while True:
        retry_name = f"{base_stem}_{index}{suffix}"
        retry_path = output_dir / retry_name
        if not retry_path.exists():
            return retry_name
        index += 1


def split_yolo_dataset(
    dataset: YOLODataset,
    output_path: Path,
    ratios: tuple[float, float, float],
    stratify: bool,
    seed: int,
) -> dict[str, int]:
    """Split a YOLO dataset into train/val/test subsets."""
    if dataset.task == TaskType.CLASSIFICATION:
        return _split_yolo_classification_dataset(
            dataset, output_path, ratios, stratify, seed
        )

    image_paths = dataset.get_image_paths()
    if not image_paths:
        raise ValueError("No images found in the dataset.")

    label_dir = dataset.path / "labels"
    if not label_dir.is_dir():
        raise ValueError("Expected labels directory at dataset root for unsplit YOLO.")

    item_to_image = {str(i): path for i, path in enumerate(sorted(image_paths))}
    labels: dict[str, set[int]] = {}

    for item_id, image_path in item_to_image.items():
        label_path = label_dir / f"{image_path.stem}.txt"
        label_set: set[int] = set()
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    label_set.add(int(parts[0]))
                except ValueError:
                    continue
        labels[item_id] = label_set

    items = list(item_to_image.keys())
    assignments = (
        _build_stratified_split(items, labels, ratios, seed)
        if stratify
        else _build_random_split(items, ratios, seed)
    )

    for split, item_ids in assignments.items():
        image_out_dir = output_path / "images" / split
        label_out_dir = output_path / "labels" / split
        _ensure_dir(image_out_dir)
        _ensure_dir(label_out_dir)

        for item_id in item_ids:
            image_src = item_to_image[item_id]
            image_dst = image_out_dir / image_src.name
            shutil.copy2(image_src, image_dst)

            label_src = label_dir / f"{image_src.stem}.txt"
            label_dst = label_out_dir / f"{image_src.stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                label_dst.write_text("")

    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dataset.class_names,
        "nc": len(dataset.class_names),
    }
    _ensure_dir(output_path)
    (output_path / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    return {split: len(item_ids) for split, item_ids in assignments.items()}


def _split_yolo_classification_dataset(
    dataset: YOLODataset,
    output_path: Path,
    ratios: tuple[float, float, float],
    stratify: bool,
    seed: int,
) -> dict[str, int]:
    class_to_idx = {name: i for i, name in enumerate(dataset.class_names)}

    item_to_image: dict[str, Path] = {}
    item_to_class: dict[str, str] = {}
    labels: dict[str, set[int]] = {}
    item_index = 0

    for class_name in dataset.class_names:
        class_dir = dataset.path / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            item_id = str(item_index)
            item_index += 1
            item_to_image[item_id] = image_path
            item_to_class[item_id] = class_name
            labels[item_id] = {class_to_idx[class_name]}

    if not item_to_image:
        raise ValueError("No images found in the dataset.")

    items = list(item_to_image.keys())
    assignments = (
        _build_stratified_split(items, labels, ratios, seed)
        if stratify
        else _build_random_split(items, ratios, seed)
    )

    for split, item_ids in assignments.items():
        for item_id in item_ids:
            image_src = item_to_image[item_id]
            class_name = item_to_class[item_id]
            image_out_dir = output_path / "images" / split / class_name
            _ensure_dir(image_out_dir)
            shutil.copy2(image_src, image_out_dir / image_src.name)

    return {split: len(item_ids) for split, item_ids in assignments.items()}


def split_mask_dataset(
    dataset: MaskDataset,
    output_path: Path,
    ratios: tuple[float, float, float],
    stratify: bool,
    seed: int,
) -> dict[str, int]:
    """Split a mask dataset into train/val/test subsets."""
    image_paths = dataset.get_image_paths()
    if not image_paths:
        raise ValueError("No images found in the dataset.")

    class_mapping = dataset.get_class_mapping()
    class_ids = set(class_mapping.keys())
    if dataset.ignore_index is not None:
        class_ids.discard(dataset.ignore_index)

    item_to_image = {str(i): path for i, path in enumerate(sorted(image_paths))}
    item_to_mask: dict[str, Path] = {}
    labels: dict[str, set[int]] = {}

    for item_id, image_path in item_to_image.items():
        mask_path = dataset.get_mask_path(image_path)
        if not mask_path or not mask_path.exists():
            raise ValueError(f"No mask found for image: {image_path.name}")
        item_to_mask[item_id] = mask_path

        label_set: set[int] = set()
        mask = dataset.load_mask(image_path)
        if mask is not None:
            for class_id in np.unique(mask).tolist():
                class_id_int = int(class_id)
                if class_id_int in class_ids:
                    label_set.add(class_id_int)
        labels[item_id] = label_set

    items = list(item_to_image.keys())
    assignments = (
        _build_stratified_split(items, labels, ratios, seed)
        if stratify
        else _build_random_split(items, ratios, seed)
    )

    for split, item_ids in assignments.items():
        image_out_dir = output_path / "images" / split
        mask_out_dir = output_path / "masks" / split
        _ensure_dir(image_out_dir)
        _ensure_dir(mask_out_dir)

        for item_id in item_ids:
            image_src = item_to_image[item_id]
            mask_src = item_to_mask[item_id]
            shutil.copy2(image_src, image_out_dir / image_src.name)
            shutil.copy2(mask_src, mask_out_dir / f"{image_src.stem}.png")

    for config_name in ("classes.yaml", "classes.yml"):
        config_src = dataset.path / config_name
        if config_src.exists():
            shutil.copy2(config_src, output_path / config_name)
            break

    return {split: len(item_ids) for split, item_ids in assignments.items()}


def split_coco_dataset(
    dataset: COCODataset,
    annotation_file: Path,
    output_path: Path,
    ratios: tuple[float, float, float],
    stratify: bool,
    seed: int,
    roboflow_layout: bool | None = None,
) -> dict[str, int]:
    """Split a COCO dataset into train/val/test annotation files and images."""
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if roboflow_layout is None:
        roboflow_layout = dataset.is_roboflow_layout

    image_annotations: dict[int, list[dict]] = {img["id"]: [] for img in images}
    labels: dict[str, set[int]] = {}

    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id in image_annotations:
            image_annotations[image_id].append(ann)

    for img in images:
        image_id = img.get("id")
        if image_id is None:
            continue
        label_set: set[int] = set()
        for ann in image_annotations.get(image_id, []):
            category_id = ann.get("category_id")
            if isinstance(category_id, int):
                label_set.add(category_id)
        labels[str(image_id)] = label_set

    items = [str(img["id"]) for img in images if "id" in img]
    assignments = (
        _build_stratified_split(items, labels, ratios, seed)
        if stratify
        else _build_random_split(items, ratios, seed)
    )

    if not roboflow_layout:
        annotations_dir = output_path / "annotations"
        images_dir = output_path / "images"
        _ensure_dir(annotations_dir)
        _ensure_dir(images_dir)

    images_by_id = {img["id"]: img for img in images if "id" in img}

    for split, image_ids in assignments.items():
        split_images = []
        split_annotations = []

        for image_id_str in image_ids:
            image_id = int(image_id_str)
            img = images_by_id.get(image_id)
            if not img:
                continue
            split_images.append(img)
            split_annotations.extend(image_annotations.get(image_id, []))

            file_name = img.get("file_name")
            if not file_name:
                continue
            source = _find_image_path(dataset.path, file_name)
            if source is None:
                source = annotation_file.parent / file_name
            if source is None or not source.exists():
                raise ValueError(f"Image file not found: {file_name}")
            if roboflow_layout:
                split_dir_name = "valid" if split == "val" else split
                split_dir = output_path / split_dir_name
            else:
                split_dir = images_dir / split
            _ensure_dir(split_dir)
            shutil.copy2(source, split_dir / Path(file_name).name)

        split_data = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": data.get("categories", []),
        }
        if roboflow_layout:
            split_dir_name = "valid" if split == "val" else split
            out_file = output_path / split_dir_name / "_annotations.coco.json"
        else:
            out_file = annotations_dir / f"instances_{split}.json"
        out_file.write_text(json.dumps(split_data))

    return {split: len(image_ids) for split, image_ids in assignments.items()}


def _find_coco_image_path(
    dataset: COCODataset, ann_file: Path, file_name: str
) -> Path | None:
    split = dataset._get_split_from_filename(ann_file.stem, ann_file.parent.name)
    split_dir = dataset.get_split_dir_name(split) if split != "unsplit" else ""
    base_name = Path(file_name).name

    candidates = [
        ann_file.parent / file_name,
        ann_file.parent / base_name,
        dataset.path / "images" / split_dir / file_name,
        dataset.path / "images" / split_dir / base_name,
        dataset.path / "images" / file_name,
        dataset.path / "images" / base_name,
        dataset.path / split_dir / file_name if split_dir else dataset.path / file_name,
        dataset.path / split_dir / base_name if split_dir else dataset.path / base_name,
        dataset.path / file_name,
        dataset.path / base_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def unsplit_yolo_dataset(
    dataset: YOLODataset,
    output_path: Path,
    collision_policy: CollisionPolicy = "error",
) -> dict[str, int]:
    """Merge a split YOLO dataset into a single unsplit dataset."""
    if dataset.task == TaskType.CLASSIFICATION:
        return _unsplit_yolo_classification_dataset(
            dataset=dataset,
            output_path=output_path,
            collision_policy=collision_policy,
        )

    images_out = output_path / "images"
    labels_out = output_path / "labels"
    _ensure_dir(images_out)
    _ensure_dir(labels_out)

    total_images = 0
    for split in dataset.splits:
        image_dir, label_dir = dataset.get_split_dirs(split)
        if not image_dir.is_dir():
            continue
        for image_src in sorted(image_dir.iterdir()):
            if image_src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            out_name = _resolve_collision_name(
                file_name=image_src.name,
                source_path=image_src,
                split=split,
                output_dir=images_out,
                collision_policy=collision_policy,
            )
            image_dst = images_out / out_name
            shutil.copy2(image_src, image_dst)

            stem = Path(out_name).stem
            label_src = label_dir / f"{image_src.stem}.txt"
            label_dst = labels_out / f"{stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                label_dst.write_text("")

            total_images += 1

    if total_images == 0:
        raise ValueError("No images found in split dataset.")

    config = {
        "path": ".",
        "names": dataset.class_names,
        "nc": len(dataset.class_names),
    }
    (output_path / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return {"total": total_images}


def _unsplit_yolo_classification_dataset(
    dataset: YOLODataset,
    output_path: Path,
    collision_policy: CollisionPolicy = "error",
) -> dict[str, int]:
    images_root = dataset.path / "images"
    total_images = 0

    for class_name in dataset.class_names:
        class_out = output_path / class_name
        _ensure_dir(class_out)

    for split in dataset.splits:
        split_dir = images_root / split
        if not split_dir.is_dir():
            continue
        for class_name in dataset.class_names:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            class_out = output_path / class_name
            for image_src in sorted(class_dir.iterdir()):
                if image_src.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                out_name = _resolve_collision_name(
                    file_name=image_src.name,
                    source_path=image_src,
                    split=split,
                    output_dir=class_out,
                    collision_policy=collision_policy,
                )
                shutil.copy2(image_src, class_out / out_name)
                total_images += 1

    if total_images == 0:
        raise ValueError("No images found in split dataset.")
    return {"total": total_images}


def unsplit_mask_dataset(
    dataset: MaskDataset,
    output_path: Path,
    collision_policy: CollisionPolicy = "error",
) -> dict[str, int]:
    """Merge a split mask dataset into flat images/masks layout."""
    images_out = output_path / "images"
    masks_out = output_path / "masks"
    _ensure_dir(images_out)
    _ensure_dir(masks_out)

    images_root = dataset.path / dataset.images_dir
    total_images = 0

    for split in dataset.splits:
        image_dir = images_root / split
        if not image_dir.is_dir():
            continue
        for image_src in sorted(image_dir.iterdir()):
            if image_src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_src = dataset.get_mask_path(image_src)
            if not mask_src or not mask_src.exists():
                continue
            out_name = _resolve_collision_name(
                file_name=image_src.name,
                source_path=image_src,
                split=split,
                output_dir=images_out,
                collision_policy=collision_policy,
            )
            image_dst = images_out / out_name
            shutil.copy2(image_src, image_dst)
            mask_dst = masks_out / f"{Path(out_name).stem}.png"
            shutil.copy2(mask_src, mask_dst)
            total_images += 1

    if total_images == 0:
        raise ValueError("No images found in split dataset.")

    for config_name in ("classes.yaml", "classes.yml"):
        config_src = dataset.path / config_name
        if config_src.exists():
            shutil.copy2(config_src, output_path / config_name)
            break

    return {"total": total_images}


def unsplit_coco_dataset(
    dataset: COCODataset,
    output_path: Path,
    collision_policy: CollisionPolicy = "error",
) -> dict[str, int]:
    """Merge split COCO annotations and images into one unsplit dataset."""
    if not dataset.annotation_files:
        raise ValueError("No annotation files found.")

    annotations_out = output_path / "annotations"
    images_out = output_path / "images"
    _ensure_dir(annotations_out)
    _ensure_dir(images_out)

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    merged_categories: list[dict] = []
    category_name_to_id: dict[str, int] = {}
    info: dict = {}
    licenses: list[dict] = []

    next_image_id = 1
    next_annotation_id = 1
    next_category_id = 1
    used_category_ids: set[int] = set()

    for ann_file in sorted(dataset.annotation_files):
        try:
            data = json.loads(ann_file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Could not read annotation file: {ann_file}") from exc

        if not info and isinstance(data.get("info"), dict):
            info = data.get("info", {})
        if not licenses and isinstance(data.get("licenses"), list):
            licenses = data.get("licenses", [])

        local_cat_map: dict[int, int] = {}
        for cat in data.get("categories", []):
            if not isinstance(cat, dict):
                continue
            old_id = cat.get("id")
            name = cat.get("name")
            if not isinstance(old_id, int) or not isinstance(name, str):
                continue
            if name in category_name_to_id:
                new_cat_id = category_name_to_id[name]
            else:
                proposed = (
                    old_id if old_id not in used_category_ids else next_category_id
                )
                while proposed in used_category_ids:
                    proposed += 1
                new_cat_id = proposed
                category_name_to_id[name] = new_cat_id
                used_category_ids.add(new_cat_id)
                next_category_id = max(next_category_id, new_cat_id + 1)

                new_cat = dict(cat)
                new_cat["id"] = new_cat_id
                merged_categories.append(new_cat)
            local_cat_map[old_id] = new_cat_id

        local_image_map: dict[int, int] = {}
        split_name = dataset._get_split_from_filename(
            ann_file.stem, ann_file.parent.name
        )
        split_hint = None if split_name == "unsplit" else split_name
        for img in data.get("images", []):
            if not isinstance(img, dict):
                continue
            old_image_id = img.get("id")
            file_name = img.get("file_name")
            if not isinstance(old_image_id, int) or not isinstance(file_name, str):
                continue

            image_src = _find_coco_image_path(dataset, ann_file, file_name)
            if image_src is None:
                raise ValueError(f"Image file not found: {file_name}")

            out_name = _resolve_collision_name(
                file_name=Path(file_name).name,
                source_path=image_src,
                split=split_hint,
                output_dir=images_out,
                collision_policy=collision_policy,
            )
            shutil.copy2(image_src, images_out / out_name)

            new_image = dict(img)
            new_image["id"] = next_image_id
            new_image["file_name"] = out_name
            merged_images.append(new_image)
            local_image_map[old_image_id] = next_image_id
            next_image_id += 1

        for ann in data.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            old_image_id = ann.get("image_id")
            if not isinstance(old_image_id, int) or old_image_id not in local_image_map:
                continue

            old_category_id = ann.get("category_id")
            new_category_id = None
            if isinstance(old_category_id, int):
                new_category_id = local_cat_map.get(old_category_id, old_category_id)

            new_ann = dict(ann)
            new_ann["id"] = next_annotation_id
            new_ann["image_id"] = local_image_map[old_image_id]
            if new_category_id is not None:
                new_ann["category_id"] = new_category_id
            merged_annotations.append(new_ann)
            next_annotation_id += 1

    if not merged_images:
        raise ValueError("No images found in split dataset.")

    unsplit_data = {
        "info": info,
        "licenses": licenses,
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }
    (annotations_out / "annotations.json").write_text(json.dumps(unsplit_data))
    return {"total": len(merged_images)}


def is_coco_roboflow_layout(dataset: COCODataset, annotation_file: Path) -> bool:
    """Return whether the dataset uses Roboflow COCO layout."""
    _ = annotation_file
    return dataset.is_roboflow_layout


def is_coco_unsplit(annotation_files: Iterable[Path]) -> bool:
    """Return whether COCO annotations do not encode split information."""
    for ann_file in annotation_files:
        name = ann_file.stem.lower()
        parent = ann_file.parent.name.lower()
        if any(split in name for split in _SPLITS):
            return False
        if parent in _COCO_SPLIT_DIRS:
            return False
    return True
