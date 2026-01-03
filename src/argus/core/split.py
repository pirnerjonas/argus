"""Dataset splitting utilities."""

from __future__ import annotations

import json
import math
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

import yaml

from argus.core.coco import COCODataset
from argus.core.yolo import YOLODataset

_SPLITS = ("train", "val", "test")


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
                donor_candidates = [j for j in range(len(base)) if j != idx and base[j] > 0]
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


def split_yolo_dataset(
    dataset: YOLODataset,
    output_path: Path,
    ratios: tuple[float, float, float],
    stratify: bool,
    seed: int,
) -> dict[str, int]:
    image_paths = dataset.get_image_paths()
    if not image_paths:
        raise ValueError("No images found in the dataset.")

    image_map = {path.stem: path for path in image_paths}
    label_dir = dataset.path / "labels"

    labels: dict[str, set[int]] = {}
    for stem, _image_path in image_map.items():
        label_path = label_dir / f"{stem}.txt"
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
        labels[stem] = label_set

    items = list(image_map.keys())
    if stratify:
        assignments = _build_stratified_split(items, labels, ratios, seed)
    else:
        assignments = _build_random_split(items, ratios, seed)

    for split, stems in assignments.items():
        image_out_dir = output_path / "images" / split
        label_out_dir = output_path / "labels" / split
        _ensure_dir(image_out_dir)
        _ensure_dir(label_out_dir)

        for stem in stems:
            image_src = image_map[stem]
            image_dst = image_out_dir / image_src.name
            shutil.copy2(image_src, image_dst)

            label_src = label_dir / f"{stem}.txt"
            label_dst = label_out_dir / f"{stem}.txt"
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

    return {split: len(items) for split, items in assignments.items()}


def split_coco_dataset(
    dataset: COCODataset,
    annotation_file: Path,
    output_path: Path,
    ratios: tuple[float, float, float],
    stratify: bool,
    seed: int,
) -> dict[str, int]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])

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
    if stratify:
        assignments = _build_stratified_split(items, labels, ratios, seed)
    else:
        assignments = _build_random_split(items, ratios, seed)

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
                raise ValueError(f"Image file not found: {file_name}")
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
        out_file = annotations_dir / f"instances_{split}.json"
        out_file.write_text(json.dumps(split_data))

    return {split: len(items) for split, items in assignments.items()}


def is_coco_unsplit(annotation_files: Iterable[Path]) -> bool:
    for ann_file in annotation_files:
        name = ann_file.stem.lower()
        if any(split in name for split in _SPLITS):
            return False
    return True
