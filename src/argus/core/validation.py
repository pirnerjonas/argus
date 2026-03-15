"""Dataset validation for detecting annotation quality issues."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from argus.core.base import IMAGE_EXTENSIONS, Dataset, DatasetFormat

if TYPE_CHECKING:
    from argus.core.coco import COCODataset
    from argus.core.mask import MaskDataset
    from argus.core.yolo import YOLODataset


@dataclass
class ValidationIssue:
    """A single validation issue found in a dataset."""

    level: Literal["error", "warning"]
    code: str
    message: str
    file: Path | None = None
    split: str | None = None


@dataclass
class ValidationReport:
    """Result of validating a dataset."""

    dataset_path: Path
    format: DatasetFormat
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_dataset(
    dataset: Dataset,
    *,
    split: str | None = None,
    check_images: bool = False,
) -> ValidationReport:
    """Validate a dataset and return a report of issues found.

    Args:
        dataset: Dataset instance to validate.
        split: If set, only validate this specific split.
        check_images: If True, verify images are readable with cv2.

    Returns:
        ValidationReport with all issues found.
    """
    report = ValidationReport(
        dataset_path=dataset.path,
        format=dataset.format,
    )

    issues = _validate_common(dataset, split=split, check_images=check_images)

    if dataset.format == DatasetFormat.YOLO:
        from argus.core.yolo import YOLODataset

        assert isinstance(dataset, YOLODataset)
        issues.extend(_validate_yolo(dataset, split=split))
    elif dataset.format == DatasetFormat.COCO:
        from argus.core.coco import COCODataset

        assert isinstance(dataset, COCODataset)
        issues.extend(_validate_coco(dataset, split=split))
    elif dataset.format == DatasetFormat.MASK:
        from argus.core.mask import MaskDataset

        assert isinstance(dataset, MaskDataset)
        issues.extend(_validate_mask(dataset, split=split))

    report.issues = issues
    return report


def _validate_common(
    dataset: Dataset,
    *,
    split: str | None = None,
    check_images: bool = False,
) -> list[ValidationIssue]:
    """Run universal validation checks applicable to all formats."""
    issues: list[ValidationIssue] = []

    splits_to_check = (
        [split] if split else (dataset.splits if dataset.splits else [None])
    )

    for s in splits_to_check:
        split_name = s or "unsplit"
        image_paths = dataset.get_image_paths(s)

        if not image_paths:
            issues.append(
                ValidationIssue(
                    level="warning",
                    code="W102",
                    message=f"Split '{split_name}' has no images.",
                    split=split_name,
                )
            )
            continue

        # Check for duplicate image filenames within split
        seen_names: dict[str, Path] = {}
        for img_path in image_paths:
            name = img_path.name
            if name in seen_names:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="E102",
                        message=(
                            f"Duplicate image filename '{name}' "
                            f"in split '{split_name}'."
                        ),
                        file=img_path,
                        split=split_name,
                    )
                )
            else:
                seen_names[name] = img_path

        # Check image existence and readability
        for img_path in image_paths:
            if not img_path.exists():
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="E101",
                        message=f"Image file does not exist: {img_path.name}",
                        file=img_path,
                        split=split_name,
                    )
                )
            elif check_images:
                import cv2

                img = cv2.imread(str(img_path))
                if img is None:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E101",
                            message=f"Image file is not readable: {img_path.name}",
                            file=img_path,
                            split=split_name,
                        )
                    )

    return issues


def _validate_yolo(
    dataset: YOLODataset,
    *,
    split: str | None = None,
) -> list[ValidationIssue]:
    """Run YOLO-specific validation checks."""
    from argus.core.base import TaskType

    issues: list[ValidationIssue] = []

    if dataset.task == TaskType.CLASSIFICATION:
        return issues

    splits_to_check = (
        [split] if split else (dataset.splits if dataset.splits else ["unsplit"])
    )

    for s in splits_to_check:
        if s == "unsplit":
            image_dir = dataset.path / "images"
            label_dir = dataset.path / "labels"
        else:
            image_dir, label_dir = dataset.get_split_dirs(s)

        if not label_dir.is_dir():
            continue

        # Collect image stems for checking orphan labels
        image_stems: set[str] = set()
        if image_dir.is_dir():
            image_stems = {
                f.stem
                for f in image_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            }

        expected_cols = 5 if dataset.task == TaskType.DETECTION else None

        for label_file in label_dir.glob("*.txt"):
            # Check orphan labels
            if label_file.stem not in image_stems:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        code="W201",
                        message=(
                            f"Label file has no corresponding image: {label_file.name}"
                        ),
                        file=label_file,
                        split=s,
                    )
                )

            try:
                content = label_file.read_text(encoding="utf-8")
            except OSError:
                continue

            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Check column count
                if expected_cols is not None and len(parts) != expected_cols:
                    if len(parts) < 5:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="E201",
                                message=(
                                    f"Line {line_num}: expected "
                                    f"{expected_cols} columns, "
                                    f"got {len(parts)}."
                                ),
                                file=label_file,
                                split=s,
                            )
                        )
                        continue
                elif len(parts) < 5:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E201",
                            message=f"Line {line_num}: too few columns ({len(parts)}).",
                            file=label_file,
                            split=s,
                        )
                    )
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E202",
                            message=f"Line {line_num}: invalid class ID '{parts[0]}'.",
                            file=label_file,
                            split=s,
                        )
                    )
                    continue

                # Check class ID range
                if class_id < 0 or class_id >= dataset.num_classes:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E202",
                            message=(
                                f"Line {line_num}: class ID {class_id} out of range "
                                f"[0, {dataset.num_classes - 1}]."
                            ),
                            file=label_file,
                            split=s,
                        )
                    )

                # Check coordinates
                try:
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E203",
                            message=f"Line {line_num}: non-numeric coordinate values.",
                            file=label_file,
                            split=s,
                        )
                    )
                    continue

                # Check bounds [0, 1]
                for val in coords:
                    if val < 0.0 or val > 1.0:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="E203",
                                message=(
                                    f"Line {line_num}: coordinate {val} out of "
                                    f"bounds [0, 1]."
                                ),
                                file=label_file,
                                split=s,
                            )
                        )
                        break  # one issue per line for bounds

                # Check box dimensions (detection only)
                if len(parts) == 5:
                    w, h = coords[2], coords[3]
                    if w <= 0 or h <= 0:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="E204",
                                message=(
                                    f"Line {line_num}: zero or negative box dimensions."
                                ),
                                file=label_file,
                                split=s,
                            )
                        )
                    elif w < 0.001 or h < 0.001:
                        issues.append(
                            ValidationIssue(
                                level="warning",
                                code="W202",
                                message=(
                                    f"Line {line_num}: very small bounding box "
                                    f"({w:.4f} x {h:.4f})."
                                ),
                                file=label_file,
                                split=s,
                            )
                        )

    return issues


def _validate_coco(
    dataset: COCODataset,
    *,
    split: str | None = None,
) -> list[ValidationIssue]:
    """Run COCO-specific validation checks."""
    issues: list[ValidationIssue] = []

    for ann_file in dataset.annotation_files:
        file_split = dataset._get_split_from_filename(
            ann_file.stem, ann_file.parent.name
        )

        if split and file_split != split:
            continue

        try:
            with open(ann_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, dict):
            continue

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        # Build lookup sets
        image_ids: set[int] = set()
        for img in images:
            if isinstance(img, dict) and "id" in img:
                img_id = img["id"]
                if img_id in image_ids:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E305",
                            message=f"Duplicate image ID: {img_id}",
                            file=ann_file,
                            split=file_split,
                        )
                    )
                image_ids.add(img_id)

        category_ids = {
            cat["id"] for cat in categories if isinstance(cat, dict) and "id" in cat
        }

        # Check for images without annotations
        annotated_image_ids: set[int] = set()

        # Check annotations
        seen_ann_ids: set[int] = set()
        for ann in annotations:
            if not isinstance(ann, dict):
                continue

            # Check duplicate annotation IDs
            ann_id = ann.get("id")
            if ann_id is not None:
                if ann_id in seen_ann_ids:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E303",
                            message=f"Duplicate annotation ID: {ann_id}",
                            file=ann_file,
                            split=file_split,
                        )
                    )
                seen_ann_ids.add(ann_id)

            # Check image_id reference
            image_id = ann.get("image_id")
            if image_id is not None and image_id not in image_ids:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="E301",
                        message=(
                            f"Annotation references non-existent image_id: {image_id}"
                        ),
                        file=ann_file,
                        split=file_split,
                    )
                )
            if image_id is not None:
                annotated_image_ids.add(image_id)

            # Check category_id reference
            cat_id = ann.get("category_id")
            if cat_id is not None and cat_id not in category_ids:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="E302",
                        message=(
                            f"Annotation references non-existent category_id: {cat_id}"
                        ),
                        file=ann_file,
                        split=file_split,
                    )
                )

            # Check bbox validity
            bbox = ann.get("bbox")
            if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                _, _, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                if w <= 0 or h <= 0:
                    neg = w < 0 or h < 0
                    dim_kind = "negative" if neg else "zero"
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E304",
                            message=(
                                f"Annotation {ann_id}: invalid bbox "
                                f"with {dim_kind} dimensions."
                            ),
                            file=ann_file,
                            split=file_split,
                        )
                    )

            # Check polygon point count
            seg = ann.get("segmentation")
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) < 6:
                        issues.append(
                            ValidationIssue(
                                level="warning",
                                code="W302",
                                message=(
                                    f"Annotation {ann_id}: polygon has fewer than "
                                    f"3 points ({len(poly) // 2} points)."
                                ),
                                file=ann_file,
                                split=file_split,
                            )
                        )

        # Warn about images without annotations
        for img in images:
            if (
                isinstance(img, dict)
                and "id" in img
                and img["id"] not in annotated_image_ids
            ):
                file_name = img.get("file_name", f"id={img['id']}")
                issues.append(
                    ValidationIssue(
                        level="warning",
                        code="W301",
                        message=f"Image has no annotations: {file_name}",
                        file=ann_file,
                        split=file_split,
                    )
                )

    return issues


def _validate_mask(
    dataset: MaskDataset,
    *,
    split: str | None = None,
) -> list[ValidationIssue]:
    """Run mask-specific validation checks."""
    import cv2

    issues: list[ValidationIssue] = []
    images_root = dataset.path / dataset.images_dir

    splits_to_check = (
        [split] if split else (dataset.splits if dataset.splits else [None])
    )

    for s in splits_to_check:
        split_name = s or "unsplit"
        image_dir = images_root / s if s else images_root

        if not image_dir.is_dir():
            continue

        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Check mask existence
            mask_path = dataset.get_mask_path(img_file)
            if mask_path is None or not mask_path.exists():
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="E401",
                        message=f"No corresponding mask for image: {img_file.name}",
                        file=img_file,
                        split=split_name,
                    )
                )
                continue

            # Check dimension match
            match, img_shape, mask_shape = dataset.validate_dimensions(img_file)
            if not match and img_shape is not None and mask_shape is not None:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="E402",
                        message=(
                            f"Dimension mismatch for {img_file.name}: "
                            f"image {img_shape} vs mask {mask_shape}."
                        ),
                        file=img_file,
                        split=split_name,
                    )
                )

            # Check mask pixel values
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_vals = set(np.unique(mask).tolist())
                valid_ids = set(dataset._class_mapping.keys())
                if dataset.ignore_index is not None:
                    valid_ids.add(dataset.ignore_index)

                unexpected = unique_vals - valid_ids
                if unexpected:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="E403",
                            message=(
                                f"Mask {mask_path.name} contains unexpected pixel "
                                f"values: {sorted(unexpected)}."
                            ),
                            file=mask_path,
                            split=split_name,
                        )
                    )

    return issues
