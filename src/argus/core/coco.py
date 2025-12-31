"""COCO dataset detection and handling."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from argus.core.base import Dataset, DatasetFormat, TaskType


@dataclass
class COCODataset(Dataset):
    """COCO format dataset.

    Supports detection and segmentation tasks.

    Structure:
        dataset/
        ├── annotations/
        │   ├── instances_train.json
        │   └── instances_val.json
        └── images/
            ├── train/
            └── val/
    """

    annotation_files: list[Path] = field(default_factory=list)
    format: DatasetFormat = field(default=DatasetFormat.COCO, init=False)

    @classmethod
    def detect(cls, path: Path) -> "COCODataset | None":
        """Detect if the given path contains a COCO dataset.

        Args:
            path: Directory path to check for dataset.

        Returns:
            COCODataset instance if detected, None otherwise.
        """
        path = Path(path)

        if not path.is_dir():
            return None

        # Find annotation JSON files
        annotation_files = cls._find_annotation_files(path)

        if not annotation_files:
            return None

        # Parse the first valid annotation file to extract metadata
        for ann_file in annotation_files:
            result = cls._parse_annotation_file(path, ann_file, annotation_files)
            if result:
                return result

        return None

    @classmethod
    def _find_annotation_files(cls, path: Path) -> list[Path]:
        """Find COCO annotation JSON files.

        Args:
            path: Dataset root path.

        Returns:
            List of annotation file paths.
        """
        annotation_files = []

        # Check annotations/ directory first
        annotations_dir = path / "annotations"
        if annotations_dir.is_dir():
            annotation_files.extend(annotations_dir.glob("*.json"))

        # Also check root directory for single annotation file
        annotation_files.extend(path.glob("*.json"))

        # Filter to only include files that might be COCO annotations
        # (exclude package.json, tsconfig.json, etc.)
        filtered_files = []
        for f in annotation_files:
            name_lower = f.name.lower()
            # Common COCO annotation patterns
            if any(
                pattern in name_lower
                for pattern in [
                    "instances",
                    "annotations",
                    "train",
                    "val",
                    "test",
                    "coco",
                ]
            ):
                filtered_files.append(f)
            elif name_lower.endswith(".json"):
                # Check if it's a COCO-format file by trying to parse
                filtered_files.append(f)

        return filtered_files

    @classmethod
    def _parse_annotation_file(
        cls, path: Path, ann_file: Path, all_annotation_files: list[Path]
    ) -> "COCODataset | None":
        """Parse a COCO annotation file and extract metadata.

        Args:
            path: Dataset root path.
            ann_file: Annotation file to parse.
            all_annotation_files: All found annotation files.

        Returns:
            COCODataset if valid COCO format, None otherwise.
        """
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return None

            # Must have images, annotations, and categories keys
            required_keys = ["images", "annotations", "categories"]
            if not all(key in data for key in required_keys):
                return None

            # Validate structure
            if not isinstance(data["images"], list):
                return None
            if not isinstance(data["annotations"], list):
                return None
            if not isinstance(data["categories"], list):
                return None

            # Extract class names from categories
            categories = data["categories"]
            class_names = []
            for cat in categories:
                if isinstance(cat, dict) and "name" in cat:
                    class_names.append(cat["name"])

            num_classes = len(class_names)

            # Determine task type from annotations
            task = cls._determine_task_type(data["annotations"])

            # Detect splits from annotation files
            splits = cls._detect_splits(all_annotation_files)

            return cls(
                path=path,
                task=task,
                num_classes=num_classes,
                class_names=class_names,
                splits=splits,
                annotation_files=all_annotation_files,
            )

        except (json.JSONDecodeError, OSError):
            return None

    def get_instance_counts(self) -> dict[str, dict[str, int]]:
        """Get the number of annotation instances per class, per split.

        Parses all annotation JSON files and counts category_id occurrences.

        Returns:
            Dictionary mapping split name to dict of class name to instance count.
        """
        counts: dict[str, dict[str, int]] = {}

        for ann_file in self.annotation_files:
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    continue

                # Build category_id -> name mapping
                categories = data.get("categories", [])
                id_to_name: dict[int, str] = {}
                for cat in categories:
                    if isinstance(cat, dict) and "id" in cat and "name" in cat:
                        id_to_name[cat["id"]] = cat["name"]

                # Determine split from filename
                split = self._get_split_from_filename(ann_file.stem)

                # Count annotations per category
                split_counts: dict[str, int] = counts.get(split, {})
                annotations = data.get("annotations", [])

                for ann in annotations:
                    if isinstance(ann, dict) and "category_id" in ann:
                        cat_id = ann["category_id"]
                        class_name = id_to_name.get(cat_id, f"class_{cat_id}")
                        split_counts[class_name] = split_counts.get(class_name, 0) + 1

                counts[split] = split_counts

            except (json.JSONDecodeError, OSError):
                continue

        return counts

    @staticmethod
    def _get_split_from_filename(filename: str) -> str:
        """Extract split name from annotation filename.

        Args:
            filename: Annotation file stem (without extension).

        Returns:
            Split name (train, val, test) or 'train' as default.
        """
        name_lower = filename.lower()
        if "train" in name_lower:
            return "train"
        elif "val" in name_lower:
            return "val"
        elif "test" in name_lower:
            return "test"
        return "train"

    @classmethod
    def _determine_task_type(cls, annotations: list) -> TaskType:
        """Determine task type from annotations.

        Args:
            annotations: List of annotation dicts.

        Returns:
            TaskType.DETECTION or TaskType.SEGMENTATION.
        """
        # Sample annotations to determine task type
        for ann in annotations[:10]:
            if not isinstance(ann, dict):
                continue

            # Check for segmentation data
            if "segmentation" in ann:
                seg = ann["segmentation"]
                # RLE or polygon segmentation
                if isinstance(seg, dict) or (isinstance(seg, list) and seg):
                    return TaskType.SEGMENTATION

        # Default to detection
        return TaskType.DETECTION

    @classmethod
    def _detect_splits(cls, annotation_files: list[Path]) -> list[str]:
        """Detect available splits from annotation filenames.

        Args:
            annotation_files: List of annotation file paths.

        Returns:
            List of detected split names.
        """
        splits = []

        for ann_file in annotation_files:
            name_lower = ann_file.stem.lower()

            if "train" in name_lower:
                if "train" not in splits:
                    splits.append("train")
            elif "val" in name_lower:
                if "val" not in splits:
                    splits.append("val")
            elif "test" in name_lower:
                if "test" not in splits:
                    splits.append("test")

        # If no splits detected from filenames, default to train
        if not splits:
            splits.append("train")

        return splits
