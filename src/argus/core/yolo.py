"""YOLO dataset detection and handling."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from argus.core.base import Dataset, DatasetFormat, TaskType


@dataclass
class YOLODataset(Dataset):
    """YOLO format dataset.

    Supports detection and segmentation tasks.

    Structure:
        dataset/
        ├── data.yaml (or *.yaml/*.yml with 'names' key)
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """

    config_file: Path | None = None
    format: DatasetFormat = field(default=DatasetFormat.YOLO, init=False)

    @classmethod
    def detect(cls, path: Path) -> "YOLODataset | None":
        """Detect if the given path contains a YOLO dataset.

        Args:
            path: Directory path to check for dataset.

        Returns:
            YOLODataset instance if detected, None otherwise.
        """
        path = Path(path)

        if not path.is_dir():
            return None

        # Try detection/segmentation (YAML-based)
        return cls._detect_yaml_based(path)

    @classmethod
    def _detect_yaml_based(cls, path: Path) -> "YOLODataset | None":
        """Detect YAML-based YOLO dataset (detection/segmentation).

        Args:
            path: Directory path to check.

        Returns:
            YOLODataset if valid YAML config found, None otherwise.
        """
        # Find all YAML files in the directory
        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                if not isinstance(config, dict):
                    continue

                # Must have 'names' key for valid YOLO dataset
                if "names" not in config:
                    continue

                names = config["names"]

                # Extract class names
                if isinstance(names, dict):
                    class_names = list(names.values())
                elif isinstance(names, list):
                    class_names = names
                else:
                    continue

                num_classes = len(class_names)

                # Detect available splits
                splits = cls._detect_splits(path, config)

                # Determine task type (detection vs segmentation)
                task = cls._determine_task_type(path, config)

                return cls(
                    path=path,
                    task=task,
                    num_classes=num_classes,
                    class_names=class_names,
                    splits=splits,
                    config_file=yaml_file,
                )

            except (yaml.YAMLError, OSError):
                continue

        return None

    def get_instance_counts(self) -> dict[str, dict[str, int]]:
        """Get the number of annotation instances per class, per split.

        Parses all label files in labels/{split}/*.txt and counts
        occurrences of each class ID. For unsplit datasets, uses "unsplit"
        as the split name.

        Returns:
            Dictionary mapping split name to dict of class name to instance count.
        """
        counts: dict[str, dict[str, int]] = {}

        # Build class_id -> class_name mapping
        id_to_name = {i: name for i, name in enumerate(self.class_names)}

        # Determine splits to process - use "unsplit" for flat structure
        splits_to_process = self.splits if self.splits else ["unsplit"]

        # Get label directories for each split
        for split in splits_to_process:
            split_counts: dict[str, int] = {}

            # Find label directory for this split
            if split == "unsplit":
                label_dir = self.path / "labels"
            else:
                label_dir = self.path / "labels" / split
                if not label_dir.is_dir():
                    # Fallback to flat structure
                    label_dir = self.path / "labels"

            if not label_dir.is_dir():
                continue

            # Parse all label files
            for txt_file in label_dir.glob("*.txt"):
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 5:  # Valid annotation line
                                try:
                                    class_id = int(parts[0])
                                    class_name = id_to_name.get(class_id, f"class_{class_id}")
                                    split_counts[class_name] = split_counts.get(class_name, 0) + 1
                                except ValueError:
                                    continue
                except OSError:
                    continue

            counts[split] = split_counts

        return counts

    def get_image_counts(self) -> dict[str, dict[str, int]]:
        """Get image counts per split, including background images.

        Counts label files in labels/{split}/*.txt. Empty files are
        counted as background images.

        Returns:
            Dictionary mapping split name to dict with "total" and "background" counts.
        """
        counts: dict[str, dict[str, int]] = {}

        # Determine splits to process - use "unsplit" for flat structure
        splits_to_process = self.splits if self.splits else ["unsplit"]

        for split in splits_to_process:
            # Find label directory for this split
            if split == "unsplit":
                label_dir = self.path / "labels"
            else:
                label_dir = self.path / "labels" / split
                if not label_dir.is_dir():
                    label_dir = self.path / "labels"

            if not label_dir.is_dir():
                continue

            total = 0
            background = 0

            for txt_file in label_dir.glob("*.txt"):
                total += 1
                try:
                    content = txt_file.read_text(encoding="utf-8").strip()
                    if not content:
                        background += 1
                except OSError:
                    continue

            counts[split] = {"total": total, "background": background}

        return counts

    @classmethod
    def _detect_splits(cls, path: Path, config: dict) -> list[str]:
        """Detect available splits from config and filesystem.

        Args:
            path: Dataset root path.
            config: Parsed YAML config.

        Returns:
            List of available split names.
        """
        splits = []

        # Check config-defined paths first
        for split_name in ["train", "val", "test"]:
            if split_name in config:
                split_path = config[split_name]
                if split_path:
                    # Handle relative paths
                    full_path = path / split_path
                    if full_path.exists():
                        splits.append(split_name)
                        continue

            # Fallback: check common directory structures
            # Pattern 1: images/train/, images/val/
            if (path / "images" / split_name).is_dir():
                splits.append(split_name)
                continue

            # Pattern 2: train/, val/ (flat structure with images/ and labels/)
            if (path / split_name).is_dir():
                # Make sure it's not a classification dataset
                if (path / "images").is_dir() or (path / "labels").is_dir():
                    continue
                splits.append(split_name)

        return splits

    @classmethod
    def _determine_task_type(cls, path: Path, config: dict) -> TaskType:
        """Determine if dataset is detection or segmentation.

        Detection labels have 5 columns: class x_center y_center width height
        Segmentation labels have >5 columns: class x1 y1 x2 y2 ... xn yn

        Args:
            path: Dataset root path.
            config: Parsed YAML config.

        Returns:
            TaskType.DETECTION or TaskType.SEGMENTATION.
        """
        # Find label files to analyze
        label_dirs = []

        # Check common label locations
        for split in ["train", "val", "test"]:
            # Pattern: labels/train/
            label_dir = path / "labels" / split
            if label_dir.is_dir():
                label_dirs.append(label_dir)

        # Pattern: labels/ (flat)
        labels_dir = path / "labels"
        if labels_dir.is_dir() and not label_dirs:
            label_dirs.append(labels_dir)

        # Sample label files
        for label_dir in label_dirs:
            txt_files = list(label_dir.glob("*.txt"))
            for txt_file in txt_files[:5]:  # Sample up to 5 files
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) > 5:
                                return TaskType.SEGMENTATION
                            elif len(parts) == 5:
                                return TaskType.DETECTION
                except OSError:
                    continue

        # Default to detection if no labels found or inconclusive
        return TaskType.DETECTION
