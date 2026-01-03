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
                with open(yaml_file, encoding="utf-8") as f:
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
                    with open(txt_file, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 5:  # Valid annotation line
                                try:
                                    class_id = int(parts[0])
                                    fallback = f"class_{class_id}"
                                    class_name = id_to_name.get(class_id, fallback)
                                    current = split_counts.get(class_name, 0)
                                    split_counts[class_name] = current + 1
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
                    with open(txt_file, encoding="utf-8") as f:
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

    def get_image_paths(self, split: str | None = None) -> list[Path]:
        """Get all image file paths for a split or the entire dataset.

        Args:
            split: Specific split to get images from. If None, returns all images.

        Returns:
            List of image file paths sorted alphabetically.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths: list[Path] = []

        # Determine which splits to search
        if split:
            splits_to_search = [split]
        elif self.splits:
            splits_to_search = self.splits
        else:
            splits_to_search = ["unsplit"]

        for s in splits_to_search:
            if s == "unsplit":
                image_dir = self.path / "images"
            else:
                image_dir = self.path / "images" / s
                if not image_dir.is_dir():
                    image_dir = self.path / "images"

            if image_dir.is_dir():
                for img_file in image_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        image_paths.append(img_file)

        return sorted(image_paths, key=lambda p: p.name)

    def get_annotations_for_image(self, image_path: Path) -> list[dict]:
        """Get annotations for a specific image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of annotation dicts with bbox/polygon in absolute coordinates.
        """
        import cv2

        annotations: list[dict] = []

        # Build class_id -> class_name mapping
        id_to_name = {i: name for i, name in enumerate(self.class_names)}

        # Find the corresponding label file
        # Image: images/train/img.jpg -> Label: labels/train/img.txt
        image_parts = image_path.parts
        try:
            images_idx = image_parts.index("images")
            label_parts = list(image_parts)
            label_parts[images_idx] = "labels"
            label_path = Path(*label_parts).with_suffix(".txt")
        except ValueError:
            # Fallback: look in labels directory with same name
            label_path = self.path / "labels" / image_path.with_suffix(".txt").name

        if not label_path.exists():
            return annotations

        # Get image dimensions for converting normalized coords to absolute
        img = cv2.imread(str(image_path))
        if img is None:
            return annotations
        img_height, img_width = img.shape[:2]

        try:
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        class_name = id_to_name.get(class_id, f"class_{class_id}")

                        if len(parts) == 5:
                            # Detection: class x_center y_center width height
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height

                            # Convert to x, y, w, h (top-left corner)
                            x = x_center - width / 2
                            y = y_center - height / 2

                            annotations.append({
                                "class_name": class_name,
                                "class_id": class_id,
                                "bbox": (x, y, width, height),
                                "polygon": None,
                            })
                        else:
                            # Segmentation: class x1 y1 x2 y2 ... xn yn
                            coords = [float(p) for p in parts[1:]]
                            polygon = []
                            for i in range(0, len(coords), 2):
                                px = coords[i] * img_width
                                py = coords[i + 1] * img_height
                                polygon.append((px, py))

                            # Calculate bounding box from polygon
                            xs = [p[0] for p in polygon]
                            ys = [p[1] for p in polygon]
                            x = min(xs)
                            y = min(ys)
                            width = max(xs) - x
                            height = max(ys) - y

                            annotations.append({
                                "class_name": class_name,
                                "class_id": class_id,
                                "bbox": (x, y, width, height),
                                "polygon": polygon,
                            })

                    except (ValueError, IndexError):
                        continue

        except OSError:
            pass

        return annotations
