"""Base dataset class for all dataset formats."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DatasetFormat(str, Enum):
    """Supported dataset formats."""

    YOLO = "yolo"
    COCO = "coco"
    MASK = "mask"


class TaskType(str, Enum):
    """Supported task types."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


@dataclass
class Dataset(ABC):
    """Base class for all dataset formats.

    Attributes:
        path: Root path of the dataset.
        format: Dataset format (yolo, coco).
        task: Task type (detection, segmentation, classification).
        num_classes: Number of classes in the dataset.
        class_names: List of class names.
        splits: Available splits (train, val, test).
    """

    path: Path
    format: DatasetFormat
    task: TaskType
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)

    @classmethod
    @abstractmethod
    def detect(cls, path: Path) -> "Dataset | None":
        """Detect if the given path contains a dataset of this format.

        Args:
            path: Directory path to check for dataset.

        Returns:
            Dataset instance if detected, None otherwise.
        """
        pass

    @abstractmethod
    def get_instance_counts(self) -> dict[str, dict[str, int]]:
        """Get the number of annotation instances per class, per split.

        Returns:
            Dictionary mapping split name to dict of class name to instance count.
            Example: {"train": {"person": 100, "car": 50}, "val": {"person": 20}}
        """
        pass

    @abstractmethod
    def get_image_counts(self) -> dict[str, dict[str, int]]:
        """Get image counts per split, including background images.

        Returns:
            Dictionary mapping split name to dict with "total" and "background" counts.
            Example: {"train": {"total": 100, "background": 10}, ...}
        """
        pass

    @abstractmethod
    def get_image_paths(self, split: str | None = None) -> list[Path]:
        """Get all image file paths for a split or the entire dataset.

        Args:
            split: Specific split to get images from. If None, returns all images.

        Returns:
            List of image file paths.
        """
        pass

    @abstractmethod
    def get_annotations_for_image(self, image_path: Path) -> list[dict]:
        """Get annotations for a specific image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of annotation dicts with keys:
            - "class_name": str - name of the class
            - "class_id": int - class ID
            - "bbox": tuple[float, float, float, float] | None - (x, y, w, h) absolute
            - "polygon": list[tuple[float, float]] | None - list of (x, y) points
        """
        pass

    def summary(self) -> dict:
        """Return a summary dict for table rendering.

        Returns:
            Dictionary with dataset information.
        """
        return {
            "path": str(self.path),
            "format": self.format.value,
            "task": self.task.value,
            "classes": self.num_classes,
            "splits": ", ".join(self.splits) if self.splits else "unsplit",
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"path={self.path}, "
            f"format={self.format.value}, "
            f"task={self.task.value}, "
            f"classes={self.num_classes})"
        )
