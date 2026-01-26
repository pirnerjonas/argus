"""Dataset validation using DINOv2 embeddings for outlier detection."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from argus.core.base import Dataset, TaskType


class AIFeaturesNotAvailable(ImportError):
    """Raised when AI features are requested but dependencies are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "AI features require additional dependencies.\n"
            "Install with: pip install argus-cv[ai]"
        )


def _check_ai_dependencies() -> None:
    """Check if AI dependencies are available."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise AIFeaturesNotAvailable() from e


@dataclass
class OutlierSample:
    """Represents an outlier sample detected during validation."""

    path: Path
    class_name: str
    distance: float  # Distance from centroid in standard deviations


@dataclass
class ClassValidationResult:
    """Validation result for a single class."""

    class_name: str
    total_samples: int
    outliers: list[OutlierSample] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result for a dataset."""

    dataset_path: Path
    task_type: TaskType
    num_classes: int
    total_images: int
    threshold: float
    split: str | None
    class_results: list[ClassValidationResult] = field(default_factory=list)

    @property
    def total_outliers(self) -> int:
        """Total number of outliers across all classes."""
        return sum(len(cr.outliers) for cr in self.class_results)

    @property
    def classes_with_outliers(self) -> int:
        """Number of classes that have at least one outlier."""
        return sum(1 for cr in self.class_results if cr.outliers)


class DatasetValidator:
    """Validates classification datasets using DINOv2 embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            model_name: HuggingFace model name for DINOv2.
            batch_size: Batch size for embedding computation.
            device: Device to run model on (cuda/cpu). Auto-detected if None.
        """
        _check_ai_dependencies()

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _compute_embeddings(self, image_paths: list[Path]) -> np.ndarray:
        """Compute DINOv2 embeddings for a list of images.

        Args:
            image_paths: List of image file paths.

        Returns:
            Numpy array of embeddings with shape (n_images, embedding_dim).
        """
        import torch
        from PIL import Image

        embeddings = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception:
                    # Skip images that can't be loaded
                    continue

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # DINOv2: use CLS token (first token of last hidden state)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                # Normalize embeddings
                cls_embeddings = cls_embeddings / cls_embeddings.norm(
                    dim=-1, keepdim=True
                )
                embeddings.append(cls_embeddings.cpu().numpy())

        if not embeddings:
            return np.array([])

        return np.vstack(embeddings)

    def _detect_outliers(
        self,
        embeddings: np.ndarray,
        image_paths: list[Path],
        class_name: str,
        threshold: float,
    ) -> list[OutlierSample]:
        """Detect outliers using distance from centroid.

        Args:
            embeddings: Embeddings for all images in the class.
            image_paths: Corresponding image paths.
            class_name: Name of the class.
            threshold: Number of standard deviations for outlier detection.

        Returns:
            List of OutlierSample objects for detected outliers.
        """
        if len(embeddings) < 2:
            # Need at least 2 samples to compute statistics
            return []

        # Compute centroid (mean embedding)
        centroid = embeddings.mean(axis=0)

        # Compute distances from centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)

        # Compute statistics
        mean_dist = distances.mean()
        std_dist = distances.std()

        if std_dist == 0:
            # All samples are identical, no outliers
            return []

        # Compute z-scores (distance in standard deviations)
        z_scores = (distances - mean_dist) / std_dist

        # Find outliers
        outliers = []
        for idx, z_score in enumerate(z_scores):
            if z_score > threshold:
                outliers.append(
                    OutlierSample(
                        path=image_paths[idx],
                        class_name=class_name,
                        distance=float(z_score),
                    )
                )

        # Sort by distance (most anomalous first)
        outliers.sort(key=lambda x: x.distance, reverse=True)
        return outliers

    def validate(
        self,
        dataset: Dataset,
        threshold: float = 2.0,
        split: str | None = None,
    ) -> ValidationResult:
        """Validate a classification dataset.

        Args:
            dataset: Dataset to validate.
            threshold: Number of standard deviations for outlier detection.
            split: Specific split to validate. If None, validates all splits.

        Returns:
            ValidationResult with outlier information.

        Raises:
            ValueError: If dataset is not a classification dataset.
        """
        if dataset.task != TaskType.CLASSIFICATION:
            raise ValueError(
                f"Validation only supports classification datasets. "
                f"Got: {dataset.task.value}"
            )

        # Get images by class
        from argus.core.yolo import YOLODataset

        if not isinstance(dataset, YOLODataset):
            raise ValueError("Validation currently only supports YOLO datasets.")

        images_by_class = dataset.get_images_by_class(split)

        # Count total images
        total_images = sum(len(imgs) for imgs in images_by_class.values())

        # Process each class
        class_results = []

        for class_name, image_paths in images_by_class.items():
            if not image_paths:
                class_results.append(
                    ClassValidationResult(
                        class_name=class_name,
                        total_samples=0,
                        outliers=[],
                    )
                )
                continue

            # Compute embeddings for this class
            embeddings = self._compute_embeddings(image_paths)

            # Detect outliers
            outliers = self._detect_outliers(
                embeddings, image_paths, class_name, threshold
            )

            class_results.append(
                ClassValidationResult(
                    class_name=class_name,
                    total_samples=len(image_paths),
                    outliers=outliers,
                )
            )

        return ValidationResult(
            dataset_path=dataset.path,
            task_type=dataset.task,
            num_classes=dataset.num_classes,
            total_images=total_images,
            threshold=threshold,
            split=split,
            class_results=class_results,
        )


def validate_dataset(
    dataset: Dataset,
    threshold: float = 2.0,
    split: str | None = None,
    batch_size: int = 32,
) -> ValidationResult:
    """Convenience function to validate a dataset.

    Args:
        dataset: Dataset to validate.
        threshold: Number of standard deviations for outlier detection.
        split: Specific split to validate. If None, validates all splits.
        batch_size: Batch size for embedding computation.

    Returns:
        ValidationResult with outlier information.
    """
    validator = DatasetValidator(batch_size=batch_size)
    return validator.validate(dataset, threshold=threshold, split=split)
