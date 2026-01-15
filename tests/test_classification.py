"""Tests for YOLO classification dataset support."""

from pathlib import Path

import pytest

from argus.core.base import DatasetFormat, TaskType
from argus.core.yolo import YOLODataset


class TestClassificationDetection:
    """Tests for detecting YOLO classification datasets."""

    def test_detect_classification_dataset(
        self, yolo_classification_dataset: Path
    ) -> None:
        """Test that classification dataset is correctly detected."""
        dataset = YOLODataset.detect(yolo_classification_dataset)

        assert dataset is not None
        assert dataset.task == TaskType.CLASSIFICATION
        assert dataset.format == DatasetFormat.YOLO
        assert set(dataset.class_names) == {"cat", "dog"}
        assert dataset.num_classes == 2
        assert set(dataset.splits) == {"train", "val"}

    def test_detect_classification_multiclass(
        self, yolo_classification_multiclass_dataset: Path
    ) -> None:
        """Test detection with multiple classes."""
        dataset = YOLODataset.detect(yolo_classification_multiclass_dataset)

        assert dataset is not None
        assert dataset.task == TaskType.CLASSIFICATION
        assert set(dataset.class_names) == {"apple", "banana", "cherry", "date"}
        assert dataset.num_classes == 4
        assert "train" in dataset.splits

    def test_classification_no_yaml_required(
        self, yolo_classification_dataset: Path
    ) -> None:
        """Test that classification datasets don't need a YAML config."""
        dataset = YOLODataset.detect(yolo_classification_dataset)

        assert dataset is not None
        assert dataset.config_file is None

    def test_classification_not_detected_for_detection(
        self, yolo_detection_dataset: Path
    ) -> None:
        """Test that detection datasets are not classified as classification."""
        dataset = YOLODataset.detect(yolo_detection_dataset)

        assert dataset is not None
        assert dataset.task != TaskType.CLASSIFICATION
        assert dataset.task == TaskType.DETECTION


class TestClassificationGetImagesByClass:
    """Tests for get_images_by_class method."""

    def test_get_images_by_class(self, yolo_classification_dataset: Path) -> None:
        """Test getting images grouped by class."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        images_by_class = dataset.get_images_by_class("train")

        assert "cat" in images_by_class
        assert "dog" in images_by_class
        assert len(images_by_class["cat"]) == 2
        assert len(images_by_class["dog"]) == 1

    def test_get_images_by_class_val_split(
        self, yolo_classification_dataset: Path
    ) -> None:
        """Test getting images from val split."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        images_by_class = dataset.get_images_by_class("val")

        assert len(images_by_class["cat"]) == 1
        assert len(images_by_class["dog"]) == 1

    def test_get_images_by_class_default_split(
        self, yolo_classification_dataset: Path
    ) -> None:
        """Test that None split uses first available split."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        images_by_class = dataset.get_images_by_class(None)

        # Should return images from first split (train)
        total_images = sum(len(imgs) for imgs in images_by_class.values())
        assert total_images > 0

    def test_get_images_by_class_non_classification(
        self, yolo_detection_dataset: Path
    ) -> None:
        """Test that non-classification datasets return empty dict."""
        dataset = YOLODataset.detect(yolo_detection_dataset)
        assert dataset is not None

        images_by_class = dataset.get_images_by_class("train")

        assert images_by_class == {}


class TestClassificationInstanceCounts:
    """Tests for instance counts in classification datasets."""

    def test_get_instance_counts(self, yolo_classification_dataset: Path) -> None:
        """Test counting images per class per split."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["cat"] == 2
        assert counts["train"]["dog"] == 1
        assert counts["val"]["cat"] == 1
        assert counts["val"]["dog"] == 1

    def test_get_image_counts(self, yolo_classification_dataset: Path) -> None:
        """Test total image counts per split."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        counts = dataset.get_image_counts()

        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["total"] == 3  # 2 cat + 1 dog
        assert counts["train"]["background"] == 0
        assert counts["val"]["total"] == 2  # 1 cat + 1 dog
        assert counts["val"]["background"] == 0


class TestClassificationImagePaths:
    """Tests for get_image_paths with classification datasets."""

    def test_get_image_paths_all(self, yolo_classification_dataset: Path) -> None:
        """Test getting all image paths."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        paths = dataset.get_image_paths()

        assert len(paths) == 5  # 3 train + 2 val

    def test_get_image_paths_train(self, yolo_classification_dataset: Path) -> None:
        """Test getting image paths for train split."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        paths = dataset.get_image_paths("train")

        assert len(paths) == 3

    def test_get_image_paths_val(self, yolo_classification_dataset: Path) -> None:
        """Test getting image paths for val split."""
        dataset = YOLODataset.detect(yolo_classification_dataset)
        assert dataset is not None

        paths = dataset.get_image_paths("val")

        assert len(paths) == 2


class TestClassificationCLI:
    """Tests for CLI integration with classification datasets."""

    def test_list_shows_classification(
        self, yolo_classification_dataset: Path
    ) -> None:
        """Test that list command shows classification datasets."""
        from typer.testing import CliRunner

        from argus.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["list", "-p", str(yolo_classification_dataset)])

        assert result.exit_code == 0
        assert "classification" in result.output.lower()
        assert "yolo" in result.output.lower()

    def test_stats_shows_classification_counts(
        self, yolo_classification_dataset: Path
    ) -> None:
        """Test that stats command shows image counts per class."""
        from typer.testing import CliRunner

        from argus.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["stats", "-d", str(yolo_classification_dataset)]
        )

        assert result.exit_code == 0
        assert "cat" in result.output
        assert "dog" in result.output
