"""Tests for dataset detection and list command."""

from pathlib import Path

from typer.testing import CliRunner

from argus.cli import _discover_datasets, app
from argus.core import COCODataset, YOLODataset
from argus.core.base import DatasetFormat, TaskType

runner = CliRunner()


class TestYOLODatasetDetection:
    """Tests for YOLO dataset detection."""

    def test_detect_yolo_detection_dataset(self, yolo_detection_dataset: Path):
        """Test detection of valid YOLO detection dataset."""
        dataset = YOLODataset.detect(yolo_detection_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.YOLO
        assert dataset.task == TaskType.DETECTION
        assert dataset.num_classes == 3
        assert dataset.class_names == ["person", "car", "bicycle"]
        assert "train" in dataset.splits
        assert "val" in dataset.splits

    def test_detect_yolo_segmentation_dataset(self, yolo_segmentation_dataset: Path):
        """Test detection of valid YOLO segmentation dataset."""
        dataset = YOLODataset.detect(yolo_segmentation_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.YOLO
        assert dataset.task == TaskType.SEGMENTATION
        assert dataset.num_classes == 2
        assert dataset.class_names == ["cat", "dog"]
        assert "train" in dataset.splits
        assert "val" in dataset.splits

    def test_detect_yolo_flat_structure(self, yolo_flat_structure_dataset: Path):
        """Test detection of YOLO dataset with flat structure."""
        dataset = YOLODataset.detect(yolo_flat_structure_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.YOLO
        assert dataset.task == TaskType.DETECTION
        assert dataset.num_classes == 2

    def test_detect_invalid_yaml_missing_names(self, invalid_yaml_missing_names: Path):
        """Test that YAML without 'names' is not detected as YOLO."""
        dataset = YOLODataset.detect(invalid_yaml_missing_names)
        assert dataset is None

    def test_detect_empty_directory(self, empty_directory: Path):
        """Test that empty directory is not detected as YOLO."""
        dataset = YOLODataset.detect(empty_directory)
        assert dataset is None

    def test_detect_random_files(self, random_files_directory: Path):
        """Test that random files are not detected as YOLO."""
        dataset = YOLODataset.detect(random_files_directory)
        assert dataset is None

    def test_detect_nonexistent_path(self, tmp_path: Path):
        """Test that nonexistent path returns None."""
        dataset = YOLODataset.detect(tmp_path / "nonexistent")
        assert dataset is None


class TestCOCODatasetDetection:
    """Tests for COCO dataset detection."""

    def test_detect_coco_detection_dataset(self, coco_detection_dataset: Path):
        """Test detection of valid COCO detection dataset."""
        dataset = COCODataset.detect(coco_detection_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.COCO
        assert dataset.task == TaskType.DETECTION
        assert dataset.num_classes == 2
        assert dataset.class_names == ["person", "car"]
        assert "train" in dataset.splits

    def test_detect_coco_segmentation_dataset(self, coco_segmentation_dataset: Path):
        """Test detection of valid COCO segmentation dataset."""
        dataset = COCODataset.detect(coco_segmentation_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.COCO
        assert dataset.task == TaskType.SEGMENTATION
        assert dataset.num_classes == 3
        assert "val" in dataset.splits

    def test_detect_invalid_coco_missing_categories(
        self, invalid_coco_missing_categories: Path
    ):
        """Test that JSON without 'categories' is not detected as COCO."""
        dataset = COCODataset.detect(invalid_coco_missing_categories)
        assert dataset is None

    def test_detect_empty_directory(self, empty_directory: Path):
        """Test that empty directory is not detected as COCO."""
        dataset = COCODataset.detect(empty_directory)
        assert dataset is None

    def test_detect_random_files(self, random_files_directory: Path):
        """Test that random files are not detected as COCO."""
        dataset = COCODataset.detect(random_files_directory)
        assert dataset is None


class TestDiscoverDatasets:
    """Tests for dataset discovery function."""

    def test_discover_nested_datasets(self, nested_datasets: Path):
        """Test discovering multiple datasets in nested structure."""
        datasets = _discover_datasets(nested_datasets, max_depth=3)

        assert len(datasets) == 2

        formats = {d.format for d in datasets}
        assert DatasetFormat.YOLO in formats
        assert DatasetFormat.COCO in formats

    def test_discover_with_max_depth_1(self, nested_datasets: Path):
        """Test that max_depth limits discovery."""
        datasets = _discover_datasets(nested_datasets, max_depth=1)

        # Should only find YOLO at depth 1, not COCO at depth 2
        assert len(datasets) == 1
        assert datasets[0].format == DatasetFormat.YOLO

    def test_discover_empty_directory(self, empty_directory: Path):
        """Test discovering in empty directory returns empty list."""
        datasets = _discover_datasets(empty_directory, max_depth=3)
        assert datasets == []

    def test_discover_single_dataset(self, yolo_detection_dataset: Path):
        """Test discovering single dataset."""
        datasets = _discover_datasets(yolo_detection_dataset, max_depth=1)

        assert len(datasets) == 1
        assert datasets[0].format == DatasetFormat.YOLO
        assert datasets[0].task == TaskType.DETECTION


class TestListCommand:
    """Tests for the list CLI command."""

    def test_list_command_finds_yolo_dataset(self, yolo_detection_dataset: Path):
        """Test list command finds YOLO dataset."""
        result = runner.invoke(app, ["list", "--path", str(yolo_detection_dataset)])

        assert result.exit_code == 0
        assert "yolo" in result.stdout.lower()
        assert "detection" in result.stdout.lower()
        assert "3" in result.stdout  # num_classes

    def test_list_command_finds_coco_dataset(self, coco_detection_dataset: Path):
        """Test list command finds COCO dataset."""
        result = runner.invoke(app, ["list", "--path", str(coco_detection_dataset)])

        assert result.exit_code == 0
        assert "coco" in result.stdout.lower()
        assert "detection" in result.stdout.lower()

    def test_list_command_finds_multiple_datasets(self, nested_datasets: Path):
        """Test list command finds multiple nested datasets."""
        result = runner.invoke(app, ["list", "--path", str(nested_datasets)])

        assert result.exit_code == 0
        assert "Found 2 dataset(s)" in result.stdout

    def test_list_command_respects_max_depth(self, nested_datasets: Path):
        """Test list command respects --max-depth option."""
        result = runner.invoke(
            app, ["list", "--path", str(nested_datasets), "--max-depth", "1"]
        )

        assert result.exit_code == 0
        assert "Found 1 dataset(s)" in result.stdout

    def test_list_command_empty_directory(self, empty_directory: Path):
        """Test list command on empty directory."""
        result = runner.invoke(app, ["list", "--path", str(empty_directory)])

        assert result.exit_code == 0
        assert "No datasets found" in result.stdout

    def test_list_command_default_path(self, monkeypatch, yolo_detection_dataset: Path):
        """Test list command uses current directory by default."""
        monkeypatch.chdir(yolo_detection_dataset)
        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "yolo" in result.stdout.lower()

    def test_list_command_short_options(self, yolo_detection_dataset: Path):
        """Test list command with short option flags."""
        result = runner.invoke(
            app, ["list", "-p", str(yolo_detection_dataset), "-d", "2"]
        )

        assert result.exit_code == 0
        assert "yolo" in result.stdout.lower()

    def test_list_command_invalid_path(self, tmp_path: Path):
        """Test list command with nonexistent path."""
        result = runner.invoke(app, ["list", "--path", str(tmp_path / "nonexistent")])

        # Should fail with error about path not existing
        assert result.exit_code == 1
        assert "does not exist" in result.stdout


class TestDatasetSummary:
    """Tests for dataset summary method."""

    def test_yolo_dataset_summary(self, yolo_detection_dataset: Path):
        """Test YOLO dataset summary output."""
        dataset = YOLODataset.detect(yolo_detection_dataset)
        summary = dataset.summary()

        assert summary["format"] == "yolo"
        assert summary["task"] == "detection"
        assert summary["classes"] == 3
        assert "train" in summary["splits"]
        assert "val" in summary["splits"]

    def test_coco_dataset_summary(self, coco_detection_dataset: Path):
        """Test COCO dataset summary output."""
        dataset = COCODataset.detect(coco_detection_dataset)
        summary = dataset.summary()

        assert summary["format"] == "coco"
        assert summary["task"] == "detection"
        assert summary["classes"] == 2
        assert "train" in summary["splits"]
