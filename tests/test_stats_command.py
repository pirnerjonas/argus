"""Tests for the stats command."""

import os
from pathlib import Path

from click.termui import strip_ansi
from typer.testing import CliRunner

from argus.cli import app
from argus.core import YOLODataset

# Set terminal width to prevent Rich from truncating output in CI
os.environ["COLUMNS"] = "200"
runner = CliRunner()


class TestYOLOInstanceCounts:
    """Tests for YOLO dataset instance counting."""

    def test_get_instance_counts_detection(self, yolo_detection_dataset: Path) -> None:
        """Test counting instances in a YOLO detection dataset."""
        dataset = YOLODataset.detect(yolo_detection_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        # Expected: train has 2 annotations (class 0, class 1), val has 1 (class 2)
        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["person"] == 1  # class 0
        assert counts["train"]["car"] == 1  # class 1
        assert counts["val"]["bicycle"] == 1  # class 2

    def test_get_instance_counts_segmentation(
        self, yolo_segmentation_dataset: Path
    ) -> None:
        """Test counting instances in a YOLO segmentation dataset."""
        dataset = YOLODataset.detect(yolo_segmentation_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        # Expected: train has 1 annotation (class 0), val has 1 (class 1)
        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["cat"] == 1  # class 0
        assert counts["val"]["dog"] == 1  # class 1

    def test_get_instance_counts_unsplit(
        self, yolo_flat_structure_dataset: Path
    ) -> None:
        """Test counting instances in an unsplit YOLO dataset."""
        dataset = YOLODataset.detect(yolo_flat_structure_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        # Expected: unsplit has 2 annotations (class 0 and class 1)
        assert "unsplit" in counts
        assert counts["unsplit"]["object1"] == 1  # class 0
        assert counts["unsplit"]["object2"] == 1  # class 1


class TestYOLOImageCounts:
    """Tests for YOLO dataset image counting."""

    def test_get_image_counts_detection(self, yolo_detection_dataset: Path) -> None:
        """Test counting images in a YOLO detection dataset."""
        dataset = YOLODataset.detect(yolo_detection_dataset)
        assert dataset is not None

        counts = dataset.get_image_counts()

        # Expected: train has 1 image, val has 1 image, no background
        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["total"] == 1
        assert counts["train"]["background"] == 0
        assert counts["val"]["total"] == 1
        assert counts["val"]["background"] == 0

    def test_get_image_counts_with_background(self, tmp_path: Path) -> None:
        """Test counting images with background-only annotations."""
        dataset_path = tmp_path / "yolo_with_bg"
        dataset_path.mkdir()

        yaml_content = """
names:
  0: object
"""
        (dataset_path / "data.yaml").write_text(yaml_content)

        (dataset_path / "images" / "train").mkdir(parents=True)
        (dataset_path / "labels" / "train").mkdir(parents=True)

        # Create 3 images: 2 with annotations, 1 empty (background)
        (dataset_path / "images" / "train" / "img1.jpg").write_bytes(b"fake")
        (dataset_path / "images" / "train" / "img2.jpg").write_bytes(b"fake")
        (dataset_path / "images" / "train" / "img3.jpg").write_bytes(b"fake")

        label1 = "0 0.5 0.5 0.2 0.3\n"
        label2 = "0 0.3 0.3 0.1 0.1\n"
        (dataset_path / "labels" / "train" / "img1.txt").write_text(label1)
        (dataset_path / "labels" / "train" / "img2.txt").write_text(label2)
        (dataset_path / "labels" / "train" / "img3.txt").write_text("")  # background

        dataset = YOLODataset.detect(dataset_path)
        assert dataset is not None

        counts = dataset.get_image_counts()

        assert counts["train"]["total"] == 3
        assert counts["train"]["background"] == 1


class TestStatsCommand:
    """Tests for the stats CLI command."""

    def test_stats_yolo_dataset(self, yolo_detection_dataset: Path) -> None:
        """Test stats command with a YOLO dataset."""
        result = runner.invoke(app, ["stats", str(yolo_detection_dataset)])

        assert result.exit_code == 0
        assert "person" in result.stdout
        assert "car" in result.stdout
        assert "bicycle" in result.stdout
        assert "train" in result.stdout
        assert "val" in result.stdout
        assert "Total" in result.stdout

    def test_stats_coco_dataset(self, coco_detection_dataset: Path) -> None:
        """Test stats command with a COCO dataset."""
        result = runner.invoke(app, ["stats", str(coco_detection_dataset)])

        assert result.exit_code == 0
        assert "person" in result.stdout
        assert "car" in result.stdout
        assert "train" in result.stdout
        assert "Total" in result.stdout

    def test_stats_nonexistent_path(self, tmp_path: Path) -> None:
        """Test stats command with non-existent path."""
        nonexistent = tmp_path / "nonexistent"
        result = runner.invoke(app, ["stats", str(nonexistent)])

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_stats_invalid_dataset(self, empty_directory: Path) -> None:
        """Test stats command with invalid dataset path."""
        result = runner.invoke(app, ["stats", str(empty_directory)])

        assert result.exit_code == 1
        assert "No dataset found" in result.stdout

    def test_stats_shows_format_and_task(self, yolo_detection_dataset: Path) -> None:
        """Test that stats shows dataset format and task type."""
        result = runner.invoke(app, ["stats", str(yolo_detection_dataset)])

        assert result.exit_code == 0
        assert "YOLO" in result.stdout
        assert "detection" in result.stdout

    def test_stats_rejects_removed_dataset_option(
        self, yolo_detection_dataset: Path
    ) -> None:
        """Test stats command no longer accepts --dataset-path."""
        result = runner.invoke(
            app,
            ["stats", "--dataset-path", str(yolo_detection_dataset)],
        )
        help_result = runner.invoke(app, ["stats", "--help"])

        assert result.exit_code == 2
        assert help_result.exit_code == 0
        assert "--dataset-path" not in strip_ansi(help_result.output)


class TestRoboflowYOLO:
    """Tests for Roboflow YOLO layout detection and stats."""

    def test_detect_roboflow_layout(self, roboflow_yolo_dataset: Path) -> None:
        """Test that Roboflow YOLO layout is detected correctly."""
        dataset = YOLODataset.detect(roboflow_yolo_dataset)
        assert dataset is not None
        assert dataset._roboflow_layout is True
        assert "train" in dataset.splits
        assert "val" in dataset.splits
        assert "test" in dataset.splits
        assert dataset.class_names == ["ball", "player", "referee"]

    def test_get_instance_counts_roboflow(self, roboflow_yolo_dataset: Path) -> None:
        """Test instance counting with Roboflow layout."""
        dataset = YOLODataset.detect(roboflow_yolo_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["ball"] == 2
        assert counts["train"]["player"] == 1
        assert counts["val"]["player"] == 1
        assert counts["val"]["referee"] == 1

    def test_get_image_counts_roboflow(self, roboflow_yolo_dataset: Path) -> None:
        """Test image counting with Roboflow layout."""
        dataset = YOLODataset.detect(roboflow_yolo_dataset)
        assert dataset is not None

        counts = dataset.get_image_counts()

        assert "train" in counts
        assert counts["train"]["total"] == 2
        assert counts["train"]["background"] == 0
        assert "val" in counts
        assert counts["val"]["total"] == 1
        assert counts["val"]["background"] == 0
        assert "test" in counts
        assert counts["test"]["total"] == 1
        assert counts["test"]["background"] == 1

    def test_get_image_paths_roboflow(self, roboflow_yolo_dataset: Path) -> None:
        """Test getting image paths with Roboflow layout."""
        dataset = YOLODataset.detect(roboflow_yolo_dataset)
        assert dataset is not None

        # All images
        all_paths = dataset.get_image_paths()
        assert len(all_paths) == 4

        # Single split
        train_paths = dataset.get_image_paths(split="train")
        assert len(train_paths) == 2

        val_paths = dataset.get_image_paths(split="val")
        assert len(val_paths) == 1

    def test_stats_command_roboflow(self, roboflow_yolo_dataset: Path) -> None:
        """Test stats CLI command with Roboflow YOLO dataset."""
        result = runner.invoke(app, ["stats", str(roboflow_yolo_dataset)])

        assert result.exit_code == 0
        assert "ball" in result.stdout
        assert "player" in result.stdout
        assert "YOLO" in result.stdout
