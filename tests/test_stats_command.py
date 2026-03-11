"""Tests for the stats command."""

import json
import os
from pathlib import Path

from click.termui import strip_ansi
from typer.testing import CliRunner

from argus.cli import app
from argus.core import COCODataset, YOLODataset

# Set terminal width to prevent Rich from truncating output in CI
os.environ["COLUMNS"] = "200"
runner = CliRunner()


class TestCOCOImageCounts:
    """Tests for COCO dataset image counting."""

    def test_get_image_counts_with_background(self, tmp_path: Path) -> None:
        """Test counting images with background-only images in COCO."""
        dataset_path = tmp_path / "coco_with_bg"
        dataset_path.mkdir()

        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
                {"id": 3, "file_name": "img3.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 50]},
                # img3 has no annotations - background
            ],
            "categories": [{"id": 1, "name": "object"}],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None

        counts = dataset.get_image_counts()

        assert counts["train"]["total"] == 3
        assert counts["train"]["background"] == 1


class TestCOCOInstanceCounts:
    """Tests for COCO dataset instance counting."""

    def test_get_instance_counts_detection(self, coco_detection_dataset: Path) -> None:
        """Test counting instances in a COCO detection dataset."""
        dataset = COCODataset.detect(coco_detection_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        # Expected: train has 2 annotations (person: 1, car: 1)
        assert "train" in counts
        assert counts["train"]["person"] == 1
        assert counts["train"]["car"] == 1

    def test_get_instance_counts_segmentation(
        self, coco_segmentation_dataset: Path
    ) -> None:
        """Test counting instances in a COCO segmentation dataset."""
        dataset = COCODataset.detect(coco_segmentation_dataset)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        # Expected: val has 1 annotation (cat: 1)
        assert "val" in counts
        assert counts["val"]["cat"] == 1

    def test_get_instance_counts_multiple_splits(self, tmp_path: Path) -> None:
        """Test counting instances across multiple splits."""
        dataset_path = tmp_path / "coco_multi"
        dataset_path.mkdir()

        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        # Create train annotations
        train_data = {
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 50, 50]},
                {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0, 50, 50, 50]},
            ],
            "categories": [
                {"id": 1, "name": "dog"},
                {"id": 2, "name": "cat"},
            ],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(train_data))

        # Create val annotations
        val_data = {
            "images": [{"id": 1, "file_name": "img2.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 50, 50]},
            ],
            "categories": [
                {"id": 1, "name": "dog"},
                {"id": 2, "name": "cat"},
            ],
        }
        (annotations_dir / "instances_val.json").write_text(json.dumps(val_data))

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        assert counts["train"]["dog"] == 2
        assert counts["train"]["cat"] == 1
        assert counts["val"]["cat"] == 1
        assert counts["val"].get("dog", 0) == 0


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
