"""Tests for the stats command."""

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.core import COCODataset, YOLODataset

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
        result = runner.invoke(
            app, ["stats", "--dataset-path", str(yolo_detection_dataset)]
        )

        assert result.exit_code == 0
        assert "person" in result.stdout
        assert "car" in result.stdout
        assert "bicycle" in result.stdout
        assert "train" in result.stdout
        assert "val" in result.stdout
        assert "Total" in result.stdout

    def test_stats_coco_dataset(self, coco_detection_dataset: Path) -> None:
        """Test stats command with a COCO dataset."""
        result = runner.invoke(
            app, ["stats", "--dataset-path", str(coco_detection_dataset)]
        )

        assert result.exit_code == 0
        assert "person" in result.stdout
        assert "car" in result.stdout
        assert "train" in result.stdout
        assert "Total" in result.stdout

    def test_stats_nonexistent_path(self, tmp_path: Path) -> None:
        """Test stats command with non-existent path."""
        nonexistent = tmp_path / "nonexistent"
        result = runner.invoke(app, ["stats", "--dataset-path", str(nonexistent)])

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_stats_invalid_dataset(self, empty_directory: Path) -> None:
        """Test stats command with invalid dataset path."""
        result = runner.invoke(app, ["stats", "--dataset-path", str(empty_directory)])

        assert result.exit_code == 1
        assert "No dataset found" in result.stdout

    def test_stats_short_option(self, yolo_detection_dataset: Path) -> None:
        """Test stats command with short -d option."""
        result = runner.invoke(app, ["stats", "-d", str(yolo_detection_dataset)])

        assert result.exit_code == 0
        assert "person" in result.stdout

    def test_stats_shows_format_and_task(self, yolo_detection_dataset: Path) -> None:
        """Test that stats shows dataset format and task type."""
        result = runner.invoke(app, ["stats", "-d", str(yolo_detection_dataset)])

        assert result.exit_code == 0
        assert "YOLO" in result.stdout
        assert "detection" in result.stdout
