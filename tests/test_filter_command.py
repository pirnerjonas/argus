"""Tests for dataset filtering utilities and CLI command."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from typer.testing import CliRunner

from argus.cli import app
from argus.core import COCODataset, MaskDataset, YOLODataset
from argus.core.filter import (
    filter_coco_dataset,
    filter_mask_dataset,
    filter_yolo_dataset,
)

runner = CliRunner()


def _create_yolo_detection_dataset(dataset_path: Path) -> None:
    """Create a YOLO detection dataset for testing."""
    dataset_path.mkdir(parents=True)
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "val").mkdir(parents=True)

    (dataset_path / "data.yaml").write_text(
        "\n".join(
            [
                "names:",
                "  0: ball",
                "  1: player",
                "  2: referee",
            ]
        )
    )

    # Create images (fake) and labels
    for idx in range(1, 5):
        (dataset_path / "images" / "train" / f"img{idx:03d}.jpg").write_bytes(
            b"fake image"
        )
        # Mix of classes in labels
        if idx == 1:
            # ball and player
            label = "0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.4\n"
        elif idx == 2:
            # only ball
            label = "0 0.6 0.6 0.15 0.15\n"
        elif idx == 3:
            # only player and referee
            label = "1 0.4 0.4 0.2 0.3\n2 0.7 0.7 0.1 0.1\n"
        else:
            # empty (background)
            label = ""
        (dataset_path / "labels" / "train" / f"img{idx:03d}.txt").write_text(label)

    # Val split
    for idx in range(1, 3):
        (dataset_path / "images" / "val" / f"img{idx:03d}.jpg").write_bytes(
            b"fake image"
        )
        if idx == 1:
            label = "0 0.5 0.5 0.1 0.1\n"
        else:
            label = "1 0.3 0.3 0.2 0.2\n2 0.6 0.6 0.1 0.1\n"
        (dataset_path / "labels" / "val" / f"img{idx:03d}.txt").write_text(label)


def _create_yolo_segmentation_dataset(dataset_path: Path) -> None:
    """Create a YOLO segmentation dataset for testing."""
    dataset_path.mkdir(parents=True)
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)

    (dataset_path / "data.yaml").write_text(
        "\n".join(
            [
                "names:",
                "  0: cat",
                "  1: dog",
            ]
        )
    )

    # Segmentation labels have polygon coordinates (>5 values)
    (dataset_path / "images" / "train" / "img001.jpg").write_bytes(b"fake")
    (dataset_path / "labels" / "train" / "img001.txt").write_text(
        "0 0.1 0.2 0.3 0.4 0.5 0.6\n1 0.2 0.3 0.4 0.5 0.6 0.7\n"
    )

    (dataset_path / "images" / "train" / "img002.jpg").write_bytes(b"fake")
    (dataset_path / "labels" / "train" / "img002.txt").write_text(
        "0 0.15 0.25 0.35 0.45 0.55 0.65\n"
    )


def _create_coco_dataset(dataset_path: Path) -> None:
    """Create a COCO dataset for testing."""
    dataset_path.mkdir(parents=True)
    (dataset_path / "annotations").mkdir()
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)

    # Train annotations
    train_coco = {
        "info": {"description": "Test"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img003.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 60, 60]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [150, 150, 40, 40]},
            {"id": 4, "image_id": 3, "category_id": 2, "bbox": [50, 50, 30, 30]},
        ],
        "categories": [
            {"id": 1, "name": "ball", "supercategory": "object"},
            {"id": 2, "name": "player", "supercategory": "object"},
        ],
    }
    (dataset_path / "annotations" / "instances_train.json").write_text(
        json.dumps(train_coco)
    )

    # Val annotations (one background image)
    val_coco = {
        "info": {"description": "Test"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "img004.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img005.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]},
            # img005 has no annotations
        ],
        "categories": [
            {"id": 1, "name": "ball", "supercategory": "object"},
            {"id": 2, "name": "player", "supercategory": "object"},
        ],
    }
    (dataset_path / "annotations" / "instances_val.json").write_text(
        json.dumps(val_coco)
    )

    # Create images
    for i in range(1, 4):
        (dataset_path / "images" / "train" / f"img{i:03d}.jpg").write_bytes(b"fake")
    for i in range(4, 6):
        (dataset_path / "images" / "val" / f"img{i:03d}.jpg").write_bytes(b"fake")


def _create_mask_dataset(dataset_path: Path) -> None:
    """Create a mask dataset for testing."""
    dataset_path.mkdir(parents=True)
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "masks" / "train").mkdir(parents=True)

    # Create classes.yaml
    (dataset_path / "classes.yaml").write_text(
        "\n".join(
            [
                "names:",
                "  0: background",
                "  1: person",
                "  2: car",
                "ignore_index: 255",
            ]
        )
    )

    # Create images and masks
    for i in range(1, 4):
        # Create real images (required for mask filtering)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_path / "images" / "train" / f"img{i:03d}.jpg"), img)

        # Create masks
        mask = np.zeros((100, 100), dtype=np.uint8)
        if i == 1:
            mask[20:40, 20:40] = 1  # person
            mask[60:80, 60:80] = 2  # car
        elif i == 2:
            mask[30:50, 30:50] = 1  # person only
        else:
            mask[40:60, 40:60] = 2  # car only
        cv2.imwrite(str(dataset_path / "masks" / "train" / f"img{i:03d}.png"), mask)


# ============================================================================
# YOLO Detection/Segmentation Filter Tests
# ============================================================================


def test_filter_yolo_detection_single_class(tmp_path: Path) -> None:
    """Test filtering YOLO detection dataset to single class."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    dataset = YOLODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    stats = filter_yolo_dataset(dataset, output_path, classes=["ball"])

    # Should have 4 train + 2 val images
    assert stats["images"] == 6

    # Check output structure
    assert (output_path / "data.yaml").exists()
    assert (output_path / "images" / "train").is_dir()
    assert (output_path / "labels" / "train").is_dir()

    # Check data.yaml has only 'ball' class
    with open(output_path / "data.yaml") as f:
        config = yaml.safe_load(f)
    assert config["names"] == {0: "ball"}

    # Check filtered labels only contain class 0 (remapped ball)
    label_file = output_path / "labels" / "train" / "img001.txt"
    content = label_file.read_text().strip()
    assert content == "0 0.5 0.5 0.1 0.1"  # Only ball, not player


def test_filter_yolo_detection_multiple_classes(tmp_path: Path) -> None:
    """Test filtering YOLO detection dataset to multiple classes."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    dataset = YOLODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    filter_yolo_dataset(dataset, output_path, classes=["ball", "referee"])

    # Check data.yaml has both classes remapped to 0 and 1
    with open(output_path / "data.yaml") as f:
        config = yaml.safe_load(f)
    assert config["names"] == {0: "ball", 1: "referee"}

    # Check label file with both classes
    label_file = output_path / "labels" / "train" / "img003.txt"
    content = label_file.read_text().strip()
    # Only referee should remain (class 2 -> 1), player (class 1) filtered out
    assert content == "1 0.7 0.7 0.1 0.1"


def test_filter_yolo_detection_no_background(tmp_path: Path) -> None:
    """Test filtering YOLO detection with --no-background flag."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    dataset = YOLODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    # Filter to 'ball' only, which means img003 and img004 become background
    stats = filter_yolo_dataset(
        dataset, output_path, classes=["ball"], no_background=True
    )

    # img003 has only player+referee, img004 is empty -> both skipped
    # img001, img002, val/img001 have ball; img003, img004, val/img002 skipped
    assert stats["skipped"] == 3
    assert stats["images"] == 3


def test_filter_yolo_detection_symlinks(tmp_path: Path) -> None:
    """Test filtering YOLO detection with symlinks."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    dataset = YOLODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    filter_yolo_dataset(dataset, output_path, classes=["ball"], use_symlinks=True)

    # Check that images are symlinks
    img_file = output_path / "images" / "train" / "img001.jpg"
    assert img_file.is_symlink()


def test_filter_yolo_segmentation(tmp_path: Path) -> None:
    """Test filtering YOLO segmentation dataset."""
    dataset_path = tmp_path / "yolo_seg"
    _create_yolo_segmentation_dataset(dataset_path)

    dataset = YOLODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    stats = filter_yolo_dataset(dataset, output_path, classes=["cat"])

    assert stats["images"] == 2

    # Check filtered label maintains polygon format
    label_file = output_path / "labels" / "train" / "img001.txt"
    content = label_file.read_text().strip()
    # Only cat (0), not dog
    assert content == "0 0.1 0.2 0.3 0.4 0.5 0.6"


# ============================================================================
# YOLO Classification Filter Tests
# ============================================================================


def test_filter_yolo_classification(
    tmp_path: Path, yolo_classification_dataset: Path
) -> None:
    """Test filtering YOLO classification dataset."""
    dataset = YOLODataset.detect(yolo_classification_dataset)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    stats = filter_yolo_dataset(dataset, output_path, classes=["cat"])

    # Check only cat class exists
    assert (output_path / "images" / "train" / "cat").is_dir()
    assert not (output_path / "images" / "train" / "dog").exists()

    # Check images were copied
    assert stats["images"] > 0


# ============================================================================
# COCO Filter Tests
# ============================================================================


def test_filter_coco_single_class(tmp_path: Path) -> None:
    """Test filtering COCO dataset to single class."""
    dataset_path = tmp_path / "coco"
    _create_coco_dataset(dataset_path)

    dataset = COCODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    stats = filter_coco_dataset(dataset, output_path, classes=["ball"])

    # Check annotations were filtered
    assert stats["images"] > 0
    assert stats["annotations"] > 0

    # Check output annotation file
    train_ann = output_path / "annotations" / "instances_train.json"
    assert train_ann.exists()

    with open(train_ann) as f:
        data = json.load(f)

    # Only ball category should exist
    assert len(data["categories"]) == 1
    assert data["categories"][0]["name"] == "ball"
    assert data["categories"][0]["id"] == 1  # Remapped to 1


def test_filter_coco_no_background(tmp_path: Path) -> None:
    """Test filtering COCO dataset with --no-background flag."""
    dataset_path = tmp_path / "coco"
    _create_coco_dataset(dataset_path)

    dataset = COCODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    # Filter to 'ball' which makes img003 (only player) and img005 background
    stats = filter_coco_dataset(
        dataset, output_path, classes=["ball"], no_background=True
    )

    assert stats["skipped"] > 0

    # Check val annotation file - img005 should be excluded
    val_ann = output_path / "annotations" / "instances_val.json"
    with open(val_ann) as f:
        data = json.load(f)
    # Only img004 should remain (it has ball annotation)
    assert len(data["images"]) == 1


def test_filter_coco_symlinks(tmp_path: Path) -> None:
    """Test filtering COCO dataset with symlinks."""
    dataset_path = tmp_path / "coco"
    _create_coco_dataset(dataset_path)

    dataset = COCODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    filter_coco_dataset(dataset, output_path, classes=["ball"], use_symlinks=True)

    # Check that images are symlinks
    img_file = output_path / "images" / "train" / "img001.jpg"
    assert img_file.is_symlink()


# ============================================================================
# Mask Filter Tests
# ============================================================================


def test_filter_mask_single_class(tmp_path: Path) -> None:
    """Test filtering mask dataset to single class."""
    dataset_path = tmp_path / "mask"
    _create_mask_dataset(dataset_path)

    dataset = MaskDataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    stats = filter_mask_dataset(dataset, output_path, classes=["person"])

    assert stats["images"] == 3
    assert stats["masks"] == 3

    # Check classes.yaml
    with open(output_path / "classes.yaml") as f:
        config = yaml.safe_load(f)
    assert config["names"] == {0: "person"}

    # Check mask values were remapped
    mask = cv2.imread(
        str(output_path / "masks" / "train" / "img001.png"), cv2.IMREAD_GRAYSCALE
    )
    unique_values = np.unique(mask)
    # Should only have 0 (person remapped) and 255 (ignore)
    assert set(unique_values) <= {0, 255}


def test_filter_mask_no_background(tmp_path: Path) -> None:
    """Test filtering mask dataset with --no-background flag."""
    dataset_path = tmp_path / "mask"
    _create_mask_dataset(dataset_path)

    dataset = MaskDataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    # Filter to 'person', which makes img003 (only car) background
    stats = filter_mask_dataset(
        dataset, output_path, classes=["person"], no_background=True
    )

    # img003 has only car, should be skipped
    assert stats["skipped"] == 1
    assert stats["images"] == 2


def test_filter_mask_multiple_classes(tmp_path: Path) -> None:
    """Test filtering mask dataset to multiple classes."""
    dataset_path = tmp_path / "mask"
    _create_mask_dataset(dataset_path)

    dataset = MaskDataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "filtered"
    filter_mask_dataset(dataset, output_path, classes=["person", "car"])

    # Check classes.yaml has both
    with open(output_path / "classes.yaml") as f:
        config = yaml.safe_load(f)
    assert config["names"] == {0: "person", 1: "car"}


# ============================================================================
# CLI Filter Command Tests
# ============================================================================


def test_filter_command_yolo(tmp_path: Path) -> None:
    """Test filter CLI command with YOLO dataset."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    output_path = tmp_path / "cli_filtered"
    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--classes",
            "ball",
        ],
    )

    assert result.exit_code == 0
    assert "Filtering complete" in result.stdout
    assert (output_path / "data.yaml").exists()


def test_filter_command_coco(tmp_path: Path) -> None:
    """Test filter CLI command with COCO dataset."""
    dataset_path = tmp_path / "coco"
    _create_coco_dataset(dataset_path)

    output_path = tmp_path / "cli_filtered"
    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--classes",
            "ball,player",
        ],
    )

    assert result.exit_code == 0
    assert "Filtering complete" in result.stdout


def test_filter_command_mask(tmp_path: Path) -> None:
    """Test filter CLI command with mask dataset."""
    dataset_path = tmp_path / "mask"
    _create_mask_dataset(dataset_path)

    output_path = tmp_path / "cli_filtered"
    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--classes",
            "person",
        ],
    )

    assert result.exit_code == 0
    assert "Filtering complete" in result.stdout


def test_filter_command_no_background(tmp_path: Path) -> None:
    """Test filter CLI command with --no-background flag."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    output_path = tmp_path / "cli_filtered"
    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--classes",
            "ball",
            "--no-background",
        ],
    )

    assert result.exit_code == 0
    assert "Skipped" in result.stdout


def test_filter_command_symlinks(tmp_path: Path) -> None:
    """Test filter CLI command with --symlinks flag."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    output_path = tmp_path / "cli_filtered"
    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--classes",
            "ball",
            "--symlinks",
        ],
    )

    assert result.exit_code == 0


def test_filter_command_no_classes_error(tmp_path: Path) -> None:
    """Test filter CLI command fails without --classes."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
        ],
    )

    assert result.exit_code == 1
    assert "No classes specified" in result.stdout


def test_filter_command_invalid_class_error(tmp_path: Path) -> None:
    """Test filter CLI command fails with invalid class name."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--classes",
            "nonexistent",
        ],
    )

    assert result.exit_code == 1
    assert "Classes not found" in result.stdout


def test_filter_command_output_exists_error(tmp_path: Path) -> None:
    """Test filter CLI command fails if output directory exists and not empty."""
    dataset_path = tmp_path / "yolo_det"
    _create_yolo_detection_dataset(dataset_path)

    output_path = tmp_path / "existing"
    output_path.mkdir()
    (output_path / "somefile.txt").write_text("existing")

    result = runner.invoke(
        app,
        [
            "filter",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
            "--classes",
            "ball",
        ],
    )

    assert result.exit_code == 1
    assert "already exists" in result.stdout
