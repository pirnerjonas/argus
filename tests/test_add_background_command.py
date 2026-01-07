"""Tests for the add-background command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.core import COCODataset, YOLODataset
from argus.core.background import (
    _assign_images_to_splits,
    _collect_source_images,
    add_background_to_coco,
    add_background_to_yolo,
)

runner = CliRunner()


@pytest.fixture
def background_images(tmp_path: Path) -> Path:
    """Create a directory with fake background images."""
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # Create 10 fake image files
    for i in range(10):
        (bg_dir / f"bg_{i:03d}.jpg").write_bytes(b"fake image data")

    return bg_dir


@pytest.fixture
def single_background_image(tmp_path: Path) -> Path:
    """Create a single fake background image."""
    img_path = tmp_path / "single_bg.png"
    img_path.write_bytes(b"fake image data")
    return img_path


@pytest.fixture
def yolo_split_dataset(tmp_path: Path) -> Path:
    """Create a YOLO dataset with train/val splits."""
    dataset_path = tmp_path / "yolo_split"
    dataset_path.mkdir()

    yaml_content = """
names:
  0: object
"""
    (dataset_path / "data.yaml").write_text(yaml_content)

    for split in ["train", "val"]:
        (dataset_path / "images" / split).mkdir(parents=True)
        (dataset_path / "labels" / split).mkdir(parents=True)
        # Add one existing image per split
        (dataset_path / "images" / split / f"existing_{split}.jpg").write_bytes(
            b"fake"
        )
        (dataset_path / "labels" / split / f"existing_{split}.txt").write_text(
            "0 0.5 0.5 0.2 0.3\n"
        )

    return dataset_path


@pytest.fixture
def yolo_unsplit_dataset(tmp_path: Path) -> Path:
    """Create a YOLO dataset without splits (flat structure)."""
    dataset_path = tmp_path / "yolo_unsplit"
    dataset_path.mkdir()

    yaml_content = """
names:
  0: object
"""
    (dataset_path / "data.yaml").write_text(yaml_content)

    (dataset_path / "images").mkdir()
    (dataset_path / "labels").mkdir()

    return dataset_path


@pytest.fixture
def coco_split_dataset(tmp_path: Path) -> Path:
    """Create a COCO dataset with train/val splits."""
    dataset_path = tmp_path / "coco_split"
    dataset_path.mkdir()

    annotations_dir = dataset_path / "annotations"
    annotations_dir.mkdir()

    for split in ["train", "val"]:
        images_dir = dataset_path / "images" / split
        images_dir.mkdir(parents=True)

        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": f"existing_{split}.jpg",
                    "width": 100,
                    "height": 100,
                }
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]}
            ],
            "categories": [{"id": 1, "name": "object"}],
        }
        (annotations_dir / f"instances_{split}.json").write_text(json.dumps(coco_data))
        (images_dir / f"existing_{split}.jpg").write_bytes(b"fake")

    return dataset_path


class TestCollectSourceImages:
    """Tests for _collect_source_images helper."""

    def test_collect_single_image(self, single_background_image: Path) -> None:
        """Test collecting a single image file."""
        images = _collect_source_images(single_background_image)
        assert len(images) == 1
        assert images[0] == single_background_image

    def test_collect_directory(self, background_images: Path) -> None:
        """Test collecting images from a directory."""
        images = _collect_source_images(background_images)
        assert len(images) == 10

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        """Test that nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            _collect_source_images(tmp_path / "nonexistent")

    def test_invalid_extension_raises(self, tmp_path: Path) -> None:
        """Test that non-image file raises ValueError."""
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("not an image")
        with pytest.raises(ValueError, match="not a supported image format"):
            _collect_source_images(txt_file)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """Test that empty directory raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No valid image files found"):
            _collect_source_images(empty_dir)


class TestAssignImagesToSplits:
    """Tests for _assign_images_to_splits helper."""

    def test_distribute_to_train_val_test(self, background_images: Path) -> None:
        """Test distribution across all three splits."""
        images = list(background_images.glob("*.jpg"))
        assignments = _assign_images_to_splits(
            images, ["train", "val", "test"], (0.8, 0.1, 0.1), seed=42
        )

        total = sum(len(imgs) for imgs in assignments.values())
        assert total == len(images)
        assert "train" in assignments
        assert len(assignments["train"]) == 8  # 80% of 10

    def test_distribute_to_train_val_only(self, background_images: Path) -> None:
        """Test distribution when only train/val exist."""
        images = list(background_images.glob("*.jpg"))
        assignments = _assign_images_to_splits(
            images, ["train", "val"], (0.8, 0.1, 0.1), seed=42
        )

        total = sum(len(imgs) for imgs in assignments.values())
        assert total == len(images)
        # Ratios should be renormalized: 0.8/(0.8+0.1) = ~0.89, 0.1/(0.8+0.1) = ~0.11
        assert "train" in assignments
        assert "val" in assignments
        assert "test" not in assignments

    def test_unsplit_dataset(self, background_images: Path) -> None:
        """Test all images go to unsplit when no splits available."""
        images = list(background_images.glob("*.jpg"))
        assignments = _assign_images_to_splits(images, [], (0.8, 0.1, 0.1), seed=42)

        assert "unsplit" in assignments
        assert len(assignments["unsplit"]) == len(images)

    def test_seed_reproducibility(self, background_images: Path) -> None:
        """Test that same seed produces same assignments."""
        images = list(background_images.glob("*.jpg"))

        assignments1 = _assign_images_to_splits(
            images, ["train", "val", "test"], (0.8, 0.1, 0.1), seed=123
        )
        assignments2 = _assign_images_to_splits(
            images, ["train", "val", "test"], (0.8, 0.1, 0.1), seed=123
        )

        assert assignments1 == assignments2


class TestAddBackgroundToYolo:
    """Tests for add_background_to_yolo function."""

    def test_add_to_split_dataset(
        self, yolo_split_dataset: Path, background_images: Path
    ) -> None:
        """Test adding backgrounds to a YOLO dataset with splits."""
        dataset = YOLODataset.detect(yolo_split_dataset)
        assert dataset is not None

        counts = add_background_to_yolo(
            dataset, background_images, (0.8, 0.1, 0.1), seed=42
        )

        total_added = sum(counts.values())
        assert total_added == 10

        # Check files were created
        for split in ["train", "val"]:
            image_dir = yolo_split_dataset / "images" / split
            label_dir = yolo_split_dataset / "labels" / split

            # Count background images (excluding existing)
            bg_images = [f for f in image_dir.glob("bg_*.jpg")]
            bg_labels = [f for f in label_dir.glob("bg_*.txt")]

            assert len(bg_images) == len(bg_labels)

            # Verify label files are empty (background)
            for label_file in bg_labels:
                assert label_file.read_text() == ""

    def test_add_to_unsplit_dataset(
        self, yolo_unsplit_dataset: Path, single_background_image: Path
    ) -> None:
        """Test adding background to unsplit YOLO dataset."""
        dataset = YOLODataset.detect(yolo_unsplit_dataset)
        assert dataset is not None

        counts = add_background_to_yolo(
            dataset, single_background_image, (0.8, 0.1, 0.1), seed=42
        )

        assert counts.get("unsplit", 0) == 1

        # Check files in root images/labels
        assert (yolo_unsplit_dataset / "images" / "single_bg.png").exists()
        assert (yolo_unsplit_dataset / "labels" / "single_bg.txt").exists()
        assert (yolo_unsplit_dataset / "labels" / "single_bg.txt").read_text() == ""

    def test_duplicate_detection(
        self, yolo_unsplit_dataset: Path, single_background_image: Path
    ) -> None:
        """Test that duplicates are detected and skipped."""
        dataset = YOLODataset.detect(yolo_unsplit_dataset)
        assert dataset is not None

        # Add once
        add_background_to_yolo(
            dataset, single_background_image, (0.8, 0.1, 0.1), seed=42
        )

        # Adding again should raise about duplicates
        with pytest.raises(ValueError, match="duplicate"):
            add_background_to_yolo(
                dataset, single_background_image, (0.8, 0.1, 0.1), seed=42
            )


class TestAddBackgroundToCoco:
    """Tests for add_background_to_coco function."""

    def test_add_to_split_dataset(
        self, coco_split_dataset: Path, background_images: Path
    ) -> None:
        """Test adding backgrounds to a COCO dataset with splits."""
        dataset = COCODataset.detect(coco_split_dataset)
        assert dataset is not None

        counts = add_background_to_coco(
            dataset, background_images, (0.8, 0.1, 0.1), seed=42
        )

        total_added = sum(counts.values())
        assert total_added == 10

        # Check annotation files were updated
        for split in ["train", "val"]:
            ann_file = coco_split_dataset / "annotations" / f"instances_{split}.json"
            with open(ann_file, encoding="utf-8") as f:
                data = json.load(f)

            # Should have more than 1 image now (1 existing + backgrounds)
            bg_images = [
                img for img in data["images"] if img["file_name"].startswith("bg_")
            ]

            # Background images should have no annotations
            bg_image_ids = {img["id"] for img in bg_images}
            bg_annotations = [
                ann for ann in data["annotations"] if ann["image_id"] in bg_image_ids
            ]
            assert len(bg_annotations) == 0


class TestAddBackgroundCommand:
    """Tests for the CLI add-background command."""

    def test_add_to_yolo_dataset(
        self, yolo_split_dataset: Path, background_images: Path
    ) -> None:
        """Test CLI command with YOLO dataset."""
        result = runner.invoke(
            app,
            [
                "add-background",
                "--dataset-path",
                str(yolo_split_dataset),
                "--source",
                str(background_images),
                "--ratio",
                "0.8,0.1,0.1",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        assert "Added" in result.stdout
        assert "background image" in result.stdout

    def test_add_to_coco_dataset(
        self, coco_split_dataset: Path, single_background_image: Path
    ) -> None:
        """Test CLI command with COCO dataset."""
        result = runner.invoke(
            app,
            [
                "add-background",
                "--dataset-path",
                str(coco_split_dataset),
                "--source",
                str(single_background_image),
            ],
        )

        assert result.exit_code == 0
        assert "Added" in result.stdout

    def test_invalid_source(self, yolo_split_dataset: Path, tmp_path: Path) -> None:
        """Test CLI with nonexistent source."""
        result = runner.invoke(
            app,
            [
                "add-background",
                "--dataset-path",
                str(yolo_split_dataset),
                "--source",
                str(tmp_path / "nonexistent"),
            ],
        )

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_invalid_dataset(
        self, tmp_path: Path, single_background_image: Path
    ) -> None:
        """Test CLI with invalid dataset path."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "add-background",
                "--dataset-path",
                str(empty_dir),
                "--source",
                str(single_background_image),
            ],
        )

        assert result.exit_code == 1
        assert "No YOLO or COCO dataset found" in result.stdout

    def test_invalid_ratio(
        self, yolo_split_dataset: Path, single_background_image: Path
    ) -> None:
        """Test CLI with invalid ratio."""
        result = runner.invoke(
            app,
            [
                "add-background",
                "--dataset-path",
                str(yolo_split_dataset),
                "--source",
                str(single_background_image),
                "--ratio",
                "invalid",
            ],
        )

        assert result.exit_code == 1
