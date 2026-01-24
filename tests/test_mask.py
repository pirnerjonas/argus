"""Tests for MaskDataset functionality."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.core import MaskDataset
from argus.core.base import DatasetFormat, TaskType

runner = CliRunner()


class TestMaskDatasetDetection:
    """Tests for MaskDataset.detect() method."""

    def test_detect_grayscale_dataset(self, mask_dataset_grayscale: Path) -> None:
        """Verify detection of images/masks directory pattern."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)

        assert dataset is not None
        assert dataset.format == DatasetFormat.MASK
        assert dataset.task == TaskType.SEGMENTATION
        assert dataset.images_dir == "images"
        assert dataset.masks_dir == "masks"
        assert set(dataset.splits) == {"train", "val"}

    def test_detect_cityscapes_pattern(self, mask_dataset_cityscapes: Path) -> None:
        """Verify detection of leftImg8bit/gtFine directory pattern."""
        dataset = MaskDataset.detect(mask_dataset_cityscapes)

        assert dataset is not None
        assert dataset.format == DatasetFormat.MASK
        assert dataset.images_dir == "leftImg8bit"
        assert dataset.masks_dir == "gtFine"
        assert "train" in dataset.splits

    def test_detect_unsplit_dataset(self, mask_dataset_unsplit: Path) -> None:
        """Verify detection of flat structure without splits."""
        dataset = MaskDataset.detect(mask_dataset_unsplit)

        assert dataset is not None
        assert dataset.format == DatasetFormat.MASK
        assert dataset.splits == []

    def test_detect_with_config(self, mask_dataset_rgb: Path) -> None:
        """Verify detection loads classes.yaml configuration."""
        dataset = MaskDataset.detect(mask_dataset_rgb)

        assert dataset is not None
        assert dataset.class_config is not None
        assert "background" in dataset.class_names
        assert "person" in dataset.class_names
        assert "car" in dataset.class_names

    def test_detect_returns_none_for_invalid_path(self, tmp_path: Path) -> None:
        """Verify returns None for non-matching directory structures."""
        # Empty directory
        empty = tmp_path / "empty"
        empty.mkdir()
        assert MaskDataset.detect(empty) is None

        # Random files
        random = tmp_path / "random"
        random.mkdir()
        (random / "file.txt").write_text("hello")
        assert MaskDataset.detect(random) is None

    def test_detect_yolo_dataset_not_detected_as_mask(
        self, yolo_detection_dataset: Path
    ) -> None:
        """Verify YOLO datasets are not detected as mask datasets."""
        dataset = MaskDataset.detect(yolo_detection_dataset)
        assert dataset is None

    def test_detect_coco_dataset_not_detected_as_mask(
        self, coco_detection_dataset: Path
    ) -> None:
        """Verify COCO datasets are not detected as mask datasets."""
        dataset = MaskDataset.detect(coco_detection_dataset)
        assert dataset is None


class TestStemMatching:
    """Tests for image-to-mask filename matching."""

    def test_get_mask_path_finds_matching_mask(
        self, mask_dataset_grayscale: Path
    ) -> None:
        """Test image-to-mask stem matching works correctly."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        image_path = mask_dataset_grayscale / "images" / "train" / "img001.jpg"
        mask_path = dataset.get_mask_path(image_path)

        assert mask_path is not None
        assert mask_path.exists()
        assert mask_path.stem == "img001"
        assert mask_path.suffix == ".png"

    def test_get_mask_path_returns_none_for_missing(
        self, mask_dataset_missing_mask: Path
    ) -> None:
        """Test returns None when no matching mask exists."""
        dataset = MaskDataset.detect(mask_dataset_missing_mask)
        assert dataset is not None

        # img002 has no mask
        image_path = mask_dataset_missing_mask / "images" / "train" / "img002.jpg"
        mask_path = dataset.get_mask_path(image_path)

        assert mask_path is None


class TestGrayscaleClassDiscovery:
    """Tests for auto-detection of class IDs from grayscale masks."""

    def test_auto_detect_classes_from_masks(self, mask_dataset_grayscale: Path) -> None:
        """Test class IDs are discovered from mask pixel values."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        class_mapping = dataset.get_class_mapping()

        # Should have discovered classes 0, 1, 2 (from mask values)
        assert 0 in class_mapping
        assert 1 in class_mapping
        assert 2 in class_mapping

        # Auto-generated names
        assert class_mapping[0] == "class_0"
        assert class_mapping[1] == "class_1"
        assert class_mapping[2] == "class_2"

    def test_config_overrides_auto_detection(self, mask_dataset_rgb: Path) -> None:
        """Test classes.yaml names override auto-detection."""
        dataset = MaskDataset.detect(mask_dataset_rgb)
        assert dataset is not None

        class_mapping = dataset.get_class_mapping()

        # Should use names from config
        assert class_mapping[0] == "background"
        assert class_mapping[1] == "person"
        assert class_mapping[2] == "car"


class TestIgnoreIndexHandling:
    """Tests for ignore index (void/unlabeled pixel) handling."""

    def test_default_ignore_index_is_255(self, mask_dataset_grayscale: Path) -> None:
        """Test default ignore index is 255."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None
        assert dataset.ignore_index == 255

    def test_custom_ignore_index_from_config(self, mask_dataset_rgb: Path) -> None:
        """Test ignore index can be configured in classes.yaml."""
        dataset = MaskDataset.detect(mask_dataset_rgb)
        assert dataset is not None
        assert dataset.ignore_index == 255  # Set in config

    def test_ignored_pixels_excluded_from_class_mapping(
        self, mask_dataset_grayscale: Path
    ) -> None:
        """Test ignore index not included in class mapping."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        class_mapping = dataset.get_class_mapping()

        # 255 should not be in the mapping
        assert 255 not in class_mapping


class TestMissingMaskWarning:
    """Tests for missing mask handling."""

    def test_missing_mask_excluded_from_image_paths(
        self, mask_dataset_missing_mask: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test images without masks are excluded and warning is logged."""
        dataset = MaskDataset.detect(mask_dataset_missing_mask)
        assert dataset is not None

        # Get image paths - should only include images with masks
        image_paths = dataset.get_image_paths("train")

        # Only img001 has a mask
        assert len(image_paths) == 1
        assert image_paths[0].name == "img001.jpg"


class TestDimensionMismatchError:
    """Tests for dimension validation between image and mask."""

    def test_dimension_mismatch_detected(
        self, mask_dataset_dimension_mismatch: Path
    ) -> None:
        """Test dimension mismatch is detected by validate_dimensions."""
        dataset = MaskDataset.detect(mask_dataset_dimension_mismatch)
        assert dataset is not None

        image_path = mask_dataset_dimension_mismatch / "images" / "train" / "img001.jpg"
        match, img_shape, mask_shape = dataset.validate_dimensions(image_path)

        assert match is False
        assert img_shape == (100, 100)
        assert mask_shape == (50, 50)


class TestPixelCounts:
    """Tests for pixel counting methods."""

    def test_get_pixel_counts(self, mask_dataset_grayscale: Path) -> None:
        """Test pixel counts are correctly calculated."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        pixel_counts = dataset.get_pixel_counts()

        # Class 0 (background) should have most pixels
        # Class 1 has 20x20 region in train, 20x20 in val
        # Class 2 has 20x20 region in train only
        assert 0 in pixel_counts
        assert 1 in pixel_counts
        assert 2 in pixel_counts
        assert pixel_counts[1] == 400 + 400  # 20x20 * 2
        assert pixel_counts[2] == 400  # 20x20 * 1

    def test_get_instance_counts(self, mask_dataset_grayscale: Path) -> None:
        """Test instance counts (pixel counts per split)."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        counts = dataset.get_instance_counts()

        assert "train" in counts
        assert "val" in counts

        # Each split should have class counts
        assert "class_1" in counts["train"]
        assert "class_2" in counts["train"]


class TestImageClassPresence:
    """Tests for image-level class presence counting."""

    def test_get_image_class_presence(self, mask_dataset_grayscale: Path) -> None:
        """Test counting images containing each class."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        presence = dataset.get_image_class_presence()

        # Class 0 (background) present in all 2 images
        # Class 1 present in both images
        # Class 2 present in 1 image (train only)
        assert presence[0] == 2
        assert presence[1] == 2
        assert presence[2] == 1


class TestMaskLoading:
    """Tests for mask loading functionality."""

    def test_load_mask(self, mask_dataset_grayscale: Path) -> None:
        """Test mask loading returns correct numpy array."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        image_path = mask_dataset_grayscale / "images" / "train" / "img001.jpg"
        mask = dataset.load_mask(image_path)

        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8

    def test_load_mask_returns_none_for_missing(
        self, mask_dataset_missing_mask: Path
    ) -> None:
        """Test load_mask returns None when mask doesn't exist."""
        dataset = MaskDataset.detect(mask_dataset_missing_mask)
        assert dataset is not None

        image_path = mask_dataset_missing_mask / "images" / "train" / "img002.jpg"
        mask = dataset.load_mask(image_path)

        assert mask is None


class TestDetectionPriority:
    """Tests for dataset format detection priority."""

    def test_yolo_detected_before_mask(
        self, yolo_detection_dataset: Path, tmp_path: Path
    ) -> None:
        """Test YOLO is detected before MaskDataset when both patterns match."""
        from argus.cli import _detect_dataset

        # YOLO dataset should be detected as YOLO, not Mask
        dataset = _detect_dataset(yolo_detection_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.YOLO

    def test_coco_detected_before_mask(self, coco_detection_dataset: Path) -> None:
        """Test COCO is detected before MaskDataset when both patterns match."""
        from argus.cli import _detect_dataset

        dataset = _detect_dataset(coco_detection_dataset)

        assert dataset is not None
        assert dataset.format == DatasetFormat.COCO


class TestStatsCommand:
    """Integration tests for stats command with mask datasets."""

    def test_stats_mask_dataset(self, mask_dataset_grayscale: Path) -> None:
        """Test stats command works with mask datasets."""
        result = runner.invoke(app, ["stats", "-d", str(mask_dataset_grayscale)])

        assert result.exit_code == 0
        assert "Class Statistics" in result.stdout
        assert "Total Pixels" in result.stdout
        assert "% Coverage" in result.stdout
        assert "Images With" in result.stdout
        assert "MASK" in result.stdout
        assert "segmentation" in result.stdout

    def test_stats_shows_pixel_counts(self, mask_dataset_grayscale: Path) -> None:
        """Test stats shows pixel count information."""
        result = runner.invoke(app, ["stats", "-d", str(mask_dataset_grayscale)])

        assert result.exit_code == 0
        # Should show class names
        assert "class_0" in result.stdout
        assert "class_1" in result.stdout


class TestListCommand:
    """Integration tests for list command with mask datasets."""

    def test_list_discovers_mask_dataset(self, mask_dataset_grayscale: Path) -> None:
        """Test list command discovers mask datasets."""
        result = runner.invoke(app, ["list", "-p", str(mask_dataset_grayscale.parent)])

        assert result.exit_code == 0
        # Dataset should be found
        assert "Found 1 dataset" in result.stdout
        # Path should be in output (may be truncated)
        assert "mask_gray" in result.stdout


class TestAbstractMethods:
    """Tests for abstract method implementations."""

    def test_get_annotations_for_image_returns_empty(
        self, mask_dataset_grayscale: Path
    ) -> None:
        """Test get_annotations_for_image returns empty list for masks."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        image_path = mask_dataset_grayscale / "images" / "train" / "img001.jpg"
        annotations = dataset.get_annotations_for_image(image_path)

        # Mask datasets don't have discrete annotations
        assert annotations == []

    def test_get_image_counts(self, mask_dataset_grayscale: Path) -> None:
        """Test get_image_counts returns correct counts."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        counts = dataset.get_image_counts()

        assert "train" in counts
        assert "val" in counts
        assert counts["train"]["total"] == 1
        assert counts["val"]["total"] == 1
        # Background always 0 for mask datasets
        assert counts["train"]["background"] == 0

    def test_summary(self, mask_dataset_grayscale: Path) -> None:
        """Test summary method returns correct info."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        summary = dataset.summary()

        assert summary["format"] == "mask"
        assert summary["task"] == "segmentation"
        assert "train" in summary["splits"]
        assert "val" in summary["splits"]
