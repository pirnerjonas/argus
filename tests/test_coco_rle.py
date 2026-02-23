"""Tests for COCO RLE segmentation support."""

import json
from pathlib import Path

import numpy as np

from argus.core.coco import COCODataset


class TestRLEDetection:
    """Test that has_rle flag is correctly set for various annotation types."""

    def test_rle_dataset_has_rle_true(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        assert dataset.has_rle is True

    def test_polygon_dataset_has_rle_false(
        self, coco_segmentation_dataset: Path
    ) -> None:
        dataset = COCODataset.detect(coco_segmentation_dataset)
        assert dataset is not None
        assert dataset.has_rle is False

    def test_detection_dataset_has_rle_false(
        self, coco_detection_dataset: Path
    ) -> None:
        dataset = COCODataset.detect(coco_detection_dataset)
        assert dataset is not None
        assert dataset.has_rle is False

    def test_mixed_dataset_has_rle_true(
        self, coco_mixed_rle_polygon_dataset: Path
    ) -> None:
        dataset = COCODataset.detect(coco_mixed_rle_polygon_dataset)
        assert dataset is not None
        assert dataset.has_rle is True


class TestRLEDecoding:
    """Test _decode_rle with known inputs."""

    def test_simple_rle(self) -> None:
        """Decode a small known RLE on a 4x4 image."""
        # 4x4 image, column-major
        # We want the first 2 pixels of column 0 to be fg:
        # counts: [0, 2, 14] -> 0 bg, 2 fg, 14 bg
        rle = {"counts": [0, 2, 14], "size": [4, 4]}
        mask = COCODataset._decode_rle(rle, 4, 4)

        assert mask.shape == (4, 4)
        assert mask[0, 0] == 1
        assert mask[1, 0] == 1
        assert mask[2, 0] == 0
        assert mask[3, 0] == 0
        assert mask.sum() == 2

    def test_empty_rle(self) -> None:
        """Empty counts list returns zero mask."""
        rle = {"counts": [], "size": [10, 10]}
        mask = COCODataset._decode_rle(rle, 10, 10)
        assert mask.shape == (10, 10)
        assert mask.sum() == 0

    def test_full_foreground(self) -> None:
        """All pixels are foreground."""
        # counts: [0, 25] -> 0 bg, 25 fg on a 5x5 image
        rle = {"counts": [0, 25], "size": [5, 5]}
        mask = COCODataset._decode_rle(rle, 5, 5)
        assert mask.shape == (5, 5)
        assert mask.sum() == 25
        assert (mask == 1).all()

    def test_column_major_order(self) -> None:
        """Verify column-major decoding order."""
        # 3x3 image, mark first pixel of each column (positions 0, 3, 6)
        # counts: [0, 1, 2, 1, 2, 1, 2] -> col0:row0, col1:row0, col2:row0
        rle = {"counts": [0, 1, 2, 1, 2, 1, 2], "size": [3, 3]}
        mask = COCODataset._decode_rle(rle, 3, 3)
        assert mask[0, 0] == 1  # row 0, col 0
        assert mask[0, 1] == 1  # row 0, col 1
        assert mask[0, 2] == 1  # row 0, col 2
        assert mask[1, 0] == 0
        assert mask[2, 0] == 0
        assert mask.sum() == 3

    def test_compressed_rle_simple(self) -> None:
        """Compressed RLE string decodes correctly."""
        # Encode a simple mask: 5x5, first 3 pixels foreground, rest bg
        # Uncompressed counts: [0, 3, 22]
        # In COCO compressed format:
        #   value 0 -> char '0' (48+0=48, no more bit)
        #   value 3 -> char '3' (48+3=51, no more bit)
        #   value 22 -> 22 in 5 bits = 10110 -> char 'V' (48+22=70, no more bit)
        # But first 2 values are direct, subsequent are differential (diff from [i-2])
        # Value at index 2: diff = 22 - 0 = 22
        # So encoded: chr(48+0) + chr(48+3) + chr(48+22) = '03V'
        rle = {"counts": "03V", "size": [5, 5]}
        mask = COCODataset._decode_rle(rle, 5, 5)
        assert mask.shape == (5, 5)
        assert mask.sum() == 3
        # Column-major: first 3 pixels of column 0
        assert mask[0, 0] == 1
        assert mask[1, 0] == 1
        assert mask[2, 0] == 1
        assert mask[3, 0] == 0

    def test_missing_counts_returns_empty(self) -> None:
        """Missing counts key returns empty mask."""
        rle = {"size": [10, 10]}
        mask = COCODataset._decode_rle(rle, 10, 10)
        assert mask.shape == (10, 10)
        assert mask.sum() == 0


class TestLoadMask:
    """Test load_mask returns correct masks."""

    def test_rle_mask_shape(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        image_paths = dataset.get_image_paths()
        assert len(image_paths) == 1

        mask = dataset.load_mask(image_paths[0])
        assert mask is not None
        assert mask.shape == (100, 100)

    def test_rle_mask_class_ids(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        image_paths = dataset.get_image_paths()

        mask = dataset.load_mask(image_paths[0])
        assert mask is not None

        # Category ID 1 should be present
        assert (mask == 1).any()
        # Background (0) should also be present
        assert (mask == 0).any()
        # No other values
        unique = set(np.unique(mask))
        assert unique == {0, 1}

    def test_rle_mask_correct_region(self, coco_rle_dataset: Path) -> None:
        """The 10x10 block at top-left should be category 1."""
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        image_paths = dataset.get_image_paths()

        mask = dataset.load_mask(image_paths[0])
        assert mask is not None

        # Top-left 10x10 should be category 1
        assert (mask[:10, :10] == 1).all()
        # Rest should be 0
        assert mask[10:, :].sum() == 0
        assert mask[:10, 10:].sum() == 0

    def test_background_is_zero(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        image_paths = dataset.get_image_paths()

        mask = dataset.load_mask(image_paths[0])
        assert mask is not None
        assert mask[50, 50] == 0

    def test_mixed_rle_polygon(self, coco_mixed_rle_polygon_dataset: Path) -> None:
        """Both RLE and polygon annotations rendered in same mask."""
        dataset = COCODataset.detect(coco_mixed_rle_polygon_dataset)
        assert dataset is not None
        image_paths = dataset.get_image_paths()

        mask = dataset.load_mask(image_paths[0])
        assert mask is not None
        assert mask.shape == (100, 100)

        # Category 1 (RLE) in top-left
        assert (mask[:10, :10] == 1).all()
        # Category 2 (polygon) in center-right area
        # The polygon is a rectangle from (50,50) to (70,70)
        # Check interior point
        assert mask[60, 60] == 2
        # Background
        assert mask[90, 90] == 0

        unique = set(np.unique(mask))
        assert unique == {0, 1, 2}

    def test_unknown_image_returns_none(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None

        result = dataset.load_mask(Path("/nonexistent/image.jpg"))
        assert result is None

    def test_empty_annotations_returns_zero_mask(self, tmp_path: Path) -> None:
        """Image with no annotations returns all-zero mask."""
        import cv2

        dataset_path = tmp_path / "coco_empty_ann"
        dataset_path.mkdir()
        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img001.jpg", "width": 50, "height": 50},
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "thing", "supercategory": "obj"},
            ],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

        images_dir = dataset_path / "images" / "train"
        images_dir.mkdir(parents=True)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img001.jpg"), img)

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None

        mask = dataset.load_mask(images_dir / "img001.jpg")
        assert mask is not None
        assert mask.shape == (50, 50)
        assert mask.sum() == 0

    def test_uint16_for_large_category_ids(self, tmp_path: Path) -> None:
        """Mask uses uint16 when category IDs exceed 255."""
        import cv2

        dataset_path = tmp_path / "coco_large_cat"
        dataset_path.mkdir()
        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img001.jpg", "width": 10, "height": 10},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 300,
                    "bbox": [0, 0, 5, 5],
                    "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]],
                    "area": 25,
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 300, "name": "rare_class", "supercategory": "obj"},
            ],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

        images_dir = dataset_path / "images" / "train"
        images_dir.mkdir(parents=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img001.jpg"), img)

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None

        mask = dataset.load_mask(images_dir / "img001.jpg")
        assert mask is not None
        assert mask.dtype == np.uint16
        assert (mask == 300).any()


class TestGetClassMapping:
    """Test get_class_mapping returns correct category mapping."""

    def test_basic_mapping(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None

        mapping = dataset.get_class_mapping()
        assert mapping == {1: "object"}

    def test_multiple_categories(self, coco_mixed_rle_polygon_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_mixed_rle_polygon_dataset)
        assert dataset is not None

        mapping = dataset.get_class_mapping()
        assert mapping == {1: "cat", 2: "dog"}

    def test_detection_dataset_mapping(self, coco_detection_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_detection_dataset)
        assert dataset is not None

        mapping = dataset.get_class_mapping()
        assert mapping == {1: "person", 2: "car"}


class TestIgnoreIndex:
    """Test ignore_index defaults for COCO datasets."""

    def test_ignore_index_is_zero(self, coco_rle_dataset: Path) -> None:
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        assert dataset.ignore_index == 0

    def test_ignore_index_default(self) -> None:
        """Default ignore_index is 0 for COCODataset."""
        dataset = COCODataset(
            path=Path("/tmp"),
            task="segmentation",
            num_classes=1,
            class_names=["obj"],
            splits=["train"],
            annotation_files=[],
        )
        assert dataset.ignore_index == 0


class TestCatZeroHandling:
    """Test handling of datasets where category ID 0 is a real class."""

    def test_has_cat_zero_detected(self, tmp_path: Path) -> None:
        """has_cat_zero is True when a category uses ID 0."""
        dataset_path = tmp_path / "cat_zero"
        dataset_path.mkdir()
        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 10, "height": 10},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [0, 0, 5, 5],
                    "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]],
                },
            ],
            "categories": [{"id": 0, "name": "thing"}],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None
        assert dataset.has_cat_zero is True
        assert dataset.ignore_index == 0

    def test_has_cat_zero_false(self, coco_rle_dataset: Path) -> None:
        """has_cat_zero is False when categories start at 1."""
        dataset = COCODataset.detect(coco_rle_dataset)
        assert dataset is not None
        assert dataset.has_cat_zero is False

    def test_cat_zero_class_mapping_shifted(self, tmp_path: Path) -> None:
        """Class mapping shifts IDs by +1 when cat 0 exists."""
        dataset_path = tmp_path / "cat_zero_map"
        dataset_path.mkdir()
        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 10, "height": 10},
            ],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "alpha"},
                {"id": 1, "name": "beta"},
            ],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None
        mapping = dataset.get_class_mapping()
        assert mapping == {1: "alpha", 2: "beta"}

    def test_cat_zero_mask_values_shifted(self, tmp_path: Path) -> None:
        """Mask pixel values are shifted by +1 when cat 0 exists."""
        import cv2

        dataset_path = tmp_path / "cat_zero_mask"
        dataset_path.mkdir()
        annotations_dir = dataset_path / "annotations"
        annotations_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img.jpg", "width": 10, "height": 10},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [0, 0, 5, 5],
                    "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]],
                },
            ],
            "categories": [{"id": 0, "name": "thing"}],
        }
        (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

        images_dir = dataset_path / "images" / "train"
        images_dir.mkdir(parents=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img.jpg"), img)

        dataset = COCODataset.detect(dataset_path)
        assert dataset is not None

        mask = dataset.load_mask(images_dir / "img.jpg")
        assert mask is not None
        # Category 0 should appear as 1 in the mask
        assert (mask == 1).any()
        # Background should be 0
        assert (mask == 0).any()
