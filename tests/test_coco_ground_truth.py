"""Cross-validation tests: argus COCO stats vs pycocotools ground truth."""

import pytest

pytest.importorskip("pycocotools", reason="pycocotools not installed")

from coco_ground_truth import get_ground_truth_stats  # noqa: E402

from argus.core.base import TaskType  # noqa: E402
from argus.core.coco import COCODataset  # noqa: E402

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# A. Task type
# ---------------------------------------------------------------------------
class TestTaskType:
    def test_detection_dataset_is_detection(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        assert ds.task == TaskType.DETECTION

    def test_segmentation_dataset_is_segmentation(self, coco_real_segmentation_dataset):
        ds = COCODataset.detect(coco_real_segmentation_dataset)
        assert ds is not None
        assert ds.task == TaskType.SEGMENTATION


# ---------------------------------------------------------------------------
# B. Class names
# ---------------------------------------------------------------------------
class TestClassNames:
    def test_detection_class_names(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        # Use train annotation as reference
        gt = get_ground_truth_stats(
            coco_real_detection_dataset / "annotations" / "instances_train.json"
        )
        assert ds.class_names == gt["category_names"]

    def test_segmentation_class_names(self, coco_real_segmentation_dataset):
        ds = COCODataset.detect(coco_real_segmentation_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_segmentation_dataset / "annotations" / "instances_train.json"
        )
        assert ds.class_names == gt["category_names"]


# ---------------------------------------------------------------------------
# C. Image counts
# ---------------------------------------------------------------------------
class TestImageCounts:
    def test_train_image_counts(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_detection_dataset / "annotations" / "instances_train.json"
        )
        counts = ds.get_image_counts()
        assert counts["train"]["total"] == gt["total"]
        assert counts["train"]["background"] == gt["background"]

    def test_val_image_counts(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_detection_dataset / "annotations" / "instances_val.json"
        )
        counts = ds.get_image_counts()
        assert counts["val"]["total"] == gt["total"]
        assert counts["val"]["background"] == gt["background"]

    def test_unsplit_image_counts(self, coco_real_unsplit_dataset):
        ds = COCODataset.detect(coco_real_unsplit_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_unsplit_dataset / "annotations" / "annotations.json"
        )
        counts = ds.get_image_counts()
        assert counts["unsplit"]["total"] == gt["total"]
        assert counts["unsplit"]["background"] == gt["background"]

    def test_roboflow_train_image_counts(self, coco_real_roboflow_dataset):
        ds = COCODataset.detect(coco_real_roboflow_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_roboflow_dataset / "train" / "_annotations.coco.json"
        )
        counts = ds.get_image_counts()
        assert counts["train"]["total"] == gt["total"]
        assert counts["train"]["background"] == gt["background"]

    def test_roboflow_val_image_counts(self, coco_real_roboflow_dataset):
        ds = COCODataset.detect(coco_real_roboflow_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_roboflow_dataset / "valid" / "_annotations.coco.json"
        )
        counts = ds.get_image_counts()
        # Roboflow "valid" maps to argus "val"
        assert counts["val"]["total"] == gt["total"]
        assert counts["val"]["background"] == gt["background"]

    def test_roboflow_test_image_counts(self, coco_real_roboflow_dataset):
        ds = COCODataset.detect(coco_real_roboflow_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_roboflow_dataset / "test" / "_annotations.coco.json"
        )
        counts = ds.get_image_counts()
        assert counts["test"]["total"] == gt["total"]
        assert counts["test"]["background"] == gt["background"]


# ---------------------------------------------------------------------------
# D. Instance counts
# ---------------------------------------------------------------------------
class TestInstanceCounts:
    def test_train_instance_counts(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_detection_dataset / "annotations" / "instances_train.json"
        )
        counts = ds.get_instance_counts()
        assert counts["train"] == gt["instance_counts"]

    def test_val_instance_counts(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_detection_dataset / "annotations" / "instances_val.json"
        )
        counts = ds.get_instance_counts()
        assert counts["val"] == gt["instance_counts"]

    def test_segmentation_instance_counts(self, coco_real_segmentation_dataset):
        ds = COCODataset.detect(coco_real_segmentation_dataset)
        assert ds is not None
        # Check both splits
        for split, filename in [
            ("train", "instances_train.json"),
            ("val", "instances_val.json"),
        ]:
            gt = get_ground_truth_stats(
                coco_real_segmentation_dataset / "annotations" / filename
            )
            counts = ds.get_instance_counts()
            assert counts[split] == gt["instance_counts"]

    def test_roboflow_instance_counts(self, coco_real_roboflow_dataset):
        ds = COCODataset.detect(coco_real_roboflow_dataset)
        assert ds is not None
        counts = ds.get_instance_counts()
        for split_dir, argus_split in [
            ("train", "train"),
            ("valid", "val"),
            ("test", "test"),
        ]:
            gt = get_ground_truth_stats(
                coco_real_roboflow_dataset / split_dir / "_annotations.coco.json"
            )
            assert counts[argus_split] == gt["instance_counts"]

    def test_unsplit_instance_counts(self, coco_real_unsplit_dataset):
        ds = COCODataset.detect(coco_real_unsplit_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_unsplit_dataset / "annotations" / "annotations.json"
        )
        counts = ds.get_instance_counts()
        assert counts["unsplit"] == gt["instance_counts"]


# ---------------------------------------------------------------------------
# E. Split detection
# ---------------------------------------------------------------------------
class TestSplitDetection:
    def test_standard_splits_detected(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        assert sorted(ds.splits) == ["train", "val"]

    def test_roboflow_splits_detected(self, coco_real_roboflow_dataset):
        ds = COCODataset.detect(coco_real_roboflow_dataset)
        assert ds is not None
        assert sorted(ds.splits) == ["test", "train", "val"]

    def test_unsplit_has_no_splits(self, coco_real_unsplit_dataset):
        ds = COCODataset.detect(coco_real_unsplit_dataset)
        assert ds is not None
        assert ds.splits == []


# ---------------------------------------------------------------------------
# F. Category zero
# ---------------------------------------------------------------------------
class TestCategoryZero:
    def test_cat_zero_detected(self, coco_real_cat_zero_dataset):
        ds = COCODataset.detect(coco_real_cat_zero_dataset)
        assert ds is not None
        assert ds.has_cat_zero is True
        assert ds.ignore_index == 0

    def test_cat_zero_not_detected_when_absent(self, coco_real_detection_dataset):
        ds = COCODataset.detect(coco_real_detection_dataset)
        assert ds is not None
        assert ds.has_cat_zero is False

    def test_cat_zero_instance_counts_match(self, coco_real_cat_zero_dataset):
        ds = COCODataset.detect(coco_real_cat_zero_dataset)
        assert ds is not None
        gt = get_ground_truth_stats(
            coco_real_cat_zero_dataset / "annotations" / "instances_train.json"
        )
        counts = ds.get_instance_counts()
        assert counts["train"] == gt["instance_counts"]

    def test_cat_zero_class_mapping_shifted(self, coco_real_cat_zero_dataset):
        """Class mapping shifts IDs by +1 when cat 0 exists."""
        ds = COCODataset.detect(coco_real_cat_zero_dataset)
        assert ds is not None
        mapping = ds.get_class_mapping()
        # Original IDs 0->alpha, 1->beta become 1->alpha, 2->beta
        assert mapping == {1: "alpha", 2: "beta"}

    def test_cat_zero_mask_values_shifted(self, tmp_path):
        """Mask pixel values are shifted by +1 when cat 0 exists."""
        import json

        import cv2
        import numpy as np

        ds_path = tmp_path / "cat_zero_mask"
        ds_path.mkdir()
        ann_dir = ds_path / "annotations"
        ann_dir.mkdir()
        img_dir = ds_path / "images" / "train"
        img_dir.mkdir(parents=True)

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
        (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img.jpg"), img)

        ds = COCODataset.detect(ds_path)
        assert ds is not None

        mask = ds.load_mask(img_dir / "img.jpg")
        assert mask is not None
        # Category 0 should appear as 1 in the mask
        assert (mask == 1).any()
        # Background should be 0
        assert (mask == 0).any()


# ---------------------------------------------------------------------------
# G. Background variants
# ---------------------------------------------------------------------------
class TestBackgroundVariants:
    def test_image_with_no_annotations_is_background(self, tmp_path):
        """Build a minimal dataset inline: 1 image, 0 annotations."""
        import json

        import cv2
        import numpy as np

        ds_path = tmp_path / "bg_only"
        ds_path.mkdir()
        ann_dir = ds_path / "annotations"
        ann_dir.mkdir()
        img_dir = ds_path / "images" / "train"
        img_dir.mkdir(parents=True)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img001.jpg"), img)

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img001.jpg", "width": 100, "height": 100}
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "obj", "supercategory": "thing"}],
        }
        (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))

        ds = COCODataset.detect(ds_path)
        assert ds is not None
        gt = get_ground_truth_stats(ann_dir / "instances_train.json")
        counts = ds.get_image_counts()
        assert counts["train"]["total"] == 1
        assert counts["train"]["background"] == 1
        assert gt["total"] == 1
        assert gt["background"] == 1
