"""Cross-validation tests: argus YOLO stats vs ultralytics ground truth.

These tests require ``ultralytics`` and are automatically skipped when it
is not installed.  Install via::

    uv sync --group integration
"""

from pathlib import Path

import pytest

ultralytics = pytest.importorskip("ultralytics", reason="ultralytics not installed")

from yolo_ground_truth import get_ground_truth_stats  # noqa: E402

from argus.core import YOLODataset  # noqa: E402
from argus.core.base import DatasetFormat, TaskType  # noqa: E402

# ---------------------------------------------------------------------------
# A. Task type agreement
# ---------------------------------------------------------------------------


class TestDetectionTaskType:
    """Argus and ultralytics agree on the task type."""

    def test_detection_dataset_is_detection(
        self, yolo_real_detection_dataset: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_real_detection_dataset)
        assert dataset is not None
        assert dataset.format == DatasetFormat.YOLO
        assert dataset.task == TaskType.DETECTION

    def test_segmentation_dataset_is_segmentation(
        self, yolo_real_segmentation_dataset: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_real_segmentation_dataset)
        assert dataset is not None
        assert dataset.format == DatasetFormat.YOLO
        assert dataset.task == TaskType.SEGMENTATION


# ---------------------------------------------------------------------------
# B. Class names
# ---------------------------------------------------------------------------


class TestClassNames:
    """Argus extracts the same class names as the data.yaml declares."""

    def test_detection_class_names(self, yolo_real_detection_dataset: Path) -> None:
        dataset = YOLODataset.detect(yolo_real_detection_dataset)
        assert dataset is not None
        assert dataset.class_names == ["person", "car", "bicycle"]

    def test_segmentation_class_names(
        self, yolo_real_segmentation_dataset: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_real_segmentation_dataset)
        assert dataset is not None
        assert dataset.class_names == ["cat", "dog"]


# ---------------------------------------------------------------------------
# C. Image counts
# ---------------------------------------------------------------------------


class TestImageCounts:
    """Cross-validate argus get_image_counts() against ultralytics."""

    def test_train_image_counts(self, yolo_real_detection_dataset: Path) -> None:
        """Train split: 3 label files + 1 missing-label image.

        Argus counts label files -> total=3, background=1.
        Ultralytics scans images -> total=4, background=2 (empty + missing).
        """
        yaml_path = yolo_real_detection_dataset / "data.yaml"
        gt = get_ground_truth_stats(yaml_path, split="train")

        dataset = YOLODataset.detect(yolo_real_detection_dataset)
        assert dataset is not None
        argus_counts = dataset.get_image_counts()

        # Ultralytics sees 4 images (scans the image directory)
        assert gt["total"] == 4
        # Ultralytics sees 2 backgrounds: empty label + missing label
        assert gt["background"] == 2

        # Argus counts label files only, so it sees 3 total, 1 background
        assert argus_counts["train"]["total"] == 3
        assert argus_counts["train"]["background"] == 1

    def test_val_image_counts(self, yolo_real_detection_dataset: Path) -> None:
        """Val split: all images have label files, counts should agree."""
        yaml_path = yolo_real_detection_dataset / "data.yaml"
        gt = get_ground_truth_stats(yaml_path, split="val")

        dataset = YOLODataset.detect(yolo_real_detection_dataset)
        assert dataset is not None
        argus_counts = dataset.get_image_counts()

        # Both should agree: 2 total, 1 background
        assert gt["total"] == argus_counts["val"]["total"]
        assert gt["background"] == argus_counts["val"]["background"]


# ---------------------------------------------------------------------------
# D. Instance counts
# ---------------------------------------------------------------------------


class TestInstanceCounts:
    """Cross-validate argus get_instance_counts() against ultralytics."""

    def test_train_instance_counts(self, yolo_real_detection_dataset: Path) -> None:
        yaml_path = yolo_real_detection_dataset / "data.yaml"
        gt = get_ground_truth_stats(yaml_path, split="train")

        dataset = YOLODataset.detect(yolo_real_detection_dataset)
        assert dataset is not None
        argus_counts = dataset.get_instance_counts()

        # Ground truth: class 0 (person) = 2, class 1 (car) = 1
        assert gt["instance_counts"] == {0: 2, 1: 1}

        # Argus uses class names
        assert argus_counts["train"]["person"] == gt["instance_counts"][0]
        assert argus_counts["train"]["car"] == gt["instance_counts"][1]

    def test_val_instance_counts(self, yolo_real_detection_dataset: Path) -> None:
        yaml_path = yolo_real_detection_dataset / "data.yaml"
        gt = get_ground_truth_stats(yaml_path, split="val")

        dataset = YOLODataset.detect(yolo_real_detection_dataset)
        assert dataset is not None
        argus_counts = dataset.get_instance_counts()

        # Ground truth: class 2 (bicycle) = 1
        assert gt["instance_counts"] == {2: 1}
        assert argus_counts["val"]["bicycle"] == gt["instance_counts"][2]

    def test_segmentation_instance_counts(
        self, yolo_real_segmentation_dataset: Path
    ) -> None:
        yaml_path = yolo_real_segmentation_dataset / "data.yaml"

        dataset = YOLODataset.detect(yolo_real_segmentation_dataset)
        assert dataset is not None
        argus_counts = dataset.get_instance_counts()

        for split in ("train", "val"):
            gt = get_ground_truth_stats(yaml_path, split=split)
            for class_id, gt_count in gt["instance_counts"].items():
                class_name = dataset.class_names[class_id]
                assert argus_counts[split][class_name] == gt_count


# ---------------------------------------------------------------------------
# E. Background variants
# ---------------------------------------------------------------------------


class TestBackgroundVariants:
    """Test that empty-label and missing-label images are handled correctly."""

    @staticmethod
    def _make_mini_dataset(tmp_path: Path) -> Path:
        """Create a minimal dataset skeleton with both train and val splits.

        Ultralytics requires both ``train`` and ``val`` in data.yaml.
        """
        import cv2
        import numpy as np

        ds = tmp_path / "ds"
        ds.mkdir()
        (ds / "data.yaml").write_text(
            f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n  0: obj\n"
        )
        for split in ("train", "val"):
            (ds / "images" / split).mkdir(parents=True)
            (ds / "labels" / split).mkdir(parents=True)

        # val needs at least one image so ultralytics doesn't complain
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images/val/dummy.jpg"), img)
        (ds / "labels/val/dummy.txt").write_text("0 0.5 0.5 0.2 0.3\n")

        return ds

    def test_empty_label_is_background(self, tmp_path: Path) -> None:
        """An empty label file is background for both argus and ultralytics."""
        import cv2
        import numpy as np

        ds = self._make_mini_dataset(tmp_path)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images/train/a.jpg"), img)
        (ds / "labels/train/a.txt").write_text("")

        gt = get_ground_truth_stats(ds / "data.yaml", "train")
        assert gt["total"] == 1
        assert gt["background"] == 1

        dataset = YOLODataset.detect(ds)
        assert dataset is not None
        counts = dataset.get_image_counts()
        assert counts["train"]["total"] == 1
        assert counts["train"]["background"] == 1

    def test_missing_label_is_background_in_ultralytics(self, tmp_path: Path) -> None:
        """An image with no label file: ultralytics counts as background,
        argus does not see the image at all (counts label files)."""
        import cv2
        import numpy as np

        ds = self._make_mini_dataset(tmp_path)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images/train/a.jpg"), img)
        # No label file created

        gt = get_ground_truth_stats(ds / "data.yaml", "train")
        assert gt["total"] == 1
        assert gt["background"] == 1

        dataset = YOLODataset.detect(ds)
        assert dataset is not None
        counts = dataset.get_image_counts()
        # Argus counts label files, so the image is invisible
        assert counts["train"]["total"] == 0

    def test_whitespace_only_label_is_background(self, tmp_path: Path) -> None:
        """A label file with only whitespace is background for both."""
        import cv2
        import numpy as np

        ds = self._make_mini_dataset(tmp_path)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images/train/a.jpg"), img)
        (ds / "labels/train/a.txt").write_text("   \n  \n")

        gt = get_ground_truth_stats(ds / "data.yaml", "train")
        assert gt["total"] == 1
        assert gt["background"] == 1

        dataset = YOLODataset.detect(ds)
        assert dataset is not None
        counts = dataset.get_image_counts()
        assert counts["train"]["total"] == 1
        assert counts["train"]["background"] == 1
