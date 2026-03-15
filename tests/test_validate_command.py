"""Tests for the validate command and validation logic."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.core.validation import ValidationReport, validate_dataset
from argus.discovery import _detect_dataset

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures for datasets with validation errors
# ---------------------------------------------------------------------------


@pytest.fixture
def yolo_bad_labels(tmp_path: Path) -> Path:
    """YOLO dataset with various label issues."""
    ds = tmp_path / "yolo_bad"
    ds.mkdir()

    yaml_content = "names:\n  0: cat\n  1: dog\n"
    (ds / "data.yaml").write_text(yaml_content)

    (ds / "images").mkdir()
    (ds / "labels").mkdir()

    # Image files
    (ds / "images" / "good.jpg").write_bytes(b"fake image")
    (ds / "images" / "bad_class.jpg").write_bytes(b"fake image")
    (ds / "images" / "bad_coords.jpg").write_bytes(b"fake image")
    (ds / "images" / "bad_cols.jpg").write_bytes(b"fake image")
    (ds / "images" / "zero_dim.jpg").write_bytes(b"fake image")
    (ds / "images" / "tiny_box.jpg").write_bytes(b"fake image")

    # Good label
    (ds / "labels" / "good.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    # Class ID out of range
    (ds / "labels" / "bad_class.txt").write_text("5 0.5 0.5 0.2 0.3\n")
    # Coordinates out of bounds
    (ds / "labels" / "bad_coords.txt").write_text("0 1.5 0.5 0.2 0.3\n")
    # Wrong column count
    (ds / "labels" / "bad_cols.txt").write_text("0 0.5 0.5\n")
    # Zero dimension
    (ds / "labels" / "zero_dim.txt").write_text("0 0.5 0.5 0.0 0.3\n")
    # Very small box
    (ds / "labels" / "tiny_box.txt").write_text("0 0.5 0.5 0.0005 0.0005\n")
    # Orphan label (no corresponding image)
    (ds / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    return ds


@pytest.fixture
def coco_bad_annotations(tmp_path: Path) -> Path:
    """COCO dataset with various annotation issues."""
    ds = tmp_path / "coco_bad"
    ds.mkdir()
    ann_dir = ds / "annotations"
    ann_dir.mkdir()

    coco_data = {
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 1, "file_name": "img002.jpg", "width": 640, "height": 480},  # dup ID
            {"id": 3, "file_name": "img003.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            # Good annotation
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
            # Duplicate annotation ID
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 20, 30, 30]},
            # Non-existent image_id
            {"id": 3, "image_id": 999, "category_id": 1, "bbox": [10, 10, 50, 50]},
            # Non-existent category_id
            {"id": 4, "image_id": 1, "category_id": 99, "bbox": [10, 10, 50, 50]},
            # Invalid bbox (zero width)
            {"id": 5, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0, 50]},
            # Short polygon
            {
                "id": 6,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "segmentation": [[10, 10, 20, 20]],  # only 2 points
            },
        ],
        "categories": [
            {"id": 1, "name": "person"},
        ],
    }

    (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))

    images_dir = ds / "images" / "train"
    images_dir.mkdir(parents=True)
    (images_dir / "img001.jpg").write_bytes(b"fake")
    (images_dir / "img002.jpg").write_bytes(b"fake")
    (images_dir / "img003.jpg").write_bytes(b"fake")

    return ds


@pytest.fixture
def mask_bad_dataset(tmp_path: Path) -> Path:
    """Mask dataset with missing masks and dimension mismatches."""
    ds = tmp_path / "mask_bad"
    ds.mkdir()

    (ds / "images" / "train").mkdir(parents=True)
    (ds / "masks" / "train").mkdir(parents=True)

    # Use classes.yaml to fix class mapping so unexpected values are detectable
    yaml_content = "names:\n  0: background\n  1: object\nignore_index: 255\n"
    (ds / "classes.yaml").write_text(yaml_content)

    # Image with matching mask (good)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(ds / "images" / "train" / "good.jpg"), img)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    cv2.imwrite(str(ds / "masks" / "train" / "good.png"), mask)

    # Image with dimension mismatch
    img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(ds / "images" / "train" / "mismatch.jpg"), img2)
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    mask2[10:30, 10:30] = 1
    cv2.imwrite(str(ds / "masks" / "train" / "mismatch.png"), mask2)

    # Image with no mask
    img3 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(ds / "images" / "train" / "no_mask.jpg"), img3)

    # Image with mask containing unexpected pixel values
    img4 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(ds / "images" / "train" / "bad_vals.jpg"), img4)
    mask4 = np.zeros((100, 100), dtype=np.uint8)
    mask4[20:40, 20:40] = 1
    mask4[60:80, 60:80] = 99  # unexpected class ID (not in classes.yaml)
    cv2.imwrite(str(ds / "masks" / "train" / "bad_vals.png"), mask4)

    return ds


# ---------------------------------------------------------------------------
# Tests: Valid datasets produce no errors
# ---------------------------------------------------------------------------


class TestValidDatasets:
    def test_valid_yolo_detection(self, yolo_detection_dataset: Path) -> None:
        dataset = _detect_dataset(yolo_detection_dataset)
        assert dataset is not None
        report = validate_dataset(dataset)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_valid_yolo_segmentation(self, yolo_segmentation_dataset: Path) -> None:
        dataset = _detect_dataset(yolo_segmentation_dataset)
        assert dataset is not None
        report = validate_dataset(dataset)
        assert report.is_valid

    def test_valid_coco_detection(self, coco_detection_dataset: Path) -> None:
        dataset = _detect_dataset(coco_detection_dataset)
        assert dataset is not None
        report = validate_dataset(dataset)
        assert report.is_valid

    def test_valid_mask_grayscale(self, mask_dataset_grayscale: Path) -> None:
        dataset = _detect_dataset(mask_dataset_grayscale)
        assert dataset is not None
        report = validate_dataset(dataset)
        assert report.is_valid


# ---------------------------------------------------------------------------
# Tests: YOLO validation errors
# ---------------------------------------------------------------------------


class TestYOLOValidation:
    def test_class_id_out_of_range(self, yolo_bad_labels: Path) -> None:
        dataset = _detect_dataset(yolo_bad_labels)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E202" in codes

    def test_coords_out_of_bounds(self, yolo_bad_labels: Path) -> None:
        dataset = _detect_dataset(yolo_bad_labels)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E203" in codes

    def test_wrong_column_count(self, yolo_bad_labels: Path) -> None:
        dataset = _detect_dataset(yolo_bad_labels)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E201" in codes

    def test_zero_dimension_box(self, yolo_bad_labels: Path) -> None:
        dataset = _detect_dataset(yolo_bad_labels)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E204" in codes

    def test_very_small_box_warning(self, yolo_bad_labels: Path) -> None:
        dataset = _detect_dataset(yolo_bad_labels)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.warnings]
        assert "W202" in codes

    def test_orphan_label_warning(self, yolo_bad_labels: Path) -> None:
        dataset = _detect_dataset(yolo_bad_labels)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.warnings]
        assert "W201" in codes


# ---------------------------------------------------------------------------
# Tests: COCO validation errors
# ---------------------------------------------------------------------------


class TestCOCOValidation:
    def test_dangling_image_id(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E301" in codes

    def test_invalid_category_id(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E302" in codes

    def test_duplicate_annotation_id(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E303" in codes

    def test_invalid_bbox(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E304" in codes

    def test_duplicate_image_id(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E305" in codes

    def test_short_polygon_warning(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.warnings]
        assert "W302" in codes

    def test_image_no_annotations_warning(self, coco_bad_annotations: Path) -> None:
        dataset = _detect_dataset(coco_bad_annotations)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.warnings]
        assert "W301" in codes


# ---------------------------------------------------------------------------
# Tests: Mask validation errors
# ---------------------------------------------------------------------------


class TestMaskValidation:
    def test_missing_mask(self, mask_bad_dataset: Path) -> None:
        dataset = _detect_dataset(mask_bad_dataset)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E401" in codes

    def test_dimension_mismatch(self, mask_bad_dataset: Path) -> None:
        dataset = _detect_dataset(mask_bad_dataset)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E402" in codes

    def test_unexpected_pixel_values(self, mask_bad_dataset: Path) -> None:
        dataset = _detect_dataset(mask_bad_dataset)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E403" in codes

    def test_existing_fixture_dimension_mismatch(
        self, mask_dataset_dimension_mismatch: Path
    ) -> None:
        dataset = _detect_dataset(mask_dataset_dimension_mismatch)
        assert dataset is not None
        report = validate_dataset(dataset)
        codes = [i.code for i in report.errors]
        assert "E402" in codes


# ---------------------------------------------------------------------------
# Tests: CLI integration
# ---------------------------------------------------------------------------


class TestCLIValidate:
    def test_valid_dataset_exit_0(self, yolo_detection_dataset: Path) -> None:
        result = runner.invoke(app, ["validate", str(yolo_detection_dataset)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "No issues" in result.output

    def test_invalid_dataset_exit_1(self, yolo_bad_labels: Path) -> None:
        result = runner.invoke(app, ["validate", str(yolo_bad_labels)])
        assert result.exit_code == 1

    def test_strict_mode(self, coco_bad_annotations: Path) -> None:
        result = runner.invoke(app, ["validate", str(coco_bad_annotations), "--strict"])
        assert result.exit_code == 1

    def test_max_issues(self, yolo_bad_labels: Path) -> None:
        result = runner.invoke(
            app, ["validate", str(yolo_bad_labels), "--max-issues", "2"]
        )
        assert "more issues" in result.output

    def test_no_dataset_found(self, empty_directory: Path) -> None:
        result = runner.invoke(app, ["validate", str(empty_directory)])
        assert result.exit_code == 1
        assert "No dataset found" in result.output

    def test_split_filter(self, yolo_detection_dataset: Path) -> None:
        result = runner.invoke(
            app, ["validate", str(yolo_detection_dataset), "--split", "train"]
        )
        assert result.exit_code == 0

    def test_coco_errors_shown(self, coco_bad_annotations: Path) -> None:
        result = runner.invoke(app, ["validate", str(coco_bad_annotations)])
        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_mask_errors_shown(self, mask_bad_dataset: Path) -> None:
        result = runner.invoke(app, ["validate", str(mask_bad_dataset)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Tests: ValidationReport properties
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_is_valid_true(self) -> None:
        from argus.core.base import DatasetFormat
        from argus.core.validation import ValidationIssue

        report = ValidationReport(
            dataset_path=Path("/tmp"),
            format=DatasetFormat.YOLO,
            issues=[
                ValidationIssue(level="warning", code="W101", message="test"),
            ],
        )
        assert report.is_valid is True
        assert len(report.warnings) == 1
        assert len(report.errors) == 0

    def test_is_valid_false(self) -> None:
        from argus.core.base import DatasetFormat
        from argus.core.validation import ValidationIssue

        report = ValidationReport(
            dataset_path=Path("/tmp"),
            format=DatasetFormat.YOLO,
            issues=[
                ValidationIssue(level="error", code="E101", message="test"),
                ValidationIssue(level="warning", code="W101", message="test"),
            ],
        )
        assert report.is_valid is False
        assert len(report.errors) == 1
        assert len(report.warnings) == 1


# ---------------------------------------------------------------------------
# Tests: Split filtering
# ---------------------------------------------------------------------------


class TestSplitFiltering:
    def test_validate_specific_split_yolo(self, yolo_detection_dataset: Path) -> None:
        dataset = _detect_dataset(yolo_detection_dataset)
        assert dataset is not None
        report = validate_dataset(dataset, split="train")
        # Should only have issues from train split (if any)
        for issue in report.issues:
            if issue.split:
                assert issue.split == "train"

    def test_validate_specific_split_coco(self, coco_detection_dataset: Path) -> None:
        dataset = _detect_dataset(coco_detection_dataset)
        assert dataset is not None
        report = validate_dataset(dataset, split="train")
        for issue in report.issues:
            if issue.split:
                assert issue.split == "train"
