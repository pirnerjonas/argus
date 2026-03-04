"""Tests for YOLO segmentation to COCO conversion."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from argus.cli import app
from argus.core.convert import (
    _parse_yolo_label_file,
    _yolo_polygon_to_coco_segmentation,
    convert_yolo_seg_to_coco,
    convert_yolo_seg_to_roboflow_coco,
)
from argus.core.yolo import YOLODataset

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def yolo_seg_dataset(tmp_path: Path) -> Path:
    """Create a YOLO segmentation dataset with real images and labels."""
    dataset_path = tmp_path / "yolo_seg"
    dataset_path.mkdir()

    # data.yaml
    yaml_content = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "cat", 1: "dog"},
    }
    (dataset_path / "data.yaml").write_text(yaml.dump(yaml_content))

    # Directory structure
    for split in ("train", "val"):
        (dataset_path / "images" / split).mkdir(parents=True)
        (dataset_path / "labels" / split).mkdir(parents=True)

    # Create 100x100 images
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_path / "images" / "train" / "img001.jpg"), img)
    cv2.imwrite(str(dataset_path / "images" / "train" / "img002.jpg"), img)
    cv2.imwrite(str(dataset_path / "images" / "val" / "img003.jpg"), img)

    # Simple polygon labels (normalized coords forming a rectangle)
    # Rectangle roughly at (10,10)-(90,90) on a 100x100 image
    rect_label = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    (dataset_path / "labels" / "train" / "img001.txt").write_text(rect_label)
    # Two annotations on one image
    two_labels = (
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )
    (dataset_path / "labels" / "train" / "img002.txt").write_text(two_labels)
    (dataset_path / "labels" / "val" / "img003.txt").write_text(
        "1 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
    )

    return dataset_path


@pytest.fixture
def yolo_seg_donut_dataset(tmp_path: Path) -> Path:
    """Create a YOLO seg dataset with a donut annotation (bridge-connected polygon).

    The polygon traces the outer boundary, bridges in to an inner hole boundary,
    traces the hole, then bridges back. When rasterized, this produces a donut shape.
    """
    dataset_path = tmp_path / "yolo_donut"
    dataset_path.mkdir()

    yaml_content = {
        "path": ".",
        "train": "images/train",
        "names": {0: "donut"},
    }
    (dataset_path / "data.yaml").write_text(yaml.dump(yaml_content))

    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)

    # Create a 200x200 image
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_path / "images" / "train" / "donut.jpg"), img)

    # Build a donut polygon: outer rect → bridge → inner rect (hole) → bridge back
    # Outer boundary: (10,10) to (190,190) on 200x200
    # Inner hole: (60,60) to (140,140)
    # Bridge connects outer top-left to inner top-left
    # Normalized coordinates (divide by 200):
    outer_tl = (0.05, 0.05)
    outer_tr = (0.95, 0.05)
    outer_br = (0.95, 0.95)
    outer_bl = (0.05, 0.95)

    inner_tl = (0.30, 0.30)
    inner_tr = (0.70, 0.30)
    inner_br = (0.70, 0.70)
    inner_bl = (0.30, 0.70)

    # Path: outer_tl → outer_tr → outer_br → outer_bl → outer_tl (close)
    # → bridge to inner_tl → inner_bl → inner_br → inner_tr → inner_tl (hole CW)
    # → bridge back to outer_tl
    donut_points = [
        outer_tl,
        outer_tr,
        outer_br,
        outer_bl,  # outer CCW
        outer_tl,  # close outer + start bridge
        inner_tl,  # bridge to hole
        inner_bl,
        inner_br,
        inner_tr,  # hole CW (opposite winding)
        inner_tl,  # close hole
        outer_tl,  # bridge back
    ]

    coords_str = " ".join(f"{x} {y}" for x, y in donut_points)
    label_line = f"0 {coords_str}\n"
    (dataset_path / "labels" / "train" / "donut.txt").write_text(label_line)

    return dataset_path


@pytest.fixture
def yolo_seg_empty_labels_dataset(tmp_path: Path) -> Path:
    """Create a YOLO seg dataset where some images have empty label files."""
    dataset_path = tmp_path / "yolo_empty"
    dataset_path.mkdir()

    yaml_content = {
        "path": ".",
        "train": "images/train",
        "names": {0: "obj"},
    }
    (dataset_path / "data.yaml").write_text(yaml.dump(yaml_content))

    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)

    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_path / "images" / "train" / "bg.jpg"), img)
    cv2.imwrite(str(dataset_path / "images" / "train" / "fg.jpg"), img)

    # Empty label (background image)
    (dataset_path / "labels" / "train" / "bg.txt").write_text("")
    # Normal label
    (dataset_path / "labels" / "train" / "fg.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    )

    return dataset_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestParseYoloLabelFile:
    """Tests for _parse_yolo_label_file."""

    def test_parse_simple(self, tmp_path: Path) -> None:
        label = tmp_path / "label.txt"
        label.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6\n")

        result = _parse_yolo_label_file(label)

        assert len(result) == 1
        class_id, points = result[0]
        assert class_id == 0
        assert len(points) == 3
        assert points[0] == pytest.approx((0.1, 0.2))

    def test_parse_multiple_lines(self, tmp_path: Path) -> None:
        label = tmp_path / "label.txt"
        label.write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n1 0.5 0.5 0.6 0.6 0.7 0.7\n")

        result = _parse_yolo_label_file(label)

        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 1

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        label = tmp_path / "label.txt"
        label.write_text("")

        result = _parse_yolo_label_file(label)

        assert result == []

    def test_parse_skips_short_lines(self, tmp_path: Path) -> None:
        """Lines with fewer than 3 coordinate pairs are skipped."""
        label = tmp_path / "label.txt"
        label.write_text("0 0.1 0.2 0.3 0.4\n")  # Only 2 pairs

        result = _parse_yolo_label_file(label)

        assert result == []

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_yolo_label_file(tmp_path / "nonexistent.txt")

        assert result == []


class TestYoloPolygonToCocoSegmentation:
    """Tests for _yolo_polygon_to_coco_segmentation."""

    def test_simple_polygon(self) -> None:
        """Simple rectangle → single COCO ring, correct bbox/area."""
        points = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        segmentation, bbox, area = _yolo_polygon_to_coco_segmentation(points, 100, 100)

        assert len(segmentation) >= 1
        # Each ring should be a flat list of coordinates
        for ring in segmentation:
            assert len(ring) >= 6
            assert len(ring) % 2 == 0

        # Bbox should roughly cover the rectangle
        x, y, w, h = bbox
        assert x == pytest.approx(10, abs=2)
        assert y == pytest.approx(10, abs=2)
        assert w == pytest.approx(80, abs=4)
        assert h == pytest.approx(80, abs=4)

        # Area should be roughly 80*80 = 6400
        assert area == pytest.approx(6400, rel=0.1)

    def test_donut_polygon_produces_multiple_rings(self) -> None:
        """Donut polygon (bridge-connected) → 2+ COCO rings (outer + hole)."""
        # Outer: (10,10)-(190,190), Inner hole: (60,60)-(140,140), on 200x200
        outer_tl = (0.05, 0.05)
        outer_tr = (0.95, 0.05)
        outer_br = (0.95, 0.95)
        outer_bl = (0.05, 0.95)
        inner_tl = (0.30, 0.30)
        inner_tr = (0.70, 0.30)
        inner_br = (0.70, 0.70)
        inner_bl = (0.30, 0.70)

        donut_points = [
            outer_tl,
            outer_tr,
            outer_br,
            outer_bl,
            outer_tl,
            inner_tl,
            inner_bl,
            inner_br,
            inner_tr,
            inner_tl,
            outer_tl,
        ]

        segmentation, bbox, area = _yolo_polygon_to_coco_segmentation(
            donut_points, 200, 200
        )

        # Should have at least 2 rings (outer + hole)
        assert len(segmentation) >= 2

        # Area should be outer area minus hole area
        outer_area = 180 * 180  # approx
        inner_area = 80 * 80  # approx
        expected_area = outer_area - inner_area
        assert area == pytest.approx(expected_area, rel=0.15)

    def test_area_from_mask(self) -> None:
        """Area is computed from pixel count of rasterized mask."""
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        _, _, area = _yolo_polygon_to_coco_segmentation(points, 50, 50)

        # Full image area
        assert area == pytest.approx(50 * 50, rel=0.05)


class TestCategoryIdMapping:
    """YOLO 0-indexed → COCO 1-indexed."""

    def test_mapping(self, yolo_seg_dataset: Path) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        assert dataset is not None

        output_path = yolo_seg_dataset.parent / "coco_out"
        convert_yolo_seg_to_coco(dataset, output_path)

        ann_file = output_path / "annotations" / "instances_train.json"
        with open(ann_file) as f:
            coco = json.load(f)

        # Categories should be 1-indexed
        cat_ids = [c["id"] for c in coco["categories"]]
        assert cat_ids == [1, 2]

        # Annotation category_ids should be 1-indexed
        for ann in coco["annotations"]:
            assert ann["category_id"] in (1, 2)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestConvertYoloSegToCocoBasic:
    """Full conversion integration tests."""

    def test_output_structure(self, yolo_seg_dataset: Path, tmp_path: Path) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        assert dataset is not None

        output = tmp_path / "coco_out"
        stats = convert_yolo_seg_to_coco(dataset, output)

        # Check directories
        assert (output / "annotations").is_dir()
        assert (output / "images" / "train").is_dir()
        assert (output / "images" / "val").is_dir()

        # Check annotation files
        assert (output / "annotations" / "instances_train.json").exists()
        assert (output / "annotations" / "instances_val.json").exists()

        # Check stats
        assert stats["images"] == 3
        assert stats["annotations"] == 4  # 1 + 2 + 1

    def test_json_validity(self, yolo_seg_dataset: Path, tmp_path: Path) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        output = tmp_path / "coco_out"
        convert_yolo_seg_to_coco(dataset, output)

        for split in ("train", "val"):
            ann_file = output / "annotations" / f"instances_{split}.json"
            with open(ann_file) as f:
                coco = json.load(f)

            assert "images" in coco
            assert "annotations" in coco
            assert "categories" in coco

            # Validate annotation fields
            for ann in coco["annotations"]:
                assert "id" in ann
                assert "image_id" in ann
                assert "category_id" in ann
                assert "segmentation" in ann
                assert "bbox" in ann
                assert "area" in ann
                assert ann["iscrowd"] == 0
                assert isinstance(ann["segmentation"], list)
                assert len(ann["bbox"]) == 4

    def test_images_copied(self, yolo_seg_dataset: Path, tmp_path: Path) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        output = tmp_path / "coco_out"
        convert_yolo_seg_to_coco(dataset, output)

        train_images = list((output / "images" / "train").iterdir())
        val_images = list((output / "images" / "val").iterdir())
        assert len(train_images) == 2
        assert len(val_images) == 1

    def test_progress_callback(self, yolo_seg_dataset: Path, tmp_path: Path) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        output = tmp_path / "coco_out"
        calls: list[tuple[int, int]] = []

        def cb(current: int, total: int) -> None:
            calls.append((current, total))

        convert_yolo_seg_to_coco(dataset, output, progress_callback=cb)

        assert len(calls) == 3  # 3 images total
        assert calls[-1][0] == calls[-1][1]


class TestConvertYoloSegToCocoDonut:
    """Tests for donut/hole handling."""

    def test_donut_produces_multiple_rings(
        self, yolo_seg_donut_dataset: Path, tmp_path: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_seg_donut_dataset)
        assert dataset is not None

        output = tmp_path / "coco_out"
        stats = convert_yolo_seg_to_coco(dataset, output)

        assert stats["images"] == 1
        assert stats["annotations"] == 1

        ann_file = output / "annotations" / "instances_train.json"
        with open(ann_file) as f:
            coco = json.load(f)

        ann = coco["annotations"][0]
        # Should have multiple polygon rings (outer + hole)
        assert len(ann["segmentation"]) >= 2


class TestConvertYoloSegToCocoMultipleSplits:
    """Tests for multi-split conversion."""

    def test_separate_annotation_files(
        self, yolo_seg_dataset: Path, tmp_path: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        output = tmp_path / "coco_out"
        convert_yolo_seg_to_coco(dataset, output)

        # Each split gets its own annotation file
        train_ann = output / "annotations" / "instances_train.json"
        val_ann = output / "annotations" / "instances_val.json"
        assert train_ann.exists()
        assert val_ann.exists()

        with open(train_ann) as f:
            train_coco = json.load(f)
        with open(val_ann) as f:
            val_coco = json.load(f)

        assert len(train_coco["images"]) == 2
        assert len(val_coco["images"]) == 1


class TestConvertYoloSegToRoboflowCoco:
    """Tests for Roboflow COCO layout conversion."""

    def test_output_structure(self, yolo_seg_dataset: Path, tmp_path: Path) -> None:
        dataset = YOLODataset.detect(yolo_seg_dataset)
        assert dataset is not None

        output = tmp_path / "roboflow_coco_out"
        stats = convert_yolo_seg_to_roboflow_coco(dataset, output)

        # train + valid split directories with colocated annotations
        assert (output / "train").is_dir()
        assert (output / "valid").is_dir()
        assert (output / "train" / "_annotations.coco.json").exists()
        assert (output / "valid" / "_annotations.coco.json").exists()

        # Images are copied directly into split dirs in Roboflow format
        assert len(list((output / "train").glob("*.jpg"))) == 2
        assert len(list((output / "valid").glob("*.jpg"))) == 1

        # Standard COCO layout directories should not be required here
        assert not (output / "annotations").exists()
        assert not (output / "images").exists()

        # Stats should match regular converter
        assert stats["images"] == 3
        assert stats["annotations"] == 4

    def test_unsplit_writes_train_folder(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "yolo_unsplit_rf"
        dataset_path.mkdir()

        yaml_content = {
            "path": ".",
            "names": {0: "thing"},
        }
        (dataset_path / "data.yaml").write_text(yaml.dump(yaml_content))

        (dataset_path / "images").mkdir()
        (dataset_path / "labels").mkdir()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_path / "images" / "img001.jpg"), img)
        cv2.imwrite(str(dataset_path / "images" / "img002.jpg"), img)
        rect_label = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        (dataset_path / "labels" / "img001.txt").write_text(rect_label)
        (dataset_path / "labels" / "img002.txt").write_text(rect_label)

        dataset = YOLODataset.detect(dataset_path)
        assert dataset is not None
        assert dataset.splits == []

        output = tmp_path / "roboflow_coco_out"
        stats = convert_yolo_seg_to_roboflow_coco(dataset, output)

        assert stats["images"] == 2
        assert stats["annotations"] == 2
        assert (output / "train" / "_annotations.coco.json").exists()
        assert len(list((output / "train").glob("*.jpg"))) == 2


class TestConvertYoloSegToCocoEmptyLabels:
    """Tests for images with no annotations."""

    def test_empty_labels_still_produce_images(
        self, yolo_seg_empty_labels_dataset: Path, tmp_path: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_seg_empty_labels_dataset)
        assert dataset is not None

        output = tmp_path / "coco_out"
        stats = convert_yolo_seg_to_coco(dataset, output)

        # Both images should be processed (including background)
        assert stats["images"] == 2
        # Only one image has annotations
        assert stats["annotations"] == 1

        ann_file = output / "annotations" / "instances_train.json"
        with open(ann_file) as f:
            coco = json.load(f)

        # Both images listed
        assert len(coco["images"]) == 2
        # Only one annotation
        assert len(coco["annotations"]) == 1


class TestConvertYoloSegToCocoUnsplit:
    """Tests for unsplit datasets (images/ and labels/ at root, no split dirs)."""

    @pytest.fixture
    def yolo_seg_unsplit_dataset(self, tmp_path: Path) -> Path:
        """Create an unsplit YOLO segmentation dataset."""
        dataset_path = tmp_path / "yolo_unsplit"
        dataset_path.mkdir()

        yaml_content = {
            "path": ".",
            "names": {0: "thing"},
        }
        (dataset_path / "data.yaml").write_text(yaml.dump(yaml_content))

        (dataset_path / "images").mkdir()
        (dataset_path / "labels").mkdir()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_path / "images" / "img001.jpg"), img)
        cv2.imwrite(str(dataset_path / "images" / "img002.jpg"), img)

        rect_label = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        (dataset_path / "labels" / "img001.txt").write_text(rect_label)
        (dataset_path / "labels" / "img002.txt").write_text(rect_label)

        return dataset_path

    def test_unsplit_annotations_found(
        self, yolo_seg_unsplit_dataset: Path, tmp_path: Path
    ) -> None:
        dataset = YOLODataset.detect(yolo_seg_unsplit_dataset)
        assert dataset is not None
        assert dataset.splits == []

        output = tmp_path / "coco_out"
        stats = convert_yolo_seg_to_coco(dataset, output)

        assert stats["images"] == 2
        assert stats["annotations"] == 2

        ann_file = output / "annotations" / "instances_train.json"
        with open(ann_file) as f:
            coco = json.load(f)

        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 2
        for ann in coco["annotations"]:
            assert ann["category_id"] == 1
            assert len(ann["segmentation"]) >= 1


class TestConvertCliCoco:
    """CLI end-to-end tests for --to coco."""

    def test_convert_to_coco_basic(
        self, yolo_seg_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "coco_out"
        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(yolo_seg_dataset),
                "-o",
                str(output),
                "--to",
                "coco",
            ],
        )

        assert result.exit_code == 0
        assert "Conversion complete" in result.stdout
        assert "Images processed" in result.stdout
        assert "Annotations created" in result.stdout
        assert output.exists()

    def test_convert_to_coco_invalid_source(self, tmp_path: Path) -> None:
        """Non-YOLO directory should fail."""
        empty = tmp_path / "empty"
        empty.mkdir()
        output = tmp_path / "out"

        result = runner.invoke(
            app,
            ["convert", "-i", str(empty), "-o", str(output), "--to", "coco"],
        )

        assert result.exit_code == 1
        assert "No YOLO dataset found" in result.stdout

    def test_convert_to_coco_detection_rejected(
        self, yolo_detection_dataset: Path, tmp_path: Path
    ) -> None:
        """Detection dataset (not segmentation) should be rejected."""
        output = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(yolo_detection_dataset),
                "-o",
                str(output),
                "--to",
                "coco",
            ],
        )

        assert result.exit_code == 1
        assert "not a segmentation dataset" in result.stdout

    def test_convert_invalid_format(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--to",
                "invalid-format",
            ],
        )

        assert result.exit_code == 1
        assert "Unsupported target format" in result.stdout


class TestConvertCliRoboflowCoco:
    """CLI end-to-end tests for --to roboflow-coco."""

    def test_convert_to_roboflow_coco_basic(
        self, yolo_seg_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "rf_coco_out"
        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(yolo_seg_dataset),
                "-o",
                str(output),
                "--to",
                "roboflow-coco",
            ],
        )

        assert result.exit_code == 0
        assert "Conversion complete" in result.stdout
        assert "Images processed" in result.stdout
        assert "Annotations created" in result.stdout
        assert (output / "train" / "_annotations.coco.json").exists()
        assert (output / "valid" / "_annotations.coco.json").exists()
