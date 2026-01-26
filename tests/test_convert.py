"""Tests for dataset conversion functionality."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from argus.cli import app
from argus.core import MaskDataset
from argus.core.convert import (
    ConversionParams,
    Polygon,
    convert_mask_to_yolo_labels,
    convert_mask_to_yolo_seg,
    mask_to_polygons,
)

runner = CliRunner()


class TestPolygon:
    """Tests for the Polygon dataclass."""

    def test_to_yolo_format(self) -> None:
        """Test YOLO format string generation."""
        poly = Polygon(
            class_id=0,
            points=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
        )
        result = poly.to_yolo()

        assert result.startswith("0 ")
        assert "0.100000 0.200000" in result
        assert "0.300000 0.400000" in result
        assert "0.500000 0.600000" in result

    def test_to_yolo_different_class(self) -> None:
        """Test YOLO format with different class ID."""
        poly = Polygon(class_id=5, points=[(0.5, 0.5)])
        result = poly.to_yolo()

        assert result.startswith("5 ")


class TestMaskToPolygons:
    """Tests for mask_to_polygons function."""

    def test_simple_rectangle(self) -> None:
        """Test polygon extraction from a simple rectangle."""
        # Create 100x100 mask with a 40x40 rectangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        params = ConversionParams(class_id=0, epsilon_factor=0.01, min_area=50)
        polygons = mask_to_polygons(mask, params)

        assert len(polygons) == 1
        assert polygons[0].class_id == 0
        # Rectangle should have ~4 points after simplification
        assert len(polygons[0].points) >= 4

    def test_filters_small_contours(self) -> None:
        """Test that small contours are filtered by min_area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Small 5x5 region (area=25)
        mask[10:15, 10:15] = 255

        params = ConversionParams(min_area=100)
        polygons = mask_to_polygons(mask, params)

        assert len(polygons) == 0

    def test_multiple_contours(self) -> None:
        """Test extraction of multiple separate regions."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # Region 1
        mask[60:80, 60:80] = 255  # Region 2

        params = ConversionParams(min_area=50)
        polygons = mask_to_polygons(mask, params)

        assert len(polygons) == 2

    def test_normalized_coordinates(self) -> None:
        """Test that coordinates are normalized to [0, 1]."""
        mask = np.zeros((200, 400), dtype=np.uint8)  # Non-square
        mask[50:150, 100:300] = 255

        params = ConversionParams(min_area=50)
        polygons = mask_to_polygons(mask, params)

        assert len(polygons) == 1
        for x, y in polygons[0].points:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_empty_mask(self) -> None:
        """Test empty mask returns no polygons."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygons = mask_to_polygons(mask)

        assert len(polygons) == 0

    def test_default_params(self) -> None:
        """Test that default parameters work."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        polygons = mask_to_polygons(mask)

        assert len(polygons) == 1


class TestConvertMaskToYoloLabels:
    """Tests for convert_mask_to_yolo_labels function."""

    def test_single_class(self) -> None:
        """Test conversion with single class."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1  # Class 1

        lines = convert_mask_to_yolo_labels(mask, class_ids=[1], min_area=50)

        assert len(lines) == 1
        assert lines[0].startswith("1 ")

    def test_multiple_classes(self) -> None:
        """Test conversion with multiple classes."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Class 1
        mask[60:80, 60:80] = 2  # Class 2

        lines = convert_mask_to_yolo_labels(mask, class_ids=[1, 2], min_area=50)

        assert len(lines) == 2
        class_ids_in_lines = [int(line.split()[0]) for line in lines]
        assert 1 in class_ids_in_lines
        assert 2 in class_ids_in_lines

    def test_empty_mask_for_class(self) -> None:
        """Test that missing classes produce no output."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1  # Only class 1

        lines = convert_mask_to_yolo_labels(mask, class_ids=[1, 2], min_area=50)

        # Only class 1 should produce output
        assert len(lines) == 1
        assert lines[0].startswith("1 ")

    def test_multiple_instances_same_class(self) -> None:
        """Test multiple separate regions of same class."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Instance 1
        mask[60:80, 60:80] = 1  # Instance 2

        lines = convert_mask_to_yolo_labels(mask, class_ids=[1], min_area=50)

        assert len(lines) == 2
        assert all(line.startswith("1 ") for line in lines)


class TestConvertMaskToYoloSeg:
    """Tests for full conversion function."""

    def test_basic_conversion(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test basic conversion of mask dataset."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        output_path = tmp_path / "yolo_output"

        stats = convert_mask_to_yolo_seg(
            dataset=dataset,
            output_path=output_path,
            epsilon_factor=0.005,
            min_area=50,
        )

        # Check output structure
        assert output_path.exists()
        assert (output_path / "data.yaml").exists()
        assert (output_path / "images").exists()
        assert (output_path / "labels").exists()

        # Check stats
        assert stats["images"] > 0
        assert stats["labels"] > 0

    def test_creates_data_yaml(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test that data.yaml is created with correct content."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        output_path = tmp_path / "yolo_output"
        convert_mask_to_yolo_seg(dataset, output_path, min_area=50)

        data_yaml = output_path / "data.yaml"
        assert data_yaml.exists()

        with open(data_yaml) as f:
            data = yaml.safe_load(f)

        assert "path" in data
        assert "names" in data
        assert "train" in data or "val" in data

    def test_creates_split_directories(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test that split directories are created."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        output_path = tmp_path / "yolo_output"
        convert_mask_to_yolo_seg(dataset, output_path, min_area=50)

        # Check train split
        assert (output_path / "images" / "train").exists()
        assert (output_path / "labels" / "train").exists()

        # Check val split
        assert (output_path / "images" / "val").exists()
        assert (output_path / "labels" / "val").exists()

    def test_label_files_content(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test that label files contain valid YOLO format."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        output_path = tmp_path / "yolo_output"
        convert_mask_to_yolo_seg(dataset, output_path, min_area=50)

        # Find a label file
        label_files = list((output_path / "labels").rglob("*.txt"))
        assert len(label_files) > 0

        # Check content format
        content = label_files[0].read_text()
        for line in content.strip().split("\n"):
            parts = line.split()
            # Class ID should be integer
            class_id = int(parts[0])
            assert class_id >= 0
            # Rest should be coordinate pairs (x, y)
            coords = [float(p) for p in parts[1:]]
            assert len(coords) >= 6  # At least 3 points
            assert len(coords) % 2 == 0  # Even number (x, y pairs)
            # All coordinates should be normalized
            assert all(0.0 <= c <= 1.0 for c in coords)

    def test_copies_images(self, mask_dataset_grayscale: Path, tmp_path: Path) -> None:
        """Test that images are copied to output."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        output_path = tmp_path / "yolo_output"
        convert_mask_to_yolo_seg(dataset, output_path, min_area=50)

        # Check images exist
        output_images = list((output_path / "images").rglob("*.jpg"))
        assert len(output_images) > 0

    def test_progress_callback(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test that progress callback is called."""
        dataset = MaskDataset.detect(mask_dataset_grayscale)
        assert dataset is not None

        output_path = tmp_path / "yolo_output"
        progress_calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        convert_mask_to_yolo_seg(
            dataset, output_path, min_area=50, progress_callback=callback
        )

        assert len(progress_calls) > 0
        # Final call should have current == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_unsplit_dataset(self, mask_dataset_unsplit: Path, tmp_path: Path) -> None:
        """Test conversion of unsplit dataset."""
        dataset = MaskDataset.detect(mask_dataset_unsplit)
        assert dataset is not None
        assert dataset.splits == []  # Unsplit

        output_path = tmp_path / "yolo_output"
        stats = convert_mask_to_yolo_seg(dataset, output_path, min_area=50)

        # Should default to 'train' split
        assert (output_path / "images" / "train").exists()
        assert (output_path / "labels" / "train").exists()
        assert stats["images"] > 0


class TestConvertCLI:
    """Integration tests for convert CLI command."""

    def test_convert_command_basic(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test basic convert command execution."""
        output_path = tmp_path / "yolo_output"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(mask_dataset_grayscale),
                "-o",
                str(output_path),
                "--to",
                "yolo-seg",
            ],
        )

        assert result.exit_code == 0
        assert "Conversion complete" in result.stdout
        assert output_path.exists()

    def test_convert_command_with_params(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test convert command with custom parameters."""
        output_path = tmp_path / "yolo_output"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(mask_dataset_grayscale),
                "-o",
                str(output_path),
                "--to",
                "yolo-seg",
                "--epsilon-factor",
                "0.01",
                "--min-area",
                "50",
            ],
        )

        assert result.exit_code == 0
        assert "Conversion complete" in result.stdout

    def test_convert_invalid_format(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test error for unsupported target format."""
        output_path = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(mask_dataset_grayscale),
                "-o",
                str(output_path),
                "--to",
                "invalid-format",
            ],
        )

        assert result.exit_code == 1
        assert "Unsupported target format" in result.stdout

    def test_convert_invalid_input(self, tmp_path: Path) -> None:
        """Test error for invalid input path."""
        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(tmp_path / "nonexistent"),
                "-o",
                str(tmp_path / "output"),
                "--to",
                "yolo-seg",
            ],
        )

        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_convert_not_mask_dataset(
        self, yolo_detection_dataset: Path, tmp_path: Path
    ) -> None:
        """Test error when input is not a MaskDataset."""
        output_path = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(yolo_detection_dataset),
                "-o",
                str(output_path),
                "--to",
                "yolo-seg",
            ],
        )

        assert result.exit_code == 1
        assert "No MaskDataset found" in result.stdout

    def test_convert_output_exists(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test error when output directory already exists and not empty."""
        output_path = tmp_path / "yolo_output"
        output_path.mkdir()
        (output_path / "some_file.txt").write_text("existing content")

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(mask_dataset_grayscale),
                "-o",
                str(output_path),
                "--to",
                "yolo-seg",
            ],
        )

        assert result.exit_code == 1
        assert "already exists and is not empty" in result.stdout

    def test_convert_shows_stats(
        self, mask_dataset_grayscale: Path, tmp_path: Path
    ) -> None:
        """Test that stats are shown in output."""
        output_path = tmp_path / "yolo_output"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(mask_dataset_grayscale),
                "-o",
                str(output_path),
                "--to",
                "yolo-seg",
            ],
        )

        assert result.exit_code == 0
        assert "Images processed" in result.stdout
        assert "Labels created" in result.stdout
        assert "Polygons extracted" in result.stdout


class TestConversionParams:
    """Tests for ConversionParams dataclass."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        params = ConversionParams()

        assert params.class_id == 0
        assert params.epsilon_factor == 0.005
        assert params.min_area == 100.0

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        params = ConversionParams(
            class_id=5,
            epsilon_factor=0.01,
            min_area=200.0,
        )

        assert params.class_id == 5
        assert params.epsilon_factor == 0.01
        assert params.min_area == 200.0


@pytest.fixture
def mask_dataset_with_shapes(tmp_path: Path) -> Path:
    """Create a mask dataset with distinct shapes for testing polygon extraction."""
    dataset_path = tmp_path / "mask_shapes"
    dataset_path.mkdir()

    # Create directories
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "masks" / "train").mkdir(parents=True)

    # Create an image
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_path / "images" / "train" / "shapes.jpg"), img)

    # Create mask with distinct shapes
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Rectangle (class 1)
    mask[20:80, 20:100] = 1
    # Square (class 2)
    mask[120:180, 120:180] = 2
    cv2.imwrite(str(dataset_path / "masks" / "train" / "shapes.png"), mask)

    return dataset_path


class TestPolygonSimplification:
    """Tests for polygon simplification behavior."""

    def test_epsilon_factor_affects_simplification(self) -> None:
        """Test that larger epsilon produces simpler polygons."""
        # Create a circle-like shape (many points)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)

        # Low epsilon = more points
        params_low = ConversionParams(epsilon_factor=0.001, min_area=50)
        polygons_low = mask_to_polygons(mask, params_low)

        # High epsilon = fewer points
        params_high = ConversionParams(epsilon_factor=0.05, min_area=50)
        polygons_high = mask_to_polygons(mask, params_high)

        assert len(polygons_low) == 1
        assert len(polygons_high) == 1
        # Lower epsilon should produce more points
        assert len(polygons_low[0].points) >= len(polygons_high[0].points)
