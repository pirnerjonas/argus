"""Tests for dataset validation functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from argus.cli import app
from argus.core.base import TaskType


class TestValidationDataStructures:
    """Tests for validation data structures."""

    def test_outlier_sample(self) -> None:
        """Test OutlierSample dataclass."""
        from argus.core.validate import OutlierSample

        sample = OutlierSample(
            path=Path("/test/img.jpg"),
            class_name="cat",
            distance=2.5,
        )

        assert sample.path == Path("/test/img.jpg")
        assert sample.class_name == "cat"
        assert sample.distance == 2.5

    def test_class_validation_result(self) -> None:
        """Test ClassValidationResult dataclass."""
        from argus.core.validate import ClassValidationResult, OutlierSample

        outlier = OutlierSample(
            path=Path("/test/img.jpg"),
            class_name="cat",
            distance=2.5,
        )

        result = ClassValidationResult(
            class_name="cat",
            total_samples=100,
            outliers=[outlier],
        )

        assert result.class_name == "cat"
        assert result.total_samples == 100
        assert len(result.outliers) == 1

    def test_validation_result(self) -> None:
        """Test ValidationResult dataclass."""
        from argus.core.validate import (
            ClassValidationResult,
            OutlierSample,
            ValidationResult,
        )

        outlier1 = OutlierSample(Path("/test/cat/img1.jpg"), "cat", 2.5)
        outlier2 = OutlierSample(Path("/test/cat/img2.jpg"), "cat", 3.0)
        outlier3 = OutlierSample(Path("/test/dog/img1.jpg"), "dog", 2.8)

        class_results = [
            ClassValidationResult("cat", 50, [outlier1, outlier2]),
            ClassValidationResult("dog", 30, [outlier3]),
            ClassValidationResult("bird", 20, []),
        ]

        result = ValidationResult(
            dataset_path=Path("/test/dataset"),
            task_type=TaskType.CLASSIFICATION,
            num_classes=3,
            total_images=100,
            threshold=2.0,
            split="train",
            class_results=class_results,
        )

        assert result.total_outliers == 3
        assert result.classes_with_outliers == 2

    def test_validation_result_no_outliers(self) -> None:
        """Test ValidationResult with no outliers."""
        from argus.core.validate import ClassValidationResult, ValidationResult

        class_results = [
            ClassValidationResult("cat", 50, []),
            ClassValidationResult("dog", 30, []),
        ]

        result = ValidationResult(
            dataset_path=Path("/test/dataset"),
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            total_images=80,
            threshold=2.0,
            split=None,
            class_results=class_results,
        )

        assert result.total_outliers == 0
        assert result.classes_with_outliers == 0


class TestAIDependencyCheck:
    """Tests for AI dependency checking."""

    def test_ai_features_not_available_error(self) -> None:
        """Test AIFeaturesNotAvailable exception."""
        from argus.core.validate import AIFeaturesNotAvailable

        error = AIFeaturesNotAvailable()
        assert "pip install argus-cv[ai]" in str(error)


class TestOutlierDetection:
    """Tests for outlier detection logic."""

    def test_detect_outliers_basic(self) -> None:
        """Test basic outlier detection with mock data."""
        from argus.core.validate import DatasetValidator

        # Create mock embeddings with one clear outlier
        # Normal samples clustered around origin
        embeddings = np.array(
            [
                [0.1, 0.1],
                [0.0, 0.1],
                [0.1, 0.0],
                [-0.1, 0.1],
                [0.1, -0.1],
                [5.0, 5.0],  # Outlier - far from cluster
            ]
        )
        image_paths = [Path(f"/test/img{i}.jpg") for i in range(6)]

        # Test the detection logic directly
        with patch.object(DatasetValidator, "__init__", lambda self, **kwargs: None):
            validator = DatasetValidator.__new__(DatasetValidator)
            outliers = validator._detect_outliers(
                embeddings, image_paths, "test_class", threshold=2.0
            )

        assert len(outliers) >= 1
        # The outlier should be the last image (5.0, 5.0)
        outlier_paths = [o.path for o in outliers]
        assert Path("/test/img5.jpg") in outlier_paths

    def test_detect_outliers_no_outliers(self) -> None:
        """Test detection when all samples are similar."""
        from argus.core.validate import DatasetValidator

        # All samples close together
        embeddings = np.array(
            [
                [0.1, 0.1],
                [0.11, 0.09],
                [0.09, 0.11],
                [0.1, 0.1],
            ]
        )
        image_paths = [Path(f"/test/img{i}.jpg") for i in range(4)]

        with patch.object(DatasetValidator, "__init__", lambda self, **kwargs: None):
            validator = DatasetValidator.__new__(DatasetValidator)
            outliers = validator._detect_outliers(
                embeddings, image_paths, "test_class", threshold=2.0
            )

        # With tight clustering, there should be no significant outliers
        assert len(outliers) == 0

    def test_detect_outliers_too_few_samples(self) -> None:
        """Test detection with less than 2 samples."""
        from argus.core.validate import DatasetValidator

        embeddings = np.array([[0.1, 0.1]])
        image_paths = [Path("/test/img0.jpg")]

        with patch.object(DatasetValidator, "__init__", lambda self, **kwargs: None):
            validator = DatasetValidator.__new__(DatasetValidator)
            outliers = validator._detect_outliers(
                embeddings, image_paths, "test_class", threshold=2.0
            )

        assert len(outliers) == 0

    def test_detect_outliers_identical_samples(self) -> None:
        """Test detection when all samples are identical."""
        from argus.core.validate import DatasetValidator

        embeddings = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
        image_paths = [Path(f"/test/img{i}.jpg") for i in range(3)]

        with patch.object(DatasetValidator, "__init__", lambda self, **kwargs: None):
            validator = DatasetValidator.__new__(DatasetValidator)
            outliers = validator._detect_outliers(
                embeddings, image_paths, "test_class", threshold=2.0
            )

        # No outliers when all samples are identical (std = 0)
        assert len(outliers) == 0


class TestValidateCLI:
    """Tests for the validate CLI command."""

    def test_validate_missing_path(self, tmp_path: Path) -> None:
        """Test validate with non-existent path."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent")])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_validate_not_directory(self, tmp_path: Path) -> None:
        """Test validate with file path instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(file_path)])

        assert result.exit_code == 1
        assert "not a directory" in result.output

    def test_validate_no_dataset(self, tmp_path: Path) -> None:
        """Test validate with directory that has no dataset."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(tmp_path)])

        assert result.exit_code == 1
        assert "No YOLO or COCO dataset found" in result.output

    def test_validate_non_classification_dataset(
        self, yolo_detection_dataset: Path
    ) -> None:
        """Test validate rejects non-classification datasets."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(yolo_detection_dataset)])

        assert result.exit_code == 1
        assert "only supports classification" in result.output

    def test_validate_invalid_split(self, yolo_classification_dataset: Path) -> None:
        """Test validate with invalid split name."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["validate", str(yolo_classification_dataset), "--split", "nonexistent"],
        )

        assert result.exit_code == 1
        assert "Split 'nonexistent' not found" in result.output

    @patch("argus.core.validate.DatasetValidator")
    def test_validate_success_no_outliers(
        self, mock_validator_class: MagicMock, yolo_classification_dataset: Path
    ) -> None:
        """Test successful validation with no outliers."""
        from argus.core.validate import ClassValidationResult, ValidationResult

        # Create mock validation result
        mock_result = ValidationResult(
            dataset_path=yolo_classification_dataset,
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            total_images=5,
            threshold=2.0,
            split=None,
            class_results=[
                ClassValidationResult("cat", 3, []),
                ClassValidationResult("dog", 2, []),
            ],
        )

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(yolo_classification_dataset)])

        assert result.exit_code == 0
        assert "No outliers found" in result.output

    @patch("argus.core.validate.DatasetValidator")
    def test_validate_success_with_outliers(
        self, mock_validator_class: MagicMock, yolo_classification_dataset: Path
    ) -> None:
        """Test successful validation with outliers."""
        from argus.core.validate import (
            ClassValidationResult,
            OutlierSample,
            ValidationResult,
        )

        # Create mock validation result with outliers
        img_path = (
            yolo_classification_dataset / "images" / "train" / "cat" / "img001.jpg"
        )
        outlier = OutlierSample(
            path=img_path,
            class_name="cat",
            distance=2.5,
        )

        mock_result = ValidationResult(
            dataset_path=yolo_classification_dataset,
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            total_images=5,
            threshold=2.0,
            split=None,
            class_results=[
                ClassValidationResult("cat", 3, [outlier]),
                ClassValidationResult("dog", 2, []),
            ],
        )

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(yolo_classification_dataset)])

        assert result.exit_code == 0
        assert "Outliers found" in result.output
        assert "cat" in result.output
        assert "2.5" in result.output

    @patch("argus.core.validate.DatasetValidator")
    def test_validate_json_output(
        self, mock_validator_class: MagicMock, yolo_classification_dataset: Path
    ) -> None:
        """Test JSON output format."""
        import json

        from argus.core.validate import (
            ClassValidationResult,
            OutlierSample,
            ValidationResult,
        )

        img_path = (
            yolo_classification_dataset / "images" / "train" / "cat" / "img001.jpg"
        )
        outlier = OutlierSample(
            path=img_path,
            class_name="cat",
            distance=2.5,
        )

        mock_result = ValidationResult(
            dataset_path=yolo_classification_dataset,
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            total_images=5,
            threshold=2.0,
            split="train",
            class_results=[
                ClassValidationResult("cat", 3, [outlier]),
                ClassValidationResult("dog", 2, []),
            ],
        )

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        runner = CliRunner()
        result = runner.invoke(
            app, ["validate", str(yolo_classification_dataset), "--format", "json"]
        )

        assert result.exit_code == 0
        # Parse JSON output
        output_data = json.loads(result.output)
        assert output_data["task_type"] == "classification"
        assert output_data["total_outliers"] == 1
        assert output_data["split"] == "train"
        assert len(output_data["class_results"]) == 2

    @patch("argus.core.validate.DatasetValidator")
    def test_validate_with_threshold(
        self, mock_validator_class: MagicMock, yolo_classification_dataset: Path
    ) -> None:
        """Test validation with custom threshold."""
        from argus.core.validate import ClassValidationResult, ValidationResult

        mock_result = ValidationResult(
            dataset_path=yolo_classification_dataset,
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            total_images=5,
            threshold=3.0,
            split=None,
            class_results=[
                ClassValidationResult("cat", 3, []),
                ClassValidationResult("dog", 2, []),
            ],
        )

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        runner = CliRunner()
        result = runner.invoke(
            app, ["validate", str(yolo_classification_dataset), "--threshold", "3.0"]
        )

        assert result.exit_code == 0
        # Verify threshold was passed
        mock_validator.validate.assert_called_once()
        call_kwargs = mock_validator.validate.call_args[1]
        assert call_kwargs["threshold"] == 3.0

    @patch("argus.core.validate.DatasetValidator")
    def test_validate_with_split(
        self, mock_validator_class: MagicMock, yolo_classification_dataset: Path
    ) -> None:
        """Test validation with specific split."""
        from argus.core.validate import ClassValidationResult, ValidationResult

        mock_result = ValidationResult(
            dataset_path=yolo_classification_dataset,
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            total_images=3,
            threshold=2.0,
            split="train",
            class_results=[
                ClassValidationResult("cat", 2, []),
                ClassValidationResult("dog", 1, []),
            ],
        )

        mock_validator = MagicMock()
        mock_validator.validate.return_value = mock_result
        mock_validator_class.return_value = mock_validator

        runner = CliRunner()
        result = runner.invoke(
            app, ["validate", str(yolo_classification_dataset), "--split", "train"]
        )

        assert result.exit_code == 0
        mock_validator.validate.assert_called_once()
        call_kwargs = mock_validator.validate.call_args[1]
        assert call_kwargs["split"] == "train"


class TestValidatorValidate:
    """Tests for DatasetValidator.validate method."""

    def test_validate_non_classification_raises(
        self, yolo_detection_dataset: Path
    ) -> None:
        """Test that validating non-classification dataset raises."""
        from argus.core.validate import DatasetValidator
        from argus.core.yolo import YOLODataset

        dataset = YOLODataset.detect(yolo_detection_dataset)
        assert dataset is not None

        with patch.object(DatasetValidator, "__init__", lambda self, **kwargs: None):
            validator = DatasetValidator.__new__(DatasetValidator)
            with pytest.raises(ValueError, match="only supports classification"):
                validator.validate(dataset)

    def test_validate_non_yolo_raises(self) -> None:
        """Test that validating non-YOLO dataset raises."""
        from argus.core.base import Dataset, DatasetFormat, TaskType
        from argus.core.validate import DatasetValidator

        # Create a mock non-YOLO classification dataset
        class MockDataset(Dataset):
            def __init__(self) -> None:
                super().__init__(
                    path=Path("/mock"),
                    format=DatasetFormat.COCO,
                    task=TaskType.CLASSIFICATION,
                )

            @classmethod
            def detect(cls, path: Path) -> "Dataset | None":
                return None

            def get_instance_counts(self) -> dict:
                return {}

            def get_image_counts(self) -> dict:
                return {}

            def get_image_paths(self, split: str | None = None) -> list:
                return []

            def get_annotations_for_image(self, image_path: Path) -> list:
                return []

        mock_dataset = MockDataset()

        with patch.object(DatasetValidator, "__init__", lambda self, **kwargs: None):
            validator = DatasetValidator.__new__(DatasetValidator)
            with pytest.raises(ValueError, match="only supports YOLO"):
                validator.validate(mock_dataset)
