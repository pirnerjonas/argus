"""Tests for dataset splitting utilities and CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from argus.cli import app
from argus.core import COCODataset, YOLODataset
from argus.core.split import parse_ratio, split_coco_dataset, split_yolo_dataset

runner = CliRunner()


def _create_unsplit_yolo_dataset(dataset_path: Path) -> None:
    dataset_path.mkdir(parents=True)
    (dataset_path / "images").mkdir()
    (dataset_path / "labels").mkdir()
    (dataset_path / "data.yaml").write_text(
        "\n".join(
            [
                "names:",
                "  0: class_a",
                "  1: class_b",
            ]
        )
    )

    for idx in range(1, 7):
        (dataset_path / "images" / f"img{idx:03d}.jpg").write_bytes(b"fake image")
        label_line = "0 0.5 0.5 0.2 0.3\n" if idx % 2 == 0 else "1 0.3 0.4 0.1 0.2\n"
        (dataset_path / "labels" / f"img{idx:03d}.txt").write_text(label_line)


def test_parse_ratio_accepts_percentages() -> None:
    ratios = parse_ratio("80,10,10")
    assert ratios == (0.8, 0.1, 0.1)


def test_split_yolo_dataset_creates_splits(tmp_path: Path) -> None:
    dataset_path = tmp_path / "yolo_unsplit"
    _create_unsplit_yolo_dataset(dataset_path)

    dataset = YOLODataset.detect(dataset_path)
    assert dataset is not None

    output_path = tmp_path / "yolo_output"
    counts = split_yolo_dataset(dataset, output_path, (0.8, 0.1, 0.1), True, seed=7)

    assert counts == {"train": 4, "val": 1, "test": 1}
    assert (output_path / "data.yaml").exists()
    for split in ("train", "val", "test"):
        assert (output_path / "images" / split).is_dir()
        assert (output_path / "labels" / split).is_dir()
        assert len(list((output_path / "images" / split).iterdir())) == counts[split]
        assert len(list((output_path / "labels" / split).iterdir())) == counts[split]


def test_split_coco_dataset_creates_splits(
    tmp_path: Path, coco_unsplit_dataset: Path
) -> None:
    dataset = COCODataset.detect(coco_unsplit_dataset)
    assert dataset is not None
    annotation_file = dataset.annotation_files[0]

    output_path = tmp_path / "coco_output"
    counts = split_coco_dataset(
        dataset, annotation_file, output_path, (0.8, 0.1, 0.1), True, seed=4
    )

    assert counts == {"train": 4, "val": 1, "test": 1}
    for split in ("train", "val", "test"):
        assert (output_path / "annotations" / f"instances_{split}.json").exists()
        assert (output_path / "images" / split).is_dir()


def test_split_command_yolo_unsplit(tmp_path: Path) -> None:
    dataset_path = tmp_path / "yolo_unsplit_cli"
    _create_unsplit_yolo_dataset(dataset_path)

    output_path = tmp_path / "cli_output"
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset-path",
            str(dataset_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert (output_path / "images" / "train").is_dir()
    assert (output_path / "labels" / "train").is_dir()


def test_split_command_rejects_existing_splits(yolo_detection_dataset: Path) -> None:
    output_path = yolo_detection_dataset / "output"
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset-path",
            str(yolo_detection_dataset),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "already has splits" in result.stdout.lower()


def test_split_command_coco_unsplit(tmp_path: Path, coco_unsplit_dataset: Path) -> None:
    output_path = tmp_path / "coco_cli_output"
    result = runner.invoke(
        app,
        [
            "split",
            "--dataset-path",
            str(coco_unsplit_dataset),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert (output_path / "annotations" / "instances_train.json").exists()
