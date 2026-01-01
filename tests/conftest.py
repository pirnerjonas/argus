"""Test fixtures for dataset detection tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def yolo_detection_dataset(tmp_path: Path) -> Path:
    """Create a valid YOLO detection dataset.

    Structure:
        dataset/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   │   └── img001.jpg
        │   └── val/
        │       └── img002.jpg
        └── labels/
            ├── train/
            │   └── img001.txt
            └── val/
                └── img002.txt
    """
    dataset_path = tmp_path / "yolo_detection"
    dataset_path.mkdir()

    # Create data.yaml
    yaml_content = """
path: .
train: images/train
val: images/val
names:
  0: person
  1: car
  2: bicycle
"""
    (dataset_path / "data.yaml").write_text(yaml_content)

    # Create directory structure
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "val").mkdir(parents=True)

    # Create dummy images (just empty files for testing)
    (dataset_path / "images" / "train" / "img001.jpg").write_bytes(b"fake image")
    (dataset_path / "images" / "val" / "img002.jpg").write_bytes(b"fake image")

    # Create detection labels (5 columns: class x_center y_center width height)
    train_label = "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2\n"
    (dataset_path / "labels" / "train" / "img001.txt").write_text(train_label)
    (dataset_path / "labels" / "val" / "img002.txt").write_text("2 0.6 0.4 0.15 0.25\n")

    return dataset_path


@pytest.fixture
def yolo_segmentation_dataset(tmp_path: Path) -> Path:
    """Create a valid YOLO segmentation dataset.

    Labels have polygon coordinates (>5 columns).
    """
    dataset_path = tmp_path / "yolo_segmentation"
    dataset_path.mkdir()

    # Create dataset.yml (different naming)
    yaml_content = """
path: .
train: images/train
val: images/val
names:
  0: cat
  1: dog
"""
    (dataset_path / "dataset.yml").write_text(yaml_content)

    # Create directory structure
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "images" / "val").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "val").mkdir(parents=True)

    # Create dummy images
    (dataset_path / "images" / "train" / "img001.jpg").write_bytes(b"fake image")
    (dataset_path / "images" / "val" / "img002.jpg").write_bytes(b"fake image")

    # Create segmentation labels (polygon: class x1 y1 x2 y2 x3 y3 ...)
    (dataset_path / "labels" / "train" / "img001.txt").write_text(
        "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1\n"
    )
    (dataset_path / "labels" / "val" / "img002.txt").write_text(
        "1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\n"
    )

    return dataset_path


@pytest.fixture
def yolo_flat_structure_dataset(tmp_path: Path) -> Path:
    """Create a YOLO dataset with flat structure (no train/val subfolders).

    Structure:
        dataset/
        ├── data.yaml
        ├── images/
        │   └── img001.jpg
        └── labels/
            └── img001.txt
    """
    dataset_path = tmp_path / "yolo_flat"
    dataset_path.mkdir()

    # Create data.yaml
    yaml_content = """
names:
  0: object1
  1: object2
"""
    (dataset_path / "data.yaml").write_text(yaml_content)

    # Create flat structure
    (dataset_path / "images").mkdir()
    (dataset_path / "labels").mkdir()

    # Add files directly in images/labels
    (dataset_path / "images" / "img001.jpg").write_bytes(b"fake image")
    (dataset_path / "images" / "img002.png").write_bytes(b"fake image")
    (dataset_path / "labels" / "img001.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    (dataset_path / "labels" / "img002.txt").write_text("1 0.3 0.7 0.1 0.2\n")

    return dataset_path


@pytest.fixture
def coco_detection_dataset(tmp_path: Path) -> Path:
    """Create a valid COCO detection dataset.

    Structure:
        dataset/
        ├── annotations/
        │   └── instances_train.json
        └── images/
            └── train/
                └── img001.jpg
    """
    dataset_path = tmp_path / "coco_detection"
    dataset_path.mkdir()

    # Create annotations directory
    annotations_dir = dataset_path / "annotations"
    annotations_dir.mkdir()

    # Create COCO annotation file
    coco_data = {
        "info": {"description": "Test dataset"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }

    (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

    # Create images directory
    images_dir = dataset_path / "images" / "train"
    images_dir.mkdir(parents=True)
    (images_dir / "img001.jpg").write_bytes(b"fake image")
    (images_dir / "img002.jpg").write_bytes(b"fake image")

    return dataset_path


@pytest.fixture
def coco_segmentation_dataset(tmp_path: Path) -> Path:
    """Create a valid COCO segmentation dataset.

    Annotations include segmentation field.
    """
    dataset_path = tmp_path / "coco_segmentation"
    dataset_path.mkdir()

    # Create annotations directory
    annotations_dir = dataset_path / "annotations"
    annotations_dir.mkdir()

    # Create COCO annotation file with segmentation
    coco_data = {
        "info": {"description": "Test segmentation dataset"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]],
                "area": 30000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"},
            {"id": 3, "name": "bird", "supercategory": "animal"},
        ],
    }

    (annotations_dir / "instances_val.json").write_text(json.dumps(coco_data))

    # Create images directory
    images_dir = dataset_path / "images" / "val"
    images_dir.mkdir(parents=True)
    (images_dir / "img001.jpg").write_bytes(b"fake image")

    return dataset_path


@pytest.fixture
def invalid_yaml_missing_names(tmp_path: Path) -> Path:
    """Create an invalid YOLO dataset (YAML missing 'names' key)."""
    dataset_path = tmp_path / "invalid_yaml"
    dataset_path.mkdir()

    # Create YAML without 'names' key
    yaml_content = """
path: .
train: images/train
val: images/val
"""
    (dataset_path / "data.yaml").write_text(yaml_content)

    # Create directory structure
    (dataset_path / "images" / "train").mkdir(parents=True)
    (dataset_path / "labels" / "train").mkdir(parents=True)

    return dataset_path


@pytest.fixture
def invalid_coco_missing_categories(tmp_path: Path) -> Path:
    """Create an invalid COCO dataset (JSON missing 'categories' key)."""
    dataset_path = tmp_path / "invalid_coco"
    dataset_path.mkdir()

    annotations_dir = dataset_path / "annotations"
    annotations_dir.mkdir()

    # Create JSON without 'categories' key
    coco_data = {
        "images": [{"id": 1, "file_name": "img001.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [100, 100, 200, 150]}],
    }

    (annotations_dir / "instances_train.json").write_text(json.dumps(coco_data))

    return dataset_path


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Create an empty directory."""
    dataset_path = tmp_path / "empty"
    dataset_path.mkdir()
    return dataset_path


@pytest.fixture
def random_files_directory(tmp_path: Path) -> Path:
    """Create a directory with random non-dataset files."""
    dataset_path = tmp_path / "random"
    dataset_path.mkdir()

    # Create various non-dataset files
    (dataset_path / "readme.txt").write_text("This is a readme")
    (dataset_path / "config.json").write_text('{"key": "value"}')
    (dataset_path / "script.py").write_text("print('hello')")
    (dataset_path / "notes.yaml").write_text("notes:\n  - item1\n  - item2")

    return dataset_path


@pytest.fixture
def nested_datasets(tmp_path: Path) -> Path:
    """Create a directory structure with multiple nested datasets."""
    root_path = tmp_path / "nested"
    root_path.mkdir()

    # Create YOLO dataset at depth 1
    yolo_path = root_path / "yolo_data"
    yolo_path.mkdir()
    yaml_content = """
names:
  0: class1
  1: class2
"""
    (yolo_path / "data.yaml").write_text(yaml_content)
    (yolo_path / "images").mkdir()
    (yolo_path / "labels").mkdir()
    (yolo_path / "images" / "img.jpg").write_bytes(b"fake")
    (yolo_path / "labels" / "img.txt").write_text("0 0.5 0.5 0.2 0.3\n")

    # Create COCO dataset at depth 2
    coco_path = root_path / "subdir" / "coco_data"
    coco_path.mkdir(parents=True)
    annotations_dir = coco_path / "annotations"
    annotations_dir.mkdir()
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]}
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    (annotations_dir / "annotations.json").write_text(json.dumps(coco_data))

    return root_path
