# Dataset formats

Argus supports YOLO, COCO, and folder-based semantic mask datasets. Detection
and segmentation are handled out of the box.

## YOLO

Argus looks for a YAML config file with a `names` key. It uses that file to
extract class names and verify the dataset layout.

Typical structure:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Unsplit YOLO datasets are also supported:

```text
dataset/
├── data.yaml
├── images/
└── labels/
```

Argus infers the task type by scanning a few label files:

- 5 values per line: detection
- more than 5 values per line: segmentation polygons

## COCO

Argus looks for COCO annotation JSON files in `annotations/` or at the dataset
root.

Typical structure:

```text
dataset/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

If your annotation filenames include `train`, `val`, or `test`, Argus will treat
those as splits. Otherwise it defaults to `train`.

### Roboflow COCO

Argus also supports the Roboflow variant of COCO format, where annotations live
inside split directories:

```text
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

Splits are detected from directory names (`train`, `valid`/`val`, `test`).

## Mask (semantic segmentation)

Mask datasets are simple image + mask folders. Argus detects a few common
patterns:

- `images/` + `masks/`
- `img/` + `gt/`
- `leftImg8bit/` + `gtFine/` (Cityscapes-style)

Split-aware layout:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── masks/
│   ├── train/
│   └── val/
└── classes.yaml  # Optional for grayscale, required for RGB palette masks
```

Unsplit layout:

```text
dataset/
├── images/
├── masks/
└── classes.yaml
```

### Mask encoding

- Grayscale masks: each pixel value is the class ID. Argus will auto-detect
  class IDs if no `classes.yaml` is provided.
- RGB palette masks: each class maps to a color. A `classes.yaml` is required.
- Mask files should be `.png` and match the image stem (e.g., `frame.png`), or
  use common suffixes like `_mask`, `_gt`, or `_label`.

Example `classes.yaml`:

```yaml
names:
  - background
  - road
  - sidewalk
ignore_index: 255
palette:
  - id: 0
    name: background
  - id: 1
    name: road
```

## Detection heuristics

If Argus does not detect your dataset, check the following:

- The dataset root is correct and readable.
- YOLO: the YAML file includes `names` as a list or dict.
- COCO: `images`, `annotations`, and `categories` exist in the JSON.
