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

### Roboflow YOLO

Argus also supports the Roboflow variant of the YOLO format, where images and
labels live inside split directories instead of under a top-level `images/` and
`labels/` folder:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Roboflow uses `valid` instead of `val` for the validation split. Argus detects
this automatically and normalises it to `val` internally. The `data.yaml` paths
typically use `../train/images` style references — Argus handles these by
falling back to filesystem probing when the relative paths do not resolve.

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

### COCO RLE segmentation

COCO annotations can encode instance masks using Run-Length Encoding (RLE)
instead of polygon point lists. Argus detects RLE annotations automatically and
renders them as pixel-level mask overlays in the viewer.

Both RLE variants are supported:

- **Uncompressed RLE** — `counts` is a list of integers:
  ```json
  {
    "segmentation": {
      "counts": [0, 5, 10, 3, 82],
      "size": [100, 100]
    }
  }
  ```
- **Compressed RLE** — `counts` is a COCO-style LEB128 string:
  ```json
  {
    "segmentation": {
      "counts": "eNpjYBgFo2AU0AsABEAAEQ==",
      "size": [100, 100]
    }
  }
  ```

Datasets that mix RLE and polygon annotations in the same file are handled
correctly. When you run `argus view` on a COCO dataset with RLE annotations,
Argus opens the mask overlay viewer instead of the bounding-box viewer.

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
