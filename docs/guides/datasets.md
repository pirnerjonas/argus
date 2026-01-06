# Dataset formats

Argus supports YOLO and COCO datasets. Detection and segmentation are handled
out of the box.

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

## Detection heuristics

If Argus does not detect your dataset, check the following:

- The dataset root is correct and readable.
- YOLO: the YAML file includes `names` as a list or dict.
- COCO: `images`, `annotations`, and `categories` exist in the JSON.
