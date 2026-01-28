# Filtering datasets

Use `argus-cv filter` to create a filtered copy of a dataset containing only specified classes.

## Basic usage

```bash
argus-cv filter -d /datasets/coco -o /datasets/coco_filtered --classes person,car
```

This creates a new dataset with only the `person` and `car` classes. Class IDs are automatically remapped to sequential values (0, 1, 2, ...).

## Filter to a single class

```bash
argus-cv filter -d /datasets/yolo -o /datasets/yolo_balls --classes ball
```

## Exclude background images

By default, images without annotations (after filtering) are kept. Use `--no-background` to exclude them:

```bash
argus-cv filter -d /datasets/coco -o /datasets/coco_filtered --classes dog --no-background
```

This is useful when you want a dataset with only images that contain your target class.

## Use symlinks for faster filtering

For large datasets, use `--symlinks` to create symbolic links instead of copying images:

```bash
argus-cv filter -d /datasets/large -o /datasets/filtered --classes cat --symlinks
```

This saves disk space and speeds up the filtering process significantly.

## Supported formats

The filter command works with all dataset formats:

| Format | Supported | Notes |
|--------|-----------|-------|
| YOLO Detection | Yes | Labels remapped to new class IDs |
| YOLO Segmentation | Yes | Polygon annotations preserved |
| YOLO Classification | Yes | Only selected class directories copied |
| COCO | Yes | Annotations and category IDs remapped |
| Mask | Yes | Pixel values remapped to new class IDs |

## Output layout

The output preserves the original dataset structure with train/val/test splits.

YOLO output:

```text
output/
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

COCO output:

```text
output/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

## Class ID remapping

When filtering, class IDs are remapped to start from 0 and be sequential. For example:

| Original | Filtered |
|----------|----------|
| 0: person | (removed) |
| 1: car | 0: car |
| 2: dog | 1: dog |
| 3: cat | (removed) |

If you filter to keep only `car` and `dog`, the new dataset will have `car` as class 0 and `dog` as class 1.

## Common errors

- "No classes specified": You must provide at least one class name with `--classes`.
- "Classes not found in dataset": Check the class names match exactly (case-sensitive). Use `argus-cv stats` to see available classes.
- "Output directory already exists": The output directory must be empty or non-existent.
