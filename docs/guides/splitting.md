# Splitting datasets

Use `argus split` to create train/val/test splits from an unsplit dataset.

## Basic split

```bash
argus split -d /datasets/animals -o /datasets/animals_splits
```

By default, Argus uses a 0.8/0.1/0.1 ratio and stratified sampling.

## Custom ratio

```bash
argus split -d /datasets/animals -o /datasets/animals_splits -r 0.7,0.2,0.1
```

Ratios can sum to 1.0 or 100.

## Disable stratification

```bash
argus split -d /datasets/animals -o /datasets/animals_splits --no-stratify
```

## Set a seed for determinism

```bash
argus split -d /datasets/animals -o /datasets/animals_splits --seed 7
```

## Output layout

YOLO splits are written like this:

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

COCO splits are written like this:

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

## Common errors

- "Dataset already has splits": Argus only splits datasets that are unsplit.
- "No images found": make sure `images/` exists and matches labels.
