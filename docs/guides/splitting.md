# Split and unsplit datasets

Use `argus-cv split` to create train/val/test splits from an unsplit dataset,
and `argus-cv unsplit` to merge split datasets back into a flat layout.

## Basic split

```bash
argus-cv split /datasets/animals -o /datasets/animals_splits
```

Argus uses a 0.8/0.1/0.1 ratio and stratified sampling by default.

## Custom ratio

```bash
argus-cv split /datasets/animals -o /datasets/animals_splits -r 0.7,0.2,0.1
```

Ratios can sum to 1.0 or 100.

## Set a seed for determinism

```bash
argus-cv split /datasets/animals -o /datasets/animals_splits --seed 7
```

## Merge back to unsplit

```bash
argus-cv unsplit /datasets/animals_splits -o /datasets/animals_unsplit
```

If your split directories contain duplicate filenames, choose a collision
strategy:

```bash
argus-cv unsplit /datasets/animals_splits -o /datasets/animals_unsplit --collision-policy prefix-split
```

`--collision-policy` options:

- `error` (default): fail on collisions
- `prefix-split`: prefix duplicates with split name
- `hash`: suffix duplicates with a deterministic short hash

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

Mask splits are written like this:

```text
output/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

## Common errors

- "Dataset already has splits": Argus only splits datasets that are unsplit.
- "Dataset is already unsplit": Argus only unsplits datasets that already have splits.
- "No images found": make sure `images/` exists and matches labels.
