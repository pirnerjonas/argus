# Validating datasets

The `validate` command checks a dataset for annotation quality issues before
training. It detects problems like missing images, invalid coordinates, broken
references, and format-specific inconsistencies.

## Quick start

```bash
argus-cv validate /datasets/retail
```

If the dataset is clean, you see:

```
Dataset is valid: retail (yolo)
No issues found.
```

Otherwise a table of errors and warnings is printed, and the process exits
with code 1 when errors are present.

## Options

| Flag | Description |
|------|-------------|
| `--strict` | Treat warnings as failures (exit code 1) |
| `--split` | Validate only a specific split (`train`, `val`, `test`) |
| `--max-issues N` | Limit displayed issues (`0` = show all, the default) |
| `--check-images` | Verify each image file is readable with OpenCV (slower) |

### Examples

```bash
# Strict mode: fail on warnings too
argus-cv validate /datasets/retail --strict

# Only validate the train split
argus-cv validate /datasets/retail --split train

# Limit output to the first 20 issues
argus-cv validate /datasets/retail --max-issues 20

# Also check that image files can actually be decoded
argus-cv validate /datasets/retail --check-images
```

## Error codes

Issues are reported with a level (**error** or **warning**) and a code.
Errors indicate problems that will likely break training; warnings flag
quality concerns worth reviewing.

### Universal checks

| Code | Level | Description |
|------|-------|-------------|
| E101 | error | Image file does not exist or is unreadable |
| E102 | error | Duplicate image filename within a split |
| W102 | warning | Split contains no images |

### YOLO-specific

| Code | Level | Description |
|------|-------|-------------|
| E201 | error | Wrong number of columns in label line |
| E202 | error | Invalid or out-of-range class ID |
| E203 | error | Coordinate value out of `[0, 1]` bounds or non-numeric |
| E204 | error | Zero or negative bounding box dimensions |
| W201 | warning | Label file has no corresponding image (orphan) |
| W202 | warning | Very small bounding box (< 0.001 in width or height) |

### COCO-specific

| Code | Level | Description |
|------|-------|-------------|
| E301 | error | Annotation references non-existent `image_id` |
| E302 | error | Annotation references non-existent `category_id` |
| E303 | error | Duplicate annotation ID |
| E304 | error | Bounding box with zero or negative dimensions |
| E305 | error | Duplicate image ID |
| W301 | warning | Image has no annotations |
| W302 | warning | Polygon has fewer than 3 points |

### Mask-specific

| Code | Level | Description |
|------|-------|-------------|
| E401 | error | No corresponding mask file for an image |
| E402 | error | Image and mask dimensions do not match |
| E403 | error | Mask contains unexpected pixel values |

## Python API

You can run validation programmatically:

```python
from argus.core import YOLODataset, validate_dataset

dataset = YOLODataset.detect("/datasets/retail")
report = validate_dataset(dataset)

print(f"Valid: {report.is_valid}")
print(f"Errors: {len(report.errors)}")
print(f"Warnings: {len(report.warnings)}")

for issue in report.issues:
    print(f"[{issue.code}] {issue.message}")
```

Use the `split` parameter to scope validation:

```python
report = validate_dataset(dataset, split="train")
```

Enable image readability checks:

```python
report = validate_dataset(dataset, check_images=True)
```
