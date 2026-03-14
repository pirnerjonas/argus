# Plan: Issue #72 — Dataset Validation Command (`argus-cv validate`)

## Goal
Add an `argus-cv validate <path>` CLI command that detects annotation quality issues (errors and warnings) before training, for YOLO, COCO, and mask format datasets.

---

## Step 1: Create validation data structures (`src/argus/core/validation.py`)

Define the core data structures used by all validators:

```python
@dataclass
class ValidationIssue:
    level: Literal["error", "warning"]
    code: str              # e.g. "E001", "W001"
    message: str           # human-readable description
    file: Path | None      # file that triggered the issue (if applicable)
    split: str | None      # split name (if applicable)

@dataclass
class ValidationReport:
    dataset_path: Path
    format: DatasetFormat
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]: ...
    @property
    def warnings(self) -> list[ValidationIssue]: ...
    @property
    def is_valid(self) -> bool:  # no errors
```

**Error codes scheme:**
- `E1xx` — Universal errors (missing images, empty dataset)
- `E2xx` — YOLO-specific errors (bad label columns, class ID out of range, coords out of bounds)
- `E3xx` — COCO-specific errors (broken references, invalid bbox)
- `E4xx` — Mask-specific errors (dimension mismatch, missing masks)
- `W1xx` — Universal warnings (images without annotations)
- `W2xx`/`W3xx`/`W4xx` — Format-specific warnings

---

## Step 2: Implement universal validation (`src/argus/core/validation.py`)

A `_validate_common(dataset) -> list[ValidationIssue]` function that applies to all formats:

- **E101**: Image file does not exist on disk (referenced but missing)
- **E102**: Duplicate image filenames within a split
- **W101**: Image has no annotations (background image — warning only)
- **W102**: Empty split (split directory exists but has no images)

Uses existing `dataset.get_image_paths(split)` and `dataset.get_annotations_for_image(path)`.

---

## Step 3: Implement YOLO-specific validation (`src/argus/core/validation.py`)

A `_validate_yolo(dataset: YOLODataset) -> list[ValidationIssue]` function:

- **E201**: Label file has wrong number of columns (not 5 for detection, not >5 for segmentation)
- **E202**: Class ID in label file is out of range (>= num_classes or < 0)
- **E203**: Normalized coordinates out of bounds (not in [0, 1])
- **E204**: Zero or negative box dimensions
- **W201**: Label file exists but corresponding image is missing
- **W202**: Very small bounding boxes (width or height < 0.001)

Reads label `.txt` files directly, parsing each line.

---

## Step 4: Implement COCO-specific validation (`src/argus/core/validation.py`)

A `_validate_coco(dataset: COCODataset) -> list[ValidationIssue]` function:

- **E301**: Annotation references non-existent image_id
- **E302**: Annotation references non-existent category_id
- **E303**: Duplicate annotation IDs
- **E304**: Invalid bbox (negative width/height or zero area)
- **E305**: Duplicate image IDs
- **W301**: Image entry has no annotations
- **W302**: Polygon has fewer than 3 points (< 6 coordinates)

Reads the JSON annotation data from `dataset._annotations` (the already-loaded COCO data).

---

## Step 5: Implement mask-specific validation (`src/argus/core/validation.py`)

A `_validate_mask(dataset: MaskDataset) -> list[ValidationIssue]` function:

- **E401**: Image has no corresponding mask file
- **E402**: Image and mask dimensions do not match
- **E403**: Mask contains pixel values not in the class mapping (unexpected class IDs)
- **W401**: Mask is entirely one class (no variation)

Uses existing `dataset.get_mask_path()`, `dataset.validate_dimensions()`, and `dataset.load_mask()`.

---

## Step 6: Create the public validation entry point (`src/argus/core/validation.py`)

A `validate_dataset(dataset, split=None, check_images=False) -> ValidationReport` function:

1. Run universal checks
2. Dispatch to format-specific validator based on `dataset.format`
3. If `split` is specified, filter issues to only that split
4. If `check_images` is True, verify image files are readable (try opening with cv2)
5. Return `ValidationReport`

Export from `src/argus/core/__init__.py`.

---

## Step 7: Create the CLI command (`src/argus/commands/validate_command.py`)

Following the existing command pattern (like `stats_command.py`):

```python
def validate_dataset(
    dataset: Path | None = typer.Argument(None, help="Path to dataset root"),
    strict: bool = typer.Option(False, "--strict", help="Fail on warnings"),
    split: str | None = typer.Option(None, "--split", help="Validate specific split"),
    max_issues: int = typer.Option(0, "--max-issues", help="Max issues to show (0=all)"),
    check_images: bool = typer.Option(False, "--check-images", help="Verify images are readable"),
) -> None:
```

**Output format** (using Rich):
- Table with columns: Level (error/warning icon), Code, File, Message
- Summary line: "X errors, Y warnings found"
- Exit code 1 if errors found, or if `--strict` and warnings found
- Exit code 0 if valid

---

## Step 8: Register the command in CLI (`src/argus/cli.py`)

Add:
```python
from argus.commands.validate_command import validate_dataset as validate_dataset_cmd
app.command(name="validate")(validate_dataset_cmd)
```

Update `__all__`.

---

## Step 9: Write tests (`tests/test_validate_command.py`)

Test categories:

1. **Valid datasets produce no errors** — use existing fixtures (`yolo_detection_dataset`, `coco_detection_dataset`, `mask_dataset_grayscale`)
2. **YOLO validation errors** — create fixtures with:
   - Wrong column counts in labels
   - Class IDs out of range
   - Coordinates > 1.0
   - Zero-dimension boxes
3. **COCO validation errors** — create fixtures with:
   - Dangling image_id references
   - Invalid category_ids
   - Duplicate annotation IDs
   - Invalid bboxes
4. **Mask validation errors** — reuse `mask_dataset_dimension_mismatch`, `mask_dataset_missing_mask`
5. **CLI integration tests** — use `typer.testing.CliRunner` to invoke `validate` and check:
   - Exit code 0 for valid datasets
   - Exit code 1 for datasets with errors
   - `--strict` makes warnings cause exit code 1
   - `--split` filters to specific split
   - `--max-issues` limits output
6. **Universal checks** — images without annotations produce warnings

---

## File Summary

| File | Action |
|------|--------|
| `src/argus/core/validation.py` | **New** — ValidationIssue, ValidationReport, all validators |
| `src/argus/core/__init__.py` | **Edit** — export `validate_dataset` |
| `src/argus/commands/validate_command.py` | **New** — CLI command implementation |
| `src/argus/cli.py` | **Edit** — register `validate` command |
| `tests/test_validate_command.py` | **New** — comprehensive test suite |

---

## Design Decisions

1. **Single validation module** rather than per-format files — keeps all validation logic together since validators are small and share data structures.
2. **Error codes** for machine-readable output and CI integration.
3. **Reuse existing dataset methods** (`get_image_paths`, `get_annotations_for_image`, `validate_dimensions`) rather than re-implementing parsing logic.
4. **Access internal dataset state** (e.g. `COCODataset._annotations`) only where existing public methods are insufficient (e.g., checking for duplicate IDs in COCO JSON).
5. **`--check-images` is opt-in** because reading every image with cv2 is slow for large datasets.
