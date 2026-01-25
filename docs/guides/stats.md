# Stats and counts

`argus-cv stats` provides per-class instance counts and image totals by split.
For mask datasets, it reports pixel coverage and how many images contain each
class.

## Example

```bash
argus-cv stats -d /datasets/retail
```

Argus prints a table by class and split. It also includes a summary with:

- total instances
- number of classes
- image totals and background images

## Why background counts matter

Empty label files or images without annotations can skew training. Argus counts
those so you can decide if you want to filter or re-label.

## Common problems

If Argus prints "No annotations found", check:

- YOLO: `labels/` exists and matches `images/`.
- COCO: annotation JSON files are valid and contain `annotations`.
- Mask: masks are `.png` files and match the image file names.
