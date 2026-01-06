# Listing datasets

Use `argus list` to scan a directory tree and discover datasets.

## Basic listing

```bash
argus list --path /datasets
```

This prints a table showing:

- Dataset path
- Format (yolo or coco)
- Task (detection or segmentation)
- Number of classes
- Detected splits

## Control scan depth

```bash
argus list --path /datasets --max-depth 2
```

This keeps scans fast when you have many nested projects.

## Tips

- Point to a broad root like `/datasets` or `~/data`.
- If you see duplicates, ensure dataset roots are not nested inside each other.
