# Converting datasets

Use `argus-cv convert` to move between supported segmentation formats.

## Supported conversions

- Mask dataset -> YOLO segmentation (`--to yolo-seg`)
- YOLO segmentation -> COCO (`--to coco`)
- YOLO segmentation -> Roboflow COCO (`--to roboflow-coco`)
- YOLO segmentation -> Roboflow COCO RLE (`--to roboflow-coco-rle`)

## 1. Mask -> YOLO segmentation

```bash
argus-cv convert -i /datasets/road_masks -o /datasets/road_yolo --to yolo-seg
```

Useful options:

- `--epsilon-factor`, `-e`: polygon simplification (default `0.005`)
- `--min-area`, `-a`: minimum contour area in pixels (default `100`)

## 2. YOLO segmentation -> COCO

```bash
argus-cv convert -i /datasets/road_yolo -o /datasets/road_coco --to coco
```

## 3. YOLO segmentation -> Roboflow COCO

```bash
argus-cv convert -i /datasets/road_yolo -o /datasets/road_rf_coco --to roboflow-coco
```

## 4. YOLO segmentation -> Roboflow COCO RLE

```bash
argus-cv convert -i /datasets/road_yolo -o /datasets/road_rf_coco_rle --to roboflow-coco-rle
```

Use this for objects with holes/exclusion zones (donut topology), where
polygon-ring interpretation can vary across training libraries.

## Output behavior

- Output directory must be empty or not exist.
- Relative output paths are resolved under the input dataset root.
- Argus keeps dataset splits when they exist.

## Common errors

- "No MaskDataset found": input is not a mask dataset for `--to yolo-seg`.
- "YOLO dataset is not a segmentation dataset": source labels are detection, not polygons.
- "Output directory already exists and is not empty": choose another path or empty it first.
