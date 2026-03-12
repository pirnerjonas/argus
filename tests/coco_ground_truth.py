"""Ground truth stats via pycocotools for cross-validation tests."""

import contextlib
import io
from pathlib import Path

from pycocotools.coco import COCO


def get_ground_truth_stats(annotation_file: Path) -> dict:
    """Load a COCO annotation JSON with pycocotools and return summary stats.

    Args:
        annotation_file: Path to a COCO-format annotation JSON.

    Returns:
        Dictionary with keys:
        - total: number of images
        - background: number of images with zero annotations
        - instance_counts: {category_name: count}
        - category_names: list of category names sorted by category ID
    """
    # Suppress pycocotools stdout logging
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_file))

    img_ids = coco.getImgIds()
    total = len(img_ids)

    background = 0
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        if len(ann_ids) == 0:
            background += 1

    # Build instance counts by category name
    instance_counts: dict[str, int] = {}
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    id_to_name = {cat["id"]: cat["name"] for cat in cats}

    for ann in coco.loadAnns(coco.getAnnIds()):
        name = id_to_name[ann["category_id"]]
        instance_counts[name] = instance_counts.get(name, 0) + 1

    # Category names sorted by ID
    sorted_cats = sorted(cats, key=lambda c: c["id"])
    category_names = [c["name"] for c in sorted_cats]

    return {
        "total": total,
        "background": background,
        "instance_counts": instance_counts,
        "category_names": category_names,
    }
