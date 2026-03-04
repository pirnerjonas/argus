"""Annotation rendering helpers used by CLI viewers."""

import hashlib

import cv2
import numpy as np


def _generate_class_colors(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """Generate consistent colors for each class name.

    Args:
        class_names: List of class names.

    Returns:
        Dictionary mapping class name to BGR color tuple.
    """
    colors: dict[str, tuple[int, int, int]] = {}

    for name in class_names:
        # Generate a consistent hash-based color
        hash_val = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF

        # Ensure colors are bright enough to be visible
        min_brightness = 100
        r = max(r, min_brightness)
        g = max(g, min_brightness)
        b = max(b, min_brightness)

        colors[name] = (b, g, r)  # BGR for OpenCV

    return colors


def _draw_annotations(
    img: np.ndarray,
    annotations: list[dict],
    class_colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Draw annotations on an image.

    Args:
        img: OpenCV image (BGR).
        annotations: List of annotation dicts.
        class_colors: Dictionary mapping class name to BGR color.

    Returns:
        Image with annotations drawn.
    """
    default_color = (0, 255, 0)  # Green default

    for ann in annotations:
        class_name = ann["class_name"]
        color = class_colors.get(class_name, default_color)
        bbox = ann.get("bbox")
        polygon = ann.get("polygon")

        # Draw polygon if available (segmentation)
        if polygon:
            pts = np.array(polygon, dtype=np.int32)
            # Collect all rings (outer + holes) for correct even-odd fill
            all_rings = [pts]
            polygon_holes = ann.get("polygon_holes", [])
            for hole in polygon_holes:
                all_rings.append(np.array(hole, dtype=np.int32))
            # Draw outlines
            for ring in all_rings:
                cv2.polylines(img, [ring], isClosed=True, color=color, thickness=2)
            # Draw semi-transparent fill with holes cut out
            overlay = img.copy()
            cv2.fillPoly(overlay, all_rings, color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            # Draw small points at polygon vertices
            for pt in pts:
                cv2.circle(img, tuple(pt), radius=3, color=color, thickness=-1)

        # Draw bounding box (only for detection, not segmentation)
        if bbox and not polygon:
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = class_name
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w + 5, y1),
                color,
                -1,
            )
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    return img
