"""Mask dataset detection and handling for semantic segmentation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from argus.core.base import Dataset, DatasetFormat, TaskType

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Error raised when dataset configuration is invalid."""

    pass


# Directory patterns for mask dataset detection (checked in order)
DIRECTORY_PATTERNS = [
    ("images", "masks"),
    ("img", "gt"),
    ("leftImg8bit", "gtFine"),
]

# Standard image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class MaskDataset(Dataset):
    """Dataset format for folder-based semantic segmentation masks.

    Supports directory structures like:
        - images/ + masks/
        - img/ + gt/
        - leftImg8bit/ + gtFine/ (Cityscapes-style)

    Each pattern expects parallel split subdirectories:
        dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── masks/
        │   ├── train/
        │   └── val/
        └── classes.yaml  # Optional for grayscale, required for RGB
    """

    images_dir: str = ""
    masks_dir: str = ""
    class_config: dict | None = None
    ignore_index: int | None = 255
    format: DatasetFormat = field(default=DatasetFormat.MASK, init=False)
    task: TaskType = field(default=TaskType.SEGMENTATION, init=False)

    # Internal: maps class_id -> class_name
    _class_mapping: dict[int, str] = field(default_factory=dict, repr=False)
    # Internal: maps RGB tuple -> (class_id, class_name) for palette masks
    _palette_mapping: dict[tuple[int, int, int], tuple[int, str]] = field(
        default_factory=dict, repr=False
    )
    # Internal: flag for RGB palette mode
    _is_rgb_palette: bool = field(default=False, repr=False)

    @classmethod
    def detect(cls, path: Path) -> "MaskDataset | None":
        """Detect if the given path contains a mask dataset.

        Args:
            path: Directory path to check for dataset.

        Returns:
            MaskDataset instance if detected, None otherwise.
        """
        path = Path(path)

        if not path.is_dir():
            return None

        # Try each directory pattern
        for images_name, masks_name in DIRECTORY_PATTERNS:
            images_root = path / images_name
            masks_root = path / masks_name

            if not (images_root.is_dir() and masks_root.is_dir()):
                continue

            # Check for split subdirectories with matching structure
            splits = cls._detect_splits(images_root, masks_root)

            # If no splits found, check if images and masks are directly in root
            if not splits:
                has_images = any(
                    f.suffix.lower() in IMAGE_EXTENSIONS for f in images_root.iterdir()
                )
                has_masks = any(
                    f.suffix.lower() == ".png" for f in masks_root.iterdir()
                )

                if has_images and has_masks:
                    # Valid unsplit structure
                    splits = []
                else:
                    continue

            # Load class configuration if available
            class_config = cls._load_class_config(path)

            # Determine if masks are grayscale or RGB palette
            is_rgb, palette_mapping = cls._detect_mask_type(path, masks_root, splits)

            # RGB masks require configuration
            if is_rgb and not class_config:
                raise ConfigurationError(
                    f"RGB palette masks detected in {path} but no classes.yaml found. "
                    "RGB masks require a classes.yaml config file with palette mapping."
                )

            # Build class mapping
            class_mapping, ignore_idx = cls._build_class_mapping(
                path, masks_root, splits, class_config, is_rgb, palette_mapping
            )

            # Extract class names from mapping
            class_names = [
                class_mapping[i]
                for i in sorted(class_mapping.keys())
                if i != ignore_idx
            ]

            dataset = cls(
                path=path,
                num_classes=len(class_names),
                class_names=class_names,
                splits=splits,
                images_dir=images_name,
                masks_dir=masks_name,
                class_config=class_config,
                ignore_index=ignore_idx,
            )
            dataset._class_mapping = class_mapping
            dataset._palette_mapping = palette_mapping if is_rgb else {}
            dataset._is_rgb_palette = is_rgb

            return dataset

        return None

    @classmethod
    def _detect_splits(cls, images_root: Path, masks_root: Path) -> list[str]:
        """Detect available splits from directory structure.

        Args:
            images_root: Root directory containing images.
            masks_root: Root directory containing masks.

        Returns:
            List of split names found in both images and masks directories.
        """
        splits = []

        for split_name in ["train", "val", "test"]:
            images_split = images_root / split_name
            masks_split = masks_root / split_name

            if images_split.is_dir() and masks_split.is_dir():
                # Verify there are actual files
                has_images = any(
                    f.suffix.lower() in IMAGE_EXTENSIONS for f in images_split.iterdir()
                )
                has_masks = any(
                    f.suffix.lower() == ".png" for f in masks_split.iterdir()
                )

                if has_images and has_masks:
                    splits.append(split_name)

        return splits

    @classmethod
    def _load_class_config(cls, path: Path) -> dict | None:
        """Load classes.yaml configuration if present.

        Args:
            path: Dataset root path.

        Returns:
            Parsed config dict or None if not found.
        """
        config_path = path / "classes.yaml"
        if not config_path.exists():
            config_path = path / "classes.yml"

        if not config_path.exists():
            return None

        try:
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            raise ConfigurationError(f"Failed to parse {config_path}: {e}") from e

    @classmethod
    def _detect_mask_type(
        cls,
        path: Path,
        masks_root: Path,
        splits: list[str],
    ) -> tuple[bool, dict[tuple[int, int, int], tuple[int, str]]]:
        """Detect if masks are grayscale or RGB palette format.

        Args:
            path: Dataset root path.
            masks_root: Masks directory root.
            splits: List of split names.

        Returns:
            Tuple of (is_rgb_palette, palette_mapping).
        """
        # Find sample mask files
        sample_masks: list[Path] = []

        if splits:
            for split in splits[:1]:  # Just check first split
                split_dir = masks_root / split
                sample_masks.extend(list(split_dir.glob("*.png"))[:5])
        else:
            sample_masks.extend(list(masks_root.glob("*.png"))[:5])

        if not sample_masks:
            return False, {}

        # Check first mask
        mask = cv2.imread(str(sample_masks[0]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return False, {}

        # Check if grayscale (single channel) or RGB (3 channels)
        if len(mask.shape) == 2 or mask.shape[2] == 1:
            return False, {}
        elif mask.shape[2] >= 3:
            return True, {}

        return False, {}

    @classmethod
    def _build_class_mapping(
        cls,
        path: Path,
        masks_root: Path,
        splits: list[str],
        class_config: dict | None,
        is_rgb: bool,
        palette_mapping: dict,
    ) -> tuple[dict[int, str], int]:
        """Build class ID to name mapping.

        Args:
            path: Dataset root path.
            masks_root: Masks directory root.
            splits: List of split names.
            class_config: Parsed classes.yaml or None.
            is_rgb: Whether masks use RGB palette encoding.
            palette_mapping: RGB to class mapping (if RGB).

        Returns:
            Tuple of (class_id_to_name dict, ignore_index or None).
        """
        ignore_index: int | None = 255

        if class_config:
            # Get ignore index from config (can be null/None to disable)
            ignore_index = class_config.get("ignore_index", 255)

            # Try to get names from config
            if "names" in class_config:
                names = class_config["names"]
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}, ignore_index
                elif isinstance(names, list):
                    return {i: name for i, name in enumerate(names)}, ignore_index

            # If RGB palette, build from palette config
            if is_rgb and "palette" in class_config:
                mapping = {}
                for entry in class_config["palette"]:
                    class_id = entry["id"]
                    class_name = entry["name"]
                    mapping[class_id] = class_name
                return mapping, ignore_index

        # Auto-detect classes from grayscale masks
        if not is_rgb:
            return cls._auto_detect_classes(masks_root, splits, ignore_index)

        # RGB without config - error should have been raised earlier
        return {}, ignore_index

    @classmethod
    def _auto_detect_classes(
        cls,
        masks_root: Path,
        splits: list[str],
        ignore_index: int | None,
    ) -> tuple[dict[int, str], int | None]:
        """Auto-detect class IDs from grayscale mask values.

        Args:
            masks_root: Masks directory root.
            splits: List of split names.
            ignore_index: Index to treat as ignored/void, or None.

        Returns:
            Tuple of (class_id_to_name dict, ignore_index or None).
        """
        unique_values: set[int] = set()

        # Sample masks to find unique values
        sample_masks: list[Path] = []

        if splits:
            for split in splits:
                split_dir = masks_root / split
                sample_masks.extend(list(split_dir.glob("*.png"))[:20])
        else:
            sample_masks.extend(list(masks_root.glob("*.png"))[:20])

        for mask_path in sample_masks[:50]:  # Limit sampling
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_values.update(np.unique(mask).tolist())

        # Build mapping with auto-generated names
        mapping = {}
        for val in sorted(unique_values):
            if val == ignore_index:
                continue
            mapping[val] = f"class_{val}"

        return mapping, ignore_index

    def get_image_paths(self, split: str | None = None) -> list[Path]:
        """Get all image file paths for a split or the entire dataset.

        Only returns images that have corresponding masks.

        Args:
            split: Specific split to get images from. If None, returns all images.

        Returns:
            List of image file paths sorted alphabetically.
        """
        images_root = self.path / self.images_dir
        image_paths: list[Path] = []

        splits_to_search = (
            [split] if split else (self.splits if self.splits else [None])
        )

        for s in splits_to_search:
            image_dir = images_root / s if s else images_root

            if not image_dir.is_dir():
                continue

            for img_file in image_dir.iterdir():
                if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Only include if mask exists
                mask_path = self.get_mask_path(img_file)
                if mask_path and mask_path.exists():
                    image_paths.append(img_file)
                else:
                    logger.warning(f"No mask found for image: {img_file}")

        return sorted(image_paths, key=lambda p: p.name)

    def get_mask_path(self, image_path: Path) -> Path | None:
        """Return corresponding mask path for an image.

        Tries multiple naming conventions in order:
        1. Exact stem match: image.jpg -> image.png
        2. With _mask suffix: image.jpg -> image_mask.png
        3. With _gt suffix: image.jpg -> image_gt.png

        Args:
            image_path: Path to the image file.

        Returns:
            Path to corresponding mask, or None if not found.
        """
        stem = image_path.stem

        # Determine the split from image path
        image_parts = image_path.parts
        images_dir_idx = None
        for i, part in enumerate(image_parts):
            if part == self.images_dir:
                images_dir_idx = i
                break

        if images_dir_idx is None:
            # Fallback: look in masks root
            masks_dir = self.path / self.masks_dir
            return self._find_mask_with_patterns(masks_dir, stem)

        # Build mask directory path with same structure
        mask_parts = list(image_parts[:-1])  # Exclude filename
        mask_parts[images_dir_idx] = self.masks_dir
        masks_dir = Path(*mask_parts)

        return self._find_mask_with_patterns(masks_dir, stem)

    def _find_mask_with_patterns(self, masks_dir: Path, stem: str) -> Path | None:
        """Find mask file trying multiple naming patterns.

        Args:
            masks_dir: Directory containing masks.
            stem: Image filename stem (without extension).

        Returns:
            Path to mask if found, None otherwise.
        """
        # Try different naming patterns in order of preference
        patterns = [
            f"{stem}.png",  # Exact match
            f"{stem}_mask.png",  # Common _mask suffix
            f"{stem}_gt.png",  # Ground truth suffix
            f"{stem}_label.png",  # Label suffix
        ]

        for pattern in patterns:
            mask_path = masks_dir / pattern
            if mask_path.exists():
                return mask_path

        return None

    def get_class_mapping(self) -> dict[int, str]:
        """Return class ID to name mapping.

        Returns:
            Dictionary mapping class IDs to class names.
        """
        return self._class_mapping.copy()

    def get_instance_counts(self) -> dict[str, dict[str, int]]:
        """Get pixel counts per class, per split.

        For mask datasets, this returns the total number of pixels
        for each class across all masks in each split.

        Returns:
            Dictionary mapping split name to dict of class name to pixel count.
        """
        counts: dict[str, dict[str, int]] = {}

        splits_to_process = self.splits if self.splits else ["unsplit"]
        masks_root = self.path / self.masks_dir

        for split in splits_to_process:
            split_counts: dict[str, int] = {name: 0 for name in self.class_names}

            mask_dir = masks_root if split == "unsplit" else masks_root / split

            if not mask_dir.is_dir():
                continue

            for mask_path in mask_dir.glob("*.png"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    logger.warning(f"Could not read mask: {mask_path}")
                    continue

                # Count pixels for each class
                unique, pixel_counts = np.unique(mask, return_counts=True)

                for class_id, count in zip(unique, pixel_counts, strict=True):
                    if class_id == self.ignore_index:
                        continue
                    class_name = self._class_mapping.get(class_id)
                    if class_name and class_name in split_counts:
                        split_counts[class_name] += int(count)

            counts[split] = split_counts

        return counts

    def get_image_counts(self) -> dict[str, dict[str, int]]:
        """Get image counts per split.

        Returns:
            Dictionary mapping split name to dict with "total" and "background" counts.
            For mask datasets, "background" is always 0.
        """
        counts: dict[str, dict[str, int]] = {}

        splits_to_process = self.splits if self.splits else ["unsplit"]

        for split in splits_to_process:
            image_paths = self.get_image_paths(split if split != "unsplit" else None)
            counts[split] = {"total": len(image_paths), "background": 0}

        return counts

    def get_image_class_presence(self, split: str | None = None) -> dict[int, int]:
        """Return count of images containing each class.

        Args:
            split: Specific split to analyze. If None, analyzes all splits.

        Returns:
            Dictionary mapping class ID to count of images containing that class.
        """
        presence: dict[int, int] = {class_id: 0 for class_id in self._class_mapping}

        splits_to_process = (
            [split] if split else (self.splits if self.splits else [None])
        )
        masks_root = self.path / self.masks_dir

        for s in splits_to_process:
            mask_dir = masks_root / s if s else masks_root

            if not mask_dir.is_dir():
                continue

            for mask_path in mask_dir.glob("*.png"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                # Find which classes are present in this mask
                unique_values = np.unique(mask)
                for class_id in unique_values:
                    if class_id != self.ignore_index and class_id in presence:
                        presence[class_id] += 1

        return presence

    def get_pixel_counts(self, split: str | None = None) -> dict[int, int]:
        """Return total pixel count per class ID.

        Args:
            split: Specific split to analyze. If None, analyzes all splits.

        Returns:
            Dictionary mapping class ID to total pixel count.
        """
        pixel_counts: dict[int | None, int] = {
            class_id: 0 for class_id in self._class_mapping
        }
        # Track ignored pixels if ignore_index is set
        if self.ignore_index is not None:
            pixel_counts[self.ignore_index] = 0

        splits_to_process = (
            [split] if split else (self.splits if self.splits else [None])
        )
        masks_root = self.path / self.masks_dir

        for s in splits_to_process:
            mask_dir = masks_root / s if s else masks_root

            if not mask_dir.is_dir():
                continue

            for mask_path in mask_dir.glob("*.png"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                unique, counts = np.unique(mask, return_counts=True)
                for class_id, count in zip(unique, counts, strict=True):
                    class_id_int = int(class_id)
                    if class_id_int in pixel_counts:
                        pixel_counts[class_id_int] += int(count)
                    elif (
                        self.ignore_index is not None
                        and class_id_int == self.ignore_index
                    ):
                        pixel_counts[self.ignore_index] += int(count)

        return pixel_counts

    def get_annotations_for_image(self, image_path: Path) -> list[dict]:
        """Get annotations for a specific image.

        For mask datasets, this returns an empty list since annotations
        are stored as pixel masks rather than discrete objects.
        Use get_mask_path() to get the mask file directly.

        Args:
            image_path: Path to the image file.

        Returns:
            Empty list (masks don't have discrete annotations).
        """
        # Mask datasets don't have discrete annotations like detection/segmentation
        # The mask itself IS the annotation
        return []

    def load_mask(self, image_path: Path) -> np.ndarray | None:
        """Load the mask for a given image.

        Args:
            image_path: Path to the image file.

        Returns:
            Mask as numpy array (grayscale), or None if not found.
        """
        mask_path = self.get_mask_path(image_path)
        if mask_path is None or not mask_path.exists():
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask

    def validate_dimensions(
        self, image_path: Path
    ) -> tuple[bool, tuple[int, int] | None, tuple[int, int] | None]:
        """Check if image and mask dimensions match.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (dimensions_match, image_shape, mask_shape).
        """
        mask_path = self.get_mask_path(image_path)
        if mask_path is None or not mask_path.exists():
            return False, None, None

        img = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            return False, None, None

        img_shape = (img.shape[0], img.shape[1])
        mask_shape = (mask.shape[0], mask.shape[1])

        return img_shape == mask_shape, img_shape, mask_shape
