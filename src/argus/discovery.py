"""Dataset discovery helpers for the CLI."""

from pathlib import Path

from argus.core import COCODataset, Dataset, MaskDataset, YOLODataset


def _discover_datasets(root_path: Path, max_depth: int) -> list[Dataset]:
    """Discover all datasets under the root path.

    Args:
        root_path: Root directory to search.
        max_depth: Maximum depth to traverse.

    Returns:
        List of discovered Dataset instances.
    """
    datasets: list[Dataset] = []
    visited_paths: set[Path] = set()

    def _walk_directory(current_path: Path, depth: int) -> None:
        """Recursively walk directories and detect datasets."""
        if depth > max_depth:
            return

        if not current_path.is_dir():
            return

        # Normalize path to avoid duplicates
        resolved_path = current_path.resolve()
        if resolved_path in visited_paths:
            return
        visited_paths.add(resolved_path)

        # Try to detect datasets at this level
        dataset = _detect_dataset(current_path)
        if dataset:
            # Check if we already have a dataset for this path
            if not any(d.path.resolve() == resolved_path for d in datasets):
                datasets.append(dataset)
            # Don't recurse into detected datasets to avoid duplicates
            return

        # Recurse into subdirectories
        try:
            for entry in current_path.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    _walk_directory(entry, depth + 1)
        except PermissionError:
            pass  # Skip directories we can't access

    _walk_directory(root_path, 0)

    # Sort datasets by path for consistent output
    datasets.sort(key=lambda d: str(d.path))

    return datasets


def _detect_dataset(path: Path) -> Dataset | None:
    """Try to detect a dataset at the given path.

    Detection priority: YOLO -> COCO -> MaskDataset
    """
    # Try YOLO first (more specific patterns - requires data.yaml)
    dataset = YOLODataset.detect(path)
    if dataset:
        return dataset

    # Try COCO (requires annotations/*.json)
    dataset = COCODataset.detect(path)
    if dataset:
        return dataset

    # Try MaskDataset (directory structure based)
    dataset = MaskDataset.detect(path)
    if dataset:
        return dataset

    return None
