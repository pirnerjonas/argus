"""View command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from argus.cli_common import console
from argus.commands._utils import _resolve_existing_directory
from argus.core import COCODataset, MaskDataset
from argus.core.base import DatasetFormat, TaskType
from argus.discovery import _detect_dataset
from argus.rendering import _generate_class_colors
from argus.viewers import _ClassificationGridViewer, _ImageViewer, _MaskViewer


def _run_mask_viewer(
    dataset: MaskDataset | COCODataset,
    dataset_path: Path,
    split: str | None,
    class_colors: dict[str, tuple[int, int, int]],
    opacity: float,
) -> None:
    """Load images and run the mask overlay viewer."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading images...", total=None)
        image_paths = dataset.get_image_paths(split)

    if not image_paths:
        console.print("[yellow]No images found in the dataset.[/yellow]")
        return

    console.print(
        f"[green]Found {len(image_paths)} images. "
        f"Opening mask viewer...[/green]\n"
        "[dim]Controls: \u2190 / \u2192 or P / N to navigate, "
        "Mouse wheel to zoom, Drag to pan, R to reset, T to toggle overlay, "
        "Q / ESC to quit[/dim]"
    )

    viewer = _MaskViewer(
        image_paths=image_paths,
        dataset=dataset,
        class_colors=class_colors,
        window_name=f"Argus Mask Viewer - {dataset_path.name}",
        opacity=opacity,
    )
    viewer.run()
    console.print("[green]Viewer closed.[/green]")


def view(
    dataset_path: Annotated[
        Path | None,
        typer.Argument(
            metavar="DATASET",
            help=(
                "Path to the dataset root directory. Defaults to the current directory."
            ),
            show_default=False,
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            "-s",
            help="Specific split to view (train, val, test).",
        ),
    ] = None,
    max_classes: Annotated[
        int | None,
        typer.Option(
            "--max-classes",
            "-m",
            help="Maximum classes to show in grid (classification only).",
        ),
    ] = None,
    opacity: Annotated[
        float,
        typer.Option(
            "--opacity",
            "-o",
            help="Mask overlay opacity (0.0-1.0, mask datasets only).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
) -> None:
    """View annotated images in a dataset.

    Opens an interactive viewer to browse images with their annotations
    (bounding boxes and segmentation masks) overlaid.

    For classification datasets, shows a grid view with one image per class.

    Controls:
        - Right Arrow / N: Next image(s)
        - Left Arrow / P: Previous image(s)
        - Mouse Wheel: Zoom in/out (detection/segmentation only)
        - Mouse Drag: Pan when zoomed (detection/segmentation only)
        - R: Reset zoom / Reset to first images
        - T: Toggle annotations (detection/segmentation only)
        - Q / ESC: Quit viewer
    """
    dataset_path = _resolve_existing_directory(dataset_path or Path("."))

    # Detect dataset
    dataset = _detect_dataset(dataset_path)
    if not dataset:
        console.print(
            f"[red]Error: No dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO), annotations/ folder (COCO), or "
            "images/ + masks/ directories (Mask).[/yellow]"
        )
        raise typer.Exit(1)

    # Validate split if specified
    if split and split not in dataset.splits:
        available = ", ".join(dataset.splits) if dataset.splits else "none"
        console.print(
            f"[red]Error: Split '{split}' not found in dataset.[/red]\n"
            f"[yellow]Available splits: {available}[/yellow]"
        )
        raise typer.Exit(1)

    # Generate consistent colors for each class
    class_colors = _generate_class_colors(dataset.class_names)

    # Handle mask datasets with overlay viewer
    if dataset.format == DatasetFormat.MASK:
        assert isinstance(dataset, MaskDataset)
        _run_mask_viewer(dataset, dataset_path, split, class_colors, opacity)
        return

    # Handle COCO RLE datasets with mask overlay viewer
    if isinstance(dataset, COCODataset) and dataset.has_rle:
        _run_mask_viewer(dataset, dataset_path, split, class_colors, opacity)
        return

    # Handle classification datasets with grid viewer
    if dataset.task == TaskType.CLASSIFICATION:
        # Use first split if specified, otherwise let get_images_by_class handle it
        view_split = split if split else (dataset.splits[0] if dataset.splits else None)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading images by class...", total=None)
            images_by_class = dataset.get_images_by_class(view_split)

        total_images = sum(len(imgs) for imgs in images_by_class.values())
        if total_images == 0:
            console.print("[yellow]No images found in the dataset.[/yellow]")
            return

        num_classes = len(dataset.class_names)
        display_classes = min(num_classes, max_classes) if max_classes else num_classes

        console.print(
            f"[green]Found {total_images} images across {num_classes} classes "
            f"(showing {display_classes}). Opening grid viewer...[/green]\n"
            "[dim]Controls: ← / → or P / N to navigate all classes, "
            "R to reset, Q / ESC to quit[/dim]"
        )

        viewer = _ClassificationGridViewer(
            images_by_class=images_by_class,
            class_names=dataset.class_names,
            window_name=f"Argus Classification Viewer - {dataset_path.name}",
            max_classes=max_classes,
        )
        viewer.run()
    else:
        # Detection/Segmentation viewer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading images...", total=None)
            image_paths = dataset.get_image_paths(split)

        if not image_paths:
            console.print("[yellow]No images found in the dataset.[/yellow]")
            return

        console.print(
            f"[green]Found {len(image_paths)} images. "
            f"Opening viewer...[/green]\n"
            "[dim]Controls: ← / → or P / N to navigate, "
            "Mouse wheel to zoom, Drag to pan, R to reset, T to toggle annotations, "
            "Q / ESC to quit[/dim]"
        )

        viewer = _ImageViewer(
            image_paths=image_paths,
            dataset=dataset,
            class_colors=class_colors,
            window_name=f"Argus Viewer - {dataset_path.name}",
        )
        viewer.run()

    console.print("[green]Viewer closed.[/green]")
