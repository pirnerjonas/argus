"""Argus CLI - Vision AI dataset toolkit."""

import hashlib
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from argus.core import COCODataset, Dataset, YOLODataset
from argus.core.base import TaskType
from argus.core.split import (
    is_coco_unsplit,
    parse_ratio,
    split_coco_dataset,
    split_yolo_dataset,
)

console = Console()

app = typer.Typer(
    name="argus",
    help="Vision AI dataset toolkit for working with YOLO and COCO datasets.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Vision AI dataset toolkit for working with YOLO and COCO datasets."""
    pass


@app.command(name="list")
def list_datasets(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Root path to search for datasets.",
        ),
    ] = Path("."),
    max_depth: Annotated[
        int,
        typer.Option(
            "--max-depth",
            "-d",
            help="Maximum directory depth to search.",
            min=1,
            max=10,
        ),
    ] = 3,
) -> None:
    """List all detected datasets in the specified path.

    Searches for YOLO and COCO format datasets within the given directory,
    up to the specified maximum depth.
    """
    # Resolve path and validate
    path = path.resolve()
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)
    if not path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        raise typer.Exit(1)

    datasets = _discover_datasets(path, max_depth)

    if not datasets:
        console.print(f"[yellow]No datasets found in {path}[/yellow]")
        return

    # Create and populate table
    table = Table(title=f"Datasets found in {path}")
    table.add_column("Path", style="cyan", no_wrap=True)
    table.add_column("Format", style="green")
    table.add_column("Task", style="magenta")
    table.add_column("Classes", justify="right", style="yellow")
    table.add_column("Splits", style="blue")

    for dataset in datasets:
        summary = dataset.summary()
        table.add_row(
            summary["path"],
            summary["format"],
            summary["task"],
            str(summary["classes"]),
            summary["splits"],
        )

    console.print(table)
    console.print(f"\n[green]Found {len(datasets)} dataset(s)[/green]")


@app.command(name="stats")
def stats(
    dataset_path: Annotated[
        Path,
        typer.Option(
            "--dataset-path",
            "-d",
            help="Path to the dataset root directory.",
        ),
    ] = Path("."),
) -> None:
    """Show instance statistics for a dataset.

    Displays the number of annotation instances per class, per split.
    The path should point to a dataset root containing data.yaml (YOLO)
    or an annotations/ folder (COCO).
    """
    # Resolve path and validate
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        console.print(f"[red]Error: Path does not exist: {dataset_path}[/red]")
        raise typer.Exit(1)
    if not dataset_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {dataset_path}[/red]")
        raise typer.Exit(1)

    # Detect dataset
    dataset = _detect_dataset(dataset_path)
    if not dataset:
        console.print(
            f"[red]Error: No YOLO or COCO dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO) or annotations/ folder (COCO).[/yellow]"
        )
        raise typer.Exit(1)

    # Get instance counts with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing dataset...", total=None)
        counts = dataset.get_instance_counts()
        image_counts = dataset.get_image_counts()

    if not counts:
        console.print("[yellow]No annotations found in the dataset.[/yellow]")
        return

    # Collect all classes and splits
    all_classes: set[str] = set()
    all_splits: list[str] = []
    for split, class_counts in counts.items():
        all_splits.append(split)
        all_classes.update(class_counts.keys())

    # Sort splits in standard order
    split_order = {"train": 0, "val": 1, "test": 2}
    all_splits.sort(key=lambda s: split_order.get(s, 99))

    # Sort classes alphabetically
    sorted_classes = sorted(all_classes)

    # Create table
    title = f"Instance Statistics: {dataset_path.name} ({dataset.format.value})"
    table = Table(title=title)
    table.add_column("Class", style="cyan")
    for split in all_splits:
        table.add_column(split, justify="right", style="green")
    table.add_column("Total", justify="right", style="yellow bold")

    # Add rows for each class
    grand_totals = {split: 0 for split in all_splits}
    grand_total = 0

    for class_name in sorted_classes:
        row = [class_name]
        class_total = 0
        for split in all_splits:
            count = counts.get(split, {}).get(class_name, 0)
            row.append(str(count) if count > 0 else "-")
            class_total += count
            grand_totals[split] += count
        row.append(str(class_total))
        grand_total += class_total
        table.add_row(*row)

    # Add totals row
    table.add_section()
    totals_row = ["[bold]Total[/bold]"]
    for split in all_splits:
        totals_row.append(f"[bold]{grand_totals[split]}[/bold]")
    totals_row.append(f"[bold]{grand_total}[/bold]")
    table.add_row(*totals_row)

    console.print(table)

    # Build image stats line
    image_parts = []
    total_images = 0
    total_background = 0
    for split in all_splits:
        if split in image_counts:
            img_total = image_counts[split]["total"]
            img_bg = image_counts[split]["background"]
            total_images += img_total
            total_background += img_bg
            if img_bg > 0:
                image_parts.append(f"{split}: {img_total} ({img_bg} background)")
            else:
                image_parts.append(f"{split}: {img_total}")

    console.print(
        f"\n[green]Dataset: {dataset.format.value.upper()} | "
        f"Task: {dataset.task.value} | "
        f"Classes: {len(sorted_classes)} | "
        f"Total instances: {grand_total}[/green]"
    )

    if image_parts:
        console.print(f"[blue]Images: {' | '.join(image_parts)}[/blue]")


@app.command(name="view")
def view(
    dataset_path: Annotated[
        Path,
        typer.Option(
            "--dataset-path",
            "-d",
            help="Path to the dataset root directory.",
        ),
    ] = Path("."),
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
    # Resolve path and validate
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        console.print(f"[red]Error: Path does not exist: {dataset_path}[/red]")
        raise typer.Exit(1)
    if not dataset_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {dataset_path}[/red]")
        raise typer.Exit(1)

    # Detect dataset
    dataset = _detect_dataset(dataset_path)
    if not dataset:
        console.print(
            f"[red]Error: No YOLO or COCO dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO) or annotations/ folder (COCO).[/yellow]"
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
            class_colors=class_colors,
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


@app.command(name="split")
def split_dataset(
    dataset_path: Annotated[
        Path,
        typer.Option(
            "--dataset-path",
            "-d",
            help="Path to the dataset root directory.",
        ),
    ] = Path("."),
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="Directory to write the split dataset.",
        ),
    ] = Path("splits"),
    ratio: Annotated[
        str,
        typer.Option(
            "--ratio",
            "-r",
            help="Train/val/test ratio (e.g. 0.8,0.1,0.1).",
        ),
    ] = "0.8,0.1,0.1",
    stratify: Annotated[
        bool,
        typer.Option(
            "--stratify/--no-stratify",
            help="Stratify by class distribution when splitting.",
        ),
    ] = True,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for deterministic splitting.",
        ),
    ] = 42,
) -> None:
    """Split an unsplit dataset into train/val/test."""
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        console.print(f"[red]Error: Path does not exist: {dataset_path}[/red]")
        raise typer.Exit(1)
    if not dataset_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {dataset_path}[/red]")
        raise typer.Exit(1)

    dataset = _detect_dataset(dataset_path)
    if not dataset:
        console.print(
            f"[red]Error: No YOLO or COCO dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO) or annotations/ folder (COCO).[/yellow]"
        )
        raise typer.Exit(1)

    try:
        ratios = parse_ratio(ratio)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    if not output_path.is_absolute():
        output_path = dataset_path / output_path
    output_path = output_path.resolve()

    if isinstance(dataset, YOLODataset):
        if dataset.splits:
            console.print(
                "[red]Error: Dataset already has splits. "
                "Use an unsplit dataset to run split.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Creating YOLO splits...", total=None)
            try:
                counts = split_yolo_dataset(
                    dataset, output_path, ratios, stratify, seed
                )
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
    else:
        coco_dataset = dataset
        if not isinstance(coco_dataset, COCODataset):
            console.print("[red]Error: Unsupported dataset type.[/red]")
            raise typer.Exit(1)
        if not is_coco_unsplit(coco_dataset.annotation_files):
            console.print(
                "[red]Error: Dataset already has splits. "
                "Use an unsplit dataset to run split.[/red]"
            )
            raise typer.Exit(1)
        if not coco_dataset.annotation_files:
            console.print("[red]Error: No annotation files found.[/red]")
            raise typer.Exit(1)
        annotation_file = coco_dataset.annotation_files[0]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Creating COCO splits...", total=None)
            try:
                counts = split_coco_dataset(
                    coco_dataset,
                    annotation_file,
                    output_path,
                    ratios,
                    stratify,
                    seed,
                )
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc

    console.print(
        "[green]Split complete.[/green] "
        f"Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}."
    )


class _ImageViewer:
    """Interactive image viewer with zoom and pan support."""

    def __init__(
        self,
        image_paths: list[Path],
        dataset: Dataset,
        class_colors: dict[str, tuple[int, int, int]],
        window_name: str,
    ):
        self.image_paths = image_paths
        self.dataset = dataset
        self.class_colors = class_colors
        self.window_name = window_name

        self.current_idx = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Mouse state for panning
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.pan_start_x = 0.0
        self.pan_start_y = 0.0

        # Current image cache
        self.current_img: np.ndarray | None = None
        self.annotated_img: np.ndarray | None = None

        # Annotation visibility toggle
        self.show_annotations = True

    def _load_current_image(self) -> bool:
        """Load and annotate the current image."""
        image_path = self.image_paths[self.current_idx]
        annotations = self.dataset.get_annotations_for_image(image_path)

        img = cv2.imread(str(image_path))
        if img is None:
            return False

        self.current_img = img
        self.annotated_img = _draw_annotations(
            img.copy(), annotations, self.class_colors
        )
        return True

    def _get_display_image(self) -> np.ndarray:
        """Get the image transformed for current zoom/pan."""
        if self.annotated_img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if self.show_annotations:
            img = self.annotated_img
        elif self.current_img is not None:
            img = self.current_img
        else:
            img = self.annotated_img
        h, w = img.shape[:2]

        if self.zoom == 1.0 and self.pan_x == 0.0 and self.pan_y == 0.0:
            display = img.copy()
        else:
            # Calculate the visible region
            view_w = int(w / self.zoom)
            view_h = int(h / self.zoom)

            # Center point with pan offset
            cx = w / 2 + self.pan_x
            cy = h / 2 + self.pan_y

            # Calculate crop bounds
            x1 = int(max(0, cx - view_w / 2))
            y1 = int(max(0, cy - view_h / 2))
            x2 = int(min(w, x1 + view_w))
            y2 = int(min(h, y1 + view_h))

            # Adjust if we hit boundaries
            if x2 - x1 < view_w:
                x1 = max(0, x2 - view_w)
            if y2 - y1 < view_h:
                y1 = max(0, y2 - view_h)

            # Crop and resize
            cropped = img[y1:y2, x1:x2]
            display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add info overlay
        image_path = self.image_paths[self.current_idx]
        idx = self.current_idx + 1
        total = len(self.image_paths)
        info_text = f"[{idx}/{total}] {image_path.name}"
        if self.zoom > 1.0:
            info_text += f" (Zoom: {self.zoom:.1f}x)"
        if not self.show_annotations:
            info_text += " [Annotations: OFF]"

        cv2.putText(
            display,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
        )

        return display

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: None
    ) -> None:
        """Handle mouse events for zoom and pan."""
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:
                self.zoom = min(10.0, self.zoom * 1.2)
            else:
                self.zoom = max(1.0, self.zoom / 1.2)

            # Reset pan if zoomed out to 1x
            if self.zoom == 1.0:
                self.pan_x = 0.0
                self.pan_y = 0.0

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_x = x
            self.drag_start_y = y
            self.pan_start_x = self.pan_x
            self.pan_start_y = self.pan_y

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.zoom > 1.0 and self.annotated_img is not None:
                h, w = self.annotated_img.shape[:2]
                # Calculate pan delta (inverted for natural feel)
                dx = (self.drag_start_x - x) / self.zoom
                dy = (self.drag_start_y - y) / self.zoom

                # Update pan with limits
                max_pan_x = w * (1 - 1 / self.zoom) / 2
                max_pan_y = h * (1 - 1 / self.zoom) / 2

                self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_start_x + dx))
                self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_start_y + dy))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _reset_view(self) -> None:
        """Reset zoom and pan to default."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def _next_image(self) -> None:
        """Go to next image."""
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self._reset_view()

    def _prev_image(self) -> None:
        """Go to previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self._reset_view()

    def run(self) -> None:
        """Run the interactive viewer."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            # Load image if needed
            if self.annotated_img is None and not self._load_current_image():
                console.print(
                    f"[yellow]Warning: Could not load "
                    f"{self.image_paths[self.current_idx]}[/yellow]"
                )
                self._next_image()
                continue

            # Display image
            display = self._get_display_image()
            cv2.imshow(self.window_name, display)

            # Wait for input (short timeout for smooth panning)
            key = cv2.waitKey(30) & 0xFF

            # Handle keyboard input
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("n") or key == 83 or key == 3:  # N or Right arrow
                self.annotated_img = None
                self._next_image()
            elif key == ord("p") or key == 81 or key == 2:  # P or Left arrow
                self.annotated_img = None
                self._prev_image()
            elif key == ord("r"):  # R to reset zoom
                self._reset_view()
            elif key == ord("t"):  # T to toggle annotations
                self.show_annotations = not self.show_annotations

        cv2.destroyAllWindows()


class _ClassificationGridViewer:
    """Grid viewer for classification datasets showing one image per class."""

    def __init__(
        self,
        images_by_class: dict[str, list[Path]],
        class_names: list[str],
        class_colors: dict[str, tuple[int, int, int]],
        window_name: str,
        max_classes: int | None = None,
        tile_size: int = 300,
    ):
        # Limit classes if max_classes specified
        if max_classes and len(class_names) > max_classes:
            self.class_names = class_names[:max_classes]
        else:
            self.class_names = class_names

        self.images_by_class = {
            cls: images_by_class.get(cls, []) for cls in self.class_names
        }
        self.class_colors = class_colors
        self.window_name = window_name
        self.tile_size = tile_size

        # Global image index (same for all classes)
        self.current_index = 0

        # Calculate max images across all classes
        self.max_images = (
            max(len(imgs) for imgs in self.images_by_class.values())
            if self.images_by_class
            else 0
        )

        # Calculate grid layout
        self.cols, self.rows = self._calculate_grid_layout()

    def _calculate_grid_layout(self) -> tuple[int, int]:
        """Calculate optimal grid layout based on number of classes."""
        n = len(self.class_names)
        if n <= 0:
            return 1, 1

        # Try to make a roughly square grid
        import math

        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        return cols, rows

    def _create_tile(
        self, class_name: str, image_path: Path | None, index: int, total: int
    ) -> np.ndarray:
        """Create a single tile for a class."""
        tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)

        if image_path is not None and image_path.exists():
            # Load and resize image
            img = cv2.imread(str(image_path))
            if img is not None:
                # Resize maintaining aspect ratio
                h, w = img.shape[:2]
                scale = min(self.tile_size / w, self.tile_size / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Center in tile
                x_offset = (self.tile_size - new_w) // 2
                y_offset = (self.tile_size - new_h) // 2
                tile[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Draw label at top: "class_name (N/M)"
        if image_path is not None:
            label = f"{class_name} ({index + 1}/{total})"
        else:
            label = f"{class_name} (-/{total})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Semi-transparent background for label
        overlay = tile.copy()
        label_bg_height = label_h + baseline + 10
        cv2.rectangle(overlay, (0, 0), (self.tile_size, label_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, tile, 0.4, 0, tile)

        cv2.putText(
            tile,
            label,
            (5, label_h + 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        # Draw thin border
        border_end = self.tile_size - 1
        cv2.rectangle(tile, (0, 0), (border_end, border_end), (80, 80, 80), 1)

        return tile

    def _compose_grid(self) -> np.ndarray:
        """Compose all tiles into a single grid image."""
        grid_h = self.rows * self.tile_size
        grid_w = self.cols * self.tile_size
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, class_name in enumerate(self.class_names):
            row = i // self.cols
            col = i % self.cols

            images = self.images_by_class[class_name]
            total = len(images)

            # Use global index - show black tile if class doesn't have this image
            if self.current_index < total:
                image_path = images[self.current_index]
                display_index = self.current_index
            else:
                image_path = None
                display_index = self.current_index

            tile = self._create_tile(class_name, image_path, display_index, total)

            y_start = row * self.tile_size
            x_start = col * self.tile_size
            y_end = y_start + self.tile_size
            x_end = x_start + self.tile_size
            grid[y_start:y_end, x_start:x_end] = tile

        return grid

    def _next_images(self) -> None:
        """Advance to next image index."""
        if self.max_images > 0:
            self.current_index = min(self.current_index + 1, self.max_images - 1)

    def _prev_images(self) -> None:
        """Go back to previous image index."""
        self.current_index = max(self.current_index - 1, 0)

    def _reset_indices(self) -> None:
        """Reset to first image."""
        self.current_index = 0

    def run(self) -> None:
        """Run the interactive grid viewer."""
        if not self.class_names:
            console.print("[yellow]No classes to display.[/yellow]")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while True:
            # Compose and display grid
            grid = self._compose_grid()
            cv2.imshow(self.window_name, grid)

            # Wait for input
            key = cv2.waitKey(30) & 0xFF

            # Handle keyboard input
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("n") or key == 83 or key == 3:  # N or Right arrow
                self._next_images()
            elif key == ord("p") or key == 81 or key == 2:  # P or Left arrow
                self._prev_images()
            elif key == ord("r"):  # R to reset
                self._reset_indices()

        cv2.destroyAllWindows()


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
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            # Draw semi-transparent fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
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
    """Try to detect a dataset at the given path."""
    # Try YOLO first (more specific patterns)
    dataset = YOLODataset.detect(path)
    if dataset:
        return dataset

    # Try COCO
    dataset = COCODataset.detect(path)
    if dataset:
        return dataset

    return None


if __name__ == "__main__":
    app()
