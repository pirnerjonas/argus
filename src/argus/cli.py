"""Argus CLI - Vision AI dataset toolkit."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from argus.core import COCODataset, Dataset, YOLODataset

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
    table = Table(title=f"Instance Statistics: {dataset_path.name} ({dataset.format.value})")
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

    console.print(f"\n[green]Dataset: {dataset.format.value.upper()} | "
                  f"Task: {dataset.task.value} | "
                  f"Classes: {len(sorted_classes)} | "
                  f"Total instances: {grand_total}[/green]")

    if image_parts:
        console.print(f"[blue]Images: {' | '.join(image_parts)}[/blue]")


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
