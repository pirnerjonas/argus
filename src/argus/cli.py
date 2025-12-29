"""Argus CLI - Vision AI dataset toolkit."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
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
