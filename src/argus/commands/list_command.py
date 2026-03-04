"""List command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from argus.cli_common import console
from argus.discovery import _discover_datasets


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
