"""View command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from argus.cli_common import console
from argus.commands._utils import _resolve_existing_directory
from argus.discovery import _detect_dataset
from argus.viewers import WebViewer


def view(
    dataset: Annotated[
        Path | None,
        typer.Argument(
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
            min=1,
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
    port: Annotated[
        int,
        typer.Option(
            "--port", min=0, max=65535, help="Local port (0 selects an available port)."
        ),
    ] = 0,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-browser",
            help="Automatically open the local viewer in a browser.",
        ),
    ] = True,
) -> None:
    """Browse annotations in a local web viewer. Stop the server with Ctrl+C.

    Supports boxes, polygons, mask overlays, and classification grids.
    """
    dataset_path = _resolve_existing_directory(dataset or Path("."))

    # Detect dataset
    detected_dataset = _detect_dataset(dataset_path)
    if not detected_dataset:
        console.print(
            f"[red]Error: No dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO), annotations/ folder (COCO), or "
            "images/ + masks/ directories (Mask).[/yellow]"
        )
        raise typer.Exit(1)

    # Validate split if specified
    if split and split not in detected_dataset.splits:
        available = (
            ", ".join(detected_dataset.splits) if detected_dataset.splits else "none"
        )
        console.print(
            f"[red]Error: Split '{split}' not found in dataset.[/red]\n"
            f"[yellow]Available splits: {available}[/yellow]"
        )
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading images...", total=None)
        viewer = WebViewer(detected_dataset, split, max_classes, opacity)

    if not viewer.image_paths:
        console.print("[yellow]No images found in the dataset.[/yellow]")
        return

    console.print(
        f"[green]Found {len(viewer.image_paths)} images. Opening web viewer...[/green]"
    )
    try:
        viewer.run(port=port, open_browser=open_browser)
    except OSError as exc:
        console.print(f"Unable to start viewer: {exc}", markup=False)
        raise typer.Exit(1) from exc
    console.print("[green]Viewer closed.[/green]")
