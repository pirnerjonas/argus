"""Split command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from argus.cli_common import console
from argus.commands._utils import _resolve_existing_directory
from argus.core import COCODataset, MaskDataset, YOLODataset
from argus.core.base import Partitioning
from argus.core.split import (
    parse_ratio,
    split_coco_dataset,
    split_mask_dataset,
    split_yolo_dataset,
)
from argus.discovery import _detect_dataset


def split_dataset(
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
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for deterministic splitting.",
        ),
    ] = 42,
) -> None:
    """Split an unsplit dataset into train/val/test."""
    dataset_path = _resolve_existing_directory(dataset_path or Path("."))

    dataset = _detect_dataset(dataset_path)
    if not dataset:
        console.print(
            f"[red]Error: No dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO), annotations/ folder (COCO), or "
            "images/ + masks/ directories (Mask).[/yellow]"
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

    if dataset.partitioning == Partitioning.SPLIT:
        console.print(
            "[red]Error: Dataset already has splits. "
            "Use an unsplit dataset to run split.[/red]"
        )
        raise typer.Exit(1)

    if isinstance(dataset, YOLODataset):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Creating YOLO splits...", total=None)
            try:
                counts = split_yolo_dataset(dataset, output_path, ratios, True, seed)
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
    elif isinstance(dataset, COCODataset):
        coco_dataset = dataset
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
                    True,
                    seed,
                    coco_dataset.is_roboflow_layout,
                )
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
    elif isinstance(dataset, MaskDataset):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Creating mask dataset splits...", total=None)
            try:
                counts = split_mask_dataset(dataset, output_path, ratios, True, seed)
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
    else:
        console.print("[red]Error: Unsupported dataset type for split command.[/red]")
        raise typer.Exit(1)

    console.print(
        "[green]Split complete.[/green] "
        f"Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}."
    )
