"""Split command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from argus.cli_common import console
from argus.core import COCODataset, YOLODataset
from argus.core.split import (
    is_coco_unsplit,
    parse_ratio,
    split_coco_dataset,
    split_yolo_dataset,
)
from argus.discovery import _detect_dataset


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
                counts = split_yolo_dataset(dataset, output_path, ratios, True, seed)
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
                    True,
                    seed,
                )
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc

    console.print(
        "[green]Split complete.[/green] "
        f"Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}."
    )
