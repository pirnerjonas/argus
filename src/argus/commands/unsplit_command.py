"""Unsplit command implementation."""

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from argus.cli_common import console
from argus.commands._utils import _resolve_existing_directory
from argus.core import COCODataset, MaskDataset, YOLODataset
from argus.core.base import Partitioning
from argus.core.split import (
    CollisionPolicy,
    unsplit_coco_dataset,
    unsplit_mask_dataset,
    unsplit_yolo_dataset,
)
from argus.discovery import _detect_dataset


def unsplit_dataset(
    dataset: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Path to the split dataset root directory. "
                "Defaults to the current directory."
            ),
            show_default=False,
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="Directory to write the unsplit dataset.",
        ),
    ] = Path("unsplit"),
    collision_policy: Annotated[
        str,
        typer.Option(
            "--collision-policy",
            help="How to handle duplicate filenames: error, prefix-split, or hash.",
            case_sensitive=False,
        ),
    ] = "error",
) -> None:
    """Merge a split dataset into an unsplit dataset."""
    dataset_path = _resolve_existing_directory(dataset or Path("."))

    dataset = _detect_dataset(dataset_path)
    if not dataset:
        console.print(
            f"[red]Error: No dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO), annotations/ folder (COCO), or "
            "images/ + masks/ directories (Mask).[/yellow]"
        )
        raise typer.Exit(1)

    if dataset.partitioning == Partitioning.UNSPLIT:
        console.print(
            "[red]Error: Dataset is already unsplit. "
            "Use a split dataset to run unsplit.[/red]"
        )
        raise typer.Exit(1)

    collision_policy = collision_policy.lower()
    if collision_policy not in ("error", "prefix-split", "hash"):
        console.print(
            "[red]Error: Invalid collision policy.[/red]\n"
            "[yellow]Expected one of: error, prefix-split, hash.[/yellow]"
        )
        raise typer.Exit(1)
    typed_policy = cast(CollisionPolicy, collision_policy)

    if not output_path.is_absolute():
        output_path = dataset_path / output_path
    output_path = output_path.resolve()

    if isinstance(dataset, YOLODataset):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Merging YOLO splits...", total=None)
            try:
                stats = unsplit_yolo_dataset(
                    dataset=dataset,
                    output_path=output_path,
                    collision_policy=typed_policy,
                )
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
    elif isinstance(dataset, COCODataset):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Merging COCO splits...", total=None)
            try:
                stats = unsplit_coco_dataset(
                    dataset=dataset,
                    output_path=output_path,
                    collision_policy=typed_policy,
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
            progress.add_task("Merging mask dataset splits...", total=None)
            try:
                stats = unsplit_mask_dataset(
                    dataset=dataset,
                    output_path=output_path,
                    collision_policy=typed_policy,
                )
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
    else:
        console.print("[red]Error: Unsupported dataset type for unsplit command.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Unsplit complete.[/green] Total images: {stats['total']}.")
