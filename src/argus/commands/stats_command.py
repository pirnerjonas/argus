"""Stats command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from argus.cli_common import console
from argus.commands._utils import _resolve_existing_directory
from argus.core import MaskDataset
from argus.core.base import DatasetFormat
from argus.discovery import _detect_dataset


def stats(
    dataset: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Path to the dataset root directory. Defaults to the current directory."
            ),
            show_default=False,
        ),
    ] = None,
) -> None:
    """Show instance statistics for a dataset.

    Displays the number of annotation instances per class, per split.
    The path should point to a dataset root containing data.yaml (YOLO)
    or an annotations/ folder (COCO).
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

    # Handle mask datasets with pixel statistics
    if detected_dataset.format == DatasetFormat.MASK:
        assert isinstance(detected_dataset, MaskDataset)
        _show_mask_stats(detected_dataset, dataset_path)
        return

    # Get instance counts with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing dataset...", total=None)
        counts = detected_dataset.get_instance_counts()
        image_counts = detected_dataset.get_image_counts()

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
    title = (
        f"Instance Statistics: {dataset_path.name} ({detected_dataset.format.value})"
    )
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
    for split in all_splits:
        if split in image_counts:
            img_total = image_counts[split]["total"]
            img_bg = image_counts[split]["background"]
            if img_bg > 0:
                image_parts.append(f"{split}: {img_total} ({img_bg} background)")
            else:
                image_parts.append(f"{split}: {img_total}")

    console.print(
        f"\n[green]Dataset: {detected_dataset.format.value.upper()} | "
        f"Task: {detected_dataset.task.value} | "
        f"Classes: {len(sorted_classes)} | "
        f"Total instances: {grand_total}[/green]"
    )

    if image_parts:
        console.print(f"[blue]Images: {' | '.join(image_parts)}[/blue]")


def _show_mask_stats(dataset: MaskDataset, dataset_path: Path) -> None:
    """Show statistics for mask datasets with pixel-level information.

    Args:
        dataset: The MaskDataset instance.
        dataset_path: Path to the dataset root.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing mask dataset...", total=None)
        pixel_counts = dataset.get_pixel_counts()
        image_presence = dataset.get_image_class_presence()
        image_counts = dataset.get_image_counts()

    # Get class mapping
    class_mapping = dataset.get_class_mapping()

    # Calculate total non-ignored pixels
    total_pixels = sum(
        count
        for class_id, count in pixel_counts.items()
        if class_id != dataset.ignore_index
    )
    ignored_pixels = pixel_counts.get(dataset.ignore_index, 0)

    # Calculate total images
    total_images = sum(ic["total"] for ic in image_counts.values())

    # Create table
    splits_str = ", ".join(dataset.splits) if dataset.splits else "unsplit"
    title = f"Class Statistics: {dataset_path.name} ({splits_str})"
    table = Table(title=title)
    table.add_column("Class", style="cyan")
    table.add_column("Total Pixels", justify="right", style="green")
    table.add_column("% Coverage", justify="right", style="magenta")
    table.add_column("Images With", justify="right", style="yellow")

    # Sort classes by class_id
    sorted_class_ids = sorted(class_mapping.keys())

    for class_id in sorted_class_ids:
        class_name = class_mapping[class_id]
        pixels = pixel_counts.get(class_id, 0)
        presence = image_presence.get(class_id, 0)

        # Calculate coverage percentage
        coverage = (pixels / total_pixels * 100) if total_pixels > 0 else 0.0

        table.add_row(
            class_name,
            f"{pixels:,}",
            f"{coverage:.1f}%",
            str(presence),
        )

    # Add ignored row if there are ignored pixels
    if ignored_pixels > 0:
        table.add_section()
        table.add_row(
            "[dim](ignored)[/dim]",
            f"[dim]{ignored_pixels:,}[/dim]",
            "[dim]-[/dim]",
            f"[dim]{total_images}[/dim]",
        )

    console.print(table)

    # Summary line
    console.print(f"\n[green]Dataset: {dataset_path}[/green]")
    console.print(
        f"[green]Format: {dataset.format.value.upper()} | "
        f"Task: {dataset.task.value}[/green]"
    )

    # Image counts per split
    image_parts = []
    for split in dataset.splits if dataset.splits else ["unsplit"]:
        if split in image_counts:
            image_parts.append(f"{split}: {image_counts[split]['total']}")

    if image_parts:
        console.print(f"[blue]Images: {' | '.join(image_parts)}[/blue]")
