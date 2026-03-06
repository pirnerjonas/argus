"""Filter command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from argus.cli_common import console
from argus.commands._utils import (
    _ensure_output_directory_empty,
    _resolve_existing_directory,
    _resolve_output_path,
)
from argus.core import COCODataset, MaskDataset, YOLODataset
from argus.core.base import DatasetFormat
from argus.core.filter import (
    filter_coco_dataset,
    filter_mask_dataset,
    filter_yolo_dataset,
)
from argus.discovery import _detect_dataset


def filter_dataset(
    dataset: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Path to the dataset root directory. Defaults to the current directory."
            ),
            show_default=False,
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output directory for filtered dataset. Relative paths resolve "
                "under the dataset root."
            ),
        ),
    ] = Path("filtered"),
    classes: Annotated[
        str,
        typer.Option(
            "--classes",
            "-c",
            help="Comma-separated list of class names to keep.",
        ),
    ] = "",
    no_background: Annotated[
        bool,
        typer.Option(
            "--no-background",
            help="Exclude images with no annotations after filtering.",
        ),
    ] = False,
    use_symlinks: Annotated[
        bool,
        typer.Option(
            "--symlinks",
            help="Use symlinks instead of copying images.",
        ),
    ] = False,
) -> None:
    """Filter a dataset by class names.

    Creates a filtered copy of the dataset containing only the specified classes.
    Class IDs are remapped to sequential values (0, 1, 2, ...).

    Examples:
        argus-cv filter dataset -o output --classes ball --no-background
        argus-cv filter dataset -o output --classes ball,player
        argus-cv filter dataset -o output --classes ball --symlinks
    """
    dataset_path = _resolve_existing_directory(dataset or Path("."))

    # Parse classes
    if not classes:
        console.print(
            "[red]Error: No classes specified. "
            "Use --classes to specify classes to keep.[/red]"
        )
        raise typer.Exit(1)

    class_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not class_list:
        console.print("[red]Error: No valid class names provided.[/red]")
        raise typer.Exit(1)

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

    # Validate classes exist in dataset
    missing_classes = [c for c in class_list if c not in detected_dataset.class_names]
    if missing_classes:
        available = ", ".join(detected_dataset.class_names)
        missing = ", ".join(missing_classes)
        console.print(
            f"[red]Error: Classes not found in dataset: {missing}[/red]\n"
            f"[yellow]Available classes: {available}[/yellow]"
        )
        raise typer.Exit(1)

    output_path = _resolve_output_path(output_path, dataset_path)
    _ensure_output_directory_empty(output_path)

    # Show filter info
    console.print(
        f"[cyan]Filtering {detected_dataset.format.value.upper()} dataset[/cyan]"
    )
    console.print(f"  Source: {dataset_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Classes to keep: {', '.join(class_list)}")
    console.print(f"  Exclude background: {no_background}")
    console.print(f"  Use symlinks: {use_symlinks}")
    console.print()

    # Run filtering with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Filtering dataset...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        try:
            if detected_dataset.format == DatasetFormat.YOLO:
                assert isinstance(detected_dataset, YOLODataset)
                stats = filter_yolo_dataset(
                    dataset=detected_dataset,
                    output_path=output_path,
                    classes=class_list,
                    no_background=no_background,
                    use_symlinks=use_symlinks,
                    progress_callback=update_progress,
                )
            elif detected_dataset.format == DatasetFormat.COCO:
                assert isinstance(detected_dataset, COCODataset)
                stats = filter_coco_dataset(
                    dataset=detected_dataset,
                    output_path=output_path,
                    classes=class_list,
                    no_background=no_background,
                    use_symlinks=use_symlinks,
                    progress_callback=update_progress,
                )
            elif detected_dataset.format == DatasetFormat.MASK:
                assert isinstance(detected_dataset, MaskDataset)
                stats = filter_mask_dataset(
                    dataset=detected_dataset,
                    output_path=output_path,
                    classes=class_list,
                    no_background=no_background,
                    use_symlinks=use_symlinks,
                    progress_callback=update_progress,
                )
            else:
                console.print(
                    "[red]Error: Unsupported dataset format: "
                    f"{detected_dataset.format}[/red]"
                )
                raise typer.Exit(1)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(f"[red]Error during filtering: {exc}[/red]")
            raise typer.Exit(1) from exc

    # Show results
    console.print()
    console.print("[green]Filtering complete![/green]")
    console.print(f"  Images: {stats.get('images', 0)}")
    if "labels" in stats:
        console.print(f"  Labels: {stats['labels']}")
    if "annotations" in stats:
        console.print(f"  Annotations: {stats['annotations']}")
    if "masks" in stats:
        console.print(f"  Masks: {stats['masks']}")
    if stats.get("skipped", 0) > 0:
        skipped = stats["skipped"]
        console.print(f"  [yellow]Skipped: {skipped} (background images)[/yellow]")

    console.print(f"\n[cyan]Output dataset: {output_path}[/cyan]")
