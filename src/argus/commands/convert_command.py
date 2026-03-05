"""Convert command implementation."""

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
from argus.core import MaskDataset, YOLODataset
from argus.core.convert import (
    convert_mask_to_yolo_seg,
    convert_yolo_seg_to_coco,
    convert_yolo_seg_to_roboflow_coco,
    convert_yolo_seg_to_roboflow_coco_rle,
)


def convert_dataset(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input-path",
            "-i",
            help="Path to the source dataset.",
        ),
    ] = Path("."),
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="Output directory for converted dataset.",
        ),
    ] = Path("converted"),
    to_format: Annotated[
        str,
        typer.Option(
            "--to",
            help=(
                "Target format: 'yolo-seg' (from mask), "
                "'coco' (from yolo-seg), or "
                "'roboflow-coco'/'roboflow-coco-rle' (from yolo-seg)."
            ),
        ),
    ] = "yolo-seg",
    epsilon_factor: Annotated[
        float,
        typer.Option(
            "--epsilon-factor",
            "-e",
            help="Polygon simplification factor (Douglas-Peucker algorithm).",
            min=0.0,
            max=1.0,
        ),
    ] = 0.005,
    min_area: Annotated[
        float,
        typer.Option(
            "--min-area",
            "-a",
            help="Minimum contour area in pixels to include.",
            min=0.0,
        ),
    ] = 100.0,
) -> None:
    """Convert a dataset from one format to another.

    Supports:
    - MaskDataset → YOLO segmentation (--to yolo-seg)
    - YOLO segmentation → COCO (--to coco)
    - YOLO segmentation → Roboflow COCO (--to roboflow-coco)
    - YOLO segmentation → Roboflow COCO RLE (--to roboflow-coco-rle)

    Examples:
        uvx argus-cv convert -i /path/to/masks -o /path/to/output --to yolo-seg
        uvx argus-cv convert -i /path/to/yolo -o /path/to/output --to coco
        uvx argus-cv convert -i /path/to/yolo -o /path/to/output --to roboflow-coco
        uvx argus-cv convert -i /path/to/yolo -o /path/to/output --to roboflow-coco-rle
    """
    # Validate format
    supported_formats = ("yolo-seg", "coco", "roboflow-coco", "roboflow-coco-rle")
    if to_format not in supported_formats:
        console.print(
            f"[red]Error: Unsupported target format '{to_format}'.[/red]\n"
            f"[yellow]Supported formats: {', '.join(supported_formats)}[/yellow]"
        )
        raise typer.Exit(1)

    input_path = _resolve_existing_directory(input_path)
    output_path = _resolve_output_path(output_path, input_path.parent)
    _ensure_output_directory_empty(output_path)

    if to_format == "yolo-seg":
        _convert_mask_to_yolo(input_path, output_path, epsilon_factor, min_area)
    elif to_format == "coco":
        _convert_yolo_to_coco(
            input_path,
            output_path,
            roboflow_layout=False,
            use_rle=False,
        )
    elif to_format == "roboflow-coco":
        _convert_yolo_to_coco(
            input_path,
            output_path,
            roboflow_layout=True,
            use_rle=False,
        )
    elif to_format == "roboflow-coco-rle":
        _convert_yolo_to_coco(
            input_path,
            output_path,
            roboflow_layout=True,
            use_rle=True,
        )


def _convert_mask_to_yolo(
    input_path: Path,
    output_path: Path,
    epsilon_factor: float,
    min_area: float,
) -> None:
    """Run mask → YOLO-seg conversion."""

    dataset = MaskDataset.detect(input_path)
    if not dataset:
        console.print(
            f"[red]Error: No MaskDataset found at {input_path}[/red]\n"
            "[yellow]Ensure the path contains images/ + masks/ directories "
            "(or equivalent patterns like img/+gt/ or leftImg8bit/+gtFine/).[/yellow]"
        )
        raise typer.Exit(1)

    console.print("[cyan]Converting MaskDataset to YOLO segmentation format[/cyan]")
    console.print(f"  Source: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Classes: {dataset.num_classes}")
    splits_str = ", ".join(dataset.splits) if dataset.splits else "unsplit"
    console.print(f"  Splits: {splits_str}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing images...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        try:
            stats = convert_mask_to_yolo_seg(
                dataset=dataset,
                output_path=output_path,
                epsilon_factor=epsilon_factor,
                min_area=min_area,
                progress_callback=update_progress,
            )
        except Exception as exc:
            console.print(f"[red]Error during conversion: {exc}[/red]")
            raise typer.Exit(1) from exc

    console.print()
    console.print("[green]Conversion complete![/green]")
    console.print(f"  Images processed: {stats['images']}")
    console.print(f"  Labels created: {stats['labels']}")
    console.print(f"  Polygons extracted: {stats['polygons']}")

    if stats["skipped"] > 0:
        skipped = stats["skipped"]
        console.print(f"  [yellow]Skipped: {skipped} (no mask or empty)[/yellow]")
    if stats["warnings"] > 0:
        console.print(f"  [yellow]Warnings: {stats['warnings']}[/yellow]")

    console.print(f"\n[cyan]Output dataset: {output_path}[/cyan]")


def _convert_yolo_to_coco(
    input_path: Path,
    output_path: Path,
    roboflow_layout: bool,
    use_rle: bool = False,
) -> None:
    """Run YOLO-seg → COCO conversion."""
    from argus.core.base import TaskType

    yolo_dataset = YOLODataset.detect(input_path)
    if not yolo_dataset:
        console.print(
            f"[red]Error: No YOLO dataset found at {input_path}[/red]\n"
            "[yellow]Ensure the path contains a data.yaml and images/labels "
            "directories.[/yellow]"
        )
        raise typer.Exit(1)

    if yolo_dataset.task != TaskType.SEGMENTATION:
        console.print(
            "[red]Error: YOLO dataset is not a segmentation dataset.[/red]\n"
            "[yellow]Only YOLO segmentation datasets can be converted to COCO formats. "
            "Labels must have polygon coordinates (>5 columns).[/yellow]"
        )
        raise typer.Exit(1)

    if roboflow_layout and use_rle:
        target_name = "Roboflow COCO RLE"
    elif roboflow_layout:
        target_name = "Roboflow COCO"
    else:
        target_name = "COCO"
    console.print(f"[cyan]Converting YOLO segmentation to {target_name} format[/cyan]")
    console.print(f"  Source: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Classes: {yolo_dataset.num_classes}")
    splits_str = ", ".join(yolo_dataset.splits) if yolo_dataset.splits else "unsplit"
    console.print(f"  Splits: {splits_str}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing images...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        try:
            if roboflow_layout and use_rle:
                stats = convert_yolo_seg_to_roboflow_coco_rle(
                    dataset=yolo_dataset,
                    output_path=output_path,
                    progress_callback=update_progress,
                )
            elif roboflow_layout:
                stats = convert_yolo_seg_to_roboflow_coco(
                    dataset=yolo_dataset,
                    output_path=output_path,
                    progress_callback=update_progress,
                )
            else:
                stats = convert_yolo_seg_to_coco(
                    dataset=yolo_dataset,
                    output_path=output_path,
                    progress_callback=update_progress,
                )
        except Exception as exc:
            console.print(f"[red]Error during conversion: {exc}[/red]")
            raise typer.Exit(1) from exc

    console.print()
    console.print("[green]Conversion complete![/green]")
    console.print(f"  Images processed: {stats['images']}")
    console.print(f"  Annotations created: {stats['annotations']}")

    if stats["skipped"] > 0:
        console.print(
            f"  [yellow]Skipped: {stats['skipped']} (could not read)[/yellow]"
        )
    if stats["warnings"] > 0:
        console.print(f"  [yellow]Warnings: {stats['warnings']}[/yellow]")

    console.print(f"\n[cyan]Output dataset: {output_path}[/cyan]")
