"""Validate command implementation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from argus.cli_common import console
from argus.commands._utils import _resolve_existing_directory
from argus.core.validation import validate_dataset
from argus.discovery import _detect_dataset


def validate(
    dataset: Annotated[
        Path | None,
        typer.Argument(
            help="Path to the dataset root directory.",
            show_default=False,
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Fail on warnings (exit code 1)."),
    ] = False,
    split: Annotated[
        str | None,
        typer.Option("--split", help="Validate only a specific split."),
    ] = None,
    max_issues: Annotated[
        int,
        typer.Option("--max-issues", help="Max issues to display (0=all)."),
    ] = 0,
    check_images: Annotated[
        bool,
        typer.Option("--check-images", help="Verify image files are readable."),
    ] = False,
) -> None:
    """Validate a dataset for annotation quality issues.

    Checks for common problems like missing images, invalid coordinates,
    broken references, and format-specific issues. Reports errors and
    warnings to help fix datasets before training.
    """
    dataset_path = _resolve_existing_directory(dataset or Path("."))

    detected = _detect_dataset(dataset_path)
    if not detected:
        console.print(
            f"[red]Error: No dataset found at {dataset_path}[/red]\n"
            "[yellow]Ensure the path points to a dataset root containing "
            "data.yaml (YOLO), annotations/ folder (COCO), or "
            "images/ + masks/ directories (Mask).[/yellow]"
        )
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Validating dataset...", total=None)
        report = validate_dataset(detected, split=split, check_images=check_images)

    if not report.issues:
        console.print(
            f"[green]Dataset is valid: {dataset_path.name} "
            f"({detected.format.value})[/green]"
        )
        console.print("[green]No issues found.[/green]")
        return

    # Build results table
    issues_to_show = report.issues
    if max_issues > 0:
        issues_to_show = issues_to_show[:max_issues]

    table = Table(title=f"Validation: {dataset_path.name} ({detected.format.value})")
    table.add_column("Level", style="bold", width=7)
    table.add_column("Code", style="cyan", width=5)
    table.add_column("Split", width=8)
    table.add_column("File")
    table.add_column("Message")

    for issue in issues_to_show:
        level_str = (
            "[red]ERROR[/red]" if issue.level == "error" else "[yellow]WARN[/yellow]"
        )
        file_str = issue.file.name if issue.file else "-"
        split_str = issue.split or "-"
        table.add_row(level_str, issue.code, split_str, file_str, issue.message)

    console.print(table)

    if max_issues > 0 and len(report.issues) > max_issues:
        console.print(
            f"[dim]... and {len(report.issues) - max_issues} more issues "
            f"(use --max-issues 0 to show all)[/dim]"
        )

    # Summary
    num_errors = len(report.errors)
    num_warnings = len(report.warnings)
    console.print(
        f"\n[{'red' if num_errors else 'green'}]{num_errors} error(s)[/], "
        f"[{'yellow' if num_warnings else 'green'}]{num_warnings} warning(s)[/] found."
    )

    if num_errors > 0 or (strict and num_warnings > 0):
        raise typer.Exit(1)
