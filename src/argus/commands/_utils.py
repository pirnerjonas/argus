"""Shared helpers for CLI command modules."""

from pathlib import Path

import typer

from argus.cli_common import console


def _resolve_existing_directory(path: Path) -> Path:
    """Resolve and validate that a path exists and is a directory."""
    resolved = path.resolve()
    if not resolved.exists():
        console.print(f"[red]Error: Path does not exist: {resolved}[/red]")
        raise typer.Exit(1)
    if not resolved.is_dir():
        console.print(f"[red]Error: Path is not a directory: {resolved}[/red]")
        raise typer.Exit(1)
    return resolved


def _resolve_output_path(output_path: Path, base_path: Path) -> Path:
    """Resolve an output path relative to a command-specific dataset root."""
    if not output_path.is_absolute():
        output_path = base_path / output_path
    return output_path.resolve()


def _ensure_output_directory_empty(output_path: Path) -> None:
    """Fail if output directory already exists and is not empty."""
    if output_path.exists() and any(output_path.iterdir()):
        console.print(
            f"[red]Error: Output directory already exists and is not empty: "
            f"{output_path}[/red]"
        )
        raise typer.Exit(1)
