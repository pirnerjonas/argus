"""Argus CLI - Vision AI dataset toolkit."""

from typing import Annotated

import typer

from argus import __version__
from argus.cli_common import console
from argus.commands.convert_command import convert_dataset
from argus.commands.filter_command import filter_dataset
from argus.commands.list_command import list_datasets
from argus.commands.split_command import split_dataset
from argus.commands.stats_command import stats
from argus.commands.unsplit_command import unsplit_dataset
from argus.commands.view_command import view
from argus.discovery import _detect_dataset, _discover_datasets
from argus.rendering import _draw_annotations, _generate_class_colors
from argus.viewers import _ClassificationGridViewer, _ImageViewer, _MaskViewer

app = typer.Typer(
    name="argus-cv",
    help="Vision AI dataset toolkit for working with YOLO, COCO, and mask datasets.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print the package version and exit before running a command."""
    if value:
        typer.echo(f"argus-cv {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Vision AI dataset toolkit for working with YOLO, COCO, and mask datasets."""
    pass


app.command(name="list")(list_datasets)
app.command(name="stats")(stats)
app.command(name="view")(view)
app.command(name="split")(split_dataset)
app.command(name="unsplit")(unsplit_dataset)
app.command(name="convert")(convert_dataset)
app.command(name="filter")(filter_dataset)

__all__ = [
    "app",
    "callback",
    "console",
    "list_datasets",
    "stats",
    "view",
    "split_dataset",
    "unsplit_dataset",
    "convert_dataset",
    "filter_dataset",
    "_discover_datasets",
    "_detect_dataset",
    "_generate_class_colors",
    "_draw_annotations",
    "_ImageViewer",
    "_ClassificationGridViewer",
    "_MaskViewer",
]


if __name__ == "__main__":
    app()
