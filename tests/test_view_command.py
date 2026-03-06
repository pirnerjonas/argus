"""Tests for the view command."""

from pathlib import Path

from click.termui import strip_ansi
from typer.testing import CliRunner

from argus.cli import app
from argus.commands import view_command

runner = CliRunner()


class _DummyImageViewer:
    """Minimal viewer stub for CLI tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self) -> None:
        pass


def test_view_command_accepts_positional_dataset(
    yolo_detection_dataset: Path, monkeypatch
) -> None:
    """Test view command accepts a positional dataset path."""
    monkeypatch.setattr(view_command, "_ImageViewer", _DummyImageViewer)

    result = runner.invoke(app, ["view", str(yolo_detection_dataset)])

    assert result.exit_code == 0
    assert "Opening viewer" in result.stdout


def test_view_command_rejects_removed_dataset_option(
    yolo_detection_dataset: Path,
) -> None:
    """Test view command no longer accepts --dataset-path."""
    result = runner.invoke(
        app,
        [
            "view",
            "--dataset-path",
            str(yolo_detection_dataset),
        ],
    )
    help_result = runner.invoke(app, ["view", "--help"])

    assert result.exit_code == 2
    assert help_result.exit_code == 0
    assert "--dataset-path" not in strip_ansi(help_result.output)
