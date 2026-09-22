"""Tests for the view command."""

from pathlib import Path

from click.termui import strip_ansi
from typer.testing import CliRunner

from argus.cli import app
from argus.commands import view_command

runner = CliRunner()


def test_view_command_accepts_positional_dataset(
    yolo_detection_dataset: Path, monkeypatch
) -> None:
    """Test view command accepts a positional dataset path."""
    monkeypatch.setattr(view_command.WebViewer, "run", lambda *a, **kw: None)

    result = runner.invoke(app, ["view", str(yolo_detection_dataset)])

    assert result.exit_code == 0
    assert "Opening web viewer" in result.stdout


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


def test_web_launch_options(yolo_detection_dataset, monkeypatch):
    calls = []
    monkeypatch.setattr(
        view_command.WebViewer, "run", lambda self, **kwargs: calls.append(kwargs)
    )
    result = runner.invoke(
        app,
        [
            "view",
            str(yolo_detection_dataset),
            "--no-browser",
            "--port",
            "8765",
        ],
    )
    assert result.exit_code == 0
    assert calls == [{"port": 8765, "open_browser": False}]


def test_invalid_split_does_not_start_server(yolo_detection_dataset, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Server should not start")

    monkeypatch.setattr(view_command.WebViewer, "run", fail)
    result = runner.invoke(
        app, ["view", str(yolo_detection_dataset), "--split", "missing"]
    )
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_port_error_is_readable(yolo_detection_dataset, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Address already in use")

    monkeypatch.setattr(view_command.WebViewer, "run", fail)
    result = runner.invoke(app, ["view", str(yolo_detection_dataset)])
    assert result.exit_code == 1
    assert "Unable to start viewer" in result.stdout
