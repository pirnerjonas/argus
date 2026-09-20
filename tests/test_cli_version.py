"""Tests for global CLI options."""

import pytest
from click import unstyle
from typer.testing import CliRunner

from argus import __version__
from argus.cli import app

runner = CliRunner()


@pytest.mark.parametrize("args", [["--version"], ["--version", "stats"]])
def test_version_exits_without_requiring_command_arguments(args: list[str]) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert result.stdout == f"argus-cv {__version__}\n"


def test_help_includes_version() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--version" in unstyle(result.stdout)
