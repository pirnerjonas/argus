# Installation

Argus is a CLI, so installation is lightweight. Choose the method that fits your
workflow.

## Quick install with uv

```bash
uvx argus
```

`uvx` runs the package in an isolated environment and keeps it up to date.

## pipx

```bash
pipx install argus
```

## From source

```bash
git clone <your-repo-url>
cd argus
pip install -e .
```

## Requirements

- Python 3.10+
- OpenCV is used for the viewer; you will need a desktop environment for it.
