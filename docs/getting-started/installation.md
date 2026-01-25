# Installation

Argus is a CLI, so installation is lightweight. Choose the method that fits your
workflow.

## Quick install with uv

```bash
uvx argus-cv
```

`uvx` runs the package in an isolated environment and keeps it up to date.

Verify it works:

```bash
argus-cv --help
```

## pipx

```bash
pipx install argus-cv
```

Verify it works:

```bash
argus-cv --help
```

## From source

```bash
git clone https://github.com/pirnerjonas/argus
cd argus
pip install -e .
```

Verify it works:

```bash
argus-cv --help
```

## Requirements

- Python 3.10+
- OpenCV is used for the viewer; you will need a desktop environment for it.
