<div class="hero">
  <div class="hero__content">
    <p class="hero__eyebrow">Argus</p>
    <h1>Vision AI dataset work, without the friction.</h1>
    <p>
      Argus is a focused CLI for YOLO and COCO datasets. List datasets, inspect
      class balance, view annotations, and split cleanly for training.
    </p>
    <div class="hero__actions">
      <a class="md-button md-button--primary" href="getting-started/quickstart/">Get started</a>
      <a class="md-button" href="guides/listing/">See the commands</a>
    </div>
  </div>
  <div class="hero__card">

```bash
# Discover datasets
argus-cv list --path /data

# Instant class stats
argus-cv stats -d /data/animals

# Visual inspection
argus-cv view -d /data/animals --split val
```

  </div>
</div>

<div class="grid cards">
  <div class="card">
    <h3>Format-aware</h3>
    <p>Detects YOLO and COCO by structure and metadata, not guesses.</p>
  </div>
  <div class="card">
    <h3>Readable statistics</h3>
    <p>Per-class counts and background image totals at a glance.</p>
  </div>
  <div class="card">
    <h3>Annotation viewer</h3>
    <p>Browse images with boxes and masks overlayed. Pan and zoom included.</p>
  </div>
  <div class="card">
    <h3>Clean splits</h3>
    <p>Stratified splitting for YOLO and COCO with deterministic seeds.</p>
  </div>
</div>

## Common workflows

=== "Quick scan"

    ```bash
    argus-cv list --path /datasets
    argus-cv stats -d /datasets/retail
    ```

=== "Find label issues"

    ```bash
    argus-cv view -d /datasets/retail --split val
    ```

=== "Create splits"

    ```bash
    argus-cv split -d /datasets/retail -o /datasets/retail_splits -r 0.8,0.1,0.1
    ```

## What Argus expects

Argus detects datasets by their structure. If you point it at the dataset root,
commands will usually just work.

- YOLO: a `data.yaml` (or `.yml`) with `names`, plus `images/` and `labels/`
- COCO: one or more `instances_*.json` files in `annotations/`

Head to the formats guide for the full layouts and edge cases.

[Explore dataset formats](guides/datasets.md){ .md-button .md-button--primary }
