# Visual inspection

The viewer overlays boxes and masks for quick spot checks. For mask datasets,
it blends the segmentation mask over the image.

## Launching the viewer

```bash
argus-cv view -d /datasets/retail
```

### View a specific split

```bash
argus-cv view -d /datasets/retail --split val
```

### Adjust mask opacity

```bash
argus-cv view -d /datasets/roads --opacity 0.3
```

## Controls

- Right arrow or `N`: next image
- Left arrow or `P`: previous image
- Mouse wheel: zoom in or out
- Drag: pan while zoomed
- `R`: reset zoom
- `Q` or `Esc`: quit
- `T`: toggle annotations or mask overlay

## Notes

- The viewer uses OpenCV; it requires a desktop environment.
- If the window does not open, make sure you are not on a headless server.
