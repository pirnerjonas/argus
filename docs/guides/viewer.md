# Visual inspection

The viewer overlays boxes and masks for quick spot checks.

## Launching the viewer

```bash
argus-cv view -d /datasets/retail
```

### View a specific split

```bash
argus-cv view -d /datasets/retail --split val
```

## Controls

- Right arrow or `N`: next image
- Left arrow or `P`: previous image
- Mouse wheel: zoom in or out
- Drag: pan while zoomed
- `R`: reset zoom
- `Q` or `Esc`: quit

## Notes

- The viewer uses OpenCV; it requires a desktop environment.
- If the window does not open, make sure you are not on a headless server.
