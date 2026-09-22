# Visual inspection

The local web viewer shows bounding boxes, polygons (including holes), and mask
overlays. Classification datasets use a responsive grid with one image per class.
The browser interface uses plain HTML, CSS, and JavaScript; no Node.js build or
additional web framework is required.

## Launching the viewer

```bash
argus-cv view /datasets/retail
argus-cv view /datasets/retail --split val
argus-cv view /datasets/roads --opacity 0.3
argus-cv view /datasets/animals --max-classes 12
```

The command opens your browser and prints the local URL. Keep the terminal running
while browsing, then press **Ctrl+C in the terminal** to stop the server. Closing
the browser tab does not stop the server. By default an available port is selected.

For a headless machine, disable automatic browser launch and choose a port:

```bash
argus-cv view /datasets/retail --no-browser --port 8765
```

The server binds to `127.0.0.1` only. For a remote machine, forward that port with
`ssh -L 8765:127.0.0.1:8765 user@host` and open the printed URL locally. The URL
contains a random per-session token; copy the entire URL. This is a local,
read-only inspection tool, not a public hosting service. Images stay on your
machine; the page loads no third-party scripts or services.

## Search and sorting

- Search filenames without changing the dataset.
- Sort filenames ascending or descending.
- For detection and polygon datasets, sort by **Largest bounding box (px²)**.
  This is the largest `width × height` in the image, in source pixels; polygon
  bounds are included and images without boxes have size zero. It is not polygon
  area or normalized image coverage.
- Object-size sorting indexes annotations on first use, which can take time on
  large datasets. Later sorts reuse that metadata for the current session.
- Semantic masks and COCO RLE overlays support filename sorting. They do not
  expose instance-size sorting because the overlay can combine objects.
- Classification sorting/searching applies within each class. Navigation advances
  all classes together; classes with fewer images show an empty tile. Without
  `--split`, classification uses the first available split, as before.

## Controls

- Previous / Next buttons, arrow keys, or `P` / `N`: navigate
- Mouse wheel over an image: zoom
- Drag: pan while zoomed
- Reset view or `R`: reset zoom and pan
- Show annotations or `T`: toggle boxes, polygons, or masks

Keyboard shortcuts are inactive while a form control has focus. Unreadable images
show an error in their tile; navigation remains available.

## Extending the viewer

`src/argus/viewers.py` adapts the existing dataset APIs and owns querying and image
rendering. Its `WebViewer.query()` returns paginated metadata with stable image IDs;
add sort keys or filters there and expose them in `config()`. HTTP routes serve only
packaged assets and images in the selected dataset's image list.

The frontend lives in `src/argus/web/`: `index.html`, `viewer.css`, and `viewer.js`.
It requests `api/config`, `api/images`, and `api/image/<id>` relative to the session
URL. It can be extended independently of the YOLO, COCO, and mask readers. OpenCV
still decodes and renders images, but the viewer no longer uses desktop GUI APIs.
The server keeps at most eight rendered image responses in memory. Dataset edits
made during a session require restarting the viewer to refresh cached results.
