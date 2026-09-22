"""Local web viewer: dataset queries, image rendering, and a read-only HTTP API."""

import json
import secrets
import webbrowser
from contextlib import suppress
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from argus.cli_common import console
from argus.core import COCODataset, Dataset, MaskDataset
from argus.core.base import TaskType
from argus.rendering import _draw_annotations, _generate_class_colors


class WebViewer:
    """Adapt existing dataset readers to a small, extensible browser API.

    Image IDs are positions in a fixed allowlist, never filesystem paths from a
    request. Add sort keys in ``query`` and controls in ``web/viewer.js``.
    """

    def __init__(
        self,
        dataset: Dataset,
        split: str | None = None,
        max_classes: int | None = None,
        opacity: float = 0.5,
    ):
        self.dataset = dataset
        self.opacity = opacity
        self.colors = _generate_class_colors(dataset.class_names)
        self.classification = dataset.task == TaskType.CLASSIFICATION
        self.mask_mode = isinstance(dataset, MaskDataset) or (
            isinstance(dataset, COCODataset) and dataset.has_rle
        )
        self.groups: dict[str, list[int]] = {}
        if self.classification:
            view_split = split or (dataset.splits[0] if dataset.splits else None)
            grouped = dataset.get_images_by_class(view_split)
            self.image_paths: list[Path] = []
            for name in dataset.class_names[:max_classes]:
                paths = grouped.get(name, [])
                start = len(self.image_paths)
                self.image_paths.extend(paths)
                self.groups[name] = list(range(start, len(self.image_paths)))
        else:
            self.image_paths = dataset.get_image_paths(split)
        self.names = [path.name for path in self.image_paths]
        self._areas: dict[int, float] = {}
        # Bound decoded/encoded image memory and avoid idle refresh work.
        self.image = lru_cache(maxsize=8)(self._image)

    def config(self) -> dict:
        return {
            "title": f"Argus — {self.dataset.path.name}",
            "total": len(self.image_paths),
            "classification": self.classification,
            "classes": list(self.groups),
            "sorts": ["filename"]
            + ([] if self.classification or self.mask_mode else ["object_size"]),
        }

    def _largest_bbox(self, image_id: int) -> float:
        if image_id not in self._areas:
            annotations = self.dataset.get_annotations_for_image(
                self.image_paths[image_id]
            )
            self._areas[image_id] = max(
                (
                    max(0.0, ann["bbox"][2]) * max(0.0, ann["bbox"][3])
                    for ann in annotations
                    if ann.get("bbox")
                ),
                default=0.0,
            )
        return self._areas[image_id]

    def query(
        self,
        search: str = "",
        sort: str = "filename",
        descending: bool = False,
        offset: int = 0,
        limit: int = 1,
    ) -> dict:
        """Return one page; classification pages contain one image per class.

        Object size is the largest bounding-box area in source pixels, including
        polygon bounds. Images without boxes have area zero. Metadata is indexed
        only when this sort is first selected, so opening large datasets is cheap.
        """
        if sort not in self.config()["sorts"]:
            raise ValueError("Unsupported sort field")
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid page bounds")

        def ordered(ids):
            matches = [i for i in ids if search.casefold() in self.names[i].casefold()]
            return sorted(
                matches,
                key=lambda i: (
                    self._largest_bbox(i) if sort == "object_size" else 0,
                    self.names[i].casefold(),
                    i,
                ),
                reverse=descending,
            )

        def item(i, class_name=None):
            return {
                "id": i,
                "filename": self.names[i],
                "class_name": class_name,
                "object_size": self._areas.get(i),
            }

        if self.classification:
            groups = {name: ordered(ids) for name, ids in self.groups.items()}
            total = max((len(ids) for ids in groups.values()), default=0)
            items = [
                item(ids[offset], name)
                if offset < len(ids)
                else {"id": None, "class_name": name, "filename": "No image"}
                for name, ids in groups.items()
            ]
        else:
            ids = ordered(range(len(self.image_paths)))
            total = len(ids)
            items = [item(i) for i in ids[offset : offset + limit]]
        return {"items": items, "total": total, "offset": offset}

    def _image(self, image_id: int, annotations: bool = True) -> bytes:
        if not 0 <= image_id < len(self.image_paths):
            raise IndexError("Unknown image")
        path = self.image_paths[image_id]
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path.name}")
        if annotations and self.mask_mode:
            mask = self.dataset.load_mask(path)
            if mask is not None:
                if mask.shape != img.shape[:2]:
                    raise ValueError(f"Image/mask dimensions differ: {path.name}")
                colored = np.zeros_like(img)
                valid = np.zeros(mask.shape, dtype=bool)
                for class_id, name in self.dataset.get_class_mapping().items():
                    if class_id == self.dataset.ignore_index:
                        continue
                    pixels = mask == class_id
                    colored[pixels] = self.colors.get(name, (0, 255, 0))
                    valid |= pixels
                blended = cv2.addWeighted(
                    img, 1 - self.opacity, colored, self.opacity, 0
                )
                img[valid] = blended[valid]
        elif annotations and not self.classification:
            img = _draw_annotations(
                img, self.dataset.get_annotations_for_image(path), self.colors
            )
        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            raise ValueError(f"Could not encode image: {path.name}")
        return encoded.tobytes()

    def make_server(self, port: int = 0) -> ThreadingHTTPServer:
        """Bind only to loopback with an unguessable, per-session URL prefix."""
        viewer = self
        prefix = "/" + secrets.token_urlsafe(24) + "/"
        assets = {
            "": ("index.html", "text/html; charset=utf-8"),
            "viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
            "viewer.css": ("viewer.css", "text/css; charset=utf-8"),
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def respond(self, body: bytes, content_type: str, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; "
                    "style-src 'self' 'unsafe-inline'; img-src 'self'; "
                    "frame-ancestors 'none'; base-uri 'none'",
                )
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlsplit(self.path)
                if not url.path.startswith(prefix):
                    self.respond(b"Not found", "text/plain", 404)
                    return
                route = url.path[len(prefix) :]
                params = parse_qs(url.query)

                def param(name, default):
                    return params.get(name, [default])[0]

                try:
                    if route in assets:
                        name, content_type = assets[route]
                        body = files("argus").joinpath("web", name).read_bytes()
                        self.respond(body, content_type)
                        return
                    if route == "api/config":
                        result = viewer.config()
                    elif route == "api/images":
                        result = viewer.query(
                            search=param("search", ""),
                            sort=param("sort", "filename"),
                            descending=param("descending", "0") == "1",
                            offset=int(param("offset", "0")),
                            limit=int(param("limit", "1")),
                        )
                    elif route.startswith("api/image/"):
                        image_id = int(route.removeprefix("api/image/"))
                        body = viewer.image(image_id, param("annotations", "1") != "0")
                        self.respond(body, "image/png")
                        return
                    else:
                        self.respond(b"Not found", "text/plain", 404)
                        return
                    self.respond(json.dumps(result).encode(), "application/json")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except IndexError:
                    self.respond(b"Unknown image", "text/plain", 404)
                except (ValueError, OSError, cv2.error) as exc:
                    # Keep dataset paths and decoder details out of HTTP errors.
                    console.print(f"Viewer request failed: {exc}", markup=False)
                    self.respond(b"Unable to load this request", "text/plain", 400)

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.viewer_url = f"http://127.0.0.1:{server.server_port}{prefix}"
        return server

    def run(self, port: int = 0, open_browser: bool = True) -> None:
        with self.make_server(port) as server:
            console.print(f"Viewer: {server.viewer_url}", markup=False)
            console.print("Press Ctrl+C in this terminal to stop the viewer.")
            if open_browser:
                try:
                    webbrowser.open(server.viewer_url)
                except webbrowser.Error:
                    console.print("Open the URL above in your browser.")
            with suppress(KeyboardInterrupt):
                server.serve_forever()
