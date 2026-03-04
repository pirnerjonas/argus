"""Interactive image viewers for CLI commands."""

from pathlib import Path

import cv2
import numpy as np

from argus.cli_common import console
from argus.core import COCODataset, Dataset, MaskDataset
from argus.rendering import _draw_annotations


class _ImageViewer:
    """Interactive image viewer with zoom and pan support."""

    def __init__(
        self,
        image_paths: list[Path],
        dataset: Dataset,
        class_colors: dict[str, tuple[int, int, int]],
        window_name: str,
    ):
        self.image_paths = image_paths
        self.dataset = dataset
        self.class_colors = class_colors
        self.window_name = window_name

        self.current_idx = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Mouse state for panning
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.pan_start_x = 0.0
        self.pan_start_y = 0.0

        # Current image cache
        self.current_img: np.ndarray | None = None
        self.annotated_img: np.ndarray | None = None

        # Annotation visibility toggle
        self.show_annotations = True

    def _load_current_image(self) -> bool:
        """Load and annotate the current image."""
        image_path = self.image_paths[self.current_idx]
        annotations = self.dataset.get_annotations_for_image(image_path)

        img = cv2.imread(str(image_path))
        if img is None:
            return False

        self.current_img = img
        self.annotated_img = _draw_annotations(
            img.copy(), annotations, self.class_colors
        )
        return True

    def _get_display_image(self) -> np.ndarray:
        """Get the image transformed for current zoom/pan."""
        if self.annotated_img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if self.show_annotations:
            img = self.annotated_img
        elif self.current_img is not None:
            img = self.current_img
        else:
            img = self.annotated_img
        h, w = img.shape[:2]

        if self.zoom == 1.0 and self.pan_x == 0.0 and self.pan_y == 0.0:
            display = img.copy()
        else:
            # Calculate the visible region
            view_w = int(w / self.zoom)
            view_h = int(h / self.zoom)

            # Center point with pan offset
            cx = w / 2 + self.pan_x
            cy = h / 2 + self.pan_y

            # Calculate crop bounds
            x1 = int(max(0, cx - view_w / 2))
            y1 = int(max(0, cy - view_h / 2))
            x2 = int(min(w, x1 + view_w))
            y2 = int(min(h, y1 + view_h))

            # Adjust if we hit boundaries
            if x2 - x1 < view_w:
                x1 = max(0, x2 - view_w)
            if y2 - y1 < view_h:
                y1 = max(0, y2 - view_h)

            # Crop and resize
            cropped = img[y1:y2, x1:x2]
            display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add info overlay
        image_path = self.image_paths[self.current_idx]
        idx = self.current_idx + 1
        total = len(self.image_paths)
        info_text = f"[{idx}/{total}] {image_path.name}"
        if self.zoom > 1.0:
            info_text += f" (Zoom: {self.zoom:.1f}x)"
        if not self.show_annotations:
            info_text += " [Annotations: OFF]"

        cv2.putText(
            display,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
        )

        return display

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: None
    ) -> None:
        """Handle mouse events for zoom and pan."""
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:
                self.zoom = min(10.0, self.zoom * 1.2)
            else:
                self.zoom = max(1.0, self.zoom / 1.2)

            # Reset pan if zoomed out to 1x
            if self.zoom == 1.0:
                self.pan_x = 0.0
                self.pan_y = 0.0

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_x = x
            self.drag_start_y = y
            self.pan_start_x = self.pan_x
            self.pan_start_y = self.pan_y

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.zoom > 1.0 and self.annotated_img is not None:
                h, w = self.annotated_img.shape[:2]
                # Calculate pan delta (inverted for natural feel)
                dx = (self.drag_start_x - x) / self.zoom
                dy = (self.drag_start_y - y) / self.zoom

                # Update pan with limits
                max_pan_x = w * (1 - 1 / self.zoom) / 2
                max_pan_y = h * (1 - 1 / self.zoom) / 2

                self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_start_x + dx))
                self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_start_y + dy))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _reset_view(self) -> None:
        """Reset zoom and pan to default."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def _next_image(self) -> None:
        """Go to next image."""
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self._reset_view()

    def _prev_image(self) -> None:
        """Go to previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self._reset_view()

    def run(self) -> None:
        """Run the interactive viewer."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            # Load image if needed
            if self.annotated_img is None and not self._load_current_image():
                console.print(
                    f"[yellow]Warning: Could not load "
                    f"{self.image_paths[self.current_idx]}[/yellow]"
                )
                self._next_image()
                continue

            # Display image
            display = self._get_display_image()
            cv2.imshow(self.window_name, display)

            # Wait for input (short timeout for smooth panning)
            key = cv2.waitKey(30) & 0xFF

            # Handle keyboard input
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("n") or key == 83 or key == 3:  # N or Right arrow
                self.annotated_img = None
                self._next_image()
            elif key == ord("p") or key == 81 or key == 2:  # P or Left arrow
                self.annotated_img = None
                self._prev_image()
            elif key == ord("r"):  # R to reset zoom
                self._reset_view()
            elif key == ord("t"):  # T to toggle annotations
                self.show_annotations = not self.show_annotations

        cv2.destroyAllWindows()


class _ClassificationGridViewer:
    """Grid viewer for classification datasets showing one image per class."""

    def __init__(
        self,
        images_by_class: dict[str, list[Path]],
        class_names: list[str],
        window_name: str,
        max_classes: int | None = None,
        tile_size: int = 300,
    ):
        # Limit classes if max_classes specified
        if max_classes and len(class_names) > max_classes:
            self.class_names = class_names[:max_classes]
        else:
            self.class_names = class_names

        self.images_by_class = {
            cls: images_by_class.get(cls, []) for cls in self.class_names
        }
        self.window_name = window_name
        self.tile_size = tile_size

        # Global image index (same for all classes)
        self.current_index = 0

        # Calculate max images across all classes
        self.max_images = (
            max(len(imgs) for imgs in self.images_by_class.values())
            if self.images_by_class
            else 0
        )

        # Calculate grid layout
        self.cols, self.rows = self._calculate_grid_layout()

    def _calculate_grid_layout(self) -> tuple[int, int]:
        """Calculate optimal grid layout based on number of classes."""
        n = len(self.class_names)
        if n <= 0:
            return 1, 1

        # Try to make a roughly square grid
        import math

        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        return cols, rows

    def _create_tile(
        self, class_name: str, image_path: Path | None, index: int, total: int
    ) -> np.ndarray:
        """Create a single tile for a class."""
        tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)

        if image_path is not None and image_path.exists():
            # Load and resize image
            img = cv2.imread(str(image_path))
            if img is not None:
                # Resize maintaining aspect ratio
                h, w = img.shape[:2]
                scale = min(self.tile_size / w, self.tile_size / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Center in tile
                x_offset = (self.tile_size - new_w) // 2
                y_offset = (self.tile_size - new_h) // 2
                tile[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Draw label at top: "class_name (N/M)"
        if image_path is not None:
            label = f"{class_name} ({index + 1}/{total})"
        else:
            label = f"{class_name} (-/{total})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Semi-transparent background for label
        overlay = tile.copy()
        label_bg_height = label_h + baseline + 10
        cv2.rectangle(overlay, (0, 0), (self.tile_size, label_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, tile, 0.4, 0, tile)

        cv2.putText(
            tile,
            label,
            (5, label_h + 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        # Draw thin border
        border_end = self.tile_size - 1
        cv2.rectangle(tile, (0, 0), (border_end, border_end), (80, 80, 80), 1)

        return tile

    def _compose_grid(self) -> np.ndarray:
        """Compose all tiles into a single grid image."""
        grid_h = self.rows * self.tile_size
        grid_w = self.cols * self.tile_size
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, class_name in enumerate(self.class_names):
            row = i // self.cols
            col = i % self.cols

            images = self.images_by_class[class_name]
            total = len(images)

            # Use global index - show black tile if class doesn't have this image
            if self.current_index < total:
                image_path = images[self.current_index]
                display_index = self.current_index
            else:
                image_path = None
                display_index = self.current_index

            tile = self._create_tile(class_name, image_path, display_index, total)

            y_start = row * self.tile_size
            x_start = col * self.tile_size
            y_end = y_start + self.tile_size
            x_end = x_start + self.tile_size
            grid[y_start:y_end, x_start:x_end] = tile

        return grid

    def _next_images(self) -> None:
        """Advance to next image index."""
        if self.max_images > 0:
            self.current_index = min(self.current_index + 1, self.max_images - 1)

    def _prev_images(self) -> None:
        """Go back to previous image index."""
        self.current_index = max(self.current_index - 1, 0)

    def _reset_indices(self) -> None:
        """Reset to first image."""
        self.current_index = 0

    def run(self) -> None:
        """Run the interactive grid viewer."""
        if not self.class_names:
            console.print("[yellow]No classes to display.[/yellow]")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while True:
            # Compose and display grid
            grid = self._compose_grid()
            cv2.imshow(self.window_name, grid)

            # Wait for input
            key = cv2.waitKey(30) & 0xFF

            # Handle keyboard input
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("n") or key == 83 or key == 3:  # N or Right arrow
                self._next_images()
            elif key == ord("p") or key == 81 or key == 2:  # P or Left arrow
                self._prev_images()
            elif key == ord("r"):  # R to reset
                self._reset_indices()

        cv2.destroyAllWindows()


class _MaskViewer:
    """Interactive viewer for semantic mask datasets with colored overlay."""

    def __init__(
        self,
        image_paths: list[Path],
        dataset: MaskDataset | COCODataset,
        class_colors: dict[str, tuple[int, int, int]],
        window_name: str,
        opacity: float = 0.5,
    ):
        self.image_paths = image_paths
        self.dataset = dataset
        self.class_colors = class_colors
        self.window_name = window_name
        self.opacity = opacity

        self.current_idx = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Mouse state for panning
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.pan_start_x = 0.0
        self.pan_start_y = 0.0

        # Current image cache
        self.current_img: np.ndarray | None = None
        self.overlay_img: np.ndarray | None = None

        # Overlay visibility toggle
        self.show_overlay = True

        # Build class_id to color mapping
        self._id_to_color: dict[int, tuple[int, int, int]] = {}
        class_mapping = dataset.get_class_mapping()
        for class_id, class_name in class_mapping.items():
            if class_name in class_colors:
                self._id_to_color[class_id] = class_colors[class_name]

    def _load_current_image(self) -> bool:
        """Load current image and create mask overlay."""
        image_path = self.image_paths[self.current_idx]

        img = cv2.imread(str(image_path))
        if img is None:
            return False

        mask = self.dataset.load_mask(image_path)
        if mask is None:
            console.print(f"[yellow]Warning: No mask for {image_path}[/yellow]")
            self.current_img = img
            self.overlay_img = img.copy()
            return True

        # Validate dimensions
        if img.shape[:2] != mask.shape[:2]:
            console.print(
                f"[red]Error: Dimension mismatch for {image_path.name}: "
                f"image={img.shape[:2]}, mask={mask.shape[:2]}[/red]"
            )
            return False

        self.current_img = img
        self.overlay_img = self._create_overlay(img, mask)
        return True

    def _create_overlay(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create colored overlay from mask.

        Args:
            img: Original image (BGR).
            mask: Grayscale mask with class IDs.

        Returns:
            Image with colored mask overlay.
        """
        # Create colored mask
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self._id_to_color.items():
            colored_mask[mask == class_id] = color

        # Blend with original image
        # Ignore pixels are fully transparent (not blended)
        ignore_mask = mask == self.dataset.ignore_index
        alpha = np.ones((h, w, 1), dtype=np.float32) * self.opacity
        alpha[ignore_mask] = 0.0

        # Blend: result = img * (1 - alpha) + colored_mask * alpha
        blended = (
            img.astype(np.float32) * (1 - alpha)
            + colored_mask.astype(np.float32) * alpha
        )
        return blended.astype(np.uint8)

    def _get_display_image(self) -> np.ndarray:
        """Get the image transformed for current zoom/pan."""
        if self.overlay_img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if self.show_overlay:
            img = self.overlay_img
        elif self.current_img is not None:
            img = self.current_img
        else:
            img = self.overlay_img

        h, w = img.shape[:2]

        if self.zoom == 1.0 and self.pan_x == 0.0 and self.pan_y == 0.0:
            display = img.copy()
        else:
            # Calculate the visible region
            view_w = int(w / self.zoom)
            view_h = int(h / self.zoom)

            # Center point with pan offset
            cx = w / 2 + self.pan_x
            cy = h / 2 + self.pan_y

            # Calculate crop bounds
            x1 = int(max(0, cx - view_w / 2))
            y1 = int(max(0, cy - view_h / 2))
            x2 = int(min(w, x1 + view_w))
            y2 = int(min(h, y1 + view_h))

            # Adjust if we hit boundaries
            if x2 - x1 < view_w:
                x1 = max(0, x2 - view_w)
            if y2 - y1 < view_h:
                y1 = max(0, y2 - view_h)

            # Crop and resize
            cropped = img[y1:y2, x1:x2]
            display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add info overlay
        image_path = self.image_paths[self.current_idx]
        idx = self.current_idx + 1
        total = len(self.image_paths)
        info_text = f"[{idx}/{total}] {image_path.name}"
        if self.zoom > 1.0:
            info_text += f" (Zoom: {self.zoom:.1f}x)"
        if not self.show_overlay:
            info_text += " [Overlay: OFF]"

        cv2.putText(
            display,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
        )

        return display

    def _mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: None
    ) -> None:
        """Handle mouse events for zoom and pan."""
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:
                self.zoom = min(10.0, self.zoom * 1.2)
            else:
                self.zoom = max(1.0, self.zoom / 1.2)

            # Reset pan if zoomed out to 1x
            if self.zoom == 1.0:
                self.pan_x = 0.0
                self.pan_y = 0.0

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_x = x
            self.drag_start_y = y
            self.pan_start_x = self.pan_x
            self.pan_start_y = self.pan_y

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.zoom > 1.0 and self.overlay_img is not None:
                h, w = self.overlay_img.shape[:2]
                # Calculate pan delta (inverted for natural feel)
                dx = (self.drag_start_x - x) / self.zoom
                dy = (self.drag_start_y - y) / self.zoom

                # Update pan with limits
                max_pan_x = w * (1 - 1 / self.zoom) / 2
                max_pan_y = h * (1 - 1 / self.zoom) / 2

                self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_start_x + dx))
                self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_start_y + dy))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _reset_view(self) -> None:
        """Reset zoom and pan to default."""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def _next_image(self) -> None:
        """Go to next image."""
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self._reset_view()

    def _prev_image(self) -> None:
        """Go to previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self._reset_view()

    def run(self) -> None:
        """Run the interactive viewer."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            # Load image if needed
            if self.overlay_img is None and not self._load_current_image():
                console.print(
                    f"[yellow]Warning: Could not load "
                    f"{self.image_paths[self.current_idx]}[/yellow]"
                )
                self._next_image()
                continue

            # Display image
            display = self._get_display_image()
            cv2.imshow(self.window_name, display)

            # Wait for input (short timeout for smooth panning)
            key = cv2.waitKey(30) & 0xFF

            # Handle keyboard input
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("n") or key == 83 or key == 3:  # N or Right arrow
                self.overlay_img = None
                self._next_image()
            elif key == ord("p") or key == 81 or key == 2:  # P or Left arrow
                self.overlay_img = None
                self._prev_image()
            elif key == ord("r"):  # R to reset zoom
                self._reset_view()
            elif key == ord("t"):  # T to toggle overlay
                self.show_overlay = not self.show_overlay

        cv2.destroyAllWindows()
