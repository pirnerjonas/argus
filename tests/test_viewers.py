"""Tests for interactive viewer behavior."""

from pathlib import Path

import numpy as np

from argus import viewers


def test_classification_grid_reuses_cached_thumbnails(
    tmp_path: Path, monkeypatch
) -> None:
    """Repeated grid composition should not reload unchanged class images."""
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.touch()
    image_b.touch()

    read_calls: list[str] = []
    resize_calls = 0

    def fake_imread(path: str) -> np.ndarray:
        read_calls.append(path)
        return np.ones((20, 40, 3), dtype=np.uint8)

    def fake_resize(
        image: np.ndarray,
        size: tuple[int, int],
        interpolation: int,
    ) -> np.ndarray:
        nonlocal resize_calls
        resize_calls += 1
        width, height = size
        return np.ones((height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(viewers.cv2, "imread", fake_imread)
    monkeypatch.setattr(viewers.cv2, "resize", fake_resize)

    viewer = viewers._ClassificationGridViewer(
        images_by_class={"a": [image_a], "b": [image_b]},
        class_names=["a", "b"],
        window_name="test",
        tile_size=100,
    )

    first = viewer._compose_grid()
    second = viewer._compose_grid()

    assert first is second
    assert read_calls == [str(image_a), str(image_b)]
    assert resize_calls == 2
