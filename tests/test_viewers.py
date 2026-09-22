"""Behavior and HTTP integration tests for the local web viewer."""

import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

from argus.discovery import _detect_dataset
from argus.viewers import WebViewer


def viewer_for(path, **kwargs):
    dataset = _detect_dataset(path)
    assert dataset is not None
    return WebViewer(dataset, **kwargs)


def decode(body):
    return cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_query_sort_filter_and_stable_ids(yolo_real_detection_dataset, monkeypatch):
    viewer = viewer_for(yolo_real_detection_dataset)
    calls = []

    def annotations(path):
        calls.append(path)
        return [{"bbox": (0, 0, viewer.image_paths.index(path) + 1, 10)}]

    monkeypatch.setattr(viewer.dataset, "get_annotations_for_image", annotations)
    page = viewer.query(limit=100)
    assert not calls  # Filename browsing does not decode annotations.
    assert [x["filename"] for x in page["items"]] == sorted(viewer.names)
    page = viewer.query(sort="object_size", descending=True, limit=100)
    assert page["items"][0]["id"] == len(viewer.image_paths) - 1
    assert page["items"][0]["object_size"] == len(viewer.image_paths) * 10
    viewer.query(sort="object_size")
    assert len(calls) == len(viewer.image_paths)
    name = viewer.names[0]
    assert all(x["filename"] == name for x in viewer.query(search=name)["items"])
    assert viewer.query(search="absent-file")["total"] == 0
    assert viewer.query(offset=10000)["items"] == []
    with pytest.raises(ValueError):
        viewer.query(sort="unknown")
    with pytest.raises(ValueError):
        viewer.query(limit=101)


@pytest.mark.parametrize(
    "fixture",
    [
        "yolo_real_detection_dataset",
        "yolo_real_segmentation_dataset",
        "mask_dataset_grayscale",
        "coco_rle_dataset",
        "coco_mixed_rle_polygon_dataset",
    ],
)
def test_render_annotation_formats(fixture, request):
    viewer = viewer_for(request.getfixturevalue(fixture))
    raw = decode(viewer.image(0, False))
    annotated = decode(viewer.image(0, True))
    assert raw.shape == annotated.shape
    assert np.any(raw != annotated)
    assert viewer.image(0, True) is viewer.image(0, True)


def test_mask_opacity_and_ignored_pixels(mask_dataset_grayscale):
    viewer = viewer_for(mask_dataset_grayscale, opacity=0)
    assert viewer.image(0, True) == viewer.image(0, False)
    viewer = viewer_for(mask_dataset_grayscale, opacity=1)
    raw = decode(viewer.image(0, False))
    annotated = decode(viewer.image(0, True))
    mask = viewer.dataset.load_mask(viewer.image_paths[0])
    ignored = mask == viewer.dataset.ignore_index
    np.testing.assert_array_equal(raw[ignored], annotated[ignored])
    assert "object_size" not in viewer.config()["sorts"]


def test_classification_grid(yolo_classification_multiclass_dataset):
    viewer = viewer_for(yolo_classification_multiclass_dataset, max_classes=2)
    assert len(viewer.groups) == 2
    page = viewer.query()
    assert len(page["items"]) == 2
    assert page["total"] == max(len(ids) for ids in viewer.groups.values())
    assert all(item["class_name"] in viewer.groups for item in page["items"])
    assert all(item["id"] is None for item in viewer.query(offset=10000)["items"])
    assert viewer.config()["sorts"] == ["filename"]


def test_missing_and_invalid_images(yolo_real_detection_dataset):
    viewer = viewer_for(yolo_real_detection_dataset)
    with pytest.raises(IndexError):
        viewer.image(-1)
    viewer.image_paths[0].write_bytes(b"invalid image")
    with pytest.raises(ValueError, match="Could not read"):
        viewer.image(0)


def test_mask_dimension_error(mask_dataset_dimension_mismatch):
    viewer = viewer_for(mask_dataset_dimension_mismatch)
    with pytest.raises(ValueError, match="dimensions differ"):
        viewer.image(0)
    assert decode(viewer.image(0, False)) is not None


def test_http_api_and_file_allowlist(yolo_real_detection_dataset):
    viewer = viewer_for(yolo_real_detection_dataset)
    with viewer.make_server() as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = server.viewer_url
            assert server.server_address[0] == "127.0.0.1"
            for route, mime in [
                ("", "text/html"),
                ("viewer.js", "text/javascript"),
                ("viewer.css", "text/css"),
                ("api/image/0", "image/png"),
            ]:
                with urlopen(base + route) as response:
                    assert response.headers["Content-Type"].startswith(mime)
                    assert response.headers["X-Content-Type-Options"] == "nosniff"
                    assert response.read()
            with urlopen(base + "api/images") as response:
                assert json.load(response)["total"] == len(viewer.image_paths)
            for route, code in [
                ("../pyproject.toml", 404),
                ("api/image/-1", 404),
                ("api/image/999999", 404),
                ("api/image/../../etc/passwd", 400),
                ("api/images?sort=invalid", 400),
                ("api/images?offset=-1", 400),
            ]:
                with pytest.raises(HTTPError) as exc:
                    urlopen(base + route)
                assert exc.value.code == code
            with pytest.raises(HTTPError) as exc:
                urlopen(f"http://127.0.0.1:{server.server_port}/api/images")
            assert exc.value.code == 404
        finally:
            server.shutdown()
            thread.join()
