from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.tracking import ObjectTracker, TrackEvent


class FakeTensor:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[Any]:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        *,
        coordinates: list[list[float]] | None = None,
        track_ids: list[float] | None = None,
    ) -> None:
        self.xyxy = FakeTensor(coordinates or [])
        self.conf = FakeTensor([0.9] * len(self.xyxy.values))
        self.cls = FakeTensor([0.0] * len(self.xyxy.values))
        self.id = FakeTensor(track_ids) if track_ids is not None else None


class FakePrediction:
    names = {0: "person"}

    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes

    def plot(self) -> np.ndarray:
        return np.zeros((20, 20, 3), dtype=np.uint8)


class FakeModel:
    def __init__(self, predictions: Iterator[FakePrediction]) -> None:
        self._predictions = predictions
        self.calls: list[dict[str, Any]] = []

    def track(self, **kwargs: Any) -> list[FakePrediction]:
        self.calls.append(kwargs)
        return [next(self._predictions)]


def prediction(track_id: int | None, x1: float = 0.0) -> FakePrediction:
    if track_id is None:
        return FakePrediction(FakeBoxes())
    return FakePrediction(
        FakeBoxes(
            coordinates=[[x1, 2.0, x1 + 10, 12.0]],
            track_ids=[float(track_id)],
        )
    )


def make_frame() -> np.ndarray:
    return np.zeros((20, 20, 3), dtype=np.uint8)


def test_tracks_objects_and_records_trajectory() -> None:
    model = FakeModel(iter([prediction(4, 0), prediction(4, 2)]))
    with patch("src.tracking.YOLO", return_value=model):
        tracker = ObjectTracker()

        first = tracker.update(make_frame())
        second = tracker.update(make_frame())

    assert first.frame_number == 1
    assert first.tracks[0].track_id == 4
    assert first.tracks[0].center == (5.0, 7.0)
    assert first.events == [TrackEvent("entered", 4, "person", 1)]
    assert second.frame_number == 2
    assert second.tracks[0].center == (7.0, 7.0)
    assert second.events == []
    assert tracker.get_trajectories() == {4: [(5.0, 7.0), (7.0, 7.0)]}
    assert all(call["persist"] is True for call in model.calls)
    assert all(call["tracker"] == "bytetrack.yaml" for call in model.calls)
    assert all(call["device"] == "cpu" for call in model.calls)
    assert all(call["imgsz"] == 640 for call in model.calls)


def test_passes_configured_inference_size_to_model() -> None:
    model = FakeModel(iter([prediction(4)]))
    with patch("src.tracking.YOLO", return_value=model):
        ObjectTracker(image_size=320).update(make_frame())

    assert model.calls[0]["imgsz"] == 320


def test_temporary_detection_loss_does_not_end_track() -> None:
    model = FakeModel(iter([prediction(4), prediction(None), prediction(4, 2)]))
    with patch("src.tracking.YOLO", return_value=model):
        tracker = ObjectTracker(max_missing_frames=1)

        tracker.update(make_frame())
        missing = tracker.update(make_frame())
        recovered = tracker.update(make_frame())

    assert missing.tracks == []
    assert missing.events == []
    assert recovered.tracks[0].track_id == 4
    assert recovered.events == []
    assert tracker.get_trajectories()[4] == [(5.0, 7.0), (7.0, 7.0)]


def test_track_exits_after_missing_frame_limit() -> None:
    model = FakeModel(iter([prediction(4), prediction(None), prediction(None)]))
    with patch("src.tracking.YOLO", return_value=model):
        tracker = ObjectTracker(max_missing_frames=1)

        tracker.update(make_frame())
        tracker.update(make_frame())
        exited = tracker.update(make_frame())

    assert exited.events == [TrackEvent("exited", 4, "person", 3)]


def test_invalid_frame_fails_before_loading_model() -> None:
    with patch("src.tracking.YOLO") as model:
        tracker = ObjectTracker()

        with pytest.raises(ValueError, match="non-empty 8-bit BGR"):
            tracker.update(np.zeros((0, 20, 3), dtype=np.uint8))

        model.assert_not_called()


def test_rejects_negative_missing_frame_limit() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ObjectTracker(max_missing_frames=-1)


@pytest.mark.parametrize("image_size", [0, 641])
def test_rejects_invalid_inference_size(image_size: int) -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        ObjectTracker(image_size=image_size)
