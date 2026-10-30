from unittest.mock import patch

import numpy as np
import pytest

from src.risk import RiskAssessor
from src.storage import SQLiteRepository
from src.tracking import TrackedObject, TrackingFrame
from src.live_camera import LiveCamera


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], opened: bool = True) -> None:
        self.frames = iter(frames)
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


class FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, frame: np.ndarray) -> TrackingFrame:
        self.calls += 1
        track = TrackedObject(
            track_id=7,
            box=(2.0, 2.0, 12.0, 12.0),
            confidence=0.9,
            class_id=2,
            label="car",
            center=(7.0, 7.0),
        )
        return TrackingFrame(
            frame_number=self.calls,
            tracks=[track],
            events=[],
            annotated_frame=frame.copy(),
        )


def test_processes_live_frames_and_releases_camera() -> None:
    capture = FakeCapture([np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)])
    tracker = FakeTracker()
    with (
        patch("src.live_camera.cv2.VideoCapture", return_value=capture),
        patch("src.live_camera.cv2.imshow"),
        patch("src.live_camera.cv2.waitKey", return_value=0),
        patch("src.live_camera.cv2.destroyAllWindows"),
    ):
        summary = LiveCamera(
            frame_skip=1,
            tracker=tracker,
            risk_assessor=RiskAssessor(),
        ).run(max_frames=3)

    assert capture.released
    assert tracker.calls == 2
    assert summary.frames_read == 3
    assert summary.frames_analyzed == 2
    assert summary.frames_skipped == 1
    assert summary.average_fps > 0


def test_persists_camera_session_and_detections(tmp_path) -> None:
    capture = FakeCapture([np.zeros((20, 20, 3), dtype=np.uint8)])
    tracker = FakeTracker()
    repository = SQLiteRepository(tmp_path / "visionpilot.sqlite")
    with (
        patch("src.live_camera.cv2.VideoCapture", return_value=capture),
        patch("src.live_camera.cv2.imshow"),
        patch("src.live_camera.cv2.waitKey", return_value=0),
        patch("src.live_camera.cv2.destroyAllWindows"),
    ):
        summary = LiveCamera(
            tracker=tracker,
            repository=repository,
        ).run(max_frames=1)

    assert summary.session_id is not None
    session = repository.get_session(summary.session_id)
    assert session.status == "completed"
    assert session.source == "camera://0"
    export_path = repository.export_session_csv(
        summary.session_id,
        tmp_path / "camera.csv",
    )
    assert "detection" in export_path.read_text(encoding="utf-8")


def test_camera_is_released_when_frame_read_fails() -> None:
    capture = FakeCapture([], opened=True)
    with (
        patch("src.live_camera.cv2.VideoCapture", return_value=capture),
        patch("src.live_camera.cv2.destroyAllWindows"),
    ):
        with pytest.raises(RuntimeError, match="stopped returning valid frames"):
            LiveCamera(tracker=FakeTracker()).run()

    assert capture.released


def test_camera_is_released_when_device_cannot_open() -> None:
    capture = FakeCapture([], opened=False)
    with patch("src.live_camera.cv2.VideoCapture", return_value=capture):
        with pytest.raises(OSError, match="Could not open camera index"):
            LiveCamera().run(max_frames=1)

    assert capture.released


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_index": -1},
        {"frame_skip": -1},
        {"confidence": 1.1},
        {"image_size": 641},
    ],
)
def test_rejects_invalid_camera_options(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        LiveCamera(**kwargs)


def test_rejects_non_positive_max_frames() -> None:
    with pytest.raises(ValueError, match="positive number"):
        LiveCamera().run(max_frames=0)
