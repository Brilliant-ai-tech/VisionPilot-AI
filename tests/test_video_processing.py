from pathlib import Path
import sqlite3

import cv2
import numpy as np
import pytest

from src.risk import RiskAssessor
from src.storage import SQLiteRepository
from src.tracking import TrackEvent, TrackedObject, TrackingFrame
from src.video_processing import VideoProcessor


class FakeTracker:
    def __init__(self) -> None:
        self.frame_number = 0
        self.calls = 0

    def update(self, frame: np.ndarray) -> TrackingFrame:
        self.frame_number += 1
        self.calls += 1
        track = TrackedObject(
            track_id=7,
            box=(2.0, 2.0, 12.0, 12.0),
            confidence=0.9,
            class_id=2,
            label="car",
            center=(7.0, 7.0),
        )
        events = (
            [TrackEvent("entered", 7, "car", self.frame_number)]
            if self.frame_number == 1
            else []
        )
        return TrackingFrame(
            frame_number=self.frame_number,
            tracks=[track],
            events=events,
            annotated_frame=frame.copy(),
        )


def write_test_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV MJPG video writer is unavailable.")
    for frame_index in range(frame_count):
        frame = np.full((24, 32, 3), frame_index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def count_video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise AssertionError(f"Could not open processed video: {path}")
    count = 0
    try:
        while True:
            success, _ = capture.read()
            if not success:
                break
            count += 1
    finally:
        capture.release()
    return count


def test_processes_video_and_preserves_duration_when_skipping_frames(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.avi"
    summary_path = tmp_path / "summary.json"
    write_test_video(input_path)
    tracker = FakeTracker()
    processor = VideoProcessor(
        frame_skip=1,
        tracker=tracker,
        risk_assessor=RiskAssessor(),
    )

    summary = processor.process(input_path, output_path)
    processor.save_summary(summary, summary_path)

    assert summary.frames_read == 4
    assert summary.frames_analyzed == 2
    assert summary.frames_skipped == 2
    assert summary.unique_tracks == 1
    assert summary.tracked_object_observations == 2
    assert tracker.calls == 2
    assert count_video_frames(output_path) == 4
    assert summary_path.is_file()
    assert '"frames_read": 4' in summary_path.read_text(encoding="utf-8")


def test_persists_video_detections_and_session_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.avi"
    write_test_video(input_path, frame_count=2)
    repository = SQLiteRepository(tmp_path / "visionpilot.sqlite")
    processor = VideoProcessor(
        tracker=FakeTracker(),
        repository=repository,
    )

    summary = processor.process(input_path, output_path)

    assert summary.session_id is not None
    session = repository.get_session(summary.session_id)
    assert session.status == "completed"
    assert session.summary is not None
    assert session.summary["frames_read"] == 2
    export_path = repository.export_session_json(
        summary.session_id,
        tmp_path / "session.json",
    )
    assert '"label": "car"' in export_path.read_text(encoding="utf-8")


def test_marks_video_session_failed_when_processing_errors(tmp_path: Path) -> None:
    class FailingTracker:
        def update(self, frame: np.ndarray) -> TrackingFrame:
            raise RuntimeError(f"synthetic tracking failure for frame {frame.shape}")

    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.avi"
    database_path = tmp_path / "visionpilot.sqlite"
    write_test_video(input_path, frame_count=1)
    repository = SQLiteRepository(database_path)
    processor = VideoProcessor(tracker=FailingTracker(), repository=repository)

    with pytest.raises(RuntimeError, match="synthetic tracking failure"):
        processor.process(input_path, output_path)

    with sqlite3.connect(database_path) as connection:
        status, error = connection.execute(
            "SELECT status, error FROM analysis_sessions"
        ).fetchone()
    assert status == "failed"
    assert error.startswith("synthetic tracking failure for frame")


def test_missing_video_fails_before_tracker_runs(tmp_path: Path) -> None:
    tracker = FakeTracker()
    processor = VideoProcessor(tracker=tracker)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        processor.process(tmp_path / "missing.mp4", tmp_path / "output.mp4")

    assert tracker.calls == 0


def test_rejects_input_as_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    write_test_video(input_path)
    processor = VideoProcessor(tracker=FakeTracker())

    with pytest.raises(ValueError, match="different from the input"):
        processor.process(input_path, input_path)


def test_rejects_unsupported_output_extension(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    write_test_video(input_path)
    processor = VideoProcessor(tracker=FakeTracker())

    with pytest.raises(ValueError, match="must use the .mp4 or .avi"):
        processor.process(input_path, tmp_path / "output.mkv")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_skip": -1},
        {"confidence": 1.1},
        {"image_size": 641},
    ],
)
def test_rejects_invalid_processing_options(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        VideoProcessor(**kwargs)
