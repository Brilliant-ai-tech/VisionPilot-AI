import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from src.detection import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH, validate_image_size
from src.risk import RiskAssessment, RiskAssessor
from src.tracking import ObjectTracker, TrackingFrame
from src.visualization import draw_risk_overlay

if TYPE_CHECKING:
    from src.storage import SQLiteRepository


class FrameTracker(Protocol):
    def update(self, frame: np.ndarray) -> TrackingFrame: ...


@dataclass(frozen=True)
class VideoProcessingSummary:
    input_path: str
    output_path: str
    frames_read: int
    frames_analyzed: int
    frames_skipped: int
    source_fps: float
    source_duration_seconds: float
    processing_duration_seconds: float
    processing_fps: float
    unique_tracks: int
    tracked_object_observations: int
    risk_assessment_count: int
    risk_level_counts: dict[str, int]
    highest_risk_assessment: RiskAssessment | None
    session_id: str | None = None


class VideoProcessor:
    """Process a video sequentially with detection tracking and risk analysis."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.25,
        frame_skip: int = 0,
        image_size: int = DEFAULT_IMAGE_SIZE,
        tracker: FrameTracker | None = None,
        risk_assessor: RiskAssessor | None = None,
        repository: "SQLiteRepository | None" = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if frame_skip < 0:
            raise ValueError("Frame skip cannot be negative.")
        validate_image_size(image_size)
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.tracker = (
            tracker
            if tracker is not None
            else ObjectTracker(
                model_path=model_path,
                confidence=confidence,
                image_size=image_size,
            )
        )
        self.risk_assessor = (
            risk_assessor if risk_assessor is not None else RiskAssessor()
        )
        self.repository = repository

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> VideoProcessingSummary:
        source = Path(input_path)
        destination = Path(output_path)
        if not source.is_file():
            raise FileNotFoundError(f"Video file does not exist: {source}")
        if source.resolve() == destination.resolve():
            raise ValueError("Output path must be different from the input video.")
        if destination.suffix.lower() not in {".mp4", ".avi"}:
            raise ValueError("Output video must use the .mp4 or .avi extension.")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video file: {source}")

        writer: cv2.VideoWriter | None = None
        session_id: str | None = None
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(source_fps) or source_fps <= 0.0:
                raise ValueError(f"Video has an invalid frame rate: {source_fps}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            if self.repository is not None:
                session_id = self.repository.start_session(str(source.resolve()))
            started_at = time.perf_counter()
            frames_read = 0
            frames_analyzed = 0
            tracked_object_observations = 0
            track_ids: set[int] = set()
            risk_assessment_count = 0
            risk_level_counts = {"low": 0, "medium": 0, "high": 0}
            highest_risk_assessment: RiskAssessment | None = None
            previous_annotated_frame: np.ndarray | None = None
            width: int | None = None
            height: int | None = None
            stride = self.frame_skip + 1

            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame is None or frame.size == 0:
                    raise RuntimeError(
                        f"Video decoder returned an empty frame at index {frames_read}."
                    )

                frame_height, frame_width = frame.shape[:2]
                if width is None or height is None:
                    width, height = frame_width, frame_height
                    writer = self._create_writer(destination, source_fps, width, height)
                elif (frame_width, frame_height) != (width, height):
                    raise RuntimeError(
                        "Video frame dimensions changed during processing."
                    )

                if frames_read % stride == 0:
                    tracking_frame = self.tracker.update(frame)
                    risk_frame = self.risk_assessor.assess(tracking_frame)
                    if self.repository is not None and session_id is not None:
                        self.repository.record_frame(
                            session_id,
                            frames_read + 1,
                            tracking_frame.tracks,
                            tracking_frame.events,
                            risk_frame.assessments,
                        )
                    annotated = draw_risk_overlay(
                        tracking_frame.annotated_frame,
                        risk_frame.assessments,
                    )
                    if annotated.shape[:2] != (height, width):
                        raise RuntimeError(
                            "Tracker returned an annotated frame with unexpected dimensions."
                        )
                    previous_annotated_frame = annotated
                    frames_analyzed += 1
                    tracked_object_observations += len(tracking_frame.tracks)
                    track_ids.update(track.track_id for track in tracking_frame.tracks)
                    risk_assessment_count += len(risk_frame.assessments)
                    for assessment in risk_frame.assessments:
                        risk_level_counts[assessment.level] += 1
                        if (
                            highest_risk_assessment is None
                            or assessment.score > highest_risk_assessment.score
                        ):
                            highest_risk_assessment = assessment
                elif previous_annotated_frame is None:
                    raise RuntimeError("The first video frame must be analyzed.")

                output_frame = previous_annotated_frame
                if output_frame is None or writer is None:
                    raise RuntimeError("Video output was not initialized.")
                writer.write(output_frame)
                frames_read += 1

            if frames_read == 0:
                raise ValueError(f"Video contains no readable frames: {source}")

            processing_duration = time.perf_counter() - started_at
            summary = VideoProcessingSummary(
                input_path=str(source.resolve()),
                output_path=str(destination.resolve()),
                frames_read=frames_read,
                frames_analyzed=frames_analyzed,
                frames_skipped=frames_read - frames_analyzed,
                source_fps=source_fps,
                source_duration_seconds=frames_read / source_fps,
                processing_duration_seconds=processing_duration,
                processing_fps=frames_read / processing_duration,
                unique_tracks=len(track_ids),
                tracked_object_observations=tracked_object_observations,
                risk_assessment_count=risk_assessment_count,
                risk_level_counts=risk_level_counts,
                highest_risk_assessment=highest_risk_assessment,
                session_id=session_id,
            )
            if self.repository is not None and session_id is not None:
                self.repository.complete_session(session_id, summary)
            return summary
        except Exception as error:
            if self.repository is not None and session_id is not None:
                self.repository.fail_session(session_id, str(error))
            raise
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    @staticmethod
    def save_summary(summary: VideoProcessingSummary, output_path: str | Path) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as summary_file:
            json.dump(asdict(summary), summary_file, indent=2)
            summary_file.write("\n")

    @staticmethod
    def _create_writer(
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        codec = "mp4v" if output_path.suffix.lower() == ".mp4" else "MJPG"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Could not create output video: {output_path}")
        return writer
