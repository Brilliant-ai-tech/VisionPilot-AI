import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.detection import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH, validate_image_size
from src.risk import RiskAssessment, RiskAssessor
from src.storage import SQLiteRepository
from src.tracking import ObjectTracker, TrackingFrame
from src.visualization import draw_live_status, draw_risk_overlay
from src.video_processing import FrameTracker


@dataclass(frozen=True)
class LiveCameraSummary:
    camera_index: int
    frames_read: int
    frames_analyzed: int
    frames_skipped: int
    unique_tracks: int
    risk_assessment_count: int
    risk_level_counts: dict[str, int]
    average_fps: float
    highest_risk_assessment: RiskAssessment | None
    session_id: str | None = None


class LiveCamera:
    """Run camera frames through detection, tracking, and risk assessment."""

    window_name = "VisionPilot AI - Live Camera"

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.25,
        frame_skip: int = 0,
        image_size: int = DEFAULT_IMAGE_SIZE,
        tracker: FrameTracker | None = None,
        risk_assessor: RiskAssessor | None = None,
        repository: SQLiteRepository | None = None,
    ) -> None:
        if camera_index < 0:
            raise ValueError("Camera index cannot be negative.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if frame_skip < 0:
            raise ValueError("Frame skip cannot be negative.")
        validate_image_size(image_size)

        self.camera_index = camera_index
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

    def run(self, max_frames: int | None = None) -> LiveCameraSummary:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("Maximum frames must be a positive number.")

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open camera index {self.camera_index}.")

        session_id: str | None = None
        try:
            if self.repository is not None:
                session_id = self.repository.start_session(
                    f"camera://{self.camera_index}"
                )
            return self._run_capture(capture, max_frames, session_id)
        except Exception as error:
            if self.repository is not None and session_id is not None:
                self.repository.fail_session(session_id, str(error))
            raise
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _run_capture(
        self,
        capture: cv2.VideoCapture,
        max_frames: int | None,
        session_id: str | None,
    ) -> LiveCameraSummary:
        started_at = time.perf_counter()
        frames_read = 0
        frames_analyzed = 0
        track_ids: set[int] = set()
        risk_assessment_count = 0
        risk_level_counts = {"low": 0, "medium": 0, "high": 0}
        highest_risk_assessment: RiskAssessment | None = None
        stride = self.frame_skip + 1

        while max_frames is None or frames_read < max_frames:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Camera {self.camera_index} stopped returning valid frames."
                )

            frames_read += 1
            analyzing = (frames_read - 1) % stride == 0
            assessments: list[RiskAssessment] = []
            if analyzing:
                tracking_frame: TrackingFrame = self.tracker.update(frame)
                risk_frame = self.risk_assessor.assess(tracking_frame)
                assessments = risk_frame.assessments
                if self.repository is not None and session_id is not None:
                    self.repository.record_frame(
                        session_id,
                        frames_read,
                        tracking_frame.tracks,
                        tracking_frame.events,
                        assessments,
                    )
                frames_analyzed += 1
                track_ids.update(track.track_id for track in tracking_frame.tracks)
                risk_assessment_count += len(assessments)
                for assessment in assessments:
                    risk_level_counts[assessment.level] += 1
                    if (
                        highest_risk_assessment is None
                        or assessment.score > highest_risk_assessment.score
                    ):
                        highest_risk_assessment = assessment
                display_frame = draw_risk_overlay(
                    tracking_frame.annotated_frame,
                    assessments,
                )
            else:
                display_frame = frame.copy()

            elapsed = time.perf_counter() - started_at
            if elapsed <= 0.0:
                raise RuntimeError("Unable to measure camera processing time.")
            fps = frames_read / elapsed
            display_frame = draw_live_status(
                display_frame,
                fps,
                frames_read,
                analyzing,
            )
            cv2.imshow(self.window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        elapsed = time.perf_counter() - started_at
        if elapsed <= 0.0:
            raise RuntimeError("Unable to measure camera processing time.")
        summary = LiveCameraSummary(
            camera_index=self.camera_index,
            frames_read=frames_read,
            frames_analyzed=frames_analyzed,
            frames_skipped=frames_read - frames_analyzed,
            unique_tracks=len(track_ids),
            risk_assessment_count=risk_assessment_count,
            risk_level_counts=risk_level_counts,
            average_fps=frames_read / elapsed,
            highest_risk_assessment=highest_risk_assessment,
            session_id=session_id,
        )
        if self.repository is not None and session_id is not None:
            self.repository.complete_session(session_id, summary)
        return summary
