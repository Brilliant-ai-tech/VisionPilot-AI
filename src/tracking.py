from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from ultralytics import YOLO

from src.detection import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH, validate_image_size


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str
    center: tuple[float, float]


@dataclass(frozen=True)
class TrackEvent:
    kind: Literal["entered", "exited"]
    track_id: int
    label: str
    frame_number: int


@dataclass(frozen=True)
class TrackingFrame:
    frame_number: int
    tracks: list[TrackedObject]
    events: list[TrackEvent]
    annotated_frame: np.ndarray


@dataclass
class _ActiveTrack:
    label: str
    last_seen_frame: int


class ObjectTracker:
    """Track objects in sequential video frames with CPU-based ByteTrack."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.25,
        max_missing_frames: int = 30,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if max_missing_frames < 0:
            raise ValueError("Maximum missing frames cannot be negative.")
        validate_image_size(image_size)

        self.model_path = str(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.max_missing_frames = max_missing_frames
        self._model: YOLO | None = None
        self._frame_number = 0
        self._active_tracks: dict[int, _ActiveTrack] = {}
        self._trajectories: dict[int, list[tuple[float, float]]] = {}

    def _get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def update(self, frame: np.ndarray) -> TrackingFrame:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            raise ValueError("Frame must be a non-empty 8-bit BGR image.")

        self._frame_number += 1
        predictions = self._get_model().track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )
        if len(predictions) != 1:
            raise RuntimeError(
                f"Expected one prediction for a single frame, got {len(predictions)}."
            )

        prediction = predictions[0]
        tracks: list[TrackedObject] = []
        events: list[TrackEvent] = []
        seen_ids: set[int] = set()
        boxes = prediction.boxes
        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            track_ids = boxes.id.cpu().tolist()
            for box, score, raw_class_id, raw_track_id in zip(
                coordinates,
                scores,
                class_ids,
                track_ids,
                strict=True,
            ):
                class_id = int(raw_class_id)
                track_id = int(raw_track_id)
                label = str(prediction.names[class_id])
                x1, y1, x2, y2 = (float(coordinate) for coordinate in box)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                if track_id not in self._active_tracks:
                    events.append(
                        TrackEvent(
                            kind="entered",
                            track_id=track_id,
                            label=label,
                            frame_number=self._frame_number,
                        )
                    )
                self._active_tracks[track_id] = _ActiveTrack(
                    label=label,
                    last_seen_frame=self._frame_number,
                )
                self._trajectories.setdefault(track_id, []).append(center)
                seen_ids.add(track_id)
                tracks.append(
                    TrackedObject(
                        track_id=track_id,
                        box=(x1, y1, x2, y2),
                        confidence=float(score),
                        class_id=class_id,
                        label=label,
                        center=center,
                    )
                )

        for track_id, active_track in list(self._active_tracks.items()):
            if track_id in seen_ids:
                continue
            missing_frames = self._frame_number - active_track.last_seen_frame
            if missing_frames > self.max_missing_frames:
                events.append(
                    TrackEvent(
                        kind="exited",
                        track_id=track_id,
                        label=active_track.label,
                        frame_number=self._frame_number,
                    )
                )
                del self._active_tracks[track_id]

        return TrackingFrame(
            frame_number=self._frame_number,
            tracks=tracks,
            events=events,
            annotated_frame=prediction.plot(),
        )

    def get_trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """Return a copy of the center-point history for each observed track."""
        return {track_id: list(points) for track_id, points in self._trajectories.items()}
