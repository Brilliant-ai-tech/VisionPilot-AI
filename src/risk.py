import math
from dataclasses import dataclass
from typing import Literal

from src.tracking import TrackedObject, TrackingFrame

_ROAD_USER_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class RiskAssessment:
    track_ids: tuple[int, int]
    labels: tuple[str, str]
    score: float
    level: Literal["low", "medium", "high"]
    current_distance_pixels: float
    predicted_distance_pixels: float | None
    frames_to_closest_approach: float | None
    explanation: str


@dataclass(frozen=True)
class RiskFrame:
    frame_number: int
    assessments: list[RiskAssessment]


class RiskAssessor:
    """Estimate image-space proximity and motion conflicts between road users."""

    def __init__(
        self,
        proximity_distance_fraction: float = 0.08,
        conflict_distance_fraction: float = 0.05,
        prediction_horizon_frames: int = 15,
    ) -> None:
        if not 0.0 < proximity_distance_fraction <= 1.0:
            raise ValueError("Proximity distance fraction must be in (0, 1].")
        if not 0.0 < conflict_distance_fraction <= 1.0:
            raise ValueError("Conflict distance fraction must be in (0, 1].")
        if prediction_horizon_frames <= 0:
            raise ValueError("Prediction horizon must be a positive number of frames.")

        self.proximity_distance_fraction = proximity_distance_fraction
        self.conflict_distance_fraction = conflict_distance_fraction
        self.prediction_horizon_frames = prediction_horizon_frames
        self._previous_positions: dict[int, tuple[int, tuple[float, float]]] = {}

    def assess(self, frame: TrackingFrame) -> RiskFrame:
        height, width = frame.annotated_frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Tracking frame must have non-zero image dimensions.")
        image_diagonal = math.hypot(width, height)
        tracks = [
            track
            for track in frame.tracks
            if track.label in _ROAD_USER_CLASSES
        ]
        velocities = {
            track.track_id: self._velocity(track, frame.frame_number)
            for track in tracks
        }

        assessments: list[RiskAssessment] = []
        for index, first in enumerate(tracks):
            for second in tracks[index + 1 :]:
                assessment = self._assess_pair(
                    first,
                    second,
                    velocities[first.track_id],
                    velocities[second.track_id],
                    image_diagonal,
                )
                if assessment is not None:
                    assessments.append(assessment)

        for track in tracks:
            self._previous_positions[track.track_id] = (
                frame.frame_number,
                track.center,
            )
        for event in frame.events:
            if event.kind == "exited":
                self._previous_positions.pop(event.track_id, None)

        return RiskFrame(frame_number=frame.frame_number, assessments=assessments)

    def _velocity(
        self,
        track: TrackedObject,
        frame_number: int,
    ) -> tuple[float, float] | None:
        previous = self._previous_positions.get(track.track_id)
        if previous is None:
            return None
        previous_frame, previous_center = previous
        elapsed_frames = frame_number - previous_frame
        if elapsed_frames <= 0:
            return None
        return (
            (track.center[0] - previous_center[0]) / elapsed_frames,
            (track.center[1] - previous_center[1]) / elapsed_frames,
        )

    def _assess_pair(
        self,
        first: TrackedObject,
        second: TrackedObject,
        first_velocity: tuple[float, float] | None,
        second_velocity: tuple[float, float] | None,
        image_diagonal: float,
    ) -> RiskAssessment | None:
        separation = (
            second.center[0] - first.center[0],
            second.center[1] - first.center[1],
        )
        current_distance = math.hypot(*separation)
        current_fraction = current_distance / image_diagonal
        closest_approach: tuple[float, float] | None = None

        if current_fraction > self.proximity_distance_fraction:
            closest_approach = self._closest_approach(
                separation,
                first_velocity,
                second_velocity,
            )
            if (
                closest_approach is None
                or closest_approach[0] / image_diagonal
                > self.conflict_distance_fraction
            ):
                return None

        pedestrian_vehicle_pair = (
            (first.label == "person" and second.label in _VEHICLE_CLASSES)
            or (second.label == "person" and first.label in _VEHICLE_CLASSES)
        )
        class_weight = 10.0 if pedestrian_vehicle_pair else 0.0
        if closest_approach is None:
            distance_factor = 1.0 - (
                current_fraction / self.proximity_distance_fraction
            )
            score = min(100.0, 40.0 + 60.0 * distance_factor + class_weight)
            explanation = (
                "Objects are currently close in the image, "
                f"about {current_fraction:.1%} of the frame diagonal apart."
            )
            predicted_distance = None
            frames_to_closest_approach = None
        else:
            closest_distance, frames_to_closest_approach = closest_approach
            closest_fraction = closest_distance / image_diagonal
            distance_factor = 1.0 - (
                closest_fraction / self.conflict_distance_fraction
            )
            urgency = 1.0 - (
                frames_to_closest_approach / self.prediction_horizon_frames
            )
            score = min(
                100.0,
                30.0 + 45.0 * distance_factor + 25.0 * urgency + class_weight,
            )
            explanation = (
                "Relative motion predicts a closest image-space separation of "
                f"{closest_fraction:.1%} of the frame diagonal in "
                f"{frames_to_closest_approach:.1f} frame(s)."
            )
            predicted_distance = closest_distance

        level: Literal["low", "medium", "high"]
        if score >= 75.0:
            level = "high"
        elif score >= 50.0:
            level = "medium"
        else:
            level = "low"

        return RiskAssessment(
            track_ids=(first.track_id, second.track_id),
            labels=(first.label, second.label),
            score=round(score, 1),
            level=level,
            current_distance_pixels=round(current_distance, 1),
            predicted_distance_pixels=(
                round(predicted_distance, 1) if predicted_distance is not None else None
            ),
            frames_to_closest_approach=(
                round(frames_to_closest_approach, 1)
                if frames_to_closest_approach is not None
                else None
            ),
            explanation=explanation,
        )

    def _closest_approach(
        self,
        separation: tuple[float, float],
        first_velocity: tuple[float, float] | None,
        second_velocity: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if first_velocity is None and second_velocity is None:
            return None

        first_dx, first_dy = first_velocity or (0.0, 0.0)
        second_dx, second_dy = second_velocity or (0.0, 0.0)
        relative_velocity = (second_dx - first_dx, second_dy - first_dy)
        speed_squared = (
            relative_velocity[0] ** 2 + relative_velocity[1] ** 2
        )
        if speed_squared == 0.0:
            return None

        frames = -(
            separation[0] * relative_velocity[0]
            + separation[1] * relative_velocity[1]
        ) / speed_squared
        if not 0.0 < frames <= self.prediction_horizon_frames:
            return None

        closest_separation = (
            separation[0] + relative_velocity[0] * frames,
            separation[1] + relative_velocity[1] * frames,
        )
        return math.hypot(*closest_separation), frames
