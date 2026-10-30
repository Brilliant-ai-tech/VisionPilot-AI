import numpy as np
import pytest

from src.risk import RiskAssessor
from src.tracking import TrackEvent, TrackedObject, TrackingFrame


def tracked_object(
    track_id: int,
    label: str,
    center_x: float,
    center_y: float = 500.0,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        box=(center_x - 5, center_y - 5, center_x + 5, center_y + 5),
        confidence=0.9,
        class_id=0,
        label=label,
        center=(center_x, center_y),
    )


def tracking_frame(
    frame_number: int,
    *tracks: TrackedObject,
    events: list[TrackEvent] | None = None,
) -> TrackingFrame:
    return TrackingFrame(
        frame_number=frame_number,
        tracks=list(tracks),
        events=events or [],
        annotated_frame=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )


def test_reports_current_proximity_with_explanation() -> None:
    assessor = RiskAssessor()

    result = assessor.assess(
        tracking_frame(
            1,
            tracked_object(1, "person", 500),
            tracked_object(2, "car", 550),
        )
    )

    assert len(result.assessments) == 1
    assessment = result.assessments[0]
    assert assessment.track_ids == (1, 2)
    assert assessment.level == "high"
    assert assessment.predicted_distance_pixels is None
    assert "currently close" in assessment.explanation


def test_predicts_approaching_objects_and_explains_timing() -> None:
    assessor = RiskAssessor()
    assessor.assess(
        tracking_frame(
            1,
            tracked_object(1, "person", 300),
            tracked_object(2, "car", 500),
        )
    )

    result = assessor.assess(
        tracking_frame(
            2,
            tracked_object(1, "person", 340),
            tracked_object(2, "car", 460),
        )
    )

    assert len(result.assessments) == 1
    assessment = result.assessments[0]
    assert assessment.predicted_distance_pixels == 0.0
    assert assessment.frames_to_closest_approach == 1.5
    assert assessment.level == "high"
    assert "Relative motion predicts" in assessment.explanation


def test_does_not_report_separating_objects_outside_proximity() -> None:
    assessor = RiskAssessor()
    assessor.assess(
        tracking_frame(
            1,
            tracked_object(1, "person", 300),
            tracked_object(2, "car", 500),
        )
    )

    result = assessor.assess(
        tracking_frame(
            2,
            tracked_object(1, "person", 260),
            tracked_object(2, "car", 540),
        )
    )

    assert result.assessments == []


def test_ignores_non_road_user_classes() -> None:
    result = RiskAssessor().assess(
        tracking_frame(
            1,
            tracked_object(1, "traffic light", 500),
            tracked_object(2, "car", 510),
        )
    )

    assert result.assessments == []


def test_exit_event_clears_previous_motion_state() -> None:
    assessor = RiskAssessor()
    assessor.assess(tracking_frame(1, tracked_object(1, "person", 300)))
    assessor.assess(
        tracking_frame(
            2,
            tracked_object(1, "person", 340),
            events=[TrackEvent("exited", 1, "person", 2)],
        )
    )

    result = assessor.assess(
        tracking_frame(
            3,
            tracked_object(1, "person", 100),
            tracked_object(2, "car", 300),
        )
    )

    assert result.assessments == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"proximity_distance_fraction": 0.0},
        {"conflict_distance_fraction": 1.1},
        {"prediction_horizon_frames": 0},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        RiskAssessor(**kwargs)
