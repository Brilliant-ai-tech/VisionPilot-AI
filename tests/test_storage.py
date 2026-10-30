import csv
import json
import sqlite3
from pathlib import Path

import pytest

from src.risk import RiskAssessment
from src.storage import SQLiteRepository
from src.tracking import TrackEvent, TrackedObject


def make_repository(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(tmp_path / "data" / "visionpilot.sqlite")


def sample_track() -> TrackedObject:
    return TrackedObject(
        track_id=12,
        box=(10.0, 20.0, 50.0, 80.0),
        confidence=0.93,
        class_id=0,
        label="person",
        center=(30.0, 50.0),
    )


def sample_risk() -> RiskAssessment:
    return RiskAssessment(
        track_ids=(12, 24),
        labels=("person", "car"),
        score=82.5,
        level="high",
        current_distance_pixels=42.0,
        predicted_distance_pixels=3.0,
        frames_to_closest_approach=4.0,
        explanation="Objects may converge.",
    )


def test_persists_session_detections_events_and_risks_and_exports(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    session_id = repository.start_session("test://sample-video")
    repository.record_frame(
        session_id,
        1,
        [sample_track()],
        [TrackEvent("entered", 12, "person", 1)],
        [sample_risk()],
    )
    repository.complete_session(session_id, {"frames_read": 1})

    session = repository.get_session(session_id)
    assert session.status == "completed"
    assert session.summary == {"frames_read": 1}
    assert session.error is None
    analytics = repository.get_analytics(session_id)
    assert analytics["detections"]["observations"] == 1
    assert analytics["detections"]["unique_tracks"] == 1
    assert analytics["track_events"]["by_kind"] == {"entered": 1}
    assert analytics["risk_events"]["by_level"] == {"high": 1}

    json_path = repository.export_session_json(session_id, tmp_path / "export.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["session"]["session_id"] == session_id
    assert payload["analytics"]["risk_events"]["highest_score"] == 82.5
    assert payload["detections"][0]["track_id"] == 12
    assert payload["track_events"][0]["kind"] == "entered"
    assert payload["risk_events"][0]["level"] == "high"

    csv_path = repository.export_session_csv(session_id, tmp_path / "export.csv")
    with csv_path.open(encoding="utf-8", newline="") as export_file:
        rows = list(csv.DictReader(export_file))
    assert {row["event_type"] for row in rows} == {
        "detection",
        "track_event",
        "risk_event",
    }
    risk_row = next(row for row in rows if row["event_type"] == "risk_event")
    assert risk_row["level"] == "high"
    assert risk_row["other_track_id"] == "24"


def test_failed_session_retains_error(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    session_id = repository.start_session("video.mp4")
    repository.fail_session(session_id, "decoder failure")

    session = repository.get_session(session_id)
    assert session.status == "failed"
    assert session.error == "decoder failure"
    assert session.ended_at is not None


def test_frame_writes_are_atomic_for_unknown_session(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    with pytest.raises(ValueError, match="not running"):
        repository.record_frame(
            "unknown-session",
            1,
            [sample_track()],
            [],
            [],
        )

    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0


def test_cannot_write_after_session_completion(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    session_id = repository.start_session("video.mp4")
    repository.complete_session(session_id, {"frames_read": 0})

    with pytest.raises(ValueError, match="not running"):
        repository.record_frame(session_id, 1, [sample_track()], [], [])


def test_rejects_empty_source_and_memory_database(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    with pytest.raises(ValueError, match="source cannot be empty"):
        repository.start_session("  ")
    with pytest.raises(ValueError, match="persistent database file"):
        SQLiteRepository(":memory:")


def test_rejects_missing_session_export(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        repository.export_session_json("unknown", tmp_path / "out.json")
