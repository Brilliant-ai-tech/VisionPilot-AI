import csv
import json
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Sequence

from src.risk import RiskAssessment
from src.tracking import TrackEvent, TrackedObject


@dataclass(frozen=True)
class AnalysisSession:
    session_id: str
    source: str
    status: str
    started_at: str
    ended_at: str | None
    summary: dict[str, object] | None
    error: str | None


class SQLiteRepository:
    """Persist analysis sessions, tracked objects, track events, and risk events."""

    def __init__(self, database_path: str | Path = Path("data") / "visionpilot.db") -> None:
        self.database_path = str(database_path)
        if self.database_path == ":memory:":
            raise ValueError("SQLiteRepository requires a persistent database file.")
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    summary_json TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES analysis_sessions(session_id),
                    frame_number INTEGER NOT NULL CHECK (frame_number > 0),
                    track_id INTEGER NOT NULL,
                    class_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    center_x REAL NOT NULL,
                    center_y REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS track_events (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES analysis_sessions(session_id),
                    frame_number INTEGER NOT NULL CHECK (frame_number > 0),
                    track_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('entered', 'exited'))
                );
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES analysis_sessions(session_id),
                    frame_number INTEGER NOT NULL CHECK (frame_number > 0),
                    track_id_a INTEGER NOT NULL,
                    track_id_b INTEGER NOT NULL,
                    label_a TEXT NOT NULL,
                    label_b TEXT NOT NULL,
                    score REAL NOT NULL CHECK (score BETWEEN 0 AND 100),
                    level TEXT NOT NULL CHECK (level IN ('low', 'medium', 'high')),
                    current_distance_pixels REAL NOT NULL,
                    predicted_distance_pixels REAL,
                    frames_to_closest_approach REAL,
                    explanation TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_detections_session_frame
                    ON detections(session_id, frame_number);
                CREATE INDEX IF NOT EXISTS idx_track_events_session_frame
                    ON track_events(session_id, frame_number);
                CREATE INDEX IF NOT EXISTS idx_risk_events_session_frame
                    ON risk_events(session_id, frame_number);
                """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_session(self, source: str) -> str:
        if not source.strip():
            raise ValueError("Analysis source cannot be empty.")
        session_id = str(uuid.uuid4())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO analysis_sessions (session_id, source, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (session_id, source, self._now()),
            )
        return session_id

    def record_frame(
        self,
        session_id: str,
        frame_number: int,
        tracks: Sequence[TrackedObject],
        events: Sequence[TrackEvent],
        assessments: Sequence[RiskAssessment],
    ) -> None:
        if frame_number <= 0:
            raise ValueError("Frame number must be positive.")
        with self._connection() as connection:
            self._require_running_session(connection, session_id)
            connection.executemany(
                """
                INSERT INTO detections (
                    session_id, frame_number, track_id, class_id, label, confidence,
                    x1, y1, x2, y2, center_x, center_y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        frame_number,
                        track.track_id,
                        track.class_id,
                        track.label,
                        track.confidence,
                        *track.box,
                        *track.center,
                    )
                    for track in tracks
                ],
            )
            connection.executemany(
                """
                INSERT INTO track_events (
                    session_id, frame_number, track_id, label, kind
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        frame_number,
                        event.track_id,
                        event.label,
                        event.kind,
                    )
                    for event in events
                ],
            )
            connection.executemany(
                """
                INSERT INTO risk_events (
                    session_id, frame_number, track_id_a, track_id_b, label_a, label_b,
                    score, level, current_distance_pixels, predicted_distance_pixels,
                    frames_to_closest_approach, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        frame_number,
                        *assessment.track_ids,
                        *assessment.labels,
                        assessment.score,
                        assessment.level,
                        assessment.current_distance_pixels,
                        assessment.predicted_distance_pixels,
                        assessment.frames_to_closest_approach,
                        assessment.explanation,
                    )
                    for assessment in assessments
                ],
            )

    def complete_session(self, session_id: str, summary: object) -> None:
        if is_dataclass(summary) and not isinstance(summary, type):
            summary_data = asdict(summary)
        elif isinstance(summary, Mapping):
            summary_data = dict(summary)
        else:
            raise TypeError("Analysis summary must be a dataclass or mapping.")
        self._finish_session(session_id, "completed", json.dumps(summary_data))

    def fail_session(self, session_id: str, error: str) -> None:
        self._finish_session(session_id, "failed", error=error)

    def _finish_session(
        self,
        session_id: str,
        status: str,
        summary_json: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_sessions
                SET status = ?, ended_at = ?, summary_json = ?, error = ?
                WHERE session_id = ? AND status = 'running'
                """,
                (status, self._now(), summary_json, error, session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Analysis session is not running: {session_id}")

    @staticmethod
    def _require_running_session(
        connection: sqlite3.Connection,
        session_id: str,
    ) -> None:
        row = connection.execute(
            "SELECT status FROM analysis_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["status"] != "running":
            raise ValueError(f"Analysis session is not running: {session_id}")

    def get_session(self, session_id: str) -> AnalysisSession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Analysis session does not exist: {session_id}")
        return AnalysisSession(
            session_id=row["session_id"],
            source=row["source"],
            status=row["status"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            summary=(
                json.loads(row["summary_json"])
                if row["summary_json"] is not None
                else None
            ),
            error=row["error"],
        )

    def list_sessions(self, limit: int = 100) -> list[AnalysisSession]:
        if limit <= 0:
            raise ValueError("Session list limit must be positive.")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_sessions
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            AnalysisSession(
                session_id=row["session_id"],
                source=row["source"],
                status=row["status"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                summary=(
                    json.loads(row["summary_json"])
                    if row["summary_json"] is not None
                    else None
                ),
                error=row["error"],
            )
            for row in rows
        ]

    def get_session_risk_events(self, session_id: str) -> list[dict[str, object]]:
        self.get_session(session_id)
        with self._connection() as connection:
            return self._rows(
                connection,
                """
                SELECT frame_number, track_id_a, track_id_b, label_a, label_b,
                       score, level, current_distance_pixels,
                       predicted_distance_pixels, frames_to_closest_approach,
                       explanation
                FROM risk_events
                WHERE session_id = ?
                ORDER BY frame_number, id
                """,
                session_id,
            )

    def get_analytics(self, session_id: str) -> dict[str, object]:
        session = self.get_session(session_id)
        with self._connection() as connection:
            detection_counts = connection.execute(
                """
                SELECT label, COUNT(*) AS observations, COUNT(DISTINCT track_id) AS tracks
                FROM detections
                WHERE session_id = ?
                GROUP BY label
                ORDER BY label
                """,
                (session_id,),
            ).fetchall()
            track_event_counts = connection.execute(
                """
                SELECT kind, COUNT(*) AS count
                FROM track_events
                WHERE session_id = ?
                GROUP BY kind
                """,
                (session_id,),
            ).fetchall()
            risk_level_counts = connection.execute(
                """
                SELECT level, COUNT(*) AS count
                FROM risk_events
                WHERE session_id = ?
                GROUP BY level
                """,
                (session_id,),
            ).fetchall()
            risk_summary = connection.execute(
                """
                SELECT COUNT(*) AS count, MAX(score) AS highest_score
                FROM risk_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            detection_summary = connection.execute(
                """
                SELECT COUNT(*) AS observations,
                       COUNT(DISTINCT track_id) AS unique_tracks,
                       COUNT(DISTINCT frame_number) AS frames
                FROM detections
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        detections_by_class = {
            row["label"]: {
                "observations": row["observations"],
                "unique_tracks": row["tracks"],
            }
            for row in detection_counts
        }
        track_events_by_kind = {
            row["kind"]: row["count"] for row in track_event_counts
        }
        risk_events_by_level = {
            row["level"]: row["count"] for row in risk_level_counts
        }
        summary = session.summary or {}
        analyzed_frames = summary.get("frames_analyzed")
        return {
            "session_id": session_id,
            "status": session.status,
            "frames_analyzed": (
                analyzed_frames
                if isinstance(analyzed_frames, int)
                else detection_summary["frames"]
            ),
            "detections": {
                "observations": detection_summary["observations"],
                "unique_tracks": detection_summary["unique_tracks"],
                "by_class": detections_by_class,
            },
            "track_events": {
                "total": sum(track_events_by_kind.values()),
                "by_kind": track_events_by_kind,
            },
            "risk_events": {
                "total": risk_summary["count"],
                "by_level": risk_events_by_level,
                "highest_score": risk_summary["highest_score"],
            },
        }

    def export_session_json(
        self,
        session_id: str,
        output_path: str | Path,
    ) -> Path:
        session = self.get_session(session_id)
        with self._connection() as connection:
            detections = self._rows(
                connection,
                "SELECT * FROM detections WHERE session_id = ? ORDER BY frame_number, id",
                session_id,
            )
            track_events = self._rows(
                connection,
                "SELECT * FROM track_events WHERE session_id = ? ORDER BY frame_number, id",
                session_id,
            )
            risk_events = self._rows(
                connection,
                "SELECT * FROM risk_events WHERE session_id = ? ORDER BY frame_number, id",
                session_id,
            )

        payload = {
            "session": asdict(session),
            "analytics": self.get_analytics(session_id),
            "detections": detections,
            "track_events": track_events,
            "risk_events": risk_events,
        }
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.write("\n")
        return destination

    def export_session_csv(
        self,
        session_id: str,
        output_path: str | Path,
    ) -> Path:
        session = self.get_session(session_id)
        with self._connection() as connection:
            detections = self._rows(
                connection,
                "SELECT * FROM detections WHERE session_id = ? ORDER BY frame_number, id",
                session_id,
            )
            track_events = self._rows(
                connection,
                "SELECT * FROM track_events WHERE session_id = ? ORDER BY frame_number, id",
                session_id,
            )
            risk_events = self._rows(
                connection,
                "SELECT * FROM risk_events WHERE session_id = ? ORDER BY frame_number, id",
                session_id,
            )

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "session_id",
            "source",
            "event_type",
            "frame_number",
            "track_id",
            "other_track_id",
            "label",
            "other_label",
            "class_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "event",
            "score",
            "level",
            "current_distance_pixels",
            "predicted_distance_pixels",
            "frames_to_closest_approach",
            "explanation",
        ]
        rows: list[dict[str, object]] = []
        rows.extend(
            {
                "session_id": session_id,
                "source": session.source,
                "event_type": "detection",
                **{key: value for key, value in row.items() if key != "id"},
            }
            for row in detections
        )
        rows.extend(
            {
                "session_id": session_id,
                "source": session.source,
                "event_type": "track_event",
                "track_id": row["track_id"],
                "label": row["label"],
                "event": row["kind"],
                "frame_number": row["frame_number"],
            }
            for row in track_events
        )
        rows.extend(
            {
                "session_id": session_id,
                "source": session.source,
                "event_type": "risk_event",
                "frame_number": row["frame_number"],
                "track_id": row["track_id_a"],
                "other_track_id": row["track_id_b"],
                "label": row["label_a"],
                "other_label": row["label_b"],
                "score": row["score"],
                "level": row["level"],
                "current_distance_pixels": row["current_distance_pixels"],
                "predicted_distance_pixels": row["predicted_distance_pixels"],
                "frames_to_closest_approach": row["frames_to_closest_approach"],
                "explanation": row["explanation"],
            }
            for row in risk_events
        )
        rows.sort(key=lambda row: (int(row["frame_number"]), str(row["event_type"])))
        with destination.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return destination

    @staticmethod
    def _rows(
        connection: sqlite3.Connection,
        query: str,
        session_id: str,
    ) -> list[dict[str, object]]:
        return [dict(row) for row in connection.execute(query, (session_id,))]
