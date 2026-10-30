import argparse
import logging
import sqlite3
from pathlib import Path

from src.detection import DEFAULT_IMAGE_SIZE
from src.live_camera import LiveCamera
from src.storage import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live object tracking and risk assessment from a webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to open (default: 0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Skip this many frames between analyzed frames (default: 0)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data") / "visionpilot.db",
        help="SQLite database path (default: data/visionpilot.db)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Inference size, a multiple of 32 (default: {DEFAULT_IMAGE_SIZE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        summary = LiveCamera(
            camera_index=args.camera_index,
            confidence=args.confidence,
            frame_skip=args.frame_skip,
            image_size=args.image_size,
            repository=SQLiteRepository(args.database),
        ).run()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        logging.error("Live camera failed: %s", error)
        return 1

    logging.info(
        "Read %d frame(s), analyzed %d, skipped %d.",
        summary.frames_read,
        summary.frames_analyzed,
        summary.frames_skipped,
    )
    logging.info(
        "Tracked %d unique object(s); generated %d risk assessment(s).",
        summary.unique_tracks,
        summary.risk_assessment_count,
    )
    logging.info("Average live processing speed: %.2f FPS", summary.average_fps)
    if summary.session_id is not None:
        logging.info("Analysis session ID: %s", summary.session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
