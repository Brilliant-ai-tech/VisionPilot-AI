import argparse
import logging
import sqlite3
from pathlib import Path

from src.detection import DEFAULT_IMAGE_SIZE
from src.storage import SQLiteRepository
from src.video_processing import VideoProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track road users, assess risks, and annotate a video."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--output",
        type=Path,
        help="Annotated output video path (default: reports/videos/<name>_annotated.mp4)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="JSON summary path (default: reports/videos/<name>_summary.json)",
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
    output_path = args.output or (
        Path("reports")
        / "videos"
        / f"{args.video.stem}_annotated.mp4"
    )
    summary_path = args.summary or (
        Path("reports") / "videos" / f"{args.video.stem}_summary.json"
    )

    try:
        processor = VideoProcessor(
            confidence=args.confidence,
            frame_skip=args.frame_skip,
            image_size=args.image_size,
            repository=SQLiteRepository(args.database),
        )
        summary = processor.process(args.video, output_path)
        processor.save_summary(summary, summary_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        logging.error("Video processing failed: %s", error)
        return 1

    logging.info(
        "Processed %d frame(s), analyzed %d, skipped %d.",
        summary.frames_read,
        summary.frames_analyzed,
        summary.frames_skipped,
    )
    logging.info(
        "Found %d unique track(s) and %d risk assessment(s).",
        summary.unique_tracks,
        summary.risk_assessment_count,
    )
    logging.info("Annotated video saved to %s", output_path)
    logging.info("Video summary saved to %s", summary_path)
    if summary.session_id is not None:
        logging.info("Analysis session ID: %s", summary.session_id)
    logging.info("Average processing speed: %.2f FPS", summary.processing_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
