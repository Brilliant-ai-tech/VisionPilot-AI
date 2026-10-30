import argparse
import logging
import sqlite3
from pathlib import Path

from src.storage import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one VisionPilot analysis session to CSV or JSON."
    )
    parser.add_argument("session_id", help="Analysis session ID to export")
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        required=True,
        help="Export file format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination file path",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data") / "visionpilot.db",
        help="SQLite database path (default: data/visionpilot.db)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        repository = SQLiteRepository(args.database)
        if args.format == "csv":
            output_path = repository.export_session_csv(
                args.session_id,
                args.output,
            )
        else:
            output_path = repository.export_session_json(
                args.session_id,
                args.output,
            )
    except (OSError, ValueError, sqlite3.Error) as error:
        logging.error("Export failed: %s", error)
        return 1
    logging.info("Export saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
