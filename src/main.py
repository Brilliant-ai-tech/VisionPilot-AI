import argparse
import logging
from pathlib import Path

from src.detection import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH, ObjectDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect objects in an image using a CPU-friendly YOLO model."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the annotated image (default: reports/detections/<name>_annotated.jpg)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"YOLO model name or path to model weights (default: {DEFAULT_MODEL_PATH})",
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
        Path("reports") / "detections" / f"{args.image.stem}_annotated.jpg"
    )

    try:
        detector = ObjectDetector(
            model_path=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
        )
        result = detector.detect_image(args.image)
        detector.save_image(result.annotated_image, output_path, source_path=args.image)
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("Detection failed: %s", error)
        return 1

    logging.info("Detected %d object(s).", len(result.detections))
    for detection in result.detections:
        logging.info(
            "%s (confidence %.2f) at %s",
            detection.label,
            detection.confidence,
            tuple(round(coordinate) for coordinate in detection.box),
        )
    logging.info("Annotated image saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
