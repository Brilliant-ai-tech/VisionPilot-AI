import argparse
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import psutil

from src.detection import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    ObjectDetector,
    validate_image_size,
)


class ImageInference(Protocol):
    def detect_image(self, image_path: str | Path) -> object: ...


@dataclass(frozen=True)
class ImageBenchmarkResult:
    image_path: str
    image_size: int
    confidence: float
    warmup_runs: int
    measured_runs: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    mean_fps: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float


def benchmark_image(
    image_path: str | Path,
    runs: int = 5,
    warmup_runs: int = 1,
    confidence: float = 0.25,
    image_size: int = DEFAULT_IMAGE_SIZE,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    detector: ImageInference | None = None,
) -> ImageBenchmarkResult:
    if runs <= 0:
        raise ValueError("Measured benchmark runs must be positive.")
    if warmup_runs < 0:
        raise ValueError("Warmup runs cannot be negative.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    validate_image_size(image_size)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    inference = (
        detector
        if detector is not None
        else ObjectDetector(
            model_path=model_path,
            confidence=confidence,
            image_size=image_size,
        )
    )
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024**2
    for _ in range(warmup_runs):
        inference.detect_image(image_path)

    latencies_ms: list[float] = []
    for _ in range(runs):
        started_at = time.perf_counter()
        inference.detect_image(image_path)
        latencies_ms.append((time.perf_counter() - started_at) * 1000)

    memory_after = process.memory_info().rss / 1024**2
    mean_latency = statistics.fmean(latencies_ms)
    return ImageBenchmarkResult(
        image_path=str(Path(image_path).resolve()),
        image_size=image_size,
        confidence=confidence,
        warmup_runs=warmup_runs,
        measured_runs=runs,
        mean_latency_ms=mean_latency,
        median_latency_ms=statistics.median(latencies_ms),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        mean_fps=1000 / mean_latency,
        memory_before_mb=memory_before,
        memory_after_mb=memory_after,
        memory_delta_mb=memory_after - memory_before,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure CPU image-detection latency, FPS, and process RSS."
    )
    parser.add_argument("image", type=Path, help="Path to the benchmark image")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of measured inference runs (default: 5)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs excluded from latency measurements (default: 1)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Inference size, a multiple of 32 (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"YOLO model name or path (default: {DEFAULT_MODEL_PATH})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        result = benchmark_image(
            image_path=args.image,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            confidence=args.confidence,
            image_size=args.image_size,
            model_path=args.model,
        )
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("Benchmark failed: %s", error)
        return 1

    logging.info("Image: %s", result.image_path)
    logging.info(
        "Image size: %d; runs: %d (+ %d warmup).",
        result.image_size,
        result.measured_runs,
        result.warmup_runs,
    )
    logging.info(
        "Latency: mean %.1f ms, median %.1f ms, p95 %.1f ms.",
        result.mean_latency_ms,
        result.median_latency_ms,
        result.p95_latency_ms,
    )
    logging.info("Mean end-to-end image FPS: %.2f.", result.mean_fps)
    logging.info(
        "Process RSS: %.1f MiB before, %.1f MiB after (delta %.1f MiB).",
        result.memory_before_mb,
        result.memory_after_mb,
        result.memory_delta_mb,
    )
    logging.info(
        "FPS includes image decoding and annotated-image rendering; "
        "RSS is sampled before and after the benchmark, not peak memory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
