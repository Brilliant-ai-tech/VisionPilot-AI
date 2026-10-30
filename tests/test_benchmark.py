from pathlib import Path

import pytest

from src.benchmark import benchmark_image


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect_image(self, image_path: str | Path) -> object:
        assert Path(image_path).is_file()
        self.calls += 1
        return object()


def test_benchmark_runs_warmups_and_reports_measured_runs(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"test image")
    detector = FakeDetector()

    result = benchmark_image(
        image_path,
        runs=3,
        warmup_runs=2,
        image_size=320,
        detector=detector,
    )

    assert detector.calls == 5
    assert result.measured_runs == 3
    assert result.warmup_runs == 2
    assert result.image_size == 320
    assert result.mean_latency_ms >= 0
    assert result.p95_latency_ms >= 0
    assert result.mean_fps > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"runs": 0},
        {"warmup_runs": -1},
        {"confidence": 1.1},
        {"image_size": 641},
    ],
)
def test_rejects_invalid_benchmark_options(
    tmp_path: Path,
    kwargs: dict[str, int | float],
) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"test image")

    with pytest.raises(ValueError):
        benchmark_image(image_path, detector=FakeDetector(), **kwargs)


def test_missing_benchmark_image_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        benchmark_image(tmp_path / "missing.jpg")
