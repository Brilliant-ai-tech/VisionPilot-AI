from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.detection import Detection, ObjectDetector


def test_rejects_confidence_outside_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ObjectDetector(confidence=1.1)


@pytest.mark.parametrize("image_size", [31, 641])
def test_rejects_invalid_image_size(image_size: int) -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        ObjectDetector(image_size=image_size)


def test_missing_image_fails_before_loading_model(tmp_path: Path) -> None:
    with patch("src.detection.YOLO") as model:
        detector = ObjectDetector()

        with pytest.raises(FileNotFoundError, match="does not exist"):
            detector.detect_image(tmp_path / "missing.jpg")

        model.assert_not_called()


def test_invalid_image_fails_before_loading_model(tmp_path: Path) -> None:
    invalid_image = tmp_path / "invalid.jpg"
    invalid_image.write_text("not an image", encoding="utf-8")

    with patch("src.detection.YOLO") as model:
        detector = ObjectDetector()

        with pytest.raises(ValueError, match="Could not decode"):
            detector.detect_image(invalid_image)

        model.assert_not_called()


def test_detects_and_saves_image_with_cpu_model(tmp_path: Path) -> None:
    class FakeTensor:
        def __init__(self, values: list[list[float]] | list[float]) -> None:
            self.values = values

        def cpu(self) -> "FakeTensor":
            return self

        def tolist(self) -> list:
            return self.values

    class FakeBoxes:
        xyxy = FakeTensor([[1.0, 2.0, 8.0, 9.0]])
        conf = FakeTensor([0.9])
        cls = FakeTensor([0.0])

    class FakePrediction:
        boxes = FakeBoxes()
        names = {0: "person"}

        def plot(self) -> np.ndarray:
            return np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeModel:
        def predict(self, **kwargs: object) -> list[FakePrediction]:
            assert kwargs["device"] == "cpu"
            assert kwargs["conf"] == 0.25
            assert kwargs["imgsz"] == 320
            return [FakePrediction()]

    input_path = tmp_path / "input.png"
    assert cv2.imwrite(str(input_path), np.zeros((10, 10, 3), dtype=np.uint8))

    with patch("src.detection.YOLO", return_value=FakeModel()):
        detector = ObjectDetector(image_size=320)
        result = detector.detect_image(input_path)

    assert result.detections == [
        Detection(box=(1.0, 2.0, 8.0, 9.0), confidence=0.9, class_id=0, label="person")
    ]

    output_path = tmp_path / "output" / "annotated.png"
    detector.save_image(result.annotated_image, output_path, source_path=input_path)
    assert output_path.is_file()
    assert cv2.imread(str(output_path)) is not None


def test_does_not_overwrite_input_image(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    with pytest.raises(ValueError, match="different from the input"):
        ObjectDetector.save_image(image, image_path, source_path=image_path)
