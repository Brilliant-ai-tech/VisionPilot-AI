from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL_PATH = Path("models") / "yolo11n.pt"
DEFAULT_IMAGE_SIZE = 640


def validate_image_size(image_size: int) -> None:
    if image_size < 32 or image_size % 32 != 0:
        raise ValueError("Image size must be a positive multiple of 32 (at least 32).")


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class DetectionResult:
    annotated_image: np.ndarray
    detections: list[Detection]


class ObjectDetector:
    """Run lightweight YOLO object detection on images using the CPU."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.25,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        validate_image_size(image_size)
        self.model_path = str(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self._model: YOLO | None = None

    def _get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect_image(self, image_path: str | Path) -> DetectionResult:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image file: {path}")

        predictions = self._get_model().predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )
        if len(predictions) != 1:
            raise RuntimeError(
                f"Expected one prediction for a single image, got {len(predictions)}."
            )

        prediction = predictions[0]
        detections: list[Detection] = []
        if prediction.boxes is not None:
            boxes = prediction.boxes.xyxy.cpu().tolist()
            scores = prediction.boxes.conf.cpu().tolist()
            class_ids = prediction.boxes.cls.cpu().tolist()
            for box, score, raw_class_id in zip(boxes, scores, class_ids, strict=True):
                class_id = int(raw_class_id)
                detections.append(
                    Detection(
                        box=tuple(float(coordinate) for coordinate in box),
                        confidence=float(score),
                        class_id=class_id,
                        label=str(prediction.names[class_id]),
                    )
                )

        return DetectionResult(
            annotated_image=prediction.plot(),
            detections=detections,
        )

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: str | Path,
        source_path: str | Path | None = None,
    ) -> None:
        destination = Path(output_path)
        if source_path is not None and destination.resolve() == Path(source_path).resolve():
            raise ValueError("Output path must be different from the input image path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            saved = cv2.imwrite(str(destination), image)
        except cv2.error as error:
            raise OSError(f"Could not save annotated image to {destination}: {error}") from error
        if not saved:
            raise OSError(f"Could not save annotated image to {destination}.")
