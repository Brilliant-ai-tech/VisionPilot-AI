import cv2
import numpy as np

from src.risk import RiskAssessment


def draw_risk_overlay(
    frame: np.ndarray,
    assessments: list[RiskAssessment],
) -> np.ndarray:
    annotated = frame.copy()
    for index, assessment in enumerate(
        sorted(assessments, key=lambda item: item.score, reverse=True)[:3]
    ):
        color = {
            "low": (0, 200, 255),
            "medium": (0, 140, 255),
            "high": (0, 0, 255),
        }[assessment.level]
        text = (
            f"RISK {assessment.level.upper()}: "
            f"{assessment.labels[0]}-{assessment.labels[1]} "
            f"({assessment.score:.0f})"
        )
        cv2.putText(
            annotated,
            text,
            (12, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_live_status(
    frame: np.ndarray,
    fps: float,
    frame_number: int,
    analyzing: bool,
) -> np.ndarray:
    annotated = frame.copy()
    status = "ANALYZING" if analyzing else "FRAME SKIPPED"
    text = f"LIVE | {status} | frame {frame_number} | {fps:.1f} FPS"
    cv2.putText(
        annotated,
        text,
        (12, max(24, annotated.shape[0] - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
