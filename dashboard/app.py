import base64
import binascii
import logging
import sqlite3
import tempfile
import uuid
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.detection import DEFAULT_IMAGE_SIZE, ObjectDetector
from src.storage import AnalysisSession, SQLiteRepository
from src.video_processing import VideoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "visionpilot.db"
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "images"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "reports" / "videos"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_BROWSER_RECORDING_BYTES = 100 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_browser_video_recorder = components.declare_component(
    "visionpilot_browser_video_recorder",
    path=str(Path(__file__).resolve().parent / "browser_recorder"),
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _apply_responsive_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --vp-ink: #14243a;
            --vp-muted: #64748b;
            --vp-border: #e3eaf1;
            --vp-accent: #087f75;
            --vp-accent-dark: #06665f;
            --vp-surface: #ffffff;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 8% 0%, rgba(196, 232, 224, 0.46) 0, transparent 36rem),
                linear-gradient(180deg, #f8fafc 0%, #f3f6fa 100%);
            color: var(--vp-ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stMain"] > div {
            width: 100%;
            max-width: 1440px;
            padding: 1.5rem clamp(1rem, 4vw, 3.5rem) 3.5rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--vp-border);
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 8px 0 28px rgba(27, 45, 70, 0.025);
        }

        [data-testid="stSidebar"] h2 {
            color: var(--vp-ink);
            font-size: 1.05rem;
            letter-spacing: -0.02em;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--vp-muted);
        }

        .vp-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            margin: 0 0 1.35rem;
            padding: clamp(1.3rem, 3vw, 2rem) clamp(1.3rem, 3vw, 2.2rem);
            overflow: hidden;
            border: 1px solid rgba(202, 224, 222, 0.9);
            border-radius: 1.35rem;
            background:
                radial-gradient(circle at 88% 18%, rgba(8, 127, 117, 0.12), transparent 13rem),
                linear-gradient(115deg, #ffffff 0%, #f1f8f7 100%);
            box-shadow: 0 16px 42px rgba(25, 52, 73, 0.07);
        }

        .vp-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .vp-mark {
            display: grid;
            flex: 0 0 3.4rem;
            width: 3.4rem;
            height: 3.4rem;
            place-items: center;
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 1.05rem;
            background: linear-gradient(145deg, #11968a, #07665f);
            color: #fff;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: -0.07em;
            box-shadow: 0 8px 18px rgba(8, 127, 117, 0.24);
        }

        .vp-eyebrow {
            margin: 0 0 0.3rem;
            color: var(--vp-accent-dark);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .vp-hero h1 {
            margin: 0;
            color: var(--vp-ink);
            font-size: clamp(1.7rem, 4vw, 2.4rem);
            line-height: 1.12;
            letter-spacing: -0.045em;
        }

        .vp-hero p {
            margin: 0.45rem 0 0;
            color: var(--vp-muted);
            font-size: clamp(0.9rem, 1.5vw, 0.98rem);
            line-height: 1.55;
        }

        .vp-badge {
            flex: 0 0 auto;
            padding: 0.5rem 0.75rem;
            border: 1px solid #d4e9e5;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            color: #176d66;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        .vp-workspace-intro {
            margin: 0.25rem 0 1.1rem;
        }

        .vp-workspace-intro h2 {
            margin: 0;
            color: var(--vp-ink);
            font-size: clamp(1.2rem, 2.5vw, 1.45rem);
            font-weight: 750;
            letter-spacing: -0.035em;
        }

        .vp-workspace-intro p {
            margin: 0.35rem 0 0;
            color: var(--vp-muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.35rem;
            padding: 0.35rem;
            border: 1px solid var(--vp-border);
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.74);
            box-shadow: 0 3px 12px rgba(24, 42, 63, 0.025);
        }

        [data-testid="stTabs"] button[role="tab"] {
            min-height: 2.55rem;
            padding: 0.55rem 1rem;
            border-radius: 0.65rem;
            color: var(--vp-muted);
            font-size: 0.9rem;
            font-weight: 650;
            transition: background 140ms ease, color 140ms ease;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            background: #e8f5f2;
            color: var(--vp-accent-dark);
            box-shadow: inset 0 0 0 1px #d4ebe6;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--vp-border) !important;
            border-radius: 1rem !important;
            background: var(--vp-surface);
            box-shadow: 0 8px 24px rgba(30, 48, 69, 0.045);
            transition: box-shadow 160ms ease, transform 160ms ease;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 12px 30px rgba(30, 48, 69, 0.075);
        }

        [data-testid="stVerticalBlockBorderWrapper"] h3 {
            margin-bottom: 0.25rem;
            color: var(--vp-ink);
            font-size: 1.15rem;
            letter-spacing: -0.025em;
        }

        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
            color: var(--vp-muted);
            line-height: 1.5;
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed #c8d5e2;
            border-radius: 0.8rem;
            background: #f9fbfd;
            transition: border-color 140ms ease, background 140ms ease;
        }

        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
            color: var(--vp-ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 0 !important;
            border-radius: 0.75rem !important;
            background: #f9fbfd !important;
            color: var(--vp-ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--vp-ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            border: 1px solid #cbd9e5 !important;
            border-radius: 0.6rem !important;
            background: #ffffff !important;
            color: var(--vp-accent-dark) !important;
        }

        [data-testid="stFileUploaderDropzone"] button svg {
            fill: var(--vp-accent-dark) !important;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #7fb9b1;
            background: #f4faf8;
        }

        [data-testid="stMetric"] {
            min-height: 6.2rem;
            padding: 1rem;
            border: 1px solid var(--vp-border);
            border-radius: 0.85rem;
            background: linear-gradient(145deg, #fff, #fbfcfe);
            box-shadow: 0 4px 12px rgba(30, 48, 69, 0.025);
        }

        [data-testid="stMetricLabel"] {
            color: var(--vp-muted);
            font-size: 0.82rem;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: var(--vp-ink);
            font-weight: 750;
            letter-spacing: -0.04em;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 0.72rem;
            font-weight: 700;
            transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
        }

        .stButton > button[kind="primary"] {
            border-color: var(--vp-accent);
            background: var(--vp-accent);
            box-shadow: 0 5px 12px rgba(8, 127, 117, 0.16);
        }

        .stButton > button[kind="primary"]:hover {
            border-color: var(--vp-accent-dark);
            background: var(--vp-accent-dark);
            box-shadow: 0 7px 16px rgba(8, 127, 117, 0.23);
            transform: translateY(-1px);
        }

        [data-testid="stRadio"] [role="radiogroup"] {
            gap: 0.35rem;
        }

        [data-testid="stAlert"] {
            border-radius: 0.8rem;
        }

        [data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--vp-border);
            border-radius: 0.8rem;
        }

        [data-testid="stExpander"] {
            border-color: var(--vp-border);
            border-radius: 0.8rem;
        }

        iframe[data-testid="stCustomComponentV1"] {
            min-height: 430px !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            max-width: 100%;
        }

        @media (max-width: 768px) {
            [data-testid="stMain"] > div {
                padding: 1rem 0.85rem 2rem;
            }

            .vp-hero {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.8rem;
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 1rem;
            }

            .vp-brand {
                gap: 0.75rem;
            }

            .vp-mark {
                flex-basis: 2.6rem;
                width: 2.6rem;
                height: 2.6rem;
                border-radius: 0.8rem;
            }

            .vp-badge {
                align-self: flex-start;
                margin-left: 3.35rem;
            }

            [data-testid="stTabs"] [role="tablist"] {
                gap: 0.15rem;
                justify-content: space-between;
                padding: 0.25rem;
            }

            [data-testid="stTabs"] button[role="tab"] {
                flex: 1 1 auto;
                min-height: 2.45rem;
                padding: 0.45rem 0.25rem;
                font-size: 0.82rem;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.75rem !important;
            }

            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])
                > [data-testid="column"] {
                flex: 1 1 calc(50% - 0.4rem) !important;
                width: calc(50% - 0.4rem) !important;
                min-width: calc(50% - 0.4rem) !important;
            }

            [data-testid="stMetric"] {
                min-height: 5.5rem;
                padding: 0.75rem;
            }

            iframe[data-testid="stCustomComponentV1"] {
                min-height: 390px !important;
            }

            .stButton > button,
            .stDownloadButton > button {
                width: 100%;
            }
        }

        @media (max-width: 360px) {
            [data-testid="stTabs"] button[role="tab"] {
                font-size: 0.74rem;
            }

            [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])
                > [data-testid="column"] {
                flex-basis: 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _session_label(session: AnalysisSession) -> str:
    return (
        f"{session.started_at[:19].replace('T', ' ')} | "
        f"{session.status.upper()} | {session.source} | {session.session_id[:8]}"
    )


def _show_analytics(repository: SQLiteRepository, session: AnalysisSession) -> None:
    analytics = repository.get_analytics(session.session_id)
    st.subheader("Session analytics")
    detection_data = analytics["detections"]
    risk_data = analytics["risk_events"]
    track_data = analytics["track_events"]

    metrics = st.columns(4)
    metrics[0].metric("Analyzed frames", analytics["frames_analyzed"])
    metrics[1].metric("Object observations", detection_data["observations"])
    metrics[2].metric("Unique tracks", detection_data["unique_tracks"])
    metrics[3].metric("Risk events", risk_data["total"])

    by_class = detection_data["by_class"]
    if by_class:
        class_frame = pd.DataFrame.from_dict(by_class, orient="index")
        class_frame.index.name = "Object class"
        st.markdown("**Detections by class**")
        st.bar_chart(class_frame[["observations"]], width="stretch")
        st.dataframe(class_frame, width="stretch")
    else:
        st.info("No tracked detections were recorded for this session.")

    risk_by_level = risk_data["by_level"]
    if risk_by_level:
        risk_frame = pd.DataFrame(
            {"Risk events": risk_by_level.values()},
            index=list(risk_by_level.keys()),
        )
        risk_frame.index.name = "Risk level"
        st.markdown("**Risk events by level**")
        st.bar_chart(risk_frame, width="stretch")

    event_counts = track_data["by_kind"]
    if event_counts:
        st.markdown("**Track entry and exit events**")
        st.bar_chart(
            pd.DataFrame(
                {"Track events": event_counts.values()},
                index=list(event_counts.keys()),
            ),
            width="stretch",
        )

    risk_events = repository.get_session_risk_events(session.session_id)
    if risk_events:
        st.markdown("**Risk event details**")
        risk_frame = pd.DataFrame(risk_events)
        st.dataframe(risk_frame, width="stretch")
        st.line_chart(
            risk_frame.set_index("frame_number")[["score"]],
            width="stretch",
        )

    if session.summary:
        with st.expander("Processing summary"):
            st.json(session.summary)
    if session.error:
        st.error(f"Session error: {session.error}")


def _analyze_image(
    uploaded_file: object,
    confidence: float,
    image_size: int,
) -> None:
    name = getattr(uploaded_file, "name", "image")
    extension = Path(name).suffix.lower()
    size = getattr(uploaded_file, "size", 0)
    if extension not in IMAGE_EXTENSIONS:
        st.error("Choose a JPG, JPEG, PNG, or BMP image.")
        return
    if size > MAX_IMAGE_BYTES:
        st.error("Image exceeds the 25 MB upload limit.")
        return

    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(uploaded_file.getvalue())

    try:
        with st.spinner("Running CPU image detection..."):
            result = ObjectDetector(
                confidence=confidence,
                image_size=image_size,
            ).detect_image(temporary_path)
        image_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
        encoded_ok, encoded_image = cv2.imencode(".jpg", result.annotated_image)
        if not encoded_ok:
            raise OSError("Could not encode the annotated image for download.")

        st.image(image_rgb, caption="Annotated image", width="stretch")
        st.metric("Objects detected", len(result.detections))
        if result.detections:
            detections = [
                {
                    "class": item.label,
                    "confidence": round(item.confidence, 4),
                    "x1": round(item.box[0], 1),
                    "y1": round(item.box[1], 1),
                    "x2": round(item.box[2], 1),
                    "y2": round(item.box[3], 1),
                }
                for item in result.detections
            ]
            st.dataframe(pd.DataFrame(detections), width="stretch")
        st.download_button(
            "Download annotated image",
            data=encoded_image.tobytes(),
            file_name=f"{Path(name).stem}_annotated.jpg",
            mime="image/jpeg",
            key="download_annotated_image",
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.exception("Image analysis failed")
        st.error(f"Image analysis failed: {error}")
    finally:
        temporary_path.unlink(missing_ok=True)


def _analyze_video(
    uploaded_file: object,
    confidence: float,
    frame_skip: int,
    image_size: int,
    repository: SQLiteRepository,
) -> None:
    name = getattr(uploaded_file, "name", "video")
    _analyze_video_bytes(
        name,
        uploaded_file.getvalue(),
        confidence,
        frame_skip,
        image_size,
        repository,
    )


def _decode_browser_recording(recording: object) -> tuple[str, bytes]:
    if not isinstance(recording, dict):
        raise ValueError("The browser recorder did not return a valid recording.")
    if recording.get("error"):
        raise ValueError(str(recording["error"]))

    encoded_video = recording.get("base64")
    extension = recording.get("extension")
    if not isinstance(encoded_video, str) or not isinstance(extension, str):
        raise ValueError("The browser recorder returned incomplete video data.")

    extension = extension.lower().lstrip(".")
    if f".{extension}" not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported browser recording format: {extension}.")
    try:
        video_bytes = base64.b64decode(encoded_video, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("The browser recording data is invalid.") from error
    if not video_bytes:
        raise ValueError("The browser recording is empty.")
    if len(video_bytes) > MAX_BROWSER_RECORDING_BYTES:
        raise ValueError("Browser recordings must be 100 MB or smaller.")

    return f"browser_recording.{extension}", video_bytes


def _analyze_video_bytes(
    name: str,
    video_bytes: bytes,
    confidence: float,
    frame_skip: int,
    image_size: int,
    repository: SQLiteRepository,
) -> None:
    extension = Path(name).suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        st.error("Choose an MP4, AVI, MOV, MKV, or WEBM video.")
        return
    if len(video_bytes) > MAX_VIDEO_BYTES:
        st.error("Video exceeds the 200 MB upload limit.")
        return

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_id = uuid.uuid4().hex
    output_path = VIDEO_OUTPUT_DIR / f"{output_id}_annotated.mp4"
    summary_path = VIDEO_OUTPUT_DIR / f"{output_id}_summary.json"
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary:
        input_path = Path(temporary.name)
        temporary.write(video_bytes)

    try:
        processor = VideoProcessor(
            confidence=confidence,
            frame_skip=frame_skip,
            image_size=image_size,
            repository=repository,
        )
        with st.spinner("Processing video on the CPU. Longer clips may take time..."):
            summary = processor.process(input_path, output_path)
            processor.save_summary(summary, summary_path)
        st.success(
            f"Processed {summary.frames_read} frames; analyzed "
            f"{summary.frames_analyzed}, skipped {summary.frames_skipped}."
        )
        metrics = st.columns(4)
        metrics[0].metric("Unique tracks", summary.unique_tracks)
        metrics[1].metric("Risk events", summary.risk_assessment_count)
        metrics[2].metric("Processing FPS", f"{summary.processing_fps:.2f}")
        metrics[3].metric("Session", (summary.session_id or "")[:8])
        st.video(str(output_path))
        st.download_button(
            "Download video summary (JSON)",
            data=summary_path.read_bytes(),
            file_name=f"{Path(name).stem}_summary.json",
            mime="application/json",
            key=f"download_summary_{output_id}",
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        LOGGER.exception("Video analysis failed")
        st.error(f"Video analysis failed: {error}")
        for generated_path in (output_path, summary_path):
            generated_path.unlink(missing_ok=True)
    finally:
        input_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(
        page_title="VisionPilot AI",
        page_icon="🚦",
        layout="wide",
    )
    _apply_responsive_styles()
    st.markdown(
        """
        <header class="vp-hero">
            <div class="vp-brand">
                <div class="vp-mark" aria-hidden="true">VP</div>
                <div>
                    <div class="vp-eyebrow">Computer vision workspace</div>
                    <h1>VisionPilot AI</h1>
                    <p>Road-scene detection, tracking, and motion-risk insights.</p>
                </div>
            </div>
            <div class="vp-badge">Research prototype</div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis settings")
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Minimum model confidence for keeping a detection.",
        )
        frame_skip = st.slider(
            "Frames to skip between analyses",
            min_value=0,
            max_value=15,
            value=0,
            help="Skipped frames reuse the last annotated frame in video output.",
        )
        image_size = st.select_slider(
            "Inference image size",
            options=list(range(320, 961, 32)),
            value=DEFAULT_IMAGE_SIZE,
            help="Smaller values usually run faster on CPU but may miss small objects.",
        )
        st.caption("Image limit: 25 MB. Video limit: 200 MB.")

    try:
        repository = SQLiteRepository(DATABASE_PATH)
        sessions = repository.list_sessions(limit=100)
    except (OSError, sqlite3.Error, ValueError) as error:
        LOGGER.exception("Could not open the analysis database")
        st.error(f"Could not open the analysis database: {error}")
        return

    analyze_tab, history_tab, about_tab = st.tabs(
        ["Analyze media", "Session history", "About"]
    )
    with analyze_tab:
        st.markdown(
            """
            <section class="vp-workspace-intro">
                <h2>Start an analysis</h2>
                <p>Choose an image for a quick scan, or use a video to track movement over time.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        _, video_source_column = st.columns(2)
        with video_source_column:
            video_source = st.radio(
                "Video source",
                ["Upload a video", "Record from camera"],
                horizontal=True,
                key="video_source",
            )

        image_column, video_column = st.columns(2)
        with image_column:
            with st.container(border=True):
                st.subheader("Image detection")
                st.caption("Upload a still image to detect and label road users.")
                image_file = st.file_uploader(
                    "Choose an image",
                    type=sorted(extension.lstrip(".") for extension in IMAGE_EXTENSIONS),
                    key="image_upload",
                )
                if image_file is not None and st.button(
                    "Analyze image",
                    key="analyze_image",
                    type="primary",
                ):
                    _analyze_image(image_file, confidence, image_size)

        with video_column:
            with st.container(border=True):
                st.subheader("Video tracking and risk analysis")
                st.caption("Track moving objects and review estimated motion conflicts.")
                if video_source == "Upload a video":
                    video_file = st.file_uploader(
                        "Choose a video",
                        type=sorted(
                            extension.lstrip(".")
                            for extension in VIDEO_EXTENSIONS
                        ),
                        key="video_upload",
                    )
                    if video_file is not None and st.button(
                        "Process video",
                        key="analyze_video",
                        type="primary",
                    ):
                        _analyze_video(
                            video_file,
                            confidence,
                            frame_skip,
                            image_size,
                            repository,
                        )
                else:
                    st.caption(
                        "Select Start camera below and allow camera access in your "
                        "browser. Then choose Start recording. Processing runs on "
                        "this computer."
                    )
                    recording = _browser_video_recorder(
                        max_bytes=MAX_BROWSER_RECORDING_BYTES,
                        key="browser_camera_recorder",
                        default=None,
                    )
                    if recording is not None:
                        try:
                            recording_name, recording_bytes = (
                                _decode_browser_recording(recording)
                            )
                        except ValueError as error:
                            st.error(f"Camera recording failed: {error}")
                        else:
                            recording_format = (
                                f"video/{Path(recording_name).suffix.lstrip('.')}"
                            )
                            st.video(recording_bytes, format=recording_format)
                            if st.button(
                                "Analyze recording",
                                key="analyze_browser_recording",
                                type="primary",
                            ):
                                _analyze_video_bytes(
                                    recording_name,
                                    recording_bytes,
                                    confidence,
                                    frame_skip,
                                    image_size,
                                    repository,
                                )

    with history_tab:
        st.subheader("Saved analysis sessions")
        if not sessions:
            st.info("No saved sessions yet. Process a video or camera stream first.")
        else:
            session_options = {session.session_id: session for session in sessions}
            selected_session_id = st.selectbox(
                "Select a session",
                options=list(session_options),
                format_func=lambda session_id: _session_label(
                    session_options[session_id]
                ),
            )
            selected_session = session_options[selected_session_id]
            st.write(f"**Source:** `{selected_session.source}`")
            _show_analytics(repository, selected_session)

            export_column, _ = st.columns([1, 3])
            with export_column:
                export_format = st.selectbox(
                    "Export format",
                    options=["JSON", "CSV"],
                    key="export_format",
                )
                if st.button("Prepare export", key="prepare_export"):
                    export_path = (
                        PROJECT_ROOT
                        / "reports"
                        / "exports"
                        / f"{selected_session_id}.{export_format.lower()}"
                    )
                    if export_format == "CSV":
                        repository.export_session_csv(
                            selected_session_id,
                            export_path,
                        )
                        mime = "text/csv"
                    else:
                        repository.export_session_json(
                            selected_session_id,
                            export_path,
                        )
                        mime = "application/json"
                    st.download_button(
                        f"Download {export_format}",
                        data=export_path.read_bytes(),
                        file_name=f"visionpilot_{selected_session_id}.{export_format.lower()}",
                        mime=mime,
                        key=f"download_session_{selected_session_id}_{export_format}",
                    )

    with about_tab:
        st.subheader("Prototype limitations")
        st.warning(
            "Risk levels are heuristic image-space estimates. They are not calibrated "
            "physical distances, collision guarantees, or safety-critical guidance."
        )
        st.markdown(
            "- Model inference runs on the CPU and may be slow for long or high-resolution videos.\n"
            "- Detection confidence is not the same as model accuracy.\n"
            "- Video processing and camera capture create session records in SQLite.\n"
            "- Image detection is displayed in the dashboard but is not saved as a tracking session."
        )


if __name__ == "__main__":
    main()
