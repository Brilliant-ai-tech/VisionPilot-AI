import logging
import sqlite3
import tempfile
import uuid
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.detection import DEFAULT_IMAGE_SIZE, ObjectDetector
from src.storage import AnalysisSession, SQLiteRepository
from src.video_processing import VideoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "visionpilot.db"
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "images"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "reports" / "videos"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


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
    extension = Path(name).suffix.lower()
    size = getattr(uploaded_file, "size", 0)
    if extension not in VIDEO_EXTENSIONS:
        st.error("Choose an MP4, AVI, MOV, or MKV video.")
        return
    if size > MAX_VIDEO_BYTES:
        st.error("Video exceeds the 200 MB upload limit.")
        return

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_id = uuid.uuid4().hex
    output_path = VIDEO_OUTPUT_DIR / f"{output_id}_annotated.mp4"
    summary_path = VIDEO_OUTPUT_DIR / f"{output_id}_summary.json"
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary:
        input_path = Path(temporary.name)
        temporary.write(uploaded_file.getvalue())

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
    st.title("VisionPilot AI")
    st.caption(
        "CPU-based road-scene detection, tracking, and experimental risk analysis."
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
        image_column, video_column = st.columns(2)
        with image_column:
            st.subheader("Image detection")
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
            st.subheader("Video tracking and risk analysis")
            video_file = st.file_uploader(
                "Choose a video",
                type=sorted(extension.lstrip(".") for extension in VIDEO_EXTENSIONS),
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
