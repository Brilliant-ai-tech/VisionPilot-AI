# VisionPilot AI

VisionPilot AI is an experimental, CPU-based computer vision prototype for road
scenes. It detects objects, tracks them in video, estimates image-space motion
conflicts, stores analysis sessions in SQLite, and provides a Streamlit
dashboard.

> **Research prototype only.** Risk scores are heuristic and use pixel
> coordinates. They are not calibrated distances, collision guarantees, or
> safety-critical guidance.

## Features

- Image detection with the lightweight YOLO11n model.
- Video object tracking using ByteTrack.
- Track IDs, center-point trajectories, and entry/exit events.
- Heuristic pairwise risk assessments with explanations.
- Annotated video output, frame skipping, and processing summaries.
- Browser camera recording for short video clips, alongside video uploads.
- Optional live webcam analysis. Press **Q while the camera window is focused**
  to stop capture.
- SQLite persistence for video and camera sessions, detections, track events,
  and risk events.
- Streamlit dashboard for uploaded images/videos and saved-session analytics.
- CSV and JSON session exports.
- CPU image benchmark reporting latency, FPS, and process resident memory.

YOLO11n inference runs on CPU and requires no CUDA or NVIDIA GPU.

## Requirements

- Windows 10/11, 64-bit.
- Python 3.11, 64-bit.
- Internet access for the initial package and model download.
- A webcam is optional; image and uploaded-video analysis do not require one.

## Setup on Windows

Open the project folder in VS Code. In its integrated PowerShell terminal, create
the environment (skip this if `.venv` already exists):

```powershell
py -3.11 -m venv .venv
```

Install the CPU-only PyTorch wheel, then the pinned project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` using **Python: Select
Interpreter**. Verify the interpreter, CPU-only PyTorch, and package health:

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.version); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pip check
```

The first model run downloads `yolo11n.pt` to `models\`. An internet connection
is needed for that first download.

## Run the application

Run commands from the project root using the virtual-environment interpreter.

### Streamlit dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The dashboard supports image detection, uploaded or browser-recorded video
tracking and risk analysis, saved session analytics, and CSV/JSON exports.
Uploads are limited to 25 MB per image and 200 MB per video; browser recordings
are limited to 100 MB. Video inference runs locally on the CPU and can take time.
The interface adapts to mobile, tablet, and desktop browser sizes. Open the
local URL printed by Streamlit (usually `http://localhost:8501`) in your browser.
To open it from a phone or another device on the same Wi-Fi network, start the
server with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 0.0.0.0
```

Then open `http://<computer-LAN-IP>:8501` on that device. Allow Python through
Windows Firewall if prompted. Analysis still runs on the computer hosting the
dashboard; this does not install a native mobile app. Camera recording requires
browser camera permission and a secure context: use `localhost` on the computer
running Streamlit, or serve the dashboard over HTTPS to record from another
device. Plain HTTP over a LAN does not allow browser camera access.

### Detect objects in an image

```powershell
.\.venv\Scripts\python.exe -m src.main "C:\path\to\image.jpg"
```

The annotated image is written to
`reports\detections\<image-name>_annotated.jpg` by default. Choose a lower
inference size to improve CPU speed:

```powershell
.\.venv\Scripts\python.exe -m src.main "C:\path\to\image.jpg" --image-size 320 --confidence 0.25
```

`--image-size` must be a multiple of 32. Lower sizes can reduce detail and may
miss small objects; test on representative images before choosing a setting.

### Process a video

```powershell
.\.venv\Scripts\python.exe -m src.video_main "C:\path\to\video.mp4" --frame-skip 1 --image-size 320
```

The default outputs are:

- `reports\videos\<video-name>_annotated.mp4`
- `reports\videos\<video-name>_summary.json`
- `data\visionpilot.db` (analysis session and per-analyzed-frame records)

`--frame-skip 1` analyzes every other frame and repeats the last annotated frame
for skipped frames, preserving the output video's frame count and duration.
Use `--database` to select another SQLite database path.

### Run the live camera

```powershell
.\.venv\Scripts\python.exe -m src.camera_main
```

Press **Q with the camera display window focused** to stop. If Windows has
multiple cameras, try `--camera-index 1`. Camera permission and other
applications using the webcam can affect access.

### Export a saved session

The video and camera commands print an analysis session ID:

```powershell
.\.venv\Scripts\python.exe -m src.export_data <session-id> --format csv --output reports\exports\session.csv
.\.venv\Scripts\python.exe -m src.export_data <session-id> --format json --output reports\exports\session.json
```

Add `--database path\to\visionpilot.db` if the session is stored in a
non-default database.

## CPU benchmark

Measure end-to-end image processing on an image of your choice:

```powershell
.\.venv\Scripts\python.exe -m src.benchmark "C:\path\to\image.jpg" --image-size 320 --warmup-runs 1 --runs 5
```

The tool reports mean, median, and p95 latency, mean FPS, and process RSS before
and after the measured runs. FPS includes image decoding and annotated-image
rendering. RSS is sampled before and after the benchmark, not peak memory.
Compare settings on the same machine and representative images; results are not
portable performance guarantees.

One local CPU-only smoke benchmark (Windows 10, Python 3.11, YOLO11n, bundled
Ultralytics `bus.jpg`, one warmup plus three measured runs) produced:

| Inference size | Mean latency | Mean image FPS | RSS after runs |
|---:|---:|---:|---:|
| 320 | 64.2 ms | 15.57 | 364.9 MiB |
| 640 | 123.8 ms | 8.08 | 389.6 MiB |

These are measurements from a single sample image and machine, not model
accuracy metrics or a prediction of video/webcam performance. No precision,
recall, F1, or mAP results are claimed because no labeled evaluation dataset has
been run.

## Tests

Run the unit and integration test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover invalid inputs, detection/tracking behavior, risk scoring,
video processing, live-camera resource release, SQLite sessions and exports,
and benchmark configuration.

## Project layout

```text
dashboard/app.py        Streamlit dashboard
src/detection.py        CPU image detection
src/tracking.py         ByteTrack integration and trajectories
src/risk.py             Image-space risk heuristics
src/video_processing.py Video pipeline and JSON summary
src/live_camera.py      Webcam capture and processing
src/storage.py          SQLite persistence, analytics, and exports
src/benchmark.py        CPU image benchmark
src/*_main.py           Command-line entry points
tests/                  Unit and integration tests
data/                   SQLite database (generated, not committed)
models/                 Downloaded model weights (not committed)
reports/                Generated images, videos, summaries, and exports
```

## Limitations and licensing

- Detection confidence is a model output threshold, not a measure of model
  accuracy. Accuracy metrics require ground-truth labels and a defined
  evaluation dataset.
- Motion and risk are estimated from image-space coordinates; camera geometry,
  calibration, depth, road context, and sensor fusion are not implemented.
- Real-time performance is not guaranteed. Lower inference size and frame
  skipping trade visual detail and temporal coverage for speed.
- Ultralytics publishes YOLO under AGPL-3.0 with a separate Enterprise option.
  Review the current [Ultralytics licensing terms](https://www.ultralytics.com/license),
  model-weight terms, dataset licenses, and the licenses of all dependencies
  before redistribution or commercial use. This project does not currently
  declare a separate project license.
- This prototype is not certified for autonomous navigation or other
  safety-critical use.

## GitHub preparation

`.gitignore` excludes the virtual environment, local SQLite databases, downloaded
model weights, and generated reports. 
