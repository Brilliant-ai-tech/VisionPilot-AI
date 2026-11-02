import base64

import pytest

from dashboard import app


def test_decodes_browser_recording() -> None:
    recording_name, video_bytes = app._decode_browser_recording(
        {
            "base64": base64.b64encode(b"sample video").decode("ascii"),
            "extension": "webm",
        }
    )

    assert recording_name == "browser_recording.webm"
    assert video_bytes == b"sample video"


@pytest.mark.parametrize(
    ("recording", "message"),
    [
        (None, "did not return a valid recording"),
        ({"error": "Camera permission denied."}, "Camera permission denied"),
        ({"base64": "not base64", "extension": "webm"}, "data is invalid"),
        ({"base64": "YQ==", "extension": "wmv"}, "Unsupported"),
        ({"base64": "", "extension": "webm"}, "is empty"),
    ],
)
def test_rejects_invalid_browser_recordings(
    recording: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        app._decode_browser_recording(recording)


def test_rejects_browser_recording_above_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(app, "MAX_BROWSER_RECORDING_BYTES", 1)

    with pytest.raises(ValueError, match="100 MB or smaller"):
        app._decode_browser_recording(
            {
                "base64": base64.b64encode(b"too large").decode("ascii"),
                "extension": "webm",
            }
        )
