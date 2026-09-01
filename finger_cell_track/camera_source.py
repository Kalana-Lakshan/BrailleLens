"""Open PC webcam or IP Webcam URL for live tip / CellMap apps."""

from __future__ import annotations

import cv2


def open_source(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        url = str(source)
        # Prefer default backend first — CAP_FFMPEG-only often hangs ~30s on IP Webcam.
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {source!r}\n"
            "Use --source 0 for PC webcam or http://PHONE_IP:8080/video for IP Webcam.\n"
            "For fingertip-only checks prefer tip_dot_test.py (MJPEG reader)."
        )
    return cap
