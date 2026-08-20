"""Easy fingertip check — no Braille page needed.

Draws a bright yellow dot where SkinContourTip (default) thinks the finger is.
Use any plain background / desk / paper.

IP Webcam streams are read as MJPEG (more reliable than OpenCV's FFMPEG open).

    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/tip_dot_test.py --source 0
    finger_cell_track\\.venv\\Scripts\\python.exe finger_cell_track/tip_dot_test.py --source http://PHONE_IP:8080/video
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hand_track import FallbackTip, MediaPipeTip, SkinContourTip  # noqa: E402


def _is_http(source: str) -> bool:
    return str(source).lower().startswith(("http://", "https://"))


def iter_shot_jpg(base_url: str, interval: float = 0.08) -> Iterator[np.ndarray]:
    """Poll IP Webcam /shot.jpg (works when /video is blocked or busy)."""
    root = base_url.rstrip("/")
    if root.endswith("/video") or root.endswith("/videofeed"):
        root = root.rsplit("/", 1)[0]
    shot = root + "/shot.jpg"
    print(f"Polling stills {shot!r} ...", flush=True)
    while True:
        req = urllib.request.Request(shot, headers={"User-Agent": "BrailleLens-tip-dot-test"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            yield frame
        time.sleep(interval)


def iter_mjpeg(url: str, timeout: float = 15.0) -> Iterator[np.ndarray]:
    """Yield BGR frames from an IP Webcam /video or /videofeed MJPEG stream."""
    req = urllib.request.Request(url, headers={"User-Agent": "BrailleLens-tip-dot-test"})
    stream = urllib.request.urlopen(req, timeout=timeout)
    buf = b""
    last_good: Optional[np.ndarray] = None
    while True:
        chunk = stream.read(16384)
        if not chunk:
            break
        buf += chunk
        # Avoid unbounded growth if markers are missing for a while.
        if len(buf) > 8_000_000:
            buf = buf[-2_000_000:]
        while True:
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start == -1 or end == -1:
                break
            jpg = buf[start : end + 2]
            buf = buf[end + 2 :]
            # Tiny "frames" are almost always torn Wi‑Fi junk — skip them.
            if len(jpg) < 8_000:
                continue
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue
            # Reject wildly different sizes mid-stream (corrupt decode artifacts).
            if last_good is not None:
                h0, w0 = last_good.shape[:2]
                h1, w1 = frame.shape[:2]
                if abs(h1 - h0) > h0 * 0.25 or abs(w1 - w0) > w0 * 0.25:
                    continue
            last_good = frame
            yield frame


def main() -> None:
    p = argparse.ArgumentParser(description="Fingertip-only live test (yellow dot)")
    p.add_argument(
        "--source",
        default="http://192.168.1.17:8080/video",
        help="IP Webcam URL only for this test (default phone stream). Do not use PC webcam.",
    )
    p.add_argument(
        "--tip-backend",
        choices=("skin", "mediapipe", "auto"),
        default="skin",
        help="skin = SkinContourTip (default for over-page / cropped hand)",
    )
    p.add_argument(
        "--display-width",
        type=int,
        default=1600,
        help="Displayed window width in pixels (default 1600, large)",
    )
    p.add_argument(
        "--display-height",
        type=int,
        default=900,
        help="Displayed window height in pixels (default 900)",
    )
    p.add_argument(
        "--show-mask",
        action="store_true",
        help="Show skin mask in a second window (debug)",
    )
    args = p.parse_args()

    if args.tip_backend == "skin":
        tipper = SkinContourTip()
        label = "SkinContourTip"
    elif args.tip_backend == "mediapipe":
        tipper = MediaPipeTip()
        label = "MediaPipeTip"
    else:
        tipper = FallbackTip(MediaPipeTip(), SkinContourTip())
        label = "MediaPipe+Skin fallback"

    print(f"Backend={label}", flush=True)
    src = str(args.source)
    if not _is_http(src):
        raise SystemExit(
            f"This tip test is IP-Webcam only. Refusing PC webcam source {src!r}.\n"
            "Use: --source http://PHONE_IP:8080/video"
        )
    print(f"Opening IP Webcam {src!r} (PC webcam disabled) ...", flush=True)

    urls = [src]
    # Common IP Webcam alternates if the given path stalls.
    base = src.rstrip("/")
    for alt in ("/video", "/videofeed"):
        root = base.rsplit("/", 1)[0] if base.count("/") >= 3 else base
        candidate = root + alt
        if candidate not in urls:
            urls.append(candidate)
    last_err: Optional[Exception] = None
    frame_iter: Optional[Iterator[np.ndarray]] = None
    for url in urls:
        try:
            print(f"Trying MJPEG {url!r} ...", flush=True)
            http_frames = iter_mjpeg(url)
            first = next(http_frames)
            print(f"Got frame {first.shape[1]}x{first.shape[0]} from MJPEG.", flush=True)
            frame_iter = _seeded(http_frames, first)
            break
        except (urllib.error.URLError, TimeoutError, StopIteration, OSError) as exc:
            last_err = exc
            print(f"  failed: {exc}", flush=True)
    if frame_iter is None:
        # Last resort: poll /shot.jpg stills (slower, but often works).
        try:
            http_frames = iter_shot_jpg(src)
            first = next(http_frames)
            print(f"Got frame {first.shape[1]}x{first.shape[0]} from shot.jpg.", flush=True)
            frame_iter = _seeded(http_frames, first)
        except (urllib.error.URLError, TimeoutError, StopIteration, OSError) as exc:
            raise SystemExit(
                f"Could not open IP Webcam stream from {src!r}.\n"
                f"MJPEG error: {last_err}\n"
                f"shot.jpg error: {exc}\n"
                "On the phone: IP Webcam → Start server, keep screen on, same Wi‑Fi.\n"
                "Close other apps using the camera, then retry."
            ) from exc

    win = "tip_dot_test — yellow = fingertip (Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.display_width, args.display_height)
    cv2.moveWindow(win, 20, 20)
    if args.show_mask:
        cv2.namedWindow("skin_mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("skin_mask", args.display_width // 2, args.display_height // 2)

    print(
        f"Large window {args.display_width}x{args.display_height}. "
        "Point finger into the frame. Yellow/red = tip. Q = quit.",
        flush=True,
    )
    t0 = time.time()
    n = 0
    hits = 0
    last_status = None

    try:
        for frame in frame_iter:
            tip, box, conf = tipper.detect(frame)
            out = frame.copy()
            n += 1
            if tip is not None:
                hits += 1
                x, y = int(tip[0]), int(tip[1])
                cv2.circle(out, (x, y), 14, (0, 255, 255), -1)
                cv2.circle(out, (x, y), 24, (0, 128, 255), 3)
                cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
                if box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cv2.putText(
                    out,
                    f"TIP ({x},{y})",
                    (x + 20, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                status = "TIP"
            else:
                status = "NO TIP"
                cv2.putText(
                    out,
                    "NO TIP — move finger into frame (from bottom/edge)",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 80, 255),
                    2,
                )

            if status != last_status:
                print(f"[{status}] tip={tip}", flush=True)
                last_status = status

            if n % 60 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                rate = hits / n
                print(
                    f"fps~{n / elapsed:.1f}  tip_rate={rate:.0%}  ({hits}/{n})",
                    flush=True,
                )

            hud = f"{label}  tip={'Y' if tip else 'N'}  rate={hits / max(n, 1):.0%}"
            cv2.putText(
                out, hud, (20, out.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2,
            )

            if args.show_mask and hasattr(tipper, "_mask"):
                cv2.imshow("skin_mask", tipper._mask(frame))
            elif args.show_mask and isinstance(tipper, FallbackTip) and hasattr(tipper.fallback, "_mask"):
                cv2.imshow("skin_mask", tipper.fallback._mask(frame))

            # Fit into the large window (letterbox to display_width x display_height).
            th, tw = args.display_height, args.display_width
            h, w = out.shape[:2]
            scale = min(tw / w, th / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            y0 = (th - nh) // 2
            x0 = (tw - nw) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
            cv2.imshow(win, canvas)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        if hasattr(tipper, "close"):
            tipper.close()
        cv2.destroyAllWindows()
        print(
            f"Stopped. tip detected on {hits}/{n} frames ({hits / max(n, 1):.0%}).",
            flush=True,
        )


def _seeded(it: Iterator[np.ndarray], first: np.ndarray) -> Iterator[np.ndarray]:
    yield first
    yield from it


if __name__ == "__main__":
    main()
