"""CLI entry point for BrailleLens live camera mode.

Run from the project root:

    # IP Webcam (recommended for phone):
    python camera_capture/run_camera.py --source http://192.168.1.x:8080/video

    # Built-in or DroidCam virtual webcam (integer index):
    python camera_capture/run_camera.py --source 0

Controls inside the preview window:
    Q   quit
    S   force a single inference on the current frame (ignores motion check)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CHECKPOINT = str(_ROOT / "braille_cnn" / "checkpoints" / "braille_cnn_dbsi_finetuned.pt")

from camera_capture.camera import run_camera  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BrailleLens — live camera Braille inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- camera source ----
    p.add_argument(
        "--source",
        default="http://192.168.1.x:8080/video",
        help=(
            "Camera source.  Use an integer (0, 1, …) for a local/DroidCam webcam, "
            "or a URL for IP Webcam (e.g. http://192.168.1.5:8080/video)."
        ),
    )

    # ---- stability / timing ----
    p.add_argument(
        "--motion-threshold",
        type=float,
        default=8.0,
        help=(
            "Mean absolute pixel difference (0–255) below which a frame is "
            "considered stable enough for inference.  Increase if results flicker "
            "on a tripod; decrease if the camera is very steady."
        ),
    )
    p.add_argument(
        "--infer-interval",
        type=float,
        default=1.5,
        help="Minimum seconds between two consecutive inferences on stable frames.",
    )
    p.add_argument(
        "--stable-frames",
        type=int,
        default=8,
        help="Consecutive stable frames required before auto-inference runs.",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="Open the live camera preview without loading a model or running inference.",
    )
    p.add_argument(
        "--display-width",
        type=int,
        default=960,
        help=(
            "Maximum preview window width in pixels. The phone stream is often much "
            "larger; without scaling, OpenCV can look zoomed/cropped on a laptop screen."
        ),
    )

    # ---- model / inference (mirrors infer_page.py args) ----
    p.add_argument(
        "--checkpoint",
        type=str,
        default=_DEFAULT_CHECKPOINT,
        help="Path to the trained model checkpoint.",
    )
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument(
        "--link-distance",
        type=float,
        default=15.0,
        help="Max pixel distance between dots to link them into the same cell.",
    )
    p.add_argument(
        "--dot-z-threshold",
        type=float,
        default=3.0,
        help="Local z-score cutoff for a peak to count as a dot (adapts per-region to lighting).",
    )
    p.add_argument(
        "--dot-footprint",
        type=int,
        default=9,
        help="Non-max-suppression window in pixels (~one dot diameter).",
    )
    p.add_argument(
        "--cell-margin-scale",
        type=float,
        default=0.8,
        help="Extra padding around each detected cell as a fraction of its dot-span.",
    )
    p.add_argument(
        "--lang",
        type=str,
        default="si",
        choices=["en", "si"],
        help="Output language for label decoding.",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.6,
        help="Cells with confidence below this are shown as '_' in the transcription.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full inference log each frame instead of in-place refresh.",
    )
    p.add_argument(
        "--debug-out",
        type=str,
        default=None,
        help="If set, save the last inference overlay image to this path.",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # run_auto expects args.auto = True; camera mode always uses auto detection
    args.auto = True

    run_camera(args)


if __name__ == "__main__":
    main()
