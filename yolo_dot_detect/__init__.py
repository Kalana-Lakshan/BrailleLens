"""YOLOv8 embossed Braille-dot detection (transfer learning + augmentation).

Public API:
    from yolo_dot_detect.detect_dots import YoloDotDetector, detect_dot_centers_yolo
"""

from .detect_dots import YoloDotDetector, detect_dot_centers_yolo

__all__ = ["YoloDotDetector", "detect_dot_centers_yolo"]
