"""Dev prescan HTTP server for BrailleLens Flutter (stage 1).

Run on your PC (same WiFi as phone):

    cd BrailleLens
    finger_cell_track\\.venv\\Scripts\\python.exe braille_lens_flutter/tools/prescan_server.py

Then in Flutter (e.g. main.dart before runApp):

    PrescanBridge.prescanServerUrl = 'http://<YOUR_PC_IP>:8765';

POST /prescan  body=JPEG  →  JSON CellMap for stage-2 hit-test.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FCT = ROOT / "finger_cell_track"
if str(FCT) not in sys.path:
    sys.path.insert(0, str(FCT))

from finger_cell_track.prescan import prescan_bgr  # noqa: E402


def _cellmap_to_json(cm, width: int, height: int) -> dict:
    cells = []
    for c in cm.cells:
        cells.append(
            {
                "id": c.id,
                "x0": c.xyxy[0],
                "y0": c.xyxy[1],
                "x1": c.xyxy[2],
                "y1": c.xyxy[3],
                "char": c.char,
                "pattern": c.pattern,
                "code": c.code,
                "conf": c.conf,
                "line": c.line,
                "col": c.col,
            }
        )
    return {"width": width, "height": height, "cells": cells}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/prescan":
            self.send_error(404)
            return
        qs = parse_qs(parsed.query)
        lang = (qs.get("lang") or ["si"])[0]
        backend = (qs.get("backend") or ["dnn"])[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        arr = np.frombuffer(body, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            self.send_error(400, "invalid JPEG")
            return
        h, w = bgr.shape[:2]
        cm = prescan_bgr(bgr, lang=lang, backend=backend)
        payload = json.dumps(_cellmap_to_json(cm, w, h)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    port = 8765
    host = "0.0.0.0"
    print(f"Prescan server http://{host}:{port}/prescan")
    print("Set PrescanBridge.prescanServerUrl on the phone to http://<PC_IP>:8765")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
