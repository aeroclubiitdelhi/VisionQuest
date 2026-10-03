"""Result map: true markers vs a team's answers, drawn on a top-down view of the field."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

HALF_EXTENT = 17.0
PPM = 30

GREEN = (80, 220, 80)
YELLOW = (0, 215, 255)
RED = (60, 60, 240)
BLUE = (255, 170, 40)
MAGENTA = (220, 60, 220)
WHITE = (245, 245, 245)

KIND_LABEL = {"wrong_dictionary": "DICT", "invalid_id": "BAD ID", "wrong_size": "SIZE",
              "duplicate": "DUP", "moving": "MOVE"}


def _px(x, y, size):
    return int(round(size / 2 + x * PPM)), int(round(size / 2 - y * PPM))


def draw_map(manifest: Dict, predictions: Iterable, score: Optional[Dict] = None,
             true_path: Optional[List] = None, background: Optional[np.ndarray] = None,
             title: str = "") -> np.ndarray:
    size = int(2 * HALF_EXTENT * PPM)
    if background is not None and background.shape[:2] == (size, size):
        canvas = (background.astype(np.float32) * 0.55).astype(np.uint8)
    else:
        canvas = np.full((size, size, 3), (40, 80, 45), np.uint8)
    hf = 15.0
    cv2.rectangle(canvas, _px(-hf, hf, size), _px(hf, -hf, size), (200, 200, 200), 1)

    if true_path:
        pts = np.array([_px(p[1], p[2], size) for p in true_path], np.int32)
        cv2.polylines(canvas, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)

    missed = set(score["missed_ids"]) if score else set()
    genuine_pos = {}
    for m in manifest["markers"]:
        half = m["size_m"] / 2
        c = _px(m["x"], m["y"], size)
        r = max(4, int(half * PPM))
        if m["genuine"]:
            genuine_pos[m["marker_id"]] = (m["x"], m["y"])
            col = YELLOW if m["marker_id"] in missed else GREEN
            cv2.rectangle(canvas, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, 2)
            cv2.putText(canvas, str(m["marker_id"]), (c[0] + r + 3, c[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        else:
            if m.get("motion"):
                mo = m["motion"]
                a, amp = mo["axis"], mo["amplitude"]
                p0 = _px(m["x"] - amp * np.cos(a), m["y"] - amp * np.sin(a), size)
                p1 = _px(m["x"] + amp * np.cos(a), m["y"] + amp * np.sin(a), size)
                cv2.line(canvas, p0, p1, RED, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), RED, 1)
            label = f"{KIND_LABEL.get(m['kind'], m['kind'])} {m['marker_id']}"
            cv2.putText(canvas, label, (c[0] + r + 3, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)

    matches = score.get("matches", {}) if score else {}
    used = set()
    for mid, x, y in predictions:
        p = _px(x, y, size)
        is_tp = False
        if mid in matches and mid not in used and mid in genuine_pos:
            gx, gy = genuine_pos[mid]
            if abs(np.hypot(x - gx, y - gy) - matches[mid]) < 1e-3:
                is_tp = True
                used.add(mid)
                cv2.line(canvas, p, _px(gx, gy, size), BLUE, 1, cv2.LINE_AA)
                cv2.circle(canvas, p, 4, BLUE, -1, cv2.LINE_AA)
        if not is_tp:
            cv2.drawMarker(canvas, p, MAGENTA, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(mid), (p[0] + 7, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MAGENTA, 1, cv2.LINE_AA)

    # Header and legend.
    cv2.rectangle(canvas, (0, 0), (size, 34), (25, 25, 25), -1)
    head = title
    if score:
        err = score["mean_error_m"]
        head += f"   score {score['score']:.1f}   found {score['found']}/{score['genuine']}" \
                f"   wrong {score['false_positives']}   mean err {err if err is not None else '-'} m"
    cv2.putText(canvas, head, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    legend = [(GREEN, "genuine (found)"), (YELLOW, "genuine (missed)"), (RED, "fake"),
              (BLUE, "your answer (correct)"), (MAGENTA, "your answer (wrong)")]
    x = 10
    cv2.rectangle(canvas, (0, size - 26), (size, size), (25, 25, 25), -1)
    for col, text in legend:
        cv2.rectangle(canvas, (x, size - 18), (x + 12, size - 6), col, -1)
        cv2.putText(canvas, text, (x + 17, size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        x += 30 + 8 * len(text)
    return canvas
