"""Scoring. The same code is used by participants (local check) and the grader.

Per mission:
  * A submitted row is a TRUE POSITIVE when its ID belongs to a genuine marker and
    it lies within MATCH_RADIUS_M (1.0 m) of it. Each genuine marker matches at most once.
  * Credit per true positive = 0.5 (found) + 0.5 x accuracy, where accuracy is 1 within
    FULL_ACCURACY_RADIUS_M (0.3 m) and falls linearly to 0 at 1.0 m.
  * Every other row is a FALSE POSITIVE (fake marker, wrong ID, duplicate row, too far off).
  * mission score = 100 x (sum of credit - FP_PENALTY x false positives) / genuine markers,
    clipped to [0, 100].
Part score = mean mission score over that part's worlds.
Technical score (out of 80) = 0.20 A + 0.32 B + 0.28 C.
Final (out of 100) = technical + 0.20 x pitch score (0..100, judged by the organizers).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import FP_PENALTY, FULL_ACCURACY_RADIUS_M, MATCH_RADIUS_M, PART_WEIGHTS, PITCH_WEIGHT

Row = Tuple[int, float, float]


def sanitize(rows: Iterable) -> Tuple[List[Row], int]:
    """Coerce submitted rows to (int id, float x, float y). Returns (clean_rows, n_invalid)."""
    clean: List[Row] = []
    invalid = 0
    for r in rows or []:
        try:
            mid, x, y = r[0], r[1], r[2]
            mid_f = float(mid)
            if not float(mid_f).is_integer():
                raise ValueError
            x, y = float(x), float(y)
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError
            clean.append((int(mid_f), x, y))
        except (TypeError, ValueError, IndexError, KeyError):
            invalid += 1
    return clean, invalid


def score_mission(pred: Iterable, truth: Sequence[Row],
                  radius: float = MATCH_RADIUS_M,
                  full_radius: float = FULL_ACCURACY_RADIUS_M,
                  fp_penalty: float = FP_PENALTY) -> Dict:
    rows, invalid = sanitize(pred)
    truth_by_id: Dict[int, Tuple[float, float]] = {int(m): (float(x), float(y)) for m, x, y in truth}

    # For each genuine ID pick the closest submitted row within the radius.
    best: Dict[int, Tuple[float, int]] = {}
    for i, (mid, x, y) in enumerate(rows):
        if mid in truth_by_id:
            tx, ty = truth_by_id[mid]
            d = math.hypot(x - tx, y - ty)
            if d <= radius and (mid not in best or d < best[mid][0]):
                best[mid] = (d, i)

    matched_rows = {i for _, i in best.values()}
    fp = invalid + sum(1 for i in range(len(rows)) if i not in matched_rows)
    credit = 0.0
    errors = []
    for mid, (d, _) in best.items():
        acc = 1.0 if d <= full_radius else max(0.0, (radius - d) / (radius - full_radius))
        credit += 0.5 + 0.5 * acc
        errors.append(d)
    n = len(truth_by_id)
    raw = 100.0 * (credit - fp_penalty * fp) / n if n else 0.0
    return {
        "genuine": n,
        "submitted": len(rows) + invalid,
        "found": len(best),
        "missed_ids": sorted(set(truth_by_id) - set(best)),
        "false_positives": fp,
        "mean_error_m": round(sum(errors) / len(errors), 4) if errors else None,
        "score": round(min(100.0, max(0.0, raw)), 3),
        "matches": {int(k): round(v[0], 4) for k, v in sorted(best.items())},
    }


def technical_score(part_scores: Dict[str, float]) -> float:
    """Weighted A/B/C score, out of 80."""
    return round(sum(PART_WEIGHTS[p] * part_scores.get(p, 0.0) for p in PART_WEIGHTS), 3)


def final_score(part_scores: Dict[str, float], pitch: float = 0.0) -> float:
    """Out of 100: technical (out of 80) + PITCH_WEIGHT x pitch score (pitch is 0..100)."""
    pitch = min(100.0, max(0.0, float(pitch)))
    return round(technical_score(part_scores) + PITCH_WEIGHT * pitch, 3)


# ----------------------------------------------------------------------------- CSV helpers

def write_csv(rows: Iterable, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean, _ = sanitize(rows)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["marker_id", "x", "y"])
        for mid, x, y in clean:
            w.writerow([mid, f"{x:.4f}", f"{y:.4f}"])


def read_csv(path) -> List[Row]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r.get("marker_id"), r.get("x"), r.get("y")))
    clean, _ = sanitize(rows)
    return clean


def mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None
