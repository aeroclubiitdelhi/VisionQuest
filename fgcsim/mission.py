"""Mission sources (live simulation or a recorded mission folder) and recording.

A recorded mission folder looks like:
    frames/00000.jpg ...     exact JPEG bytes the camera produced
    telemetry.csv            one row per camera tick (also for dropped frames)
    meta.json                camera_info, rules, info
    (no answers are stored; for practice worlds the score is computed by regenerating
     the world from its part and seed, which are in meta.json)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import __version__
from .env import FieldSim, Observation

TELEMETRY_FIELDS = ["t", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"]


class LiveSource:
    def __init__(self, sim: FieldSim):
        self.sim = sim
        self.camera_info = sim.camera_info
        self.rules = sim.rules
        self.info = sim.info

    def reset(self) -> Observation:
        return self.sim.reset()

    def step(self) -> Observation:
        return self.sim.step()

    @property
    def flight_time(self) -> float:
        return self.sim.t


class RecordedSource:
    def __init__(self, folder):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "meta.json").read_text())
        self.camera_info = meta["camera_info"]
        self.rules = meta["rules"]
        self.info = meta["info"]
        self.meta = meta
        self.rows: List[Dict] = []
        with open(self.folder / "telemetry.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                self.rows.append(r)
        if not self.rows:
            raise ValueError(f"{folder}: empty telemetry.csv")
        self.i = 0

    def _obs(self, i: int) -> Observation:
        r = self.rows[i]
        idx = int(r["index"])
        jpeg = None
        if r["has_frame"] == "1":
            jpeg = (self.folder / "frames" / f"{idx:05d}.jpg").read_bytes()
        telem = {k: float(r[k]) for k in TELEMETRY_FIELDS}
        return Observation(idx, telem["t"], jpeg, telem, i == len(self.rows) - 1)

    def reset(self) -> Observation:
        self.i = 0
        return self._obs(0)

    def step(self) -> Observation:
        self.i = min(self.i + 1, len(self.rows) - 1)
        return self._obs(self.i)

    @property
    def flight_time(self) -> float:
        return float(self.rows[self.i]["t"])

    def truth(self) -> Optional[Dict]:
        """Answers for scoring: truth.json if present (organizer copies), else regenerate a
        PRACTICE world from meta.json. Hidden missions have neither."""
        p = self.folder / "truth.json"
        if p.exists():
            return json.loads(p.read_text())
        w = self.meta.get("practice_world")
        if w:
            from .world import generate_world
            world = generate_world(w["part"], w["seed"])
            return {"ground_truth": [list(r) for r in world.ground_truth()], "manifest": world.manifest(),
                    "true_path": None}
        return None


def truth_payload(sim: FieldSim) -> Dict:
    return {
        "ground_truth": [list(r) for r in sim.world.ground_truth()],
        "manifest": sim.world.manifest(),
        "true_path": [list(p) for p in sim.true_path],
        "flight_time": sim.t,
    }


def record_mission(sim: FieldSim, out_dir, truth_dir=None, progress=None) -> Dict:
    """Fly the preset path and save every camera tick. Returns summary info.

    Answers (truth.json) are written only to `truth_dir` (organizers, hidden worlds).
    Practice recordings store just the part and seed, so no answer key sits on disk.
    """
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    obs = sim.reset()
    while True:
        if obs.jpeg is not None:
            (out / "frames" / f"{obs.index:05d}.jpg").write_bytes(obs.jpeg)
        rows.append({"index": obs.index, "has_frame": int(obs.jpeg is not None),
                     **{k: obs.telemetry[k] for k in TELEMETRY_FIELDS}})
        if progress:
            progress(obs.index)
        if obs.done:
            break
        obs = sim.step()
    with open(out / "telemetry.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "has_frame"] + TELEMETRY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    meta = {"camera_info": sim.camera_info, "rules": sim.rules, "info": sim.info,
            "n_ticks": len(rows), "n_frames": sum(r["has_frame"] for r in rows),
            "flight_time": sim.t, "fgcsim_version": __version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    truth = truth_payload(sim)
    if truth_dir is not None:
        td = Path(truth_dir)
        td.mkdir(parents=True, exist_ok=True)
        (td / "truth.json").write_text(json.dumps(truth))
        cv2.imwrite(str(td / "topdown.png"), topdown_background(sim.world))
    elif sim.world.salt_is_public:
        meta["practice_world"] = {"part": sim.world.part, "seed": sim.world.seed}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def topdown_background(world, half_extent_m: float = 17.0, ppm: int = 30) -> np.ndarray:
    tex = world.texture
    tppm = world.cfg.field.texture_ppm
    n = tex.shape[0]
    c = (n - 1) / 2
    r = int(round(half_extent_m * tppm))
    crop = tex[int(c) - r:int(c) + r, int(c) - r:int(c) + r]
    size = int(2 * half_extent_m * ppm)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
