"""Top-down replay video of a mission.

Left: the field from above, the drone flying its path (with trail), the camera's view
area sweeping the ground, markers lighting up the first time the camera sees them
(green = genuine, red = fake). At the end the team's answers appear on the map
(blue = correct, magenta = wrong) with the score.
Right: the drone's camera image and live stats.

Used by `python run.py ... --replay-video`. Pure OpenCV; uses ffmpeg (H.264) when
available so the video plays in browsers, else OpenCV's mp4v.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import ground_homography, project_points, rot_world_cam

MAP_PX = 720
HALF = 17.0
PPM = MAP_PX / (2 * HALF)
PANEL_W = 480
W, H = MAP_PX + PANEL_W, MAP_PX

GREEN = (90, 225, 90)
RED = (70, 70, 235)
GREY = (150, 150, 150)
BLUE = (255, 170, 40)
MAGENTA = (220, 70, 220)
YELLOW = (0, 215, 255)
WHITE = (240, 240, 240)
DARK = (28, 28, 28)


def _px(x, y):
    return int(round(MAP_PX / 2 + x * PPM)), int(round(MAP_PX / 2 - y * PPM))


class ReplayTap:
    """Wraps a mission source and records what is needed for the replay video."""

    def __init__(self, source, true_path=None):
        self.src = source
        self.camera_info, self.rules, self.info = source.camera_info, source.rules, source.info
        self.ticks: List[Dict] = []
        self.sim = getattr(source, "sim", None)
        # Recorded missions: true positions (t, x, y, z) from truth.json, if available.
        self.true_pos = {round(r[0], 1): np.array(r[1:4], float) for r in (true_path or [])}

    def _record(self, obs):
        tick = {"t": obs.t, "tel": dict(obs.telemetry), "jpeg": obs.jpeg}
        if self.sim is not None:                       # live run: true pose available
            tick["p"] = self.sim.drone.p.copy()
            tick["att"] = tuple(self.sim.drone.attitude)
        elif round(obs.t, 1) in self.true_pos:         # recorded run: true position, reported attitude
            tick["p"] = self.true_pos[round(obs.t, 1)]
        self.ticks.append(tick)
        return obs

    def reset(self):
        return self._record(self.src.reset())

    def step(self):
        return self._record(self.src.step())

    @property
    def flight_time(self):
        return self.src.flight_time

    def __getattr__(self, name):          # forward anything else (e.g. inner taps)
        return getattr(self.src, name)


class _Writer:
    def __init__(self, path, fps, size=(W, H)):
        self.proc = None
        self.cv = None
        w, h = size
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast", str(path)],
                stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def _footprint(K, size, p, att):
    """Ground polygon (world x, y) seen by the camera, or None if it looks above the horizon."""
    w, h = size
    R = rot_world_cam(*att)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = np.column_stack([(corners - K[:2, 2]) / [K[0, 0], K[1, 1]], np.ones(4)]) @ R.T
    if np.any(rays[:, 2] >= -1e-6):
        return None
    lam = -p[2] / rays[:, 2]
    return p[None, :2] + lam[:, None] * rays[:, :2]


def _visible(K, size, p, att, xy):
    uv, z = project_points(K, rot_world_cam(*att), p, np.array([[xy[0], xy[1], 0.0]]))
    u, v = uv[0]
    return z[0] > 0 and 0 <= u < size[0] and 0 <= v < size[1]


def _draw_drone(img, p, yaw):
    c = _px(p[0], p[1])
    r = 11
    for k in range(4):                                 # rotors
        a = yaw + math.pi / 4 + k * math.pi / 2
        q = (int(c[0] + r * math.cos(a)), int(c[1] - r * math.sin(a)))
        cv2.line(img, c, q, WHITE, 2, cv2.LINE_AA)
        cv2.circle(img, q, 5, WHITE, 1, cv2.LINE_AA)
    nose = (int(c[0] + 16 * math.cos(yaw)), int(c[1] - 16 * math.sin(yaw)))
    cv2.arrowedLine(img, c, nose, YELLOW, 2, cv2.LINE_AA, tipLength=0.4)


def render_replay(tap: ReplayTap, manifest: Dict, background: Optional[np.ndarray], answers,
                  score: Optional[Dict], out_path, title: str = "", speed: float = 3.0,
                  out_fps: int = 15) -> None:
    ci = tap.camera_info
    K = np.array(ci["K"], float)
    size = (ci["width"], ci["height"])
    markers = manifest["markers"]

    if background is not None:
        base = cv2.resize(background, (MAP_PX, MAP_PX), interpolation=cv2.INTER_AREA)
        base = (base.astype(np.float32) * 0.6).astype(np.uint8)
    else:
        base = np.full((MAP_PX, MAP_PX, 3), (40, 80, 45), np.uint8)
    cv2.rectangle(base, _px(-15, 15), _px(15, -15), (205, 205, 205), 1, cv2.LINE_AA)

    cam_fps = float(ci.get("fps", 10))
    step = max(1, int(round(speed * cam_fps / out_fps)))
    writer = _Writer(out_path, out_fps)
    seen_at: Dict[int, float] = {}
    trail: List[tuple] = []
    last_cam = np.zeros((size[1], size[0], 3), np.uint8)
    n_gen = sum(1 for m in markers if m["genuine"])

    def marker_pos(m, t):
        x, y = m["x"], m["y"]
        if m.get("motion"):
            mo = m["motion"]
            s = mo["amplitude"] * math.sin(mo["omega"] * t + mo["phase"])
            x, y = x + s * math.cos(mo["axis"]), y + s * math.sin(mo["axis"])
        return x, y

    def frame(i, final=False):
        tick = tap.ticks[i]
        t = tick["t"]
        tel = tick["tel"]
        p = tick.get("p", np.array([tel["x"], tel["y"], tel["z"]]))
        att = tick.get("att", (tel["roll"], tel["pitch"], tel["yaw"]))
        img = base.copy()

        poly = None if final else _footprint(K, size, p, att)
        if poly is not None:
            pts = np.array([_px(x, y) for x, y in poly], np.int32)
            overlay = img.copy()
            cv2.fillConvexPoly(overlay, pts, (120, 200, 255))
            img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
            cv2.polylines(img, [pts], True, (120, 200, 255), 1, cv2.LINE_AA)

        if len(trail) > 1:
            cv2.polylines(img, [np.array(trail, np.int32)], False, (210, 210, 210), 1, cv2.LINE_AA)

        for idx, m in enumerate(markers):
            x, y = marker_pos(m, t)
            c = _px(x, y)
            r = max(4, int(m["size_m"] / 2 * PPM) + 2)
            lit = idx in seen_at
            col = (GREEN if m["genuine"] else RED) if lit else GREY
            cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, 2 if lit else 1, cv2.LINE_AA)
            if lit:
                age = t - seen_at[idx]
                if age < 1.0 and not final:          # flash when first seen
                    cv2.circle(img, c, int(r + 6 + 14 * age), col, 1, cv2.LINE_AA)
                label = str(m["marker_id"]) if m["genuine"] else f"x{m['marker_id']}"
                cv2.putText(img, label, (c[0] + r + 3, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                            cv2.LINE_AA)

        if final and answers is not None:
            matches = score.get("matches", {}) if score else {}
            gpos = {m["marker_id"]: (m["x"], m["y"]) for m in markers if m["genuine"]}
            used = set()
            for mid, x, y in answers:
                q = _px(x, y)
                ok = (mid in matches and mid not in used and mid in gpos and
                      abs(math.hypot(x - gpos[mid][0], y - gpos[mid][1]) - matches[mid]) < 1e-3)
                if ok:
                    used.add(mid)
                    cv2.line(img, q, _px(*gpos[mid]), BLUE, 1, cv2.LINE_AA)
                    cv2.circle(img, q, 5, BLUE, -1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(img, q, MAGENTA, cv2.MARKER_TILTED_CROSS, 13, 2, cv2.LINE_AA)
        else:
            _draw_drone(img, p, att[2])

        # Right panel: camera + stats
        panel = np.full((H, PANEL_W, 3), DARK, np.uint8)
        cam_h = int(PANEL_W * size[1] / size[0])
        panel[40:40 + cam_h] = cv2.resize(last_cam, (PANEL_W, cam_h), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, "drone camera" + ("  [frame dropped]" if tick["jpeg"] is None else ""),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        n_seen_gen = sum(1 for k in seen_at if markers[k]["genuine"])
        n_seen_fake = len(seen_at) - n_seen_gen
        lines = [f"t = {t:5.1f} s", f"altitude {p[2]:.1f} m",
                 f"genuine markers seen  {n_seen_gen}/{n_gen}", f"fake markers seen     {n_seen_fake}"]
        if final and score:
            err = score["mean_error_m"]
            lines += ["", "TEAM ANSWERS", f"score {score['score']:.1f} / 100",
                      f"found {score['found']}/{score['genuine']}   wrong {score['false_positives']}",
                      f"mean error {err if err is not None else '-'} m"]
        y0 = 40 + cam_h + 32
        for k, txt in enumerate(lines):
            col = YELLOW if txt in ("TEAM ANSWERS",) else WHITE
            cv2.putText(panel, txt, (14, y0 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)
        leg = [(GREY, "not seen yet"), (GREEN, "genuine (seen)"), (RED, "fake (seen)")]
        if final:
            leg = [(GREEN, "genuine"), (RED, "fake"), (BLUE, "team: correct"), (MAGENTA, "team: wrong")]
        for k, (col, txt) in enumerate(leg):            # two-column legend at the bottom
            xx = 14 + (k % 2) * 230
            yy = H - 40 + (k // 2) * 22
            cv2.rectangle(panel, (xx, yy - 11), (xx + 12, yy + 1), col, -1)
            cv2.putText(panel, txt, (xx + 20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        out = np.hstack([img, panel])
        cv2.rectangle(out, (0, 0), (MAP_PX, 30), DARK, -1)
        cv2.putText(out, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        return out

    try:
        last = None
        for i, tick in enumerate(tap.ticks):
            tel = tick["tel"]
            p = tick.get("p", np.array([tel["x"], tel["y"], tel["z"]]))
            att = tick.get("att", (tel["roll"], tel["pitch"], tel["yaw"]))
            if tick["jpeg"] is not None:
                last_cam = cv2.imdecode(np.frombuffer(tick["jpeg"], np.uint8), cv2.IMREAD_COLOR)
                for idx, m in enumerate(markers):
                    if idx not in seen_at and _visible(K, size, p, att, marker_pos(m, tick["t"])):
                        seen_at[idx] = tick["t"]
            trail.append(_px(p[0], p[1]))
            if i % step == 0:
                last = frame(i)
                writer.write(last)
        end = frame(len(tap.ticks) - 1, final=True)
        for _ in range(out_fps * 4):                  # hold the result for 4 s
            writer.write(end)
    finally:
        writer.close()
