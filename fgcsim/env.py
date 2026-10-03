"""FieldSim: the simulation participants run.

    sim = FieldSim(part="B", seed=3)
    obs = sim.reset()
    while not obs.done:
        ...                       # obs.frame (BGR image or None), obs.telemetry (dict)
        obs = sim.step()

The drone flies the preset lawnmower path. Physics runs at 50 Hz, the camera at
`fps` (10 Hz by default).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from .config import PUBLIC_SALT, PartConfig, get_part
from .disturbances import ImagePipeline
from .dynamics import Drone, LawnmowerAutopilot, Wind
from .geometry import R_BODY_CAM, camera_matrix, project_points, rot_world_cam
from .render import Renderer
from .sensors import TelemetrySensor
from .world import World, generate_world


@dataclass
class Observation:
    index: int
    t: float
    jpeg: Optional[bytes]          # None when the frame was dropped
    telemetry: Dict[str, float]
    done: bool

    @property
    def frame(self) -> Optional[np.ndarray]:
        return decode_jpeg(self.jpeg)


def decode_jpeg(jpeg: Optional[bytes]) -> Optional[np.ndarray]:
    if jpeg is None:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class FieldSim:
    PHYS_DT = 0.02

    def __init__(self, part: str = "A", seed: int = 0, salt: str = PUBLIC_SALT,
                 cfg: Optional[PartConfig] = None):
        self.cfg = cfg or get_part(part)
        self.world: World = generate_world(self.cfg.name, seed, salt, self.cfg)
        self.renderer = Renderer(self.world)
        cam = self.cfg.camera
        self.steps_per_frame = int(round(1.0 / (cam.fps * self.PHYS_DT)))
        self.frame_dt = self.steps_per_frame * self.PHYS_DT
        self.time_limit = self.cfg.flight.time_limit_s

        self.camera_info = {
            "width": cam.width,
            "height": cam.height,
            "K": camera_matrix(cam.fx, cam.fy, cam.cx, cam.cy).tolist(),
            "dist": [float(v) for v in self.cfg.image.distortion],
            "R_body_cam": R_BODY_CAM.tolist(),
            "fps": cam.fps,
        }
        self.rules = {
            "dictionary": self.cfg.markers.dictionary,
            "valid_ids": list(self.cfg.markers.valid_ids),
            "marker_size_m": self.cfg.markers.size_m,
            "field_half_size_m": self.cfg.field.half_size_m,
        }
        self.info = {
            "part": self.cfg.name,
            "title": self.cfg.title,
            "frame_rate": cam.fps,
            "altitude_m": self.cfg.flight.altitude_m,
            "speed_mps": self.cfg.flight.speed_mps,
        }
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> Observation:
        w = self.world
        fc = self.cfg.flight
        self.autopilot = LawnmowerAutopilot(fc)
        start = self.autopilot.start
        self.drone = Drone([start[0], start[1], fc.altitude_m], 0.0, fc, w.rng("jitter"))
        self.wind = Wind(w.conditions["wind_mean"], self.cfg.wind, w.rng("wind"))
        self.sensor = TelemetrySensor(self.cfg.telemetry, w.rng("telemetry"))
        self.pipeline = ImagePipeline(w, w.rng("image"))
        self.t = 0.0
        self.index = 0
        self.done = False
        self.true_path = []
        self.history = deque(maxlen=64)
        self._record_history()
        self.last = self._observe()
        return self.last

    def step(self) -> Observation:
        if self.done:
            return self.last
        for _ in range(self.steps_per_frame):
            v_cmd, yaw_rate = self.autopilot.command(self.sensor.nav_estimate(self.drone.p), self.drone.yaw)
            wind = self.wind.step(self.PHYS_DT)
            self.drone.step(self.PHYS_DT, v_cmd, yaw_rate, wind)
            self.sensor.advance(self.PHYS_DT)
            self.t = round(self.t + self.PHYS_DT, 6)
            self._record_history()
        self.index += 1
        if self.autopilot.finished or self.t >= self.time_limit - 1e-9:
            self.done = True
        self.last = self._observe()
        return self.last

    @property
    def flight_time(self) -> float:
        return self.t

    # ------------------------------------------------------------------ internals
    def _record_history(self) -> None:
        roll, pitch, yaw = self.drone.attitude
        self.history.append((self.t, self.drone.p.copy(), np.array([roll, pitch, yaw])))
        if not self.true_path or self.t - self.true_path[-1][0] >= 0.1 - 1e-9:
            self.true_path.append((self.t, float(self.drone.p[0]), float(self.drone.p[1]), float(self.drone.p[2])))

    def _pose_at(self, t: float):
        hist = self.history
        if t <= hist[0][0]:
            return hist[0][1], hist[0][2]
        for (t0, p0, a0), (t1, p1, a1) in zip(list(hist)[:-1], list(hist)[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / max(t1 - t0, 1e-9)
                da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
                return p0 + k * (p1 - p0), a0 + k * da
        return hist[-1][1], hist[-1][2]

    def _observe(self) -> Observation:
        attitude = self.drone.attitude
        telem = self.sensor.measure(self.t, self.drone.p, self.drone.v, attitude)
        jpeg = None
        if not self.pipeline.should_drop() or self.index == 0:
            jpeg = self._render_frame(self.drone.p.copy(), attitude)
        return Observation(self.index, round(self.t, 4), jpeg, telem, self.done)

    def _render_frame(self, p: np.ndarray, attitude) -> bytes:
        r = self.renderer
        R_wc = rot_world_cam(*attitude)
        img = r.render(p, R_wc, self.t)
        strength = self.cfg.image.cloud_shadow_strength
        if strength > 0:
            img = r.cloud_shadow(img, p, R_wc, self.t, strength)
        exposure = self.world.conditions["exposure_s"]
        axis = R_wc[:, 2]
        lam = -p[2] / axis[2]
        ground = p + lam * axis
        p_old, a_old = self._pose_at(self.t - exposure)
        uv_old, _ = project_points(r.K_pad, rot_world_cam(*a_old), p_old, ground[None, :])
        centre = np.array([r.K_pad[0, 2], r.K_pad[1, 2]])
        img = self.pipeline.blur(img, centre - uv_old[0])
        img = r.distort(img)
        img = self.pipeline.photometric(img, self.t)
        return self.pipeline.encode(img)
