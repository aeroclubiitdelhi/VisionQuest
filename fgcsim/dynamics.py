"""Quadcopter + wind model and the preset lawnmower autopilot.

The model is intentionally simple but physically consistent:
  * a velocity controller commands horizontal acceleration (bounded),
  * wind acts through linear drag, so the drone drifts downwind and must tilt into it,
  * roll/pitch follow the thrust direction needed for that acceleration (first-order lag),
  * a fast random vibration is added to the attitude seen by the camera.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .config import FlightConfig, WindConfig
from .geometry import wrap_angle

G = 9.81


class Wind:
    def __init__(self, mean_xy, cfg: WindConfig, rng: np.random.Generator):
        self.mean = np.asarray(mean_xy, dtype=float)
        self.sigma = cfg.gust_sigma_mps
        self.tau = max(cfg.gust_tau_s, 1e-3)
        self.rng = rng
        self.gust = self.rng.normal(0, self.sigma, 2) if self.sigma > 0 else np.zeros(2)

    def step(self, dt: float) -> np.ndarray:
        if self.sigma > 0:
            self.gust += (-self.gust / self.tau) * dt + \
                self.sigma * math.sqrt(2 * dt / self.tau) * self.rng.normal(0, 1, 2)
        return self.mean + self.gust


class Drone:
    KP_V = 2.0          # velocity loop gain (1/s)
    KP_Z = 2.0
    K_DRAG = 0.35       # linear drag coefficient (1/s)
    TAU_ATT = 0.12      # attitude response time constant (s)
    MAX_ACC = 4.0       # m/s^2
    MAX_TILT = math.radians(25)

    def __init__(self, p0, yaw0: float, cfg: FlightConfig, jitter_rng: np.random.Generator):
        self.cfg = cfg
        self.p = np.asarray(p0, dtype=float).copy()
        self.v = np.zeros(3)
        self.yaw = float(yaw0)
        self.roll = 0.0
        self.pitch = 0.0
        self.rng = jitter_rng
        self.jitter = np.zeros(3)
        self.j_sigma = math.radians(cfg.attitude_jitter_deg)
        self.j_tau = max(cfg.attitude_jitter_tau_s, 1e-3)

    @property
    def attitude(self) -> Tuple[float, float, float]:
        """True camera attitude (roll, pitch, yaw) including vibration."""
        return (self.roll + self.jitter[0], self.pitch + self.jitter[1],
                wrap_angle(self.yaw + self.jitter[2] * 0.5))

    def step(self, dt: float, v_cmd, yaw_rate_cmd: float, wind_xy) -> None:
        c = self.cfg
        v_cmd = np.asarray(v_cmd, dtype=float).copy()
        hs = np.linalg.norm(v_cmd[:2])
        if hs > c.max_speed_mps:
            v_cmd[:2] *= c.max_speed_mps / hs
        v_cmd[2] = float(np.clip(v_cmd[2], -c.max_climb_mps, c.max_climb_mps))
        yaw_rate = float(np.clip(yaw_rate_cmd, -c.max_yaw_rate, c.max_yaw_rate))

        a_ctrl = self.KP_V * (v_cmd[:2] - self.v[:2])
        an = np.linalg.norm(a_ctrl)
        if an > self.MAX_ACC:
            a_ctrl *= self.MAX_ACC / an
        a_h = a_ctrl + self.K_DRAG * (np.asarray(wind_xy) - self.v[:2])
        a_z = float(np.clip(self.KP_Z * (v_cmd[2] - self.v[2]), -3.0, 3.0))

        self.v += np.array([a_h[0], a_h[1], a_z]) * dt
        self.p += self.v * dt
        self._enforce_limits()
        self.yaw = wrap_angle(self.yaw + yaw_rate * dt)

        # Attitude needed to produce the commanded acceleration (thrust direction).
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        fx = cy * a_ctrl[0] + sy * a_ctrl[1]
        fy = -sy * a_ctrl[0] + cy * a_ctrl[1]
        fz = G + a_z
        fn = math.sqrt(fx * fx + fy * fy + fz * fz)
        roll_des = math.asin(float(np.clip(-fy / fn, -1, 1)))
        pitch_des = math.atan2(fx, fz)
        roll_des = float(np.clip(roll_des, -self.MAX_TILT, self.MAX_TILT))
        pitch_des = float(np.clip(pitch_des, -self.MAX_TILT, self.MAX_TILT))
        k = min(1.0, dt / self.TAU_ATT)
        self.roll += (roll_des - self.roll) * k
        self.pitch += (pitch_des - self.pitch) * k

        if self.j_sigma > 0:
            self.jitter += (-self.jitter / self.j_tau) * dt + \
                self.j_sigma * math.sqrt(2 * dt / self.j_tau) * self.rng.normal(0, 1, 3)

    def _enforce_limits(self) -> None:
        c = self.cfg
        g = c.geofence_half_size_m
        for i in (0, 1):
            if abs(self.p[i]) > g:
                self.p[i] = math.copysign(g, self.p[i])
                if self.p[i] * self.v[i] > 0:
                    self.v[i] = 0.0
        if self.p[2] < c.min_altitude_m:
            self.p[2] = c.min_altitude_m
            self.v[2] = max(self.v[2], 0.0)
        elif self.p[2] > c.max_altitude_m:
            self.p[2] = c.max_altitude_m
            self.v[2] = min(self.v[2], 0.0)


class LawnmowerAutopilot:
    """Pure-pursuit waypoint follower over back-and-forth lanes (the preset path)."""

    ACCEPT_RADIUS = 1.0
    LOOKAHEAD = 3.0

    def __init__(self, cfg: FlightConfig):
        self.cfg = cfg
        L = cfg.lane_half_length_m
        wps: List[Tuple[float, float]] = []
        for i, y in enumerate(cfg.lane_y_m):
            xs = (-L, L) if i % 2 == 0 else (L, -L)
            wps += [(xs[0], y), (xs[1], y)]
        self.waypoints = np.array(wps, dtype=float)
        self.idx = 1
        self.finished = False

    @property
    def start(self) -> np.ndarray:
        return self.waypoints[0]

    def command(self, nav_p: np.ndarray, yaw: float):
        """Returns (v_cmd[3], yaw_rate). Uses the (possibly biased) navigation estimate."""
        c = self.cfg
        vz = float(np.clip(1.2 * (c.altitude_m - nav_p[2]), -c.max_climb_mps, c.max_climb_mps))
        if self.finished:
            return np.array([0.0, 0.0, vz]), 0.0
        pos = nav_p[:2]
        target = self.waypoints[self.idx]
        if np.linalg.norm(target - pos) < self.ACCEPT_RADIUS:
            self.idx += 1
            if self.idx >= len(self.waypoints):
                self.finished = True
                return np.array([0.0, 0.0, vz]), 0.0
            target = self.waypoints[self.idx]
        prev = self.waypoints[self.idx - 1]
        seg = target - prev
        seg_len = float(np.linalg.norm(seg))
        u = seg / seg_len if seg_len > 1e-9 else np.zeros(2)
        along = float(np.clip(np.dot(pos - prev, u), 0.0, seg_len))
        look = prev + u * min(seg_len, along + self.LOOKAHEAD)
        d = look - pos
        dist_to_target = float(np.linalg.norm(target - pos))
        speed = c.speed_mps * float(np.clip(dist_to_target / 3.0, 0.35, 1.0))
        dn = float(np.linalg.norm(d))
        v_xy = d / dn * speed if dn > 1e-9 else np.zeros(2)

        heading = math.atan2(u[1], u[0]) if seg_len > 1e-9 else yaw
        yaw_rate = float(np.clip(1.5 * wrap_angle(heading - yaw), -c.max_yaw_rate, c.max_yaw_rate))
        return np.array([v_xy[0], v_xy[1], vz]), yaw_rate
