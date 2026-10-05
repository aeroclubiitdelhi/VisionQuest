"""Telemetry model: what the drone *reports* about itself (true state + errors)."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import TelemetryNoiseConfig
from .geometry import wrap_angle


class TelemetrySensor:
    def __init__(self, cfg: TelemetryNoiseConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.bias_xy = rng.normal(0, cfg.pos_bias_sigma_m, 2) if cfg.pos_bias_sigma_m > 0 else np.zeros(2)
        self.bias_alt = float(rng.normal(0, cfg.alt_bias_sigma_m)) if cfg.alt_bias_sigma_m > 0 else 0.0
        s = math.radians(cfg.att_bias_sigma_deg)
        self.att_bias = rng.normal(0, s, 3) if s > 0 else np.zeros(3)

    def advance(self, dt: float) -> None:
        """Let the slowly wandering biases evolve (Ornstein-Uhlenbeck)."""
        c = self.cfg
        tau = max(c.pos_bias_tau_s, 1e-3)
        k = math.sqrt(2 * dt / tau)
        if c.pos_bias_sigma_m > 0:
            self.bias_xy += (-self.bias_xy / tau) * dt + c.pos_bias_sigma_m * k * self.rng.normal(0, 1, 2)
        if c.alt_bias_sigma_m > 0:
            self.bias_alt += (-self.bias_alt / tau) * dt + c.alt_bias_sigma_m * k * float(self.rng.normal())

    def nav_estimate(self, p: np.ndarray) -> np.ndarray:
        """Filtered navigation estimate used by the onboard autopilot (bias, no white noise)."""
        return np.array([p[0] + self.bias_xy[0], p[1] + self.bias_xy[1], p[2] + self.bias_alt])

    def measure(self, t: float, p: np.ndarray, v: np.ndarray, attitude) -> Dict[str, float]:
        c = self.cfg
        r = self.rng
        att_s = math.radians(c.att_sigma_deg)
        n_pos = r.normal(0, c.pos_sigma_m, 2) if c.pos_sigma_m > 0 else np.zeros(2)
        n_alt = float(r.normal(0, c.alt_sigma_m)) if c.alt_sigma_m > 0 else 0.0
        n_att = r.normal(0, att_s, 3) if att_s > 0 else np.zeros(3)
        n_vel = r.normal(0, c.vel_sigma_mps, 3) if c.vel_sigma_mps > 0 else np.zeros(3)
        if c.glitch_prob > 0 and r.random() < c.glitch_prob:
            n_pos = n_pos + r.normal(0, c.glitch_sigma_m, 2)
        roll, pitch, yaw = attitude
        return {
            "t": round(float(t), 4),
            "x": float(p[0] + self.bias_xy[0] + n_pos[0]),
            "y": float(p[1] + self.bias_xy[1] + n_pos[1]),
            "z": float(p[2] + self.bias_alt + n_alt),
            "roll": float(roll + self.att_bias[0] + n_att[0]),
            "pitch": float(pitch + self.att_bias[1] + n_att[1]),
            "yaw": float(wrap_angle(yaw + self.att_bias[2] + n_att[2])),
            "vx": float(v[0] + n_vel[0]),
            "vy": float(v[1] + n_vel[1]),
            "vz": float(v[2] + n_vel[2]),
        }
