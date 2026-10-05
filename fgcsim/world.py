"""Seeded world generation: marker layout (genuine + fakes), ground texture, conditions.

A world is fully determined by (salt, part, seed). Participants use the public
salt; hidden evaluation worlds use a secret salt, so knowing the simulator code
does not reveal the hidden marker layouts.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .aruco_utils import SAFE_FAKE_IDS
from .config import PUBLIC_SALT, PartConfig, get_part

STREAMS = ("layout", "texture", "conditions", "wind", "telemetry", "jitter", "image", "clouds")


def derive_seed(salt: str, part: str, seed: int) -> int:
    digest = hashlib.sha256(f"{salt}|{part.upper()}|{int(seed)}".encode()).digest()
    return int.from_bytes(digest[:16], "little")


@dataclass
class Marker:
    marker_id: int
    dictionary: str
    size_m: float
    x: float
    y: float
    yaw: float
    genuine: bool
    kind: str                       # genuine | wrong_dictionary | invalid_id | wrong_size | duplicate | moving
    motion: Optional[Dict] = None   # moving markers: axis, amplitude, omega, phase

    def pose_at(self, t: float) -> Tuple[float, float, float]:
        if not self.motion:
            return self.x, self.y, self.yaw
        m = self.motion
        s = m["amplitude"] * np.sin(m["omega"] * t + m["phase"])
        return self.x + s * np.cos(m["axis"]), self.y + s * np.sin(m["axis"]), self.yaw

    def path_points(self, n: int = 9) -> np.ndarray:
        if not self.motion:
            return np.array([[self.x, self.y]])
        m = self.motion
        s = np.linspace(-m["amplitude"], m["amplitude"], n)
        return np.column_stack([self.x + s * np.cos(m["axis"]), self.y + s * np.sin(m["axis"])])


@dataclass
class World:
    part: str
    seed: int
    salt_is_public: bool
    cfg: PartConfig
    root_seed: int
    markers: List[Marker]
    conditions: Dict
    texture: np.ndarray = field(repr=False)
    cloud_field: Optional[np.ndarray] = field(default=None, repr=False)

    def rng(self, stream: str) -> np.random.Generator:
        idx = STREAMS.index(stream)
        return np.random.default_rng(np.random.SeedSequence(self.root_seed).spawn(len(STREAMS))[idx])

    def ground_truth(self) -> List[Tuple[int, float, float]]:
        return sorted((m.marker_id, float(m.x), float(m.y)) for m in self.markers if m.genuine)

    def manifest(self) -> Dict:
        return {
            "part": self.part,
            "seed": self.seed,
            "public": self.salt_is_public,
            "conditions": self.conditions,
            "markers": [asdict(m) for m in self.markers],
        }


# ----------------------------------------------------------------------------- layout

def _clear(pts: np.ndarray, placed: List[np.ndarray], spacing: float) -> bool:
    for other in placed:
        d = np.linalg.norm(pts[:, None, :] - other[None, :, :], axis=2)
        if d.min() < spacing:
            return False
    return True


def _place(rng, cfg: PartConfig, placed: List[np.ndarray], motion_amp: float = 0.0):
    lim = cfg.field.half_size_m - cfg.field.marker_margin_m
    for _ in range(4000):
        x, y = rng.uniform(-lim, lim, 2)
        if motion_amp > 0:
            axis = rng.uniform(0, np.pi)
            s = np.linspace(-motion_amp, motion_amp, 9)
            pts = np.column_stack([x + s * np.cos(axis), y + s * np.sin(axis)])
            if np.abs(pts).max() > lim:
                continue
        else:
            axis = 0.0
            pts = np.array([[x, y]])
        if _clear(pts, placed, cfg.markers.min_spacing_m):
            placed.append(pts)
            return float(x), float(y), float(axis)
    raise RuntimeError("Could not place all markers; field too crowded for this config")


def _layout(cfg: PartConfig, rng: np.random.Generator) -> List[Marker]:
    mc, fc = cfg.markers, cfg.fakes
    placed: List[np.ndarray] = []
    markers: List[Marker] = []

    n_gen = int(rng.integers(mc.n_genuine[0], mc.n_genuine[1] + 1))
    valid = np.array(mc.valid_ids)
    gen_ids = rng.choice(valid, size=n_gen, replace=False)
    spare = [int(i) for i in rng.permutation(valid) if i not in set(gen_ids.tolist())]

    def spare_id() -> int:
        if not spare:
            raise RuntimeError("Not enough unused valid IDs for fakes; enlarge valid_ids")
        return spare.pop()

    # Moving fakes first (they need the most room).
    for _ in range(fc.moving):
        amp = float(rng.uniform(*fc.moving_amplitude_m))
        speed = float(rng.uniform(*fc.moving_speed_mps))
        x, y, axis = _place(rng, cfg, placed, motion_amp=amp)
        motion = {"axis": axis, "amplitude": amp, "omega": speed / amp,
                  "phase": float(rng.uniform(0, 2 * np.pi))}
        markers.append(Marker(spare_id(), mc.dictionary, mc.size_m, x, y,
                              float(rng.uniform(0, 2 * np.pi)), False, "moving", motion))

    for mid in gen_ids:
        x, y, _ = _place(rng, cfg, placed)
        markers.append(Marker(int(mid), mc.dictionary, mc.size_m, x, y,
                              float(rng.uniform(0, 2 * np.pi)), True, "genuine"))

    for _ in range(fc.wrong_dictionary):
        dname = str(rng.choice(fc.fake_dictionaries))
        ids = SAFE_FAKE_IDS[dname]
        x, y, _ = _place(rng, cfg, placed)
        markers.append(Marker(int(rng.choice(ids)), dname, mc.size_m, x, y,
                              float(rng.uniform(0, 2 * np.pi)), False, "wrong_dictionary"))

    for mid in rng.choice(np.array(mc.invalid_ids), size=fc.invalid_id, replace=False):
        x, y, _ = _place(rng, cfg, placed)
        markers.append(Marker(int(mid), mc.dictionary, mc.size_m, x, y,
                              float(rng.uniform(0, 2 * np.pi)), False, "invalid_id"))

    for _ in range(fc.wrong_size):
        factor = float(rng.choice(fc.wrong_size_factors))
        x, y, _ = _place(rng, cfg, placed)
        markers.append(Marker(spare_id(), mc.dictionary, mc.size_m * factor, x, y,
                              float(rng.uniform(0, 2 * np.pi)), False, "wrong_size"))

    for _ in range(fc.duplicate_pairs):
        mid = spare_id()
        for _k in range(2):
            x, y, _ = _place(rng, cfg, placed)
            markers.append(Marker(mid, mc.dictionary, mc.size_m, x, y,
                                  float(rng.uniform(0, 2 * np.pi)), False, "duplicate"))
    return markers


# ----------------------------------------------------------------------------- textures

def _fbm(rng, n: int, octaves: Tuple[Tuple[int, float], ...]) -> np.ndarray:
    """Smooth multi-scale noise in [-1, 1] (roughly), size n x n."""
    out = np.zeros((n, n), np.float32)
    for cells, amp in octaves:
        small = rng.normal(0, 1, (cells, cells)).astype(np.float32)
        out += amp * cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    return out / max(1e-6, sum(a for _, a in octaves))


def make_ground_texture(cfg: PartConfig, rng: np.random.Generator) -> np.ndarray:
    fcfg = cfg.field
    ppm = fcfg.texture_ppm
    n = int(round(2 * fcfg.texture_half_extent_m * ppm))
    coords = (np.arange(n) - (n - 1) / 2) / ppm
    X, Y = np.meshgrid(coords, -coords)

    base = np.array([62, 128, 74], np.float32)          # BGR grass
    tone = _fbm(rng, n, ((6, 1.0), (24, 0.6), (96, 0.35)))
    img = base[None, None, :] * (1.0 + 0.16 * tone[..., None])

    # Mowing stripes inside the field.
    inside = (np.abs(X) <= fcfg.half_size_m + 1.0) & (np.abs(Y) <= fcfg.half_size_m + 1.0)
    stripes = np.where((np.floor((Y + 100) / 3.0) % 2) == 0, 1.05, 0.95).astype(np.float32)
    img *= np.where(inside, stripes, 0.88)[..., None]

    # Dirt patches.
    dirt_mask = np.clip((_fbm(rng, n, ((5, 1.0), (20, 0.5))) - 0.55) * 4.0, 0, 1)
    dirt = np.array([70, 100, 128], np.float32)
    img = img * (1 - dirt_mask[..., None]) + dirt[None, None, :] * dirt_mask[..., None]

    # Fine grain.
    img += rng.normal(0, 7, (n, n, 1)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    def to_px(x, y):
        return int(round((n - 1) / 2 + x * ppm)), int(round((n - 1) / 2 - y * ppm))

    # Painted boundary line around the field.
    h = fcfg.half_size_m + 0.6
    cv2.rectangle(img, to_px(-h, h), to_px(h, -h), (225, 228, 228), max(2, int(0.08 * ppm)))

    # Distractors: blank white sheets, dark tiles, small stones / flowers.
    lim = fcfg.texture_half_extent_m - 2
    for _ in range(10):
        cx, cy = rng.uniform(-lim, lim, 2)
        s = rng.uniform(0.25, 0.6) * ppm / 2
        ang = rng.uniform(0, 180)
        box = cv2.boxPoints(((*to_px(cx, cy),), (2 * s, 2 * s), ang)).astype(np.int32)
        cv2.fillConvexPoly(img, box, (230, 232, 232) if rng.random() < 0.6 else (40, 42, 45))
    for _ in range(900):
        cx, cy = rng.uniform(-lim, lim, 2)
        col = (200, 210, 215) if rng.random() < 0.7 else (90, 180, 230)
        cv2.circle(img, to_px(cx, cy), int(rng.integers(1, 3)), col, -1)
    return img


def make_cloud_field(rng: np.random.Generator, n: int = 256) -> np.ndarray:
    """Tileable soft cloud-shadow mask in [0, 1] (1 = full shadow)."""
    noise = rng.normal(0, 1, (n, n))
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    spectrum = np.fft.fft2(noise) * np.exp(-(f / 0.02) ** 2)
    field_ = np.real(np.fft.ifft2(spectrum))
    field_ = (field_ - field_.mean()) / (field_.std() + 1e-9)
    mask = np.clip((field_ - 0.3) / 0.8, 0, 1)
    mask = mask * mask * (3 - 2 * mask)   # smoothstep
    return mask.astype(np.float32)


# ----------------------------------------------------------------------------- conditions

def _sample_conditions(cfg: PartConfig, rng: np.random.Generator) -> Dict:
    ic = cfg.image
    wind_dir = float(rng.uniform(0, 2 * np.pi))
    wind_speed = float(rng.uniform(*cfg.wind.mean_speed_mps))
    return {
        "wind_mean": [wind_speed * np.cos(wind_dir), wind_speed * np.sin(wind_dir)],
        "wind_speed": wind_speed,
        "exposure_s": float(rng.uniform(*ic.exposure_s)),
        "noise_sigma": float(rng.uniform(*ic.noise_sigma)),
        "contrast": float(rng.uniform(*ic.contrast)),
        "jpeg_quality": int(rng.integers(ic.jpeg_quality[0], ic.jpeg_quality[1] + 1)),
        "gain_phase": float(rng.uniform(0, 2 * np.pi)),
        "gain_period_s": float(rng.uniform(25, 45)),
        "color_cast": [float(v) for v in rng.uniform(0.94, 1.06, 3)],
        "cloud_velocity": [float(v) for v in rng.uniform(-2.5, 2.5, 2)],
        "cloud_offset": [float(v) for v in rng.uniform(0, 128, 2)],
    }


def generate_world(part: str, seed: int, salt: str = PUBLIC_SALT,
                   cfg: Optional[PartConfig] = None) -> World:
    cfg = cfg or get_part(part)
    root = derive_seed(salt, cfg.name, seed)
    world = World(part=cfg.name, seed=int(seed), salt_is_public=(salt == PUBLIC_SALT), cfg=cfg,
                  root_seed=root, markers=[], conditions={}, texture=np.zeros((1, 1, 3), np.uint8))
    world.markers = _layout(cfg, world.rng("layout"))
    world.conditions = _sample_conditions(cfg, world.rng("conditions"))
    world.texture = make_ground_texture(cfg, world.rng("texture"))
    if cfg.image.cloud_shadow_strength > 0:
        world.cloud_field = make_cloud_field(world.rng("clouds"))
    return world
