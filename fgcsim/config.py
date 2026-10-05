"""All simulation parameters for every part, in one place.

Ranges written as (low, high) are sampled once per world from the world's
random stream, so every seed gets slightly different conditions while the
overall difficulty of a part stays the same.

Change values here BEFORE releasing the participant kit. After release,
keep them frozen: hidden evaluation worlds use exactly these settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, Tuple

Range = Tuple[float, float]

PUBLIC_SALT = "fgc-public-v1"


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fx: float = 520.0
    fy: float = 520.0
    cx: float = 319.5
    cy: float = 239.5
    fps: float = 10.0
    supersample: int = 2          # render at 2x then downscale (anti-aliasing)


@dataclass(frozen=True)
class FieldConfig:
    half_size_m: float = 15.0     # markers live inside [-15, 15] x [-15, 15]
    marker_margin_m: float = 1.5  # keep markers this far from the field edge
    texture_half_extent_m: float = 26.0
    texture_ppm: int = 40         # ground texture resolution (pixels per meter)


@dataclass(frozen=True)
class MarkerConfig:
    dictionary: str = "DICT_4X4_50"
    valid_ids: Tuple[int, ...] = tuple(range(35))       # 0..34 are valid
    invalid_ids: Tuple[int, ...] = tuple(range(35, 50))  # same dictionary, not valid
    size_m: float = 0.5           # side of the black square
    n_genuine: Tuple[int, int] = (14, 18)
    min_spacing_m: float = 2.2


@dataclass(frozen=True)
class FakeConfig:
    wrong_dictionary: int = 0
    invalid_id: int = 0
    wrong_size: int = 0
    duplicate_pairs: int = 0      # each pair = 2 markers sharing one valid ID
    moving: int = 0
    wrong_size_factors: Tuple[float, ...] = (0.6, 1.6, 2.0)
    moving_speed_mps: Range = (0.7, 1.3)
    moving_amplitude_m: Range = (2.5, 4.0)
    fake_dictionaries: Tuple[str, ...] = ("DICT_5X5_50", "DICT_6X6_50", "DICT_APRILTAG_36h11")


@dataclass(frozen=True)
class WindConfig:
    mean_speed_mps: Range = (0.0, 0.0)
    gust_sigma_mps: float = 0.0
    gust_tau_s: float = 3.0


@dataclass(frozen=True)
class TelemetryNoiseConfig:
    pos_sigma_m: float = 0.0          # white noise on x, y
    pos_bias_sigma_m: float = 0.0     # slowly wandering bias on x, y (GNSS-like)
    pos_bias_tau_s: float = 40.0
    alt_sigma_m: float = 0.0
    alt_bias_sigma_m: float = 0.0
    att_sigma_deg: float = 0.0        # white noise on roll, pitch, yaw
    att_bias_sigma_deg: float = 0.0   # constant per-world bias on roll, pitch, yaw
    vel_sigma_mps: float = 0.0
    glitch_prob: float = 0.0          # chance per frame of a GNSS multipath jump
    glitch_sigma_m: float = 0.0       # size of those jumps


@dataclass(frozen=True)
class ImageNoiseConfig:
    exposure_s: Range = (0.004, 0.004)    # motion blur comes from motion during exposure
    vibration_blur_px: float = 0.0
    noise_sigma: Range = (1.5, 1.5)
    gain_drift: float = 0.0               # slow brightness changes (+/- fraction)
    contrast: Range = (1.0, 1.0)
    cloud_shadow_strength: float = 0.0    # 0 = no shadows, 0.4 = shadows darken by up to 40 %
    vignetting: float = 0.0
    distortion: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    jpeg_quality: Tuple[int, int] = (92, 92)
    drop_prob: float = 0.0


@dataclass(frozen=True)
class FlightConfig:
    altitude_m: float = 8.0
    speed_mps: float = 2.5
    lane_half_length_m: float = 14.0
    lane_y_m: Tuple[float, ...] = (-12.0, -6.0, 0.0, 6.0, 12.0)
    time_limit_s: float = 240.0          # safety cap; the preset path takes ~80-110 s
    attitude_jitter_deg: float = 0.0      # high-frequency vibration of the airframe
    attitude_jitter_tau_s: float = 0.08
    max_speed_mps: float = 5.0
    max_climb_mps: float = 1.5
    max_yaw_rate: float = 0.8             # rad/s
    min_altitude_m: float = 2.0
    max_altitude_m: float = 20.0
    geofence_half_size_m: float = 20.0


@dataclass(frozen=True)
class PartConfig:
    name: str
    title: str
    camera: CameraConfig = dc_field(default_factory=CameraConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    markers: MarkerConfig = dc_field(default_factory=MarkerConfig)
    fakes: FakeConfig = dc_field(default_factory=FakeConfig)
    wind: WindConfig = dc_field(default_factory=WindConfig)
    telemetry: TelemetryNoiseConfig = dc_field(default_factory=TelemetryNoiseConfig)
    image: ImageNoiseConfig = dc_field(default_factory=ImageNoiseConfig)
    flight: FlightConfig = dc_field(default_factory=FlightConfig)


_DISTURBED_TELEMETRY = TelemetryNoiseConfig(
    pos_sigma_m=0.10, pos_bias_sigma_m=0.08, pos_bias_tau_s=40.0,
    alt_sigma_m=0.06, alt_bias_sigma_m=0.05,
    att_sigma_deg=0.5, att_bias_sigma_deg=0.25, vel_sigma_mps=0.05,
    glitch_prob=0.03, glitch_sigma_m=2.5,
)

_DISTURBED_IMAGE = ImageNoiseConfig(
    exposure_s=(0.015, 0.028),
    vibration_blur_px=1.2,
    noise_sigma=(6.0, 10.0),
    gain_drift=0.2,
    contrast=(0.75, 0.95),
    cloud_shadow_strength=0.4,
    vignetting=0.3,
    distortion=(-0.28, 0.08, 0.0, 0.0, 0.0),
    jpeg_quality=(45, 65),
    drop_prob=0.06,
)

PARTS: Dict[str, PartConfig] = {
    "A": PartConfig(
        name="A",
        title="Direct Identification",
    ),
    "B": PartConfig(
        name="B",
        title="Wind and Sensor Disturbance",
        wind=WindConfig(mean_speed_mps=(2.0, 4.0), gust_sigma_mps=1.2, gust_tau_s=3.0),
        telemetry=_DISTURBED_TELEMETRY,
        image=_DISTURBED_IMAGE,
        flight=FlightConfig(attitude_jitter_deg=0.8),
    ),
    "C": PartConfig(
        name="C",
        title="Fake Markers",
        fakes=FakeConfig(wrong_dictionary=2, invalid_id=3, wrong_size=2, duplicate_pairs=1, moving=2),
        wind=WindConfig(mean_speed_mps=(1.5, 3.0), gust_sigma_mps=1.0, gust_tau_s=3.0),
        telemetry=_DISTURBED_TELEMETRY,
        image=_DISTURBED_IMAGE,
        flight=FlightConfig(attitude_jitter_deg=0.8),
    ),
}

# Scoring: technical parts (80 %) + judged pitch of the idea and approach (20 %)
PART_WEIGHTS = {"A": 0.20, "B": 0.32, "C": 0.28}
PITCH_WEIGHT = 0.20
MATCH_RADIUS_M = 1.0          # a marker counts as found only within this distance
FULL_ACCURACY_RADIUS_M = 0.3  # full accuracy credit within this distance
FP_PENALTY = 0.5              # penalty per wrong row, relative to one marker's full credit


def get_part(name: str) -> PartConfig:
    key = name.strip().upper()
    if key not in PARTS:
        raise ValueError(f"Unknown part {name!r}. Choose from {', '.join(PARTS)}")
    return PARTS[key]
