"""Image degradation: motion blur, photometric changes, sensor noise, JPEG.

The order follows a real camera: scene (with cloud shadows) -> blur during the
exposure -> lens -> vignetting / exposure / colour -> sensor noise -> JPEG.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .world import World


def directional_blur(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Motion blur along (dx, dy) pixels: rotate so the motion is horizontal,
    apply a 1-D anti-aliased line filter, rotate back. Much faster than a 2-D kernel."""
    L = math.hypot(dx, dy)
    if L < 0.6:
        return img
    h, w = img.shape[:2]
    ang = math.degrees(math.atan2(dy, dx))
    diag = int(math.ceil(math.hypot(w, h))) + 4
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), ang, 1.0)
    M[0, 2] += (diag - w) / 2
    M[1, 2] += (diag - h) / 2
    rot = cv2.warpAffine(img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    k = int(2 * math.ceil(L / 2) + 1)
    xs = np.arange(k, dtype=np.float32) - (k - 1) / 2
    kern = np.clip(L / 2 + 0.5 - np.abs(xs), 0.0, 1.0)
    kern = (kern / kern.sum()).reshape(1, -1)
    rot = cv2.filter2D(rot, -1, kern, borderType=cv2.BORDER_REFLECT)
    Minv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(rot, Minv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class ImagePipeline:
    NOISE_MARGIN = 48

    def __init__(self, world: World, rng: np.random.Generator):
        self.world = world
        self.cfg = world.cfg.image
        self.cond = world.conditions
        self.rng = rng
        cam = world.cfg.camera
        h, w = cam.height, cam.width
        self.h, self.w = h, w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - cam.cx) ** 2 + (yy - cam.cy) ** 2) / (cam.cx ** 2 + cam.cy ** 2)
        vignette = (1.0 - self.cfg.vignetting * r2).astype(np.float32)[..., None]
        disturbed = self.cfg.gain_drift > 0 or self.cfg.vignetting > 0
        color = np.array(self.cond["color_cast"], np.float32)[None, None, :] if disturbed \
            else np.ones((1, 1, 3), np.float32)
        self.shading = np.ascontiguousarray(vignette * color, dtype=np.float32)
        m = self.NOISE_MARGIN
        self.noise_luma = rng.standard_normal((h + m, w + m), dtype=np.float32)
        self.noise_rgb = rng.standard_normal((h + m, w + m, 3), dtype=np.float32)

    def should_drop(self) -> bool:
        return self.cfg.drop_prob > 0 and self.rng.random() < self.cfg.drop_prob

    def blur(self, img: np.ndarray, disp_px) -> np.ndarray:
        dx, dy = float(disp_px[0]), float(disp_px[1])
        vib = self.cfg.vibration_blur_px
        if vib > 0:
            dx += float(self.rng.normal(0, vib))
            dy += float(self.rng.normal(0, vib))
        out = directional_blur(img, dx, dy)
        if vib > 0:
            out = cv2.GaussianBlur(out, (0, 0), 0.35 * vib)
        return out

    def _noise(self) -> np.ndarray:
        m = self.NOISE_MARGIN
        ox, oy = (int(v) for v in self.rng.integers(0, m, 2))
        flip = int(self.rng.integers(0, 4))
        luma = self.noise_luma[oy:oy + self.h, ox:ox + self.w]
        rgb = self.noise_rgb[oy:oy + self.h, ox:ox + self.w]
        if flip & 1:
            luma, rgb = luma[::-1], rgb[::-1]
        if flip & 2:
            luma, rgb = luma[:, ::-1], rgb[:, ::-1]
        sigma = self.cond["noise_sigma"]
        return (0.6 * sigma) * luma[..., None] + (0.8 * sigma) * rgb

    def photometric(self, img: np.ndarray, t: float) -> np.ndarray:
        c = self.cfg
        out = img.astype(np.float32)
        contrast = self.cond["contrast"]
        if contrast != 1.0:
            out = (out - 128.0) * contrast + 128.0
        gain = 1.0
        if c.gain_drift > 0:
            gain += c.gain_drift * math.sin(2 * math.pi * t / self.cond["gain_period_s"] + self.cond["gain_phase"])
        out *= self.shading * gain
        if self.cond["noise_sigma"] > 0:
            out += self._noise()
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def encode(self, img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cond["jpeg_quality"])])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()
