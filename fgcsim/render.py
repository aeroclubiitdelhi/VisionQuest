"""Camera renderer: warps the ground texture and every marker into the drone's view.

Everything is plain OpenCV + NumPy. The image is rendered at `supersample` x the
final resolution and downscaled with area averaging (anti-aliasing). When the
part uses lens distortion, the ideal image is rendered with a margin and then
remapped through the OpenCV distortion model, so participants can undo it with
cv2.undistortPoints using the published coefficients.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .aruco_utils import marker_patch
from .geometry import apply_homography, camera_matrix, ground_homography
from .world import World


class Renderer:
    def __init__(self, world: World):
        cam = world.cfg.camera
        self.world = world
        self.w, self.h = cam.width, cam.height
        self.s = int(cam.supersample)
        self.K = camera_matrix(cam.fx, cam.fy, cam.cx, cam.cy)
        self.dist = np.array(world.cfg.image.distortion, dtype=float)
        self.has_dist = bool(np.any(self.dist != 0))

        self.pad = 0
        self.map_x = self.map_y = None
        if self.has_dist:
            u, v = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
            pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
            ideal = cv2.undistortPointsIter(pts, self.K, self.dist, None, self.K, crit).reshape(-1, 2)
            ix = ideal[:, 0].reshape(self.h, self.w)
            iy = ideal[:, 1].reshape(self.h, self.w)
            over = max(-ix.min(), ix.max() - (self.w - 1), -iy.min(), iy.max() - (self.h - 1), 0.0)
            self.pad = int(math.ceil(over)) + 3
            self.map_x = (ix + self.pad).astype(np.float32)
            self.map_y = (iy + self.pad).astype(np.float32)

        self.W_pad, self.H_pad = self.w + 2 * self.pad, self.h + 2 * self.pad
        self.K_pad = camera_matrix(cam.fx, cam.fy, cam.cx + self.pad, cam.cy + self.pad)
        s = self.s
        self.K_ss = camera_matrix(s * cam.fx, s * cam.fy,
                                  s * (cam.cx + self.pad) + (s - 1) / 2,
                                  s * (cam.cy + self.pad) + (s - 1) / 2)
        self.W_ss, self.H_ss = s * self.W_pad, s * self.H_pad

        tex = world.texture
        n = tex.shape[0]
        ppm = world.cfg.field.texture_ppm
        c = (n - 1) / 2
        self.A_tex = np.array([[1 / ppm, 0, -c / ppm], [0, -1 / ppm, c / ppm], [0, 0, 1]])

    # ------------------------------------------------------------------ public
    def render(self, p_w: np.ndarray, R_wc: np.ndarray, t: float) -> np.ndarray:
        """Ideal (undistorted, noise-free) image at the padded final resolution."""
        H_ss = ground_homography(self.K_ss, R_wc, p_w)
        canvas = cv2.warpPerspective(self.world.texture, H_ss @ self.A_tex, (self.W_ss, self.H_ss),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for m in self.world.markers:
            self._composite_marker(canvas, H_ss, m, t)
        if self.s > 1:
            canvas = cv2.resize(canvas, (self.W_pad, self.H_pad), interpolation=cv2.INTER_AREA)
        return canvas

    def cloud_shadow(self, img: np.ndarray, p_w, R_wc, t: float, strength: float) -> np.ndarray:
        cloud = self.world.cloud_field
        if cloud is None or strength <= 0:
            return img
        cond = self.world.conditions
        res = 0.5  # meters per cloud-tile pixel
        vx, vy = cond["cloud_velocity"]
        ox, oy = cond["cloud_offset"]
        A = np.array([[res, 0, vx * t - ox * res], [0, res, vy * t - oy * res], [0, 0, 1]])
        H = ground_homography(self.K_pad, R_wc, p_w) @ A
        mask = cv2.warpPerspective(cloud, H, (self.W_pad, self.H_pad),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out = img.astype(np.float32) * (1.0 - strength * mask)[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def distort(self, img: np.ndarray) -> np.ndarray:
        """Padded ideal image -> final-size image with lens distortion (or a crop)."""
        if self.has_dist:
            return cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        p = self.pad
        return img[p:p + self.h, p:p + self.w]

    # ------------------------------------------------------------------ internals
    def _composite_marker(self, canvas: np.ndarray, H_ss: np.ndarray, m, t: float) -> None:
        x, y, yaw = m.pose_at(t)
        patch, side_px = marker_patch(m.dictionary, m.marker_id)
        P = patch.shape[0]
        mpp = m.size_m / side_px
        c = (P - 1) / 2
        S = np.array([[mpp, 0, -c * mpp], [0, -mpp, c * mpp], [0, 0, 1]])
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        Rt = np.array([[cy_, -sy_, x], [sy_, cy_, y], [0, 0, 1]])
        H = H_ss @ Rt @ S

        corners = np.array([[-0.5, -0.5], [P - 0.5, -0.5], [P - 0.5, P - 0.5], [-0.5, P - 0.5]])
        ph = np.column_stack([corners, np.ones(4)]) @ H.T
        if np.any(ph[:, 2] <= 0):
            return
        img_pts = ph[:, :2] / ph[:, 2:3]
        x0 = int(math.floor(img_pts[:, 0].min())) - 1
        y0 = int(math.floor(img_pts[:, 1].min())) - 1
        x1 = int(math.ceil(img_pts[:, 0].max())) + 2
        y1 = int(math.ceil(img_pts[:, 1].max())) + 2
        x0c, y0c = max(x0, 0), max(y0, 0)
        x1c, y1c = min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
        if x1c <= x0c or y1c <= y0c:
            return
        T = np.array([[1, 0, -x0c], [0, 1, -y0c], [0, 0, 1]], dtype=float)
        Hr = T @ H
        size = (x1c - x0c, y1c - y0c)
        warped = cv2.warpPerspective(np.asarray(patch), Hr, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha = cv2.warpPerspective(np.ones((P, P), np.float32), Hr, size, flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        roi = canvas[y0c:y1c, x0c:x1c].astype(np.float32)
        a = alpha[..., None]
        # `warped` is already premultiplied by alpha (the border value is 0).
        roi = roi * (1 - a) + warped.astype(np.float32)[..., None]
        canvas[y0c:y1c, x0c:x1c] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)

    def ground_point_pixel(self, p_w, R_wc, ground_xy) -> np.ndarray:
        """Pixel (padded ideal image) of a ground point. Used for motion-blur direction."""
        H = ground_homography(self.K_pad, R_wc, p_w)
        return apply_homography(H, np.asarray(ground_xy, dtype=float)[None, :])[0]
