"""Frames, rotations and the pinhole camera model.

Conventions (identical in the simulator, the primer and the reference solution):

World frame W   ENU: x = east, y = north, z = up. Origin = field centre, ground is z = 0.
Body frame B    FLU: x = forward, y = left, z = up.
Attitude        R_WB = Rz(yaw) @ Ry(pitch) @ Rx(roll)   (ZYX Euler angles, radians).
                yaw = 0 means the drone faces east; positive pitch = nose down.
Camera frame C  OpenCV: x = image right, y = image down, z = optical axis.
                The camera points straight down; the top of the image faces
                the drone's forward direction.
                R_BC (columns are camera axes in body coordinates) = R_BODY_CAM.
Projection      X_C = R_WC^T (X_W - p_W),   R_WC = R_WB @ R_BC
                pixel = K @ (X_C / X_C.z), then OpenCV lens distortion.
"""
from __future__ import annotations

import numpy as np

R_BODY_CAM = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rot_world_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_world_cam(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_world_body(roll, pitch, yaw) @ R_BODY_CAM


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def ground_homography(K: np.ndarray, R_wc: np.ndarray, p_w: np.ndarray) -> np.ndarray:
    """3x3 H mapping ground points (x, y, 1) with z = 0 to ideal (undistorted) pixels."""
    M = R_wc.T
    t = -M @ np.asarray(p_w, dtype=float)
    return K @ np.column_stack([M[:, 0], M[:, 1], t])


def project_points(K: np.ndarray, R_wc: np.ndarray, p_w: np.ndarray, pts_w: np.ndarray):
    """Project world points (N, 3) to ideal pixels (N, 2). Also returns depth (N,)."""
    pts_w = np.atleast_2d(np.asarray(pts_w, dtype=float))
    pc = (pts_w - np.asarray(p_w, dtype=float)) @ R_wc  # == (R_wc.T @ (X - p).T).T
    z = pc[:, 2]
    uv = (pc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]
