# Technical specification

Exact conventions and data formats. Everything your `Solution` receives follows this document.

## 1. Coordinate frames

| Frame | Axes | Notes |
|---|---|---|
| **World (W)** | x = east, y = north, z = up | Origin at the field centre. Ground is the plane z = 0. |
| **Body (B)** | x = forward, y = left, z = up | Attached to the drone. |
| **Camera (C)** | x = image right, y = image down, z = optical axis | Standard OpenCV camera frame. |

**Attitude.** Telemetry gives `roll`, `pitch`, `yaw` in radians (ZYX Euler angles):

```
R_WB = Rz(yaw) · Ry(pitch) · Rx(roll)
```
- `yaw = 0`: the drone faces east; `yaw = +π/2`: faces north.
- `pitch > 0`: nose down. `roll > 0`: right side down.

**Camera mounting.** The camera looks straight down out of the drone's belly. The top of the image faces the drone's forward direction. Columns of `R_BC` are the camera axes written in body coordinates (also given as `camera_info["R_body_cam"]`):

```
R_BC = [[ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0, -1]]          R_WC = R_WB · R_BC
```

## 2. Camera model
A world point `X` (3-vector) appears at pixel `(u, v)`:

```
X_c = R_WC^T (X - p)            p = drone position (x, y, z)
(x_n, y_n) = (X_c.x / X_c.z, X_c.y / X_c.z)      normalized image coordinates
then OpenCV lens distortion with dist = [k1, k2, p1, p2, k3]
(u, v) = (fx · x_d + cx,  fy · y_d + cy)
```
`K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` is `camera_info["K"]`; `dist` is `camera_info["dist"]` (all zeros in Part A).

## 3. What your code receives

**`camera_info`**: `width`, `height`, `K` (3×3), `dist` (5 numbers), `R_body_cam` (3×3), `fps`.

**`rules`**: `dictionary` ("DICT_4X4_50"), `valid_ids` (0–34), `marker_size_m` (0.5), `field_half_size_m` (15).

**`info`**: `part`, `title`, `frame_rate`, `altitude_m` (planned flight height), `speed_mps` (planned speed).

**`telemetry`** (every tick, even when the frame is dropped):

| key | meaning | unit |
|---|---|---|
| `t` | time since start | s |
| `x`, `y`, `z` | reported position (z = height above ground) | m |
| `roll`, `pitch`, `yaw` | reported attitude at the moment of the frame | rad |
| `vx`, `vy`, `vz` | reported velocity (world frame) | m/s |

In Parts B/C telemetry contains noise, a slowly drifting position error and occasional multi-meter position jumps. The frame and its telemetry are taken at the same instant.
