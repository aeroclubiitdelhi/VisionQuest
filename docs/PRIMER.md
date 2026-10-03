# Primer: everything you need to know

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

## 3. From a pixel to a point on the ground
Reverse the steps:
1. **Undistort** the pixel to normalized coordinates: `cv2.undistortPoints(pts, K, dist)` returns `(x_n, y_n)`.
2. Make the camera ray `d_c = (x_n, y_n, 1)` and rotate it to the world: `d_w = R_WC · d_c`.
3. **Intersect with the ground** (z = 0): `λ = -p_z / d_w.z`, point = `p + λ · d_w`.

```python
import math, cv2, numpy as np

def rot_world_body(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

R_BC = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)

def pixels_to_ground(pixels, tel, K, dist):
    """pixels: (N, 2) array -> (N, 2) ground points (x, y) in meters."""
    p = np.array([tel["x"], tel["y"], tel["z"]])
    R_wc = rot_world_body(tel["roll"], tel["pitch"], tel["yaw"]) @ R_BC
    n = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.column_stack([n, np.ones(len(n))]) @ R_wc.T
    lam = -p[2] / rays[:, 2]
    return p[:2] + lam[:, None] * rays[:, :2]
```
Tip: project **all four corners** of a marker to the ground. Their mean is the exact marker centre, and the distances between them give the marker's **real size** (useful against wrong-size fakes).

Why the starter is inaccurate: it assumes the camera looks straight down. At 8 m height a tilt of only 5° shifts the ground point by 8 × tan 5° ≈ 0.7 m. A drone cruising at constant speed is always tilted forward a little (to overcome drag), and in wind it tilts much more.

## 4. What your code receives

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

## 5. Ideas that score points
**Detection (Parts B/C)**
- Use a grayscale image; try `cv2.createCLAHE` for low contrast and cloud shadows.
- Tune `cv2.aruco.DetectorParameters()`, e.g. `cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX`, adaptive threshold window sizes.
- You get many looks at each marker; missing some frames is fine.

**Combining sightings**
- Keep every sighting: `(id, time, x, y, size, ...)`. Decide in `finalize()`.
- Use a **median** or a trimmed mean, not a plain mean: a GNSS jump or a false detection ruins a mean.
- Sightings near the image centre are usually more accurate than near the edges.
- Require several sightings before trusting an ID (a single one may be a false detection in the grass texture).

**Rejecting fakes (Part C)**
- Invalid IDs: check against `rules["valid_ids"]`.
- Wrong dictionary: a DICT_4X4_50 detector normally ignores them.
- Wrong size: measure the marker's size on the ground (see Section 3).
- Duplicates: the same ID seen at two clearly separate places, so drop it.
- Moving markers: their estimated position changes with time; static markers stay put.

## 6. Debugging workflow
1. `python run.py --part B --seed 3 --record data/B3` once (rendering is the slow part).
2. `python run.py --replay data/B3` as often as you like: much faster.
3. Open `outputs/<mission>_map.png`: green = found, yellow = missed, red = fakes, blue dot = your correct answer, magenta cross = your wrong answer.
4. `--replay-video` saves a top-down replay (`_replay.mp4`): drone path, the camera's view area, markers lighting up when first seen, and your answers at the end. Also works with `--replay`. `--replay-3d` makes the same flight as a 3D video (`_replay3d.mp4`) from a camera circling the field. `--video` saves the raw camera view; `--view` shows it live.
5. Before submitting, run `python run.py --part A B C --seeds 1 2 3 --isolated` to test exactly like the grader.

## 7. Preparing the pitch (20 % of the score)
Keep notes while you work, so the pitch is easy to put together at the end:
- what makes each part hard, in one sentence each;
- your pipeline as 4–6 steps (detect → pixel to ground → combine sightings → reject fakes);
- one or two result maps (`outputs/*_map.png`) or a replay video (`--replay-video`) showing what works and what still fails;
- your practice scores per part, before and after your main improvements;
- what you would do next with more time.
