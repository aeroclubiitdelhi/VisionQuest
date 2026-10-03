"""3D replay video: the mission seen by a virtual camera that slowly circles the field.

Uses the simulator's own renderer (ground texture + every marker, moving ones included),
then draws the drone, its path, the camera's view pyramid and the area it sees, markers
lighting up when first seen (green = genuine, red = fake), and the team's answers at the
end. The drone's camera image is shown picture-in-picture. Pure OpenCV + NumPy.

Used by `python run.py ... --replay-3d`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import apply_homography, camera_matrix, ground_homography, project_points, rot_world_body
from .render import Renderer
from .replay import BLUE, DARK, GREEN, GREY, MAGENTA, RED, WHITE, YELLOW, _footprint, _visible, _Writer

W3, H3 = 1280, 720
SS = 2                      # supersampling for the ground + markers
FOCAL = 950.0
DRONE_SCALE = 3.0           # draw the drone 3x larger than life so it is visible from far away
SKY_TOP = np.array([120, 70, 35], np.float32)      # BGR
SKY_HORIZON = np.array([205, 175, 150], np.float32)
FAR_GRASS = (44, 92, 52)


def look_at(eye, target):
    """OpenCV camera rotation R_wc (columns = camera right, down, forward in world)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.column_stack([r, d, f])


class Scene3D:
    def __init__(self, world):
        self.world = world
        self.r = Renderer(world)
        self.K = camera_matrix(FOCAL, FOCAL, (W3 - 1) / 2, (H3 - 1) / 2)
        s = SS
        self.K_ss = camera_matrix(s * FOCAL, s * FOCAL, s * (W3 - 1) / 2 + (s - 1) / 2,
                                  s * (H3 - 1) / 2 + (s - 1) / 2)
        u, v = np.meshgrid(np.arange(W3, dtype=np.float32), np.arange(H3, dtype=np.float32))
        self.uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
        t = np.linspace(0, 1, H3, dtype=np.float32)[:, None, None]
        self.sky = (SKY_TOP * (1 - t) + SKY_HORIZON * t).repeat(W3, axis=1)

    def background(self, eye, R_wc, t):
        """Sky + textured ground + all markers, from the virtual camera."""
        H_ss = ground_homography(self.K_ss, R_wc, eye)
        canvas = cv2.warpPerspective(self.world.texture, H_ss @ self.r.A_tex, (W3 * SS, H3 * SS),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=FAR_GRASS)
        for m in self.world.markers:
            self.r._composite_marker(canvas, H_ss, m, t)
        img = cv2.resize(canvas, (W3, H3), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Rays that point upwards see the sky; fade the far ground into haze.
        rays = self.uv1 @ (R_wc @ np.linalg.inv(self.K)).T
        dz = rays[..., 2]
        ground = dz < -1e-4
        dist = np.where(ground, -eye[2] / np.minimum(dz, -1e-4), 1e9)
        haze = np.clip((dist - 45.0) / 120.0, 0, 1)[..., None]
        img = img * (1 - haze) + SKY_HORIZON * haze
        img = np.where(ground[..., None], img, self.sky)
        return np.clip(img, 0, 255).astype(np.uint8)

    def proj(self, eye, R_wc, pts):
        uv, z = project_points(self.K, R_wc, eye, np.atleast_2d(pts))
        return uv, z


def _pt(uv):
    return int(round(uv[0])), int(round(uv[1]))


def render_replay_3d(tap, world, answers, score: Optional[Dict], out_path, title: str = "",
                     speed: float = 3.0, out_fps: int = 15) -> None:
    sc = Scene3D(world)
    ci = tap.camera_info
    K_cam = np.array(ci["K"], float)
    size = (ci["width"], ci["height"])
    markers = world.markers
    n_gen = sum(1 for m in markers if m.genuine)
    cam_fps = float(ci.get("fps", 10))
    step = max(1, int(round(speed * cam_fps / out_fps)))
    n_ticks = len(tap.ticks)

    # Pose of every tick (true position when known, else telemetry).
    poses = []
    for tick in tap.ticks:
        tel = tick["tel"]
        p = np.asarray(tick.get("p", [tel["x"], tel["y"], tel["z"]]), float)
        att = tick.get("att", (tel["roll"], tel["pitch"], tel["yaw"]))
        poses.append((p, att))

    def orbit(i):
        a = math.radians(-115 + 70 * i / max(1, n_ticks - 1))     # slow 70 degree orbit
        dist, height = 34.0, 26.0
        eye = np.array([dist * math.cos(a), dist * math.sin(a), height])
        return eye, look_at(eye, [0.0, 0.0, 0.0])

    seen_at: Dict[int, float] = {}
    trail: List[np.ndarray] = []
    last_cam = np.zeros((size[1], size[0], 3), np.uint8)
    writer = _Writer(out_path, out_fps, size=(W3, H3))

    def draw(i, final=False):
        t = tap.ticks[i]["t"]
        p, att = poses[i]
        eye, R = orbit(i)
        img = sc.background(eye, R, t)

        # Camera footprint on the ground + view pyramid.
        if not final:
            poly = _footprint(K_cam, size, p, att)
            if poly is not None:
                g = np.column_stack([poly, np.zeros(4)])
                uv, z = sc.proj(eye, R, g)
                if np.all(z > 0):
                    pts = np.array([_pt(q) for q in uv], np.int32)
                    ov = img.copy()
                    cv2.fillConvexPoly(ov, pts, (120, 200, 255))
                    img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
                    cv2.polylines(img, [pts], True, (120, 200, 255), 1, cv2.LINE_AA)
                    apex, za = sc.proj(eye, R, p)
                    if za[0] > 0:
                        for q in pts:
                            cv2.line(img, _pt(apex[0]), tuple(int(c) for c in q), (120, 200, 255), 1, cv2.LINE_AA)

        # Markers: outline once seen.
        for idx, m in enumerate(markers):
            if idx not in seen_at:
                continue
            x, y, yaw = m.pose_at(t)
            h = m.size_m / 2 + 0.25
            c, s = math.cos(yaw), math.sin(yaw)
            sq = np.array([[x + c * a - s * b, y + s * a + c * b, 0.0]
                           for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))])
            uv, z = sc.proj(eye, R, sq)
            if np.any(z <= 0):
                continue
            col = GREEN if m.genuine else RED
            pts = np.array([_pt(q) for q in uv], np.int32)
            cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
            age = t - seen_at[idx]
            if age < 1.0 and not final:
                cen = uv.mean(axis=0)
                cv2.circle(img, _pt(cen), int(10 + 25 * age), col, 1, cv2.LINE_AA)
            lab = str(m.marker_id) if m.genuine else f"x{m.marker_id}"
            cv2.putText(img, lab, (pts[:, 0].max() + 3, pts[:, 1].min() + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv2.LINE_AA)

        # Path in the air and its shadow on the ground.
        if len(trail) > 1:
            tr = np.array(trail)
            layers = [(np.column_stack([tr[:, :2], np.zeros(len(tr))]), (60, 60, 60))]
            if not final:
                layers.append((tr, (235, 235, 235)))
            for pts3, col in layers:
                uv, z = sc.proj(eye, R, pts3)
                ok = z > 0
                if ok.sum() > 1:
                    cv2.polylines(img, [uv[ok].round().astype(np.int32)], False, col, 1, cv2.LINE_AA)

        # Team answers at the end.
        if final and answers is not None:
            matches = score.get("matches", {}) if score else {}
            gpos = {m.marker_id: (m.x, m.y) for m in markers if m.genuine}
            used = set()
            for mid, x, y in answers:
                uv, z = sc.proj(eye, R, [x, y, 0.0])
                if z[0] <= 0:
                    continue
                q = _pt(uv[0])
                ok = (mid in matches and mid not in used and mid in gpos and
                      abs(math.hypot(x - gpos[mid][0], y - gpos[mid][1]) - matches[mid]) < 1e-3)
                if ok:
                    used.add(mid)
                    cv2.circle(img, q, 6, BLUE, -1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(img, q, MAGENTA, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

        # Drone (scaled up), its shadow and a drop line for depth.
        if not final:
            Rb = rot_world_body(*att)
            arm = 0.35 * DRONE_SCALE
            centre, zc = sc.proj(eye, R, p)
            shadow, zs = sc.proj(eye, R, [p[0], p[1], 0.0])
            if zc[0] > 0 and zs[0] > 0:
                cv2.line(img, _pt(centre[0]), _pt(shadow[0]), (90, 90, 90), 1, cv2.LINE_AA)
                cv2.circle(img, _pt(shadow[0]), 5, (30, 30, 30), -1, cv2.LINE_AA)
                for k in range(4):
                    a = math.pi / 4 + k * math.pi / 2
                    tip = p + Rb @ np.array([arm * math.cos(a), arm * math.sin(a), 0.0])
                    ring = np.array([tip + Rb @ np.array([0.18 * DRONE_SCALE * math.cos(b),
                                                          0.18 * DRONE_SCALE * math.sin(b), 0.0])
                                     for b in np.linspace(0, 2 * math.pi, 14)])
                    uvt, _ = sc.proj(eye, R, tip)
                    uvr, zr = sc.proj(eye, R, ring)
                    cv2.line(img, _pt(centre[0]), _pt(uvt[0]), WHITE, 2, cv2.LINE_AA)
                    if np.all(zr > 0):
                        cv2.polylines(img, [uvr.round().astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
                nose, _ = sc.proj(eye, R, p + Rb @ np.array([arm * 1.4, 0.0, 0.0]))
                cv2.arrowedLine(img, _pt(centre[0]), _pt(nose[0]), YELLOW, 2, cv2.LINE_AA, tipLength=0.35)

        # Overlays: title, stats, picture-in-picture camera, legend.
        cv2.rectangle(img, (0, 0), (W3, 34), DARK, -1)
        cv2.putText(img, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 1, cv2.LINE_AA)
        n_sg = sum(1 for k in seen_at if markers[k].genuine)
        lines = [f"t = {t:5.1f} s   altitude {p[2]:.1f} m",
                 f"genuine seen {n_sg}/{n_gen}   fakes seen {len(seen_at) - n_sg}"]
        if final and score:
            err = score["mean_error_m"]
            lines += [f"TEAM: score {score['score']:.1f}/100   found {score['found']}/{score['genuine']}"
                      f"   wrong {score['false_positives']}   err {err if err is not None else '-'} m"]
        box_h = 14 + 26 * len(lines)
        ov = img.copy()
        cv2.rectangle(ov, (10, 44), (560, 44 + box_h), DARK, -1)
        img = cv2.addWeighted(ov, 0.7, img, 0.3, 0)
        for k, txt in enumerate(lines):
            col = YELLOW if txt.startswith("TEAM") else WHITE
            cv2.putText(img, txt, (20, 68 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)
        if not final:
            pw, ph = 320, 240
            pip = cv2.resize(last_cam, (pw, ph), interpolation=cv2.INTER_AREA)
            x0, y0 = W3 - pw - 14, H3 - ph - 14
            img[y0:y0 + ph, x0:x0 + pw] = pip
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), WHITE, 1)
            cv2.putText(img, "drone camera", (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        leg = [(GREEN, "genuine (seen)"), (RED, "fake (seen)"), ((120, 200, 255), "camera view")]
        if final:
            leg = [(GREEN, "genuine"), (RED, "fake"), (BLUE, "team: correct"), (MAGENTA, "team: wrong")]
        x = 14
        for col, txt in leg:
            cv2.rectangle(img, (x, H3 - 26), (x + 12, H3 - 14), col, -1)
            cv2.putText(img, txt, (x + 18, H3 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            x += 40 + 9 * len(txt)
        cv2.putText(img, f"drone drawn {DRONE_SCALE:.0f}x larger", (W3 - 220, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    try:
        for i, tick in enumerate(tap.ticks):
            p, att = poses[i]
            if tick["jpeg"] is not None:
                last_cam = cv2.imdecode(np.frombuffer(tick["jpeg"], np.uint8), cv2.IMREAD_COLOR)
                for idx, m in enumerate(markers):
                    if idx not in seen_at and _visible(K_cam, size, p, att, m.pose_at(tick["t"])[:2]):
                        seen_at[idx] = tick["t"]
            trail.append(p.copy())
            if i % step == 0:
                writer.write(draw(i))
        end = draw(n_ticks - 1, final=True)
        for _ in range(out_fps * 4):
            writer.write(end)
    finally:
        writer.close()
