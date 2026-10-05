# Eyes Over the Field
**AeroClub IIT Delhi · Tech FGC · Problem Statement**
Team size: **2–3** · Mode: **simulation only (no hardware)**

**Format:** briefing → coding → code submission deadline → pitches to the judges. The deadline is announced at the briefing.

---

## 1. Background
Drones cannot always trust GPS. In warehouses, disaster zones or under signal jamming, a drone has to understand the ground below it by *looking*. A common trick is to lay **ArUco markers** (square black-and-white tags, like simple QR codes) on the ground. A drone flying overhead detects them with its camera, reads their IDs, and works out exactly where they are.

Real flights are messy. Wind pushes and tilts the drone, cameras blur, lenses bend straight lines, sensors drift, and not every marker can be trusted. Your job is to write the software that turns a drone's camera feed into an accurate map of the markers on the ground.

## 2. What you will do
You run a **drone simulator** on your own laptop (plain Python). A quadcopter flies over a 30 m × 30 m field with a **downward-facing camera**. Every camera frame (10 per second) is handed to your code together with the drone's **telemetry** (its reported position and orientation).

Your code must output a **map of every genuine marker: its ID and its ground position (x, y) in meters.** At the end you **pitch your idea and approach** to the judges.

## 3. Prerequisites
**Please finish setup before the event. No setup help is given during the event.**
- A laptop (Windows, macOS or Linux) with **Python 3.9 or newer**.
- `pip install -r requirements.txt` (OpenCV and NumPy). Google Colab also works.
- Knowledge: Python.

**Setup check** (every team must be able to do this before comp day):
```bash
python run.py --part A --seed 1
```
It should fly a mission and print a score (0 for the unfilled template).

## 4. The world
- **Coordinates:** flat ground (z = 0). x = east, y = north, origin = field centre, meters. All markers lie inside |x|, |y| ≤ 15 m.
- **Drone:** it flies a fixed back-and-forth ("lawnmower") path at about 8 m height.
- **Camera:** 640 × 480, 10 frames per second, mounted on the drone's belly. Its model and mounting are defined in `docs/SPEC.md`.
- **Telemetry per frame:** `t, x, y, z, roll, pitch, yaw, vx, vy, vz` (meters, radians, m/s). Conventions are in `docs/SPEC.md`.
- **Dropped frames:** sometimes the camera drops a frame; your code then gets `frame = None` (telemetry still arrives).

## 5. Valid markers
A marker is **genuine** only if **all** of these hold:
1. It is from the ArUco dictionary **DICT_4X4_50**.
2. Its ID is on the **valid list: 0 to 34**.
3. Its black square is exactly **0.50 m** wide.
4. It does **not move**.
5. Its ID appears at **only one place** on the field. If an ID appears at two places, **both** are fake.

Anything else is a **fake** and must not appear in your answer.

## 6. Tasks

### Part A: Direct Identification (20 %)
Calm air, clean images, accurate telemetry. Detect every marker, read its ID, and compute its ground position. Each marker is seen in many frames; report it once.

### Part B: Wind and Sensor Disturbance (32 %)
- **Wind and gusts** push the drone around.
- **Camera problems:** motion blur, noise, moving cloud shadows, brightness changes, strong lens distortion, JPEG artefacts, dropped frames.
- **Sensor problems:** noisy position and attitude, a slowly drifting position error, and occasional GNSS jumps of a few meters.

Produce an accurate map anyway.

### Part C: Fake Markers (28 %)
Part B conditions **plus decoys**: markers from other dictionaries, invalid IDs, the wrong physical size, duplicated IDs, and markers that move (carried by vehicles). Map every genuine marker and **reject every fake**.

### Part D: Pitch your idea and approach (20 %)
After the submission deadline, each team presents its solution to the judges: **a short pitch followed by questions** (exact length announced at the briefing). Every team member should be able to answer questions. Slides are optional (at most 4); you may instead show your code on your laptop.

Explain the problem in your own words, your approach for each part, what worked, what did not, and why. Judges score with this rubric:

| Criterion | Points |
|---|---|
| **Problem understanding**: what makes each part difficult | 20 |
| **Approach and reasoning**: your method for each part, and why you chose it | 35 |
| **Results and self-analysis**: practice scores, failure cases, honest limits | 20 |
| **Originality and engineering decisions**: smart ideas, trade-offs, testing | 15 |
| **Communication**: clarity, staying within the time limit, answers to questions | 10 |
| **Total** | **100** |

The pitch must describe the code you submitted. A clear explanation of a partial solution can still score well here.

## 7. Your code
Write your code in `solution/solution.py` (a template with TODOs is provided). It must define:
```python
class Solution:
    def __init__(self, camera_info, rules, info): ...
    def on_frame(self, frame, telemetry):
        # frame: BGR image (numpy uint8), or None if dropped; telemetry: dict
        ...
    def finalize(self):
        return [(marker_id, x, y), ...]
```
**Commands**
```bash
python run.py --part B --seed 3              # one practice world (any seed)
python run.py --part A B C --seeds 1 2 3     # several practice worlds
python run.py --part B --seed 3 --record data/B3   # save a mission
python run.py --replay data/B3               # run your code on a saved mission
python run.py --part A --seed 1 --isolated   # run exactly like the grader
python run.py --part A --seed 1 --video      # save the drone camera video
```
Each run prints your score and writes your answers to `outputs/<mission>.csv`.

## 8. Submission
Before the deadline, submit a **zip of your `solution/` folder** through [submission link].

- **One member submits for the whole team.** Only the team's last submission before the deadline counts.
- **Name the zip after your team**, e.g. `SkyWatchers.zip`. Use the same team name you registered with and the one you give the judges at the pitch.
- The zip must contain `solution.py` plus any helper `.py` files it imports. Nothing else is needed.
- Keep the class name `Solution` and its three methods (`__init__`, `on_frame`, `finalize`) exactly as in the template. Renaming them makes your code fail to run.
- **Late submissions are not accepted.** Submit a working version early and update it later.
- Before submitting, check your code runs with `python run.py --part A B C --isolated`. A solution that crashes on our laptop scores 0 for that mission.
- Pitch slides (optional) are not submitted. Present them from your own laptop.

Your code must:
- run with only Python, NumPy and OpenCV (the packages in `requirements.txt`),
- use only the frames and telemetry given to it: no reading or writing files, no OS, process or network access, no importing the simulator (`fgcsim`). During grading this is enforced and every submission is checked,
- finish each mission within **300 s** of computation on the organizers' laptop (your code runs in its own process with OpenCV limited to one thread; `python run.py ... --isolated` runs it exactly this way).

## 9. Evaluation
**Technical (80 %).** We run your solution on **hidden worlds** (new layouts, same conditions as practice), with each part flown on several worlds. For each mission:
- A reported marker is **correct** if the ID is genuine and it is within **1.0 m** of the true position.
- Each correct marker earns **0.5** for being found + up to **0.5** for accuracy (full within 0.3 m, falling to 0 at 1.0 m).
- Every other row (fake, wrong ID, duplicate, too far) costs **0.5**.
- **Mission score = 100 × (earned − penalties) / number of genuine markers**, between 0 and 100.

Part score = mean mission score over its worlds. A crash or timeout scores 0 for that mission. `run.py` uses exactly this scoring on the practice worlds.

**Pitch (20 %).** Judges' rubric score (0–100), averaged over judges.

**Final score (out of 100) = 0.20 × A + 0.32 × B + 0.28 × C + 0.20 × Pitch.**
Ties are broken by the lower average position error.

## 10. Rules
- Any method is allowed: classical OpenCV, your own algorithms, machine learning (if it runs with the allowed packages and time budget).
- Reading ground truth or simulator internals from inside your solution (e.g. importing `fgcsim` to regenerate the world, or reading files) is **not allowed**. Every submission is scanned and the suspicious ones reviewed by hand; violations mean disqualification.
- Sharing code between teams is not allowed.
- Organizers' decisions are final.

## 11. Resources
- `docs/SPEC.md`: coordinate frames, camera model, data formats.
- Public documentation (OpenCV, NumPy, Python) may be used freely.
