"""Runs a participant solution on a mission.

The solution is a Python file defining:

    class Solution:
        def __init__(self, camera_info: dict, rules: dict, info: dict): ...
        def on_frame(self, frame, telemetry: dict):   # frame = BGR np.ndarray, or None if dropped
            ...                                       # return value is ignored
        def finalize(self) -> list:                   # [(marker_id, x, y), ...]
            ...

Two hosts:
  InProcessHost  - runs the solution in this process (easy debugging, prints, breakpoints).
  IsolatedHost   - runs it in a separate process that only ever receives frames and
                   telemetry (this is how grading works), with a time budget.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from .env import decode_jpeg
from .scoring import sanitize


class SolutionError(Exception):
    pass


class SolutionTimeout(Exception):
    pass


def load_solution_class(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise SolutionError(f"Solution file not found: {path}")
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = f"fgc_solution_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "Solution", None)
    if cls is None:
        raise SolutionError(f"{path.name} must define a class named 'Solution'")
    return cls


def clean_rows(rows: Any) -> List[tuple]:
    try:
        rows = list(rows or [])
    except TypeError:
        return []
    return [tuple(r) for r in sanitize(rows)[0]]


# ----------------------------------------------------------------------------- hosts

class InProcessHost:
    def __init__(self, solution_path):
        self.cls = load_solution_class(solution_path)
        self.sol = None
        self.solution_time = 0.0

    def _timed(self, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            raise SolutionError(traceback.format_exc()) from e
        finally:
            self.solution_time += time.perf_counter() - t0

    def start(self, camera_info, rules, info):
        self.sol = self._timed(self.cls, camera_info, rules, info)

    def on_frame(self, jpeg, telemetry):
        frame = decode_jpeg(jpeg)
        self._timed(self.sol.on_frame, frame, dict(telemetry))

    def finalize(self):
        return clean_rows(self._timed(self.sol.finalize))

    def close(self):
        pass


BLOCKED_MODULES = ("fgcsim", "grading", "common", "run", "rerun_view", "tools", "reference", "tests")
MEMORY_LIMIT_BYTES = 6 * 1024 ** 3


class _BlockSimulator:
    """Import hook: solutions may not import the simulator or organizer code."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED_MODULES:
            raise ImportError(f"importing '{name}' is not allowed in a solution")
        return None


def _enter_sandbox(solution_path):
    """Called in the child before the team's code is loaded.

    * copies the team's folder to a fresh temporary directory and runs from there,
    * removes the simulator / organizer code from sys.modules and sys.path and blocks
      importing it again,
    * caps memory where the OS supports it.
    Grading should ALSO follow the two-phase procedure (answers and secret not on the
    machine while team code runs); this is a second layer, not the only one.
    """
    import shutil
    import tempfile

    import atexit

    src = Path(solution_path).resolve()
    work = Path(tempfile.mkdtemp(prefix="fgc_team_"))
    atexit.register(shutil.rmtree, work, True)
    shutil.copytree(src.parent, work / "team", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work / "team")
    repo_root = Path(__file__).resolve().parent.parent
    keep = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            continue
        if resolved == repo_root or repo_root in resolved.parents or entry in ("", "."):
            continue
        keep.append(entry)
    sys.path[:] = keep
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED_MODULES or name == "__mp_main__":
            del sys.modules[name]
    sys.meta_path.insert(0, _BlockSimulator())
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError, AttributeError):
        pass                                  # not available on this OS (e.g. Windows, macOS)
    return str(work / "team" / src.name)


def _child_main(conn, solution_path, log_path, sandbox=False):
    """Runs inside the isolated process. Every reply carries the time the solution itself
    spent (measured here, so a busy grading machine does not count against the team)."""
    if log_path:
        f = open(log_path, "a", buffering=1)
        sys.stdout = sys.stderr = f
    # One OpenCV thread per solution: fair timing, and no thread oversubscription
    # when several solutions run side by side on the grading machine.
    import cv2
    cv2.setNumThreads(1)

    def fail():
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        conn.send(("error", tb, 0.0))

    try:
        if sandbox:
            solution_path = _enter_sandbox(solution_path)
        cls = load_solution_class(solution_path)
        conn.send(("ok", None, 0.0))
    except BaseException:  # noqa: BLE001
        fail()
        return
    sol = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        kind = msg[0]
        if kind == "stop":
            return
        try:
            if kind == "init":
                t0 = time.perf_counter()
                sol = cls(msg[1], msg[2], msg[3])
                conn.send(("ok", None, time.perf_counter() - t0))
            elif kind == "frame":
                frame = decode_jpeg(msg[1])
                t0 = time.perf_counter()
                sol.on_frame(frame, msg[2])
                conn.send(("ok", None, time.perf_counter() - t0))
            elif kind == "finalize":
                t0 = time.perf_counter()
                rows = sol.finalize()
                conn.send(("ok", clean_rows(rows), time.perf_counter() - t0))
        except BaseException:  # noqa: BLE001
            fail()
            return


class IsolatedHost:
    def __init__(self, solution_path, budget_s: float = 300.0, log_path=None, start_timeout_s: float = 60.0,
                 sandbox: bool = True):
        self.budget = budget_s
        self.solution_time = 0.0
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        # The child never needs organizer secrets.
        saved = os.environ.pop("FGC_SECRET", None)
        try:
            self.proc = ctx.Process(target=_child_main,
                                    args=(child, str(Path(solution_path).resolve()),
                                          str(log_path) if log_path else None, sandbox), daemon=True)
            self.proc.start()
        finally:
            if saved is not None:
                os.environ["FGC_SECRET"] = saved
        child.close()
        self._wait(start_timeout_s, count=False)

    WALL_SLACK_S = 60.0   # extra wall-clock allowance for a busy grading machine

    def _wait(self, limit, count=True):
        ready = self.conn.poll(max(0.0, limit))
        if not ready:
            self.close()
            raise SolutionTimeout(f"solution exceeded its time budget ({self.budget:.0f} s)")
        try:
            status, payload, dt = self.conn.recv()
        except EOFError as e:
            self.close()
            raise SolutionError("solution process crashed") from e
        if status == "error":
            self.close()
            raise SolutionError(payload)
        if count:
            self.solution_time += dt
            if self.solution_time > self.budget:
                self.close()
                raise SolutionTimeout(f"solution exceeded its time budget ({self.budget:.0f} s)")
        return payload

    def _call(self, msg):
        try:
            self.conn.send(msg)
        except (BrokenPipeError, OSError) as e:
            self.close()
            raise SolutionError("solution process is not running") from e
        return self._wait(self.budget - self.solution_time + self.WALL_SLACK_S)

    def start(self, camera_info, rules, info):
        self._call(("init", camera_info, rules, info))

    def on_frame(self, jpeg, telemetry):
        self._call(("frame", jpeg, dict(telemetry)))

    def finalize(self):
        return self._call(("finalize",))

    def close(self):
        try:
            if self.proc.is_alive():
                try:
                    self.conn.send(("stop",))
                except (BrokenPipeError, OSError):
                    pass
                self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=2)
        finally:
            self.conn.close()


# ----------------------------------------------------------------------------- mission loop

@dataclass
class MissionResult:
    part: str
    status: str                      # ok | error | timeout
    markers: List[tuple] = field(default_factory=list)
    error: Optional[str] = None
    n_ticks: int = 0
    flight_time: float = 0.0
    solution_time: float = 0.0
    wall_time: float = 0.0


def run_mission(host, source, part: str) -> MissionResult:
    t0 = time.perf_counter()
    res = MissionResult(part=part, status="ok")
    try:
        host.start(source.camera_info, source.rules, source.info)
        obs = source.reset()
        while True:
            host.on_frame(obs.jpeg, obs.telemetry)
            res.n_ticks += 1
            if obs.done:
                break
            obs = source.step()
        res.markers = host.finalize()
    except SolutionTimeout as e:
        res.status, res.error = "timeout", str(e)
    except SolutionError as e:
        res.status, res.error = "error", str(e)
    finally:
        res.flight_time = source.flight_time
        res.solution_time = host.solution_time
        res.wall_time = time.perf_counter() - t0
        host.close()
    return res
