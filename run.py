#!/usr/bin/env python3
"""Run your solution in the simulator on practice worlds and print its score.

  python run.py --part A --seed 1
  python run.py --part A B C --seeds 1 2 3
  python run.py --part B --seed 3 --record data/B3     # save a mission (no solution)
  python run.py --replay data/B3                       # run your solution on a saved mission
  python run.py --part A --seed 1 --isolated           # run exactly like the grader (sandboxed)
  python run.py --part A --seed 1 --video              # save the drone camera video
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from fgcsim import FieldSim
from fgcsim.config import get_part
from fgcsim.mission import LiveSource, RecordedSource, record_mission
from fgcsim.runner import InProcessHost, IsolatedHost, MissionResult, SolutionError, SolutionTimeout, run_mission
from fgcsim.scoring import score_mission, write_csv


class VideoTap:
    """Saves the drone's camera frames to an mp4."""

    def __init__(self, source, path=None):
        self.src = source
        self.camera_info, self.rules, self.info = source.camera_info, source.rules, source.info
        self.writer = None
        if path:
            ci = source.camera_info
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(ci["fps"]),
                                          (ci["width"], ci["height"]))
        self.blank = np.zeros((source.camera_info["height"], source.camera_info["width"], 3), np.uint8)

    def _tap(self, obs):
        if self.writer is not None:
            frame = obs.frame
            self.writer.write(frame if frame is not None else self.blank)
        return obs

    def reset(self):
        return self._tap(self.src.reset())

    def step(self):
        return self._tap(self.src.step())

    @property
    def flight_time(self):
        return self.src.flight_time

    def close(self):
        if self.writer is not None:
            self.writer.release()


def run_one(args, source, part):
    try:
        host = (IsolatedHost(args.solution, budget_s=args.budget, sandbox=True) if args.isolated
                else InProcessHost(args.solution))
    except (SolutionError, SolutionTimeout) as e:
        return MissionResult(part=part, status="error", error=str(e))
    except Exception:  # noqa: BLE001  (e.g. a syntax error in solution.py)
        import traceback
        return MissionResult(part=part, status="error", error=traceback.format_exc())
    return run_mission(host, source, part)


def show(tag, res, truth, out_dir):
    write_csv(res.markers, out_dir / f"{tag}.csv")
    if res.status != "ok":
        print(f"{tag:<12} {res.status}")
        print("  " + (res.error or "").strip().replace("\n", "\n  "))
        return 0.0
    if truth is None:
        print(f"{tag:<12} ok   ({len(res.markers)} markers submitted, no score available)")
        return None
    s = score_mission(res.markers, truth)["score"]
    print(f"{tag:<12} score {s:6.2f}   (solution time {res.solution_time:.1f} s)")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="+", default=["A"], help="A, B and/or C")
    ap.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--solution", default="solution/solution.py")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--record", metavar="DIR")
    ap.add_argument("--replay", metavar="DIR")
    ap.add_argument("--isolated", action="store_true")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replay:
        src = RecordedSource(args.replay)
        tag = Path(args.replay).name
        truth = src.truth()
        tap = VideoTap(src, out_dir / f"{tag}.mp4" if args.video else None)
        res = run_one(args, tap, src.info["part"])
        tap.close()
        show(tag, res, truth["ground_truth"] if truth else None, out_dir)
        return 0

    parts = [get_part(p).name for p in args.part]
    if args.record:
        for part in parts:
            for seed in args.seeds:
                dest = Path(args.record) if len(parts) * len(args.seeds) == 1 else Path(args.record) / f"{part}_seed{seed}"
                meta = record_mission(FieldSim(part, seed), dest)
                print(f"recorded {part} seed {seed}: {meta['n_frames']} frames -> {dest}")
        return 0

    for part in parts:
        scores = []
        for seed in args.seeds:
            tag = f"{part}_seed{seed}"
            sim = FieldSim(part, seed)
            tap = VideoTap(LiveSource(sim), out_dir / f"{tag}.mp4" if args.video else None)
            res = run_one(args, tap, part)
            tap.close()
            scores.append(show(tag, res, sim.world.ground_truth(), out_dir) or 0.0)
        if len(args.seeds) > 1:
            print(f"Part {part} mean score {sum(scores) / len(scores):6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
