"""Eyes Over the Field: your solution.

Fill in the TODOs. Do not rename the class or the three methods.
The task is described in docs/PS.md; the exact data formats and conventions are in docs/SPEC.md.

How it is used:
    sol = Solution(camera_info, rules, info)     # once, before the flight
    sol.on_frame(frame, telemetry)               # for every camera tick (10 per second)
    answers = sol.finalize()                     # once, after the flight

Allowed packages: Python standard library (no file, OS or network access), numpy, opencv.
Helper files placed in the same folder can be imported.
"""
import numpy as np
import cv2


class Solution:
    def __init__(self, camera_info, rules, info):
        """Called once before the flight.

        camera_info: dict with width, height, K, dist, R_body_cam, fps    (docs/SPEC.md)
        rules:       dict with dictionary, valid_ids, marker_size_m, field_half_size_m
        info:        dict with part ("A", "B" or "C"), frame_rate, altitude_m, speed_mps
        """
        self.camera_info = camera_info
        self.rules = rules
        self.info = info

        # TODO: set up anything you need during the flight
        #       (for example a marker detector, and somewhere to store your observations).

    def on_frame(self, frame, telemetry):
        """Called for every camera tick.

        frame:     BGR image, numpy uint8 array of shape (480, 640, 3),
                   or None when the camera dropped this frame
        telemetry: dict with t, x, y, z, roll, pitch, yaw, vx, vy, vz    (docs/SPEC.md)
        The return value is ignored.
        """
        if frame is None:
            return

        # TODO: find the markers visible in this frame.

        # TODO: work out where each of them is on the ground (x, y in meters).

        # TODO: store whatever you will need in finalize().

    def finalize(self):
        """Called once after the flight.

        Return one entry per GENUINE marker: [(marker_id, x, y), ...] with x, y in meters.
        Fake markers and duplicate rows cost points (see docs/PS.md, Evaluation).
        """
        answers = []

        # TODO: decide which markers are genuine and where each one is, and fill `answers`.

        return answers
