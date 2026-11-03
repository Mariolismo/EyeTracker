"""E's eye data capture: the frames the detector saw, before the overlay drew on them."""

import csv
import json
import os
import tempfile

import cv2
import numpy as np

from frontcam_app.eye_capture import EyeCapture
from frontcam_app.pipeline import Pipeline
from gazecore.camera.camera_reader import FramePacket
from gazecore.pupil_detection import crop_to_aspect_ratio
from gazecore.tests import test_front_calibration as rendered
from gazecore.tests.golden import eye_golden

FRAMES = 12


def test_the_capture_saves_undrawn_frames_and_lists_every_frame():
    if not os.path.exists(eye_golden.EYE_VIDEO):
        return
    frames = eye_golden.read_frames(FRAMES)
    with tempfile.TemporaryDirectory() as folder:
        pipeline = Pipeline(rendered.CAMERA, gaze_vector_path=None)
        pipeline.eye_capture = capture = EyeCapture(folder, "P01", {"cameras": {}}, every=3)
        for index, frame in enumerate(frames):
            t = float(index)
            pipeline.step({"left": FramePacket(index + 1, t, frame, (640, 480)),
                           "right": FramePacket(index + 1, t, cv2.resize(frame, (1280, 720)), (1280, 720))}, now=t)
        pipeline.eye_capture = None
        capture.close()

        with open(os.path.join(folder, "frames.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        saved = [row for row in rows if row["image"]]
        assert len(rows) == 2 * FRAMES
        assert len(saved) == 2 * 4 and capture.dropped == 0  # Frames 0, 3, 6 and 9 of each eye
        assert meta["subject"] == "P01" and meta["saved"] == 8 and meta["frames"] == {"left": FRAMES, "right": FRAMES}

        row = next(row for row in saved if row["eye"] == "left" and row["seq"] == "4")
        image = cv2.imread(os.path.join(folder, row["image"]), cv2.IMREAD_UNCHANGED)
        expected = cv2.cvtColor(crop_to_aspect_ratio(frames[3]), cv2.COLOR_BGR2GRAY)
        assert image.shape == (480, 640) and np.array_equal(image, expected)  # No overlay on it
        if row["status"] == "ok":
            assert 0 < float(row["cx"]) < 640 and 0 < float(row["confidence"]) <= 1
