"""The Qt app end to end, offscreen, on video files (app_tour.py in a subprocess).

A valid calibration must go straight to tracking; without a lens calibration the wizard must
open at the lens step. Every page is shown, fed and painted on the way.
"""

import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from gazecore.tests import test_front_calibration as rendered
from gazecore.tests.golden import eye_golden

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "FrontCameraTracker")


def front_video(path):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for i in range(60):
        writer.write(rendered.render_marker([0.05 * np.sin(i / 10), 0.0, 0.45]))
    writer.release()


def tour(folder, expect):
    video = os.path.join(folder, "front.mp4")
    if not os.path.exists(video):
        front_video(video)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", GAZECORE_DATA_DIR=os.path.join(folder, "data"),
               FRONTCAM_SETTINGS_DIR=os.path.join(folder, "settings"))
    process = subprocess.run([sys.executable, os.path.join(HERE, "app_tour.py"), eye_golden.EYE_VIDEO,
                              eye_golden.EYE_VIDEO, video, expect], env=env, capture_output=True, text=True,
                             timeout=180, cwd=folder)
    output = process.stdout + process.stderr
    assert process.returncode == 0 and "TOUR OK" in process.stdout, output[-4000:]
    return output


def test_a_valid_calibration_goes_straight_to_tracking():
    if not (os.path.exists(eye_golden.EYE_VIDEO) and os.path.exists(os.path.join(TRACKER, "gaze_calibration.json"))):
        return
    with tempfile.TemporaryDirectory() as folder:
        os.makedirs(os.path.join(folder, "data"))
        # As legacy files: once the app has saved there, app_meta names the real USB cameras, not the videos
        for name in ("front_camera_calibration.json", "gaze_calibration.json"):
            with open(os.path.join(TRACKER, name)) as f:
                data = json.load(f)
            data.pop("app_meta", None)
            with open(os.path.join(folder, "data", name), "w") as f:
                json.dump(data, f)
        output = tour(folder, "tracking")
    assert "TOUR page review ok" in output


def test_without_a_lens_calibration_the_wizard_opens_at_the_lens():
    if not os.path.exists(eye_golden.EYE_VIDEO):
        return
    with tempfile.TemporaryDirectory() as folder:
        os.makedirs(os.path.join(folder, "data"))
        tour(folder, "screen,intrinsics")
