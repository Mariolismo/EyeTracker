"""Where the per-headset calibration files live; the tracker and the app share one set."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The front-camera tracker has always kept them next to itself. GAZECORE_DATA_DIR overrides it (tests).
DATA_DIR = os.environ.get("GAZECORE_DATA_DIR") or os.path.join(REPO_ROOT, "FrontCameraTracker")

# Front camera model and fitted eye offsets (see front_calibration.load/save_calibration).
CALIBRATION_FILE = os.path.join(DATA_DIR, "front_camera_calibration.json")
# The last successful gaze calibration, restored at startup.
GAZE_CALIBRATION_FILE = os.path.join(DATA_DIR, "gaze_calibration.json")
# Every capture session, successful or not, for analysis and recovery.
GAZE_SESSIONS_DIR = os.path.join(DATA_DIR, "calibrations")
# Per-frame gaze CSV; relative to the working directory, where 3DTracker/GazeFollower.cs reads it.
GAZE_VECTOR_FILE = "gaze_vector.txt"
