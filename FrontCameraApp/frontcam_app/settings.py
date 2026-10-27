"""The app's own files (calibrations are shared with the tracker, see gazecore.paths)."""

import json
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_DIR = os.environ.get("FRONTCAM_SETTINGS_DIR") or os.path.join(APP_DIR, "settings")
CAMERA_SETUP_FILE = os.path.join(SETTINGS_DIR, "camera_setup.json")
APP_SETTINGS_FILE = os.path.join(SETTINGS_DIR, "app_settings.json")
MONITORS_FILE = os.path.join(SETTINGS_DIR, "monitors.json")
SESSIONS_DIR = os.path.join(APP_DIR, "sessions")
LOG_DIR = os.path.join(APP_DIR, "logs")

DEFAULTS = {
    "capture_mode": "manual",  # "manual" (Space/C/clicker) or "dwell" (hold the gaze ~1 s)
    "dwell_seconds": 1.0,
    "frames_averaged": 30,  # Frames averaged per calibration sample; 1 = the tracker's single frame
    # Locked eyes: only pupils centered inside the eye circle (with pupil_seeds > 1: one pass, no
    # whole-frame retry, so it costs nothing; a fit outside the circle is dropped, U resets the model)
    "pupil_search_in_eye_circle": True,
    "pupil_seeds": 4,  # Dark areas tried per frame; 1 is the original single darkest-area search
    "target_screen": None,  # QScreen name the targets and the overlay use (None: primary)
    "overlay_heatmap": True,
    "overlay_midpoints": False,  # Edge-midpoint markers too, for a monitor wider than the front camera's view
    "gaze_vector_path": "gaze_vector.txt",  # Relative to the working directory, like the tracker; "" disables
    "last_verify": None,  # {"saved_at": calibration saved_at, "summary": {band: [n, error, baseline]}}
}


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """Atomic: a crash never leaves half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary, path)


class AppSettings:
    def __init__(self, path=APP_SETTINGS_FILE):
        self.path = path
        self.values = {**DEFAULTS, **(read_json(path, {}) or {})}

    def __getitem__(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value
        write_json(self.path, self.values)
