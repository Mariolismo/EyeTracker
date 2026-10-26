"""Loading, solving and saving a gaze calibration, plus the app's metadata about it.

The files keep the front-camera tracker's format (front_calibration.save_*), so either program
restores what the other saved. Everything the app adds sits under one "app_meta" key, which
the tracker ignores (and drops when it rewrites a file: such a file then counts as legacy).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from . import front_calibration as fc
from . import paths

APP_META_KEY = "app_meta"
APP_META_VERSION = 1


def fingerprint(data):
    """Short hash of JSON-ready data, e.g. to tie a gaze calibration to the intrinsics it used."""
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def intrinsics_fingerprint(camera):
    return None if camera is None or not camera.calibrated else fingerprint(camera.to_json())


def samples_fingerprint(calibration, eye_states):
    return fingerprint({**calibration.to_json(), "eye_states": eye_states})


@dataclass
class FrontCalibration:
    camera: object  # CameraModel, or None when the lens is not calibrated
    eye_origins: dict
    app_meta: dict  # None for files written by the tracker


@dataclass
class StoredGaze:
    calibration: fc.GazeCalibration
    eye_states: dict
    saved_at: str
    app_meta: dict  # None for files written by the tracker


@dataclass
class Solved:
    result: fc.CalibrationResult
    eye_origins: dict  # Origins to keep: fitted ones, or the saved ones reused


def load_front(path=None, image_size=(640, 480)):
    path = path or paths.CALIBRATION_FILE
    camera, origins = fc.load_calibration(path, image_size)
    return FrontCalibration(camera, origins, fc.load_extra(path, APP_META_KEY))


def load_gaze(path=None):
    """Raises FileNotFoundError when there is none, ValueError/KeyError/OSError when unreadable."""
    path = path or paths.GAZE_CALIBRATION_FILE
    calibration, eye_states, saved_at = fc.load_gaze_session(path)
    return StoredGaze(calibration, eye_states, saved_at, fc.load_extra(path, APP_META_KEY))


def solve(calibration, saved_origins):
    """Fit the eye models; raises ValueError. Writes nothing."""
    result = calibration.solve(saved_origins)
    origins = dict(saved_origins)
    if result.origins_fitted:
        origins.update({eye_id: model["origin"].copy() for eye_id, model in result.models.items()})
    return Solved(result, origins)


def save_front(front, path=None):
    fc.save_calibration(path or paths.CALIBRATION_FILE, front.camera, front.eye_origins,
                        None if front.app_meta is None else {APP_META_KEY: front.app_meta})


def save_gaze(calibration, eye_states, app_meta, path=None):
    fc.save_gaze_session(path or paths.GAZE_CALIBRATION_FILE, calibration, eye_states,
                         {APP_META_KEY: app_meta})


def session_path():
    return os.path.join(paths.GAZE_SESSIONS_DIR, time.strftime("gaze_%Y%m%d_%H%M%S.json"))


def summary_lines(result):
    """Per-eye fit quality in plain numbers."""
    lines = []
    for eye_id, model in result.models.items():
        bands = ", ".join(f"{band} {rms:.2f}°" for band, rms in model["band_rms_deg"].items())
        dropped = np.flatnonzero(~model["inliers"]) + 1
        text = f"{eye_id.title()} eye: RMS {model['rms_deg']:.2f}° ({bands})"
        if dropped.size:
            text += f", ignored sample(s) {dropped.tolist()}"
        lines.append(text)
    if result.eye_separation_mm is not None:
        lines.append(f"Eye separation {result.eye_separation_mm:.0f} mm")
    return lines
