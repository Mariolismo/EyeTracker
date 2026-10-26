"""One gaze calibration in progress: samples by band, undo, the session file, solve, accept.

No Qt. The wizard's gaze pages and the quick recalibration feed it capture_gates.Samples.
Every sample is written to a session file in calibrations/ at once (for recovery); the
calibration files the tracker and the app restore from are only written by accept().
"""

import time

import numpy as np

from gazecore import calibration_store as store
from gazecore import front_calibration as fc
from gazecore import gaze_geometry as gg

from . import capture_gates

EYE_IDS = ("left", "right")


class CalibrationSession:
    def __init__(self, eye_states, saved_origins, quick=False, settings=None, session_path=None):
        self.eye_states = eye_states  # Locked eye states the raw gaze depends on
        self.saved_origins = dict(saved_origins or {})
        self.quick = quick  # One depth (near), reusing the saved eye origins
        self.settings = dict(settings or {})  # capture_mode, frames_averaged, monitor...
        self.session_path = session_path or store.session_path()
        self.calibration = fc.GazeCalibration(EYE_IDS)
        self.samples = []  # capture_gates.Sample per calibration sample, same order
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")

    # --- Samples ---

    def band_of(self, sample):
        return gg.depth_band(sample.depth)

    def pixels(self, band):
        return [s.pixel for s in self.samples if self.band_of(s) == band]

    def counts(self):
        return self.calibration.band_counts()

    def last_in_band(self, band):
        matching = [s for s in self.samples if self.band_of(s) == band]
        return matching[-1] if matching else None

    def add(self, sample):
        """Returns a plausibility warning or None."""
        warning = capture_gates.plausibility(self.last_in_band(self.band_of(sample)), sample)
        self.calibration.add_sample(sample.gaze, fc.MarkerDetection(None, sample.point, sample.pixel))
        self.samples.append(sample)
        self.save_session()
        return warning

    def undo(self):
        if not self.samples:
            return False
        self.samples.pop()
        self.calibration.samples.pop()
        self.save_session()
        return True

    def remove_band(self, band):
        keep = [i for i, s in enumerate(self.samples) if self.band_of(s) != band]
        self.samples = [self.samples[i] for i in keep]
        self.calibration.samples = [self.calibration.samples[i] for i in keep]
        self.save_session()

    def clear(self):
        self.samples, self.calibration.samples = [], []

    def band_ready(self, band):
        return capture_gates.band_ready(self.pixels(band))

    @property
    def ready(self):
        if self.quick:
            return self.band_ready("near") and all(eye in self.saved_origins for eye in EYE_IDS)
        return self.band_ready("near") and self.band_ready("far")

    # --- Files ---

    def sample_meta(self):
        return [s.meta() for s in self.samples]

    def app_meta(self, cameras, camera):
        """cameras: {role: RoleBinding.geometry}; camera: the front CameraModel."""
        return {"version": store.APP_META_VERSION, "cameras": cameras,
                "intrinsics_fingerprint": store.intrinsics_fingerprint(camera),
                "capture_mode": self.settings.get("capture_mode"),
                "frames_averaged": self.settings.get("frames_averaged"),
                "monitor": self.settings.get("monitor"), "quick": self.quick,
                "started": self.started, "sample_meta": self.sample_meta(),
                "fingerprint": store.samples_fingerprint(self.calibration, self.eye_states)}

    def save_session(self):
        try:
            fc.save_gaze_session(self.session_path, self.calibration, self.eye_states,
                                 {store.APP_META_KEY: {"session": True, "sample_meta": self.sample_meta()}})
        except OSError:
            pass  # Recovery only; never block a calibration over it

    def solve(self):
        """store.Solved; raises ValueError (thread-safe: works on a copy of the samples)."""
        calibration = fc.GazeCalibration(EYE_IDS, list(self.calibration.samples))
        return store.solve(calibration, self.saved_origins)

    def accept(self, solved, front, cameras, gaze_path=None, front_path=None):
        """Write the calibration files. front: calibration_store.FrontCalibration (updated in place)."""
        store.save_gaze(self.calibration, self.eye_states, self.app_meta(cameras, front.camera), gaze_path)
        if solved.result.origins_fitted:
            front.eye_origins = {eye: np.asarray(o, dtype=float) for eye, o in solved.eye_origins.items()}
            store.save_front(front, front_path)
