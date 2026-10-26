"""Front-camera calibration workflow, independent of any UI or tracker state.

The tracker feeds frames and raw gaze directions in and draws whatever these classes
return; nothing here opens windows, reads keys or touches globals. The math lives in
gaze_geometry; see GAZE_GEOMETRY.md for the derivations.

    CameraModel           intrinsics + projection; load/save_calibration persist it
                          (and the eye offsets); load/save_gaze_session persist samples
    MarkerTarget          ArUco gaze target: detection and metric 3D pose
    IntrinsicsCalibrator  ChArUco views -> CameraModel (Zhang's method)
    GazeCalibration       (gaze per eye, marker point) samples -> fitted eye models
    BinocularGazeMapper   raw gaze per eye -> 3D gaze point, depth and pixel
    AccuracyTest          angular error at the marker, binned by depth
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

import gaze_geometry as gg

# The gaze target and the intrinsics board use different dictionaries,
# so board markers are never mistaken for the target.
TARGET_DICTIONARY = cv2.aruco.DICT_4X4_50
BOARD_DICTIONARY = cv2.aruco.DICT_5X5_100


class CameraModel:
    """Pinhole camera with optional OpenCV distortion, for one fixed image size."""

    def __init__(self, K, dist=None, image_size=(640, 480), calibrated=False):
        self.K = np.asarray(K, dtype=float)
        self.dist = np.zeros(5) if dist is None else np.ravel(np.asarray(dist, dtype=float))
        self.image_size = tuple(int(v) for v in image_size)
        self.calibrated = calibrated

    @classmethod
    def from_fov(cls, image_size, diagonal_fov_degrees):
        """Uncalibrated fallback: square pixels, centered principal point."""
        return cls(gg.default_intrinsics(*image_size, diagonal_fov_degrees), None, image_size, False)

    def project(self, point):
        return gg.project_point(self.K, self.dist, point)

    def ray(self, pixel):
        return gg.pixel_ray(self.K, self.dist, pixel)

    def angular_error_degrees(self, pixel_a, pixel_b):
        return float(gg.angular_errors_degrees([self.ray(pixel_a)], [self.ray(pixel_b)])[0])

    def to_json(self):
        return {"image_size": list(self.image_size), "K": self.K.tolist(), "dist": self.dist.tolist()}

    @classmethod
    def from_json(cls, data):
        return cls(data["K"], data["dist"], data["image_size"], calibrated=True)


def load_calibration(path, image_size):
    """Returns (CameraModel or None, {eye_id: origin}). A missing file is not an error.

    Intrinsics saved for a different image size are ignored. Eye origins are kept:
    they depend on the headset, not on the image size.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, {}
    camera = None
    if data.get("intrinsics") and tuple(data["intrinsics"]["image_size"]) == tuple(image_size):
        camera = CameraModel.from_json(data["intrinsics"])
    origins = {eye_id: np.asarray(origin, dtype=float) for eye_id, origin in data.get("eye_origins", {}).items()}
    return camera, origins


def save_calibration(path, camera, eye_origins):
    data = {"eye_origins": {eye_id: np.asarray(origin).tolist() for eye_id, origin in eye_origins.items()}}
    if camera is not None and camera.calibrated:
        data["intrinsics"] = camera.to_json()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_gaze_session(path, calibration, eye_states):
    """Persist a gaze calibration's samples with the eye-tracker state they were taken in.

    eye_states is opaque, JSON-ready data from the eye tracker (e.g. locked eye centers):
    the samples' raw gaze is only meaningful with the same eye model restored.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    data = {"saved_at": datetime.now().isoformat(timespec="seconds"), "eye_states": eye_states,
            **calibration.to_json()}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_gaze_session(path):
    """Returns (GazeCalibration, eye_states, saved_at); raises OSError/ValueError/KeyError."""
    with open(path) as f:
        data = json.load(f)
    return GazeCalibration.from_json(data), data["eye_states"], data.get("saved_at")


@dataclass
class MarkerDetection:
    corners: np.ndarray  # (1, 4, 2) as returned by the ArUco detector
    point: np.ndarray  # marker center in the camera frame, metres
    pixel: np.ndarray  # projected marker center


class MarkerTarget:
    """A printed ArUco marker whose center the user fixates; its pose gives a metric 3D target."""

    def __init__(self, length_m=0.10, marker_id=0):
        self.length_m = length_m  # side of the black square, as printed
        self.marker_id = marker_id
        self.dictionary = cv2.aruco.getPredefinedDictionary(TARGET_DICTIONARY)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())

    def detect(self, frame, camera):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None
        for marker_corners, marker_id in zip(corners, ids.ravel()):
            if marker_id != self.marker_id:
                continue
            half = self.length_m / 2.0
            # Corner order expected by SOLVEPNP_IPPE_SQUARE: TL, TR, BR, BL.
            object_points = np.array(
                [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32
            )
            ok, _, tvec = cv2.solvePnP(
                object_points, marker_corners.reshape(4, 2), camera.K, camera.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or tvec[2, 0] <= 0:
                return None
            point = tvec.reshape(3)
            return MarkerDetection(marker_corners, point, camera.project(point))
        return None

    def printable_image(self, side_pixels=800, border_pixels=100):
        """Marker with the white quiet zone that detection needs."""
        marker = cv2.aruco.generateImageMarker(self.dictionary, self.marker_id, side_pixels)
        return cv2.copyMakeBorder(marker, *[border_pixels] * 4, cv2.BORDER_CONSTANT, value=255)


class IntrinsicsCalibrator:
    """Collects ChArUco board views and calibrates the camera (Zhang's method).

    The board's printed scale does not matter for the intrinsics.
    """

    def __init__(self, image_size, views_needed=15, min_corners=12, min_interval_s=1.0):
        self.image_size = tuple(image_size)
        self.views_needed = views_needed
        self.min_corners = min_corners
        self.min_interval_s = min_interval_s
        self.board = cv2.aruco.CharucoBoard(
            (7, 5), 0.03, 0.022, cv2.aruco.getPredefinedDictionary(BOARD_DICTIONARY)
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.views = []
        self.last_capture_time = None

    @property
    def done(self):
        return len(self.views) >= self.views_needed

    def add_frame(self, frame, now):
        """Detect the board; keep the view if enough corners and time passed.

        Returns (charuco_corners, charuco_ids) for drawing; either may be None.
        """
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(frame)
        # OpenCV 5 may return empty or mismatched arrays when the board is not (fully) visible.
        if (charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0
                or len(charuco_ids) != len(charuco_corners)):
            charuco_corners, charuco_ids = None, None
        else:  # OpenCV 5 returns (N, 2) / (N,); the aruco drawing functions need (N, 1, 2) / (N, 1)
            charuco_corners = np.asarray(charuco_corners, np.float32).reshape(-1, 1, 2)
            charuco_ids = np.asarray(charuco_ids, np.int32).reshape(-1, 1)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        interval_ok = self.last_capture_time is None or now - self.last_capture_time >= self.min_interval_s
        if not self.done and corner_count >= self.min_corners and interval_ok:
            object_points, image_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
            self.views.append((object_points, image_points))
            self.last_capture_time = now
        return charuco_corners, charuco_ids

    def calibrate(self):
        """Returns (CameraModel, RMS reprojection error in pixels)."""
        if not self.done:
            raise ValueError(f"Need {self.views_needed} board views, have {len(self.views)}.")
        # k1, k2 only: with few views k3 and tangential terms overfit and blow up at the image corners.
        rms, K, dist, _, _ = cv2.calibrateCamera(
            [view[0] for view in self.views], [view[1] for view in self.views], self.image_size, None, None,
            flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        )
        width, height = self.image_size
        corner_r2 = max(((u - K[0, 2]) / K[0, 0]) ** 2 + ((v - K[1, 2]) / K[1, 1]) ** 2
                        for u in (0, width) for v in (0, height))
        k1, k2 = dist.ravel()[:2]
        radial_factor = 1 + k1 * corner_r2 + k2 * corner_r2 ** 2
        if not 0.6 <= radial_factor <= 1.4:
            raise ValueError(f"lens distortion is implausible (factor {radial_factor:.2f} at the image corners); "
                             "repeat with the board in every corner of the image")
        return CameraModel(K, dist, self.image_size, calibrated=True), float(rms)

    def board_image(self, size=(2100, 1500), margin=80):
        return self.board.generateImage(size, marginSize=margin)


@dataclass
class CalibrationResult:
    models: dict  # eye_id -> gaze_geometry model with the eye origin fitted (or reused)
    baselines: dict  # eye_id -> same samples fitted with the eye at the camera; None if that fit failed
    origins_fitted: bool  # False when saved origins were reused for a single-depth set

    @property
    def eye_separation_mm(self):
        if len(self.models) != 2:
            return None
        first, second = self.models.values()
        return float(1000 * np.linalg.norm(first["origin"] - second["origin"]))

    def summary(self):
        rms = " ".join(f"{eye_id[0].upper()} {model['rms_deg']:.2f}" for eye_id, model in self.models.items())
        text = f"Calibrated, RMS deg {rms}"
        if self.eye_separation_mm is not None:
            text += f", eye separation {self.eye_separation_mm:.0f} mm"
        return text


@dataclass
class GazeCalibration:
    """Samples pair each calibrated eye's raw gaze with the marker's 3D point."""

    eye_ids: tuple
    samples: list = field(default_factory=list)

    def add_sample(self, gaze_dirs, detection):
        """gaze_dirs: {eye_id: raw gaze}. Raises ValueError if a calibrated eye has no gaze."""
        missing = [eye_id for eye_id in self.eye_ids if gaze_dirs.get(eye_id) is None]
        if missing:
            raise ValueError(f"No {' or '.join(missing)} eye gaze")
        self.samples.append({
            "gaze": {eye_id: np.asarray(gaze_dirs[eye_id], dtype=float).copy() for eye_id in self.eye_ids},
            "point": np.asarray(detection.point, dtype=float).copy(),
            "pixel": np.asarray(detection.pixel, dtype=float).copy(),
        })

    def band_counts(self):
        counts = {"near": 0, "mid": 0, "far": 0}
        for sample in self.samples:
            counts[gg.depth_band(sample["point"][2])] += 1
        return counts

    @property
    def ready(self):
        """Enough NEAR and FAR samples to fit the eye origins."""
        counts = self.band_counts()
        return counts["near"] >= gg.MIN_SAMPLES and counts["far"] >= gg.MIN_SAMPLES

    def progress_text(self):
        counts = self.band_counts()
        return (f"NEAR(<{gg.NEAR_BAND_MAX_M} m) {counts['near']}/{gg.MIN_SAMPLES}  "
                f"FAR(>{gg.FAR_BAND_MIN_M} m) {counts['far']}/{gg.MIN_SAMPLES}  mid {counts['mid']}")

    def solve(self, saved_origins=None):
        """Fit every eye. Without enough depth span, reuse saved_origins; raises ValueError."""
        saved_origins = saved_origins or {}
        if len(self.samples) < gg.MIN_SAMPLES:
            raise ValueError(f"need at least {gg.MIN_SAMPLES} samples")
        targets = np.array([sample["point"] for sample in self.samples])
        origins_fitted = np.ptp(1.0 / targets[:, 2]) >= gg.MIN_INVERSE_DEPTH_SPAN
        models, baselines = {}, {}
        for eye_id in self.eye_ids:
            gazes = np.array([sample["gaze"][eye_id] for sample in self.samples])
            if origins_fitted:
                fixed_origin = None
            elif eye_id in saved_origins:
                fixed_origin = saved_origins[eye_id]
            else:
                raise ValueError("eye offsets are unknown; add FAR samples")
            try:
                models[eye_id] = gg.fit_eye_model(gazes, targets, fixed_origin=fixed_origin)
            except ValueError as exc:
                raise ValueError(f"{eye_id} eye: {exc}") from exc
            try:  # Comparison only: its failure must not block the calibration.
                baselines[eye_id] = gg.fit_eye_model(gazes, targets, fixed_origin=np.zeros(3))
            except ValueError:
                pass
        if set(baselines) != set(self.eye_ids):
            baselines = None
        return CalibrationResult(models, baselines, bool(origins_fitted))

    def to_json(self):
        return {"eye_ids": list(self.eye_ids), "samples": [
            {"gaze": {eye_id: gaze.tolist() for eye_id, gaze in sample["gaze"].items()},
             "point": sample["point"].tolist(), "pixel": sample["pixel"].tolist()}
            for sample in self.samples
        ]}

    @classmethod
    def from_json(cls, data):
        samples = [{"gaze": {eye_id: np.asarray(gaze, dtype=float) for eye_id, gaze in sample["gaze"].items()},
                    "point": np.asarray(sample["point"], dtype=float),
                    "pixel": np.asarray(sample["pixel"], dtype=float)}
                   for sample in data["samples"]]
        return cls(tuple(data["eye_ids"]), samples)


@dataclass
class GazeEstimate:
    point: np.ndarray
    depth: float
    pixel: np.ndarray


class BinocularGazeMapper:
    """Raw gaze per eye -> gaze point in the camera, with vergence depth smoothed over frames."""

    def __init__(self, models, camera, alpha=0.2):
        self.models = models
        self.camera = camera
        self.depth_filter = gg.InverseDepthFilter(alpha=alpha)

    def update(self, gaze_dirs):
        """gaze_dirs: {eye_id: raw gaze or None}. Returns GazeEstimate or None."""
        point, depth = gg.binocular_gaze_point(self.models, gaze_dirs, self.depth_filter)
        if point is None:
            return None
        pixel = self.camera.project(point)
        return None if pixel is None else GazeEstimate(point, depth, pixel)


class AccuracyTest:
    """Front-camera angular error between gaze and marker, with the no-offset model for comparison."""

    def __init__(self):
        self.samples = []

    def record(self, camera, detection, gaze_pixel, baseline_pixel=None):
        sample = {
            "depth": float(detection.point[2]),
            "error": camera.angular_error_degrees(gaze_pixel, detection.pixel),
            "baseline_error": None if baseline_pixel is None
            else camera.angular_error_degrees(baseline_pixel, detection.pixel),
        }
        self.samples.append(sample)
        return sample

    def summary_by_band(self):
        """{band: (count, mean error, mean baseline error or None)} for bands with samples."""
        summary = {}
        for band in ("near", "mid", "far"):
            samples = [s for s in self.samples if gg.depth_band(s["depth"]) == band]
            if not samples:
                continue
            baselines = [s["baseline_error"] for s in samples if s["baseline_error"] is not None]
            summary[band] = (
                len(samples),
                float(np.mean([s["error"] for s in samples])),
                float(np.mean(baselines)) if baselines else None,
            )
        return summary
