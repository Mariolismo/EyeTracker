"""One processing step per pair of eye frames: eye trackers, binocular gaze, front camera.

No Qt here, so it runs headless in tests. The worker thread owns the Pipeline: the UI only
reads the Bundles it publishes and sends commands (callables run between steps).

The order is the front-camera tracker's: left eye, right eye (both share one random
generator, so the order matters), then one binocular update, then the front frame.
"""

import collections
import random
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from gazecore import front_calibration as fc
from gazecore import paths
from gazecore.eye_overlay import draw_eye_overlay
from gazecore.eye_tracker import CenterOverrideDone, EyeTracker

from .capture_gates import cell_of

EYE_IDS = ("left", "right")
FRONT_SIZE = (640, 480)  # The front camera intrinsics are calibrated at this size
EYE_FRESH_S = 0.2  # An older eye frame gives no gaze (that eye is offline)
HISTORY_LENGTH = 120  # Frames kept for averaging calibration samples (~4 s)
CHARUCO_VIEWS_PER_CELL = 2  # Lens views kept per cell of a 3x3 grid (3 once 6 cells have some)

# Front camera modes
FRONT_NONE, FRONT_MARKER, FRONT_CHARUCO = "none", "marker", "charuco"


def orient(frame, binding):
    """Apply a role's flip (upside down) and mirror (left-right), like the tracker's checkbox."""
    if binding is None:
        return frame
    if binding.flip:
        frame = cv2.flip(frame, 0)
    if binding.mirror:
        frame = cv2.flip(frame, 1)
    return frame


@dataclass
class HistoryEntry:
    """What a calibration sample averages over: one entry per step."""

    t: float
    gaze: dict  # {eye: raw gaze or None}
    confidence: dict  # {eye: pupil quality (EyeResult.quality), 0 when not tracked}
    marker_point: np.ndarray  # Marker center in the front camera frame (m), or None
    marker_pixel: np.ndarray
    marker_generation: int  # Marker size version the detection used
    front_t: float  # Front frame timestamp, None without one
    eye_t: dict
    marker_corners: np.ndarray = None  # (4, 2) image corners of the marker


@dataclass
class Bundle:
    """One step's results, immutable once published."""

    t: float
    seq: int
    eyes: dict  # {eye: EyeResult or None}; result.frame is the drawn eye view
    front: np.ndarray  # Oriented 640x480 front frame (not drawn on), or None
    marker: object  # MarkerDetection or None
    estimate: object  # GazeEstimate or None
    charuco: tuple  # (corners, ids) found by the intrinsics capture, or None
    intrinsics_views: int
    events: list = field(default_factory=list)
    processing_fps: float = 0.0
    screen: object = None  # ScreenGaze when the desktop overlay runs
    history: tuple = ()  # Recent HistoryEntry items, oldest first (capture gates average them)
    charuco_cells: dict = None  # {(row, column): views kept} during the lens calibration


class Pipeline:
    def __init__(self, camera, bindings=None, rng=random, gaze_vector_path=paths.GAZE_VECTOR_FILE):
        self.camera = camera  # CameraModel of the front camera
        self.bindings = bindings or {}  # {role: RoleBinding}, for flip/mirror
        self.trackers = {eye_id: EyeTracker(eye_id, rng) for eye_id in EYE_IDS}
        self.gaze_outputs = {eye_id: [0.0] * 6 for eye_id in EYE_IDS}
        self.mapper = None
        self.baseline_mapper = None
        self.marker_target = fc.MarkerTarget()
        self.marker_generation = 0
        self.marker_valid_from = 0.0
        self.front_mode = FRONT_MARKER
        self.intrinsics_calibrator = None
        self.charuco_cells = {}
        self.screen_tracker = None  # Set while the desktop overlay runs (screen_geometry)
        self.gaze_vector_path = gaze_vector_path
        self.history = collections.deque(maxlen=HISTORY_LENGTH)
        self.seq = 0
        self._last_eye = {}  # {eye: (packet seq, EyeResult)} of the last frame tracked
        self._fps_window = collections.deque(maxlen=30)

    # --- Commands (run on the worker thread between steps) ---

    def set_mapping(self, result):
        """Map gaze with a solved CalibrationResult, or stop mapping with None."""
        if result is None:
            self.mapper = self.baseline_mapper = None
            return
        self.mapper = fc.BinocularGazeMapper(result.models, self.camera)
        self.baseline_mapper = (None if result.baselines is None
                                else fc.BinocularGazeMapper(result.baselines, self.camera))

    def set_marker_side(self, side_m, now=None):
        """A new on-screen marker size; detections from older frames are ignored (0.3 s)."""
        if abs(side_m - self.marker_target.length_m) < 1e-9:
            return
        self.marker_target.length_m = side_m
        self.marker_generation += 1
        self.marker_valid_from = (time.perf_counter() if now is None else now) + 0.3

    def set_camera(self, camera):
        """New front camera intrinsics: the marker poses and gaze mapping change with them."""
        self.camera = camera
        self.set_mapping(None)

    def start_intrinsics(self, calibrator):
        self.intrinsics_calibrator = calibrator
        self.charuco_cells = {}
        self.front_mode = FRONT_CHARUCO

    def stop_intrinsics(self):
        self.intrinsics_calibrator = None
        self.front_mode = FRONT_MARKER

    def _charuco_view_wanted(self, corners):
        """A view counts as new only where the grid needs more, so the views spread over the image."""
        row, column = cell_of(np.asarray(corners).reshape(-1, 2).mean(axis=0), FRONT_SIZE)
        limit = CHARUCO_VIEWS_PER_CELL + (len(self.charuco_cells) >= 6)
        return self.charuco_cells.get((row, column), 0) < limit, (row, column)

    def apply_eye_states(self, eye_states):
        for eye_id, state in eye_states.items():
            self.trackers[eye_id].apply_locked_state(state)

    def locked_eye_states(self):
        return {eye_id: tracker.locked_state() for eye_id, tracker in self.trackers.items()}

    # --- Per step ---

    def step(self, packets, now=None):
        """packets: {role: FramePacket or None}. Returns a Bundle."""
        now = time.perf_counter() if now is None else now
        events = []
        eyes, gaze_dirs, confidence, eye_t = {}, {}, {}, {}
        for eye_id in EYE_IDS:
            packet = packets.get(eye_id)
            tracker = self.trackers[eye_id]
            if packet is None or now - packet.t > EYE_FRESH_S:
                eyes[eye_id], gaze_dirs[eye_id], confidence[eye_id], eye_t[eye_id] = None, None, 0.0, None
                continue
            seen_seq, result = self._last_eye.get(eye_id, (None, None))
            if packet.seq != seen_seq:  # A frame is tracked once: tracking changes the eye model
                result = tracker.process(orient(packet.frame, self.bindings.get(eye_id)))
                if result.gaze_output is not None:
                    self.gaze_outputs[eye_id] = result.gaze_output
                events += result.events
                draw_eye_overlay(result, show_quality=True)
                self._last_eye[eye_id] = (packet.seq, result)
            eyes[eye_id] = result
            gaze_dirs[eye_id] = tracker.last_gaze_dir
            confidence[eye_id] = result.quality if result.status == "ok" else 0.0
            eye_t[eye_id] = packet.t
        if any(isinstance(event, CenterOverrideDone) for event in events):
            self.set_mapping(None)  # Built from the old eye center; the controller decides what next

        # Both eyes are current now, so the vergence depth uses one moment in time.
        estimate = None if self.mapper is None else self.mapper.update(gaze_dirs)

        front, marker, charuco = None, None, None
        packet = packets.get("front")
        if packet is not None:
            front = cv2.resize(orient(packet.frame, self.bindings.get("front")), FRONT_SIZE)
            if self.front_mode == FRONT_CHARUCO and self.intrinsics_calibrator is not None:
                calibrator = self.intrinsics_calibrator
                corners, ids, _, _ = calibrator.detector.detectBoard(front)
                if ids is not None and corners is not None and len(ids) >= calibrator.min_corners:
                    wanted, cell = self._charuco_view_wanted(corners)
                    if wanted:  # add_frame detects again and applies its own corner and interval rules
                        views = len(calibrator.views)
                        corners, ids = calibrator.add_frame(front, now)
                        if len(calibrator.views) > views:
                            self.charuco_cells[cell] = self.charuco_cells.get(cell, 0) + 1
                if ids is not None and corners is not None and len(ids) and len(ids) == len(corners):
                    charuco = (np.asarray(corners, np.float32).reshape(-1, 1, 2), np.asarray(ids, np.int32).reshape(-1, 1))
            elif self.front_mode == FRONT_MARKER and packet.t >= self.marker_valid_from:
                marker = self.marker_target.detect(front, self.camera)
        screen = None
        if self.screen_tracker is not None and front is not None:
            screen = self.screen_tracker.update(front, self.mapper, gaze_dirs, now)

        self.history.append(HistoryEntry(
            now, {eye: None if g is None else g.copy() for eye, g in gaze_dirs.items()}, confidence,
            None if marker is None else marker.point, None if marker is None else marker.pixel,
            self.marker_generation, None if packet is None else packet.t, eye_t,
            None if marker is None else marker.corners.reshape(4, 2)))
        self.write_gaze_vector(estimate)

        self.seq += 1
        self._fps_window.append(now)
        fps = 0.0 if len(self._fps_window) < 2 else (len(self._fps_window) - 1) / max(
            1e-6, self._fps_window[-1] - self._fps_window[0])
        return Bundle(now, self.seq, eyes, front, marker, estimate, charuco,
                      0 if self.intrinsics_calibrator is None else len(self.intrinsics_calibrator.views),
                      events, fps, screen, tuple(self.history),
                      dict(self.charuco_cells) if self.front_mode == FRONT_CHARUCO else None)

    def write_gaze_vector(self, estimate):
        """CSV: left origin+direction, right origin+direction (eye model units), then the
        gaze point (front camera frame, metres) and its depth; zeros when unavailable."""
        if self.gaze_vector_path is None:
            return
        if estimate is None:
            point, depth = [0.0, 0.0, 0.0], 0.0
        else:
            point, depth = list(estimate.point), estimate.depth
        values = self.gaze_outputs["left"] + self.gaze_outputs["right"] + point + [depth]
        try:
            with open(self.gaze_vector_path, "w") as f:
                f.write(",".join(f"{value:.6f}" for value in values) + "\n")
        except OSError:
            pass  # In use by the reader; the next frame writes again
