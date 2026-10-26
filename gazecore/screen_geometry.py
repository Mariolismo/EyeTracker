"""Where on a monitor the user looks: corner markers -> screen pose -> eye rays hit the screen.

Screen frame: origin at the screen's top-left pixel, X right, Y down, Z into the screen
(right-handed), metres. A viewer in front of the screen therefore has a negative Z. The front
camera frame is OpenCV's (x right, y down, z forward), so a camera facing the screen squarely
has the identity rotation: p_camera = R @ p_screen + t.

The monitor shows DICT_4X4_50 markers 1-4 at its corners (5-8 at the edge midpoints, optional);
id 0 stays the gaze-calibration target and the intrinsics board uses another dictionary.
The pose is re-solved every frame (no smoothing: a moving head is real motion); the gaze
point is smoothed afterwards, in mm on the screen (filters.OneEuroFilter2D).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import gaze_geometry as gg
from .front_calibration import TARGET_DICTIONARY

CORNER_IDS = (1, 2, 3, 4)  # Top-left, top-right, bottom-right, bottom-left
MIDPOINT_IDS = (5, 6, 7, 8)  # Top, right, bottom, left edge midpoints
MARKER_SIDE_MM = 35.0
QUIET_ZONE = 0.25  # White border around each marker, as a fraction of its side
MIN_MARKERS = 2  # One square alone has a two-fold pose ambiguity at a distance
DISTANCE_RANGE_M = (0.15, 3.0)
MAX_REPROJECTION_PX = 2.5


@dataclass
class ScreenLayout:
    """Marker placement on one monitor, in that monitor's pixels (QScreen geometry)."""

    width_px: int
    height_px: int
    mm_per_px: float
    markers: dict = field(default_factory=dict)  # {id: (center_x, center_y, side_px)}

    @classmethod
    def corners(cls, width_px, height_px, mm_per_px, area=None, side_mm=MARKER_SIDE_MM, midpoints=False):
        """Markers inside area (x, y, w, h: the available geometry, screen-local), quiet zone included."""
        x, y, w, h = area or (0, 0, width_px, height_px)
        side = int(round(side_mm / mm_per_px))
        inset = side / 2 + int(np.ceil(side * QUIET_ZONE)) + 2
        left, right, top, bottom = x + inset, x + w - inset, y + inset, y + h - inset
        markers = {1: (left, top), 2: (right, top), 3: (right, bottom), 4: (left, bottom)}
        if midpoints:
            middle_x, middle_y = (left + right) / 2, (top + bottom) / 2
            markers.update({5: (middle_x, top), 6: (right, middle_y), 7: (middle_x, bottom), 8: (left, middle_y)})
        return cls(width_px, height_px, mm_per_px, {i: (float(cx), float(cy), side) for i, (cx, cy) in markers.items()})

    def to_metres(self, x_px, y_px):
        return np.array([x_px, y_px, 0.0]) * self.mm_per_px / 1000.0

    def to_pixels(self, point_m):
        return np.asarray(point_m[:2], dtype=float) * 1000.0 / self.mm_per_px

    def object_corners(self, marker_id):
        """The marker's corners in the screen frame, in ArUco order (TL, TR, BR, BL)."""
        cx, cy, side = self.markers[marker_id]
        half = side / 2
        return np.array([self.to_metres(cx + dx, cy + dy)
                         for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))])

    def marker_image(self, marker_id):
        """The marker with its white quiet zone, at its exact on-screen pixel size (grey uint8)."""
        side = self.markers[marker_id][2]
        dictionary = cv2.aruco.getPredefinedDictionary(TARGET_DICTIONARY)
        border = int(np.ceil(side * QUIET_ZONE))
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        return cv2.copyMakeBorder(image, *[border] * 4, cv2.BORDER_CONSTANT, value=255)


@dataclass
class ScreenPose:
    rotation: np.ndarray  # p_camera = rotation @ p_screen + translation
    translation: np.ndarray
    reprojection_px: float
    marker_ids: tuple

    @property
    def camera_position(self):
        """The front camera in the screen frame (negative z: in front of the screen)."""
        return -self.rotation.T @ self.translation

    @property
    def distance_m(self):
        return float(np.linalg.norm(self.camera_position))

    def to_screen(self, point):
        return self.rotation.T @ (np.asarray(point, dtype=float) - self.translation)

    def direction_to_screen(self, direction):
        return self.rotation.T @ np.asarray(direction, dtype=float)


def _reprojection_rms(object_points, image_points, rvec, tvec, camera):
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera.K, camera.dist)
    return float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))


def estimate_pose(detections, layout, camera, previous=None):
    """detections: {marker_id: (4, 2) image corners}. Returns the best plausible ScreenPose or None.

    Candidates: both IPPE solutions of the planar point set, plus an iterative solve seeded from
    the previous pose, each refined with Levenberg-Marquardt. Kept only if the camera is in front of the screen, at a plausible
    distance, with every marker in front of the camera, and reprojecting within 2.5 px.
    """
    ids = tuple(sorted(i for i in detections if i in layout.markers))
    if len(ids) < MIN_MARKERS:
        return None
    object_points = np.concatenate([layout.object_corners(i) for i in ids]).astype(np.float64)
    image_points = np.concatenate([np.asarray(detections[i], dtype=np.float64).reshape(4, 2) for i in ids])
    candidates = []
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_points, image_points, camera.K, camera.dist,
                                                 flags=cv2.SOLVEPNP_IPPE)
        candidates += list(zip(rvecs, tvecs))
    except cv2.error:
        pass
    if previous is not None:
        rvec0 = cv2.Rodrigues(previous.rotation)[0]
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera.K, camera.dist, rvec0.copy(),
                                          previous.translation.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                candidates.append((rvec, tvec))
        except cv2.error:
            pass
    best = None
    for rvec, tvec in candidates:
        # IPPE alone can stop several degrees off when the screen is seen square-on.
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera.K, camera.dist,
                                          np.array(rvec, dtype=np.float64).reshape(3, 1),
                                          np.array(tvec, dtype=np.float64).reshape(3, 1))
        rotation = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
        translation = np.asarray(tvec, dtype=float).reshape(3)
        pose = ScreenPose(rotation, translation, 0.0, ids)
        if pose.camera_position[2] >= 0:  # Behind the screen: the mirrored solution
            continue
        if not DISTANCE_RANGE_M[0] <= pose.distance_m <= DISTANCE_RANGE_M[1]:
            continue
        if np.any((object_points @ rotation.T + translation)[:, 2] <= 0):
            continue
        pose.reprojection_px = _reprojection_rms(object_points, image_points, rvec, tvec, camera)
        if pose.reprojection_px > MAX_REPROJECTION_PX:
            continue
        if best is None or pose.reprojection_px < best.reprojection_px:
            best = pose
    return best


def ray_hit(pose, origin, direction):
    """Where a camera-frame ray meets the screen plane (screen frame, metres), or None."""
    o, d = pose.to_screen(origin), pose.direction_to_screen(direction)
    if abs(d[2]) < 1e-9:
        return None
    s = -o[2] / d[2]
    return None if s <= 0 else o + s * d


@dataclass
class ScreenGaze:
    pose: ScreenPose  # None when the screen is not found
    marker_ids: tuple  # Screen markers detected this frame
    pixel: np.ndarray = None  # Smoothed gaze point, screen-local pixels (may be off-screen)
    raw_pixel: np.ndarray = None  # Before smoothing
    eye_pixels: dict = field(default_factory=dict)  # {eye: pixel} per eye ray
    disagreement_mm: float = None  # Distance between the two eyes' hits; None with one eye
    on_screen: bool = False

    @property
    def eyes_used(self):
        return len(self.eye_pixels)


def gaze_hits(pose, models, gaze_dirs, layout):
    """{eye: pixel} where each eye's calibrated ray meets the screen."""
    hits = {}
    for eye, model in (models or {}).items():
        gaze = gaze_dirs.get(eye)
        if model is None or gaze is None:
            continue
        try:
            origin, direction = gg.eye_ray(model, gaze)
        except ValueError:
            continue
        hit = ray_hit(pose, origin, direction)
        if hit is not None:
            hits[eye] = layout.to_pixels(hit)
    return hits


class ScreenTracker:
    """Per front frame: find the screen, then where the eyes' rays hit it (the worker calls update)."""

    def __init__(self, layout, camera, smoother=None):
        self.layout = layout
        self.camera = camera
        self.smoother = smoother  # filters.OneEuroFilter2D (in mm) or None
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TARGET_DICTIONARY),
                                                cv2.aruco.DetectorParameters())
        self.previous = None

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in self.layout.markers}

    def update(self, frame, mapper, gaze_dirs, now):
        detections = self.detect(frame)
        pose = estimate_pose(detections, self.layout, self.camera, self.previous)
        self.previous = pose
        result = ScreenGaze(pose, tuple(sorted(detections)))
        if pose is None or mapper is None:
            return result
        hits = gaze_hits(pose, mapper.models, gaze_dirs, self.layout)
        if not hits:
            return result
        result.eye_pixels = hits
        result.raw_pixel = np.mean(list(hits.values()), axis=0)
        if len(hits) == 2:
            left, right = hits.values()
            result.disagreement_mm = float(np.linalg.norm(left - right) * self.layout.mm_per_px)
        if self.smoother is None:
            result.pixel = result.raw_pixel
        else:
            result.pixel = self.smoother(result.raw_pixel * self.layout.mm_per_px, now) / self.layout.mm_per_px
        x, y = result.pixel
        result.on_screen = bool(0 <= x < self.layout.width_px and 0 <= y < self.layout.height_px)
        return result
