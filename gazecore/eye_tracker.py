"""Eye model per eye camera: eyeball center from pupil normals, sphere radius, gaze direction.

The math is Orlosky's 3D tracker, moved unchanged from the front-camera tracker, where it ran
on module globals swapped per eye. EyeTracker keeps the same state under the same names
(STATE_NAMES) and updates it in the same order, so its output is bit-identical
(gazecore/tests/golden). It neither draws (see eye_overlay) nor touches the gaze calibration:
what used to reset it from inside the tracking code is now an event for the caller.
"""

import logging
import math
import random
from dataclasses import dataclass, field

import numpy as np

from .pupil_detection import PupilObservation, detect_pupil, pupil_quality

log = logging.getLogger(__name__)

MAX_RAYS = 100  # Limit the number of stored pupil rays
MIN_MODEL_CENTERS = 100  # Minimum model centers required before updating sphere radius
PUPIL_CONFIDENCE_THRESHOLD = 0.85  # Minimum pupil confidence for storing a ray
PUPIL_CONFIDENCE_THRESHOLD_SPHERE = 0.90  # Minimum pupil confidence for determining eye sphere radius
INTERSECTION_RAY_COUNT = 4  # Rays sampled for each intersection estimate
MINIMUM_INTERSECTION_ANGLE_DEGREES = 8  # Minimum angle between sampled rays
MAX_STORED_INTERSECTIONS = 1500
MODEL_CENTER_AVERAGE_COUNT = 200
CENTER_OVERRIDE_FRAMES_NEEDED = 30  # High-confidence pupil centers averaged per override
INITIAL_MODEL_CENTER = (320, 240)  # Preserve the last valid eye center

STATE_NAMES = (
    "ray_lines", "model_centers", "prev_model_center_avg", "max_observed_distance",
    "last_sphere_radius_ellipse", "last_gaze_vector_text", "last_tracking_result",
    "stored_intersections", "last_sphere_center", "last_gaze_dir", "calibrated_sphere_center",
    "sphere_center_locked_2d", "locked_model_center_avg", "eye_sphere_adjustment_enabled",
    "center_override_collecting", "center_override_samples",
)


def distance_to_pupil_outer_edge(eye_center, pupil_ellipse):
    pupil_center, axes, angle_degrees = pupil_ellipse
    direction_x = pupil_center[0] - eye_center[0]
    direction_y = pupil_center[1] - eye_center[1]
    center_distance = math.hypot(direction_x, direction_y)

    semi_axis_x = axes[0] / 2
    semi_axis_y = axes[1] / 2
    if center_distance == 0 or semi_axis_x <= 0 or semi_axis_y <= 0:
        return None

    unit_x = direction_x / center_distance
    unit_y = direction_y / center_distance
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)

    local_x = cosine * unit_x + sine * unit_y
    local_y = -sine * unit_x + cosine * unit_y
    edge_offset = 1 / math.sqrt(
        (local_x / semi_axis_x) ** 2
        + (local_y / semi_axis_y) ** 2
    )

    return center_distance + edge_offset

def update_and_average_point(point_list, new_point, N):
    """
    Adds a new point to the list, keeps only the last N points, 
    and returns the average of those points.
    
    Parameters:
    - point_list: Global list storing past points [(x1, y1), (x2, y2), ...]
    - new_point: Tuple (x, y) representing the new point to add.
    - N: Maximum number of points to keep in the list.
    
    Returns:
    - (avg_x, avg_y): The average point as a tuple of integers.
    - None if the list is empty.
    """
    point_list.append(new_point)  # Add new point

    if len(point_list) > N:
        point_list.pop(0)  # Remove the oldest point to maintain size N

    if not point_list:
        return None  # No points available

    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))

    return (avg_x, avg_y)

#Removes the oldest intersections to ensure only the last M intersections remain.
def prune_intersections(intersections, maximum_intersections):

    if len(intersections) <= maximum_intersections:
        return intersections  # No need to prune if within the limit

    # Keep only the last M intersections
    pruned_intersections = intersections[-maximum_intersections:]

    return pruned_intersections

def find_line_intersection(ellipse1, ellipse2):
    """
    Computes the intersection of two lines that are orthogonal to the surface of given ellipses.
    
    Parameters:
    - ellipse1, ellipse2: Ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    
    Returns:
    - (x, y): Intersection point of the two lines, or None if parallel.
    """

    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

    # Convert angles to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    # Compute direction vectors for the two lines
    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

    # Line equations in parametric form:
    # (x1, y1) + t1 * (dx1, dy1) = (x2, y2) + t2 * (dx2, dy2)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])

    # Solve for t1, t2 using linear algebra (if the determinant is nonzero)
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel and do not intersect

    t1, t2 = np.linalg.solve(A, B)

    # Compute intersection point
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1

    return (int(intersection_x), int(intersection_y))

def gaze_vector_from_pixels(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Compute 3D gaze direction from pupil and sphere center screen coordinates.
    x, y: pupil center; center_x, center_y: projected eyeball center (eye frame pixels).
    Returns:
        sphere_center (np.ndarray): 3D position of the sphere center in world space
        gaze_direction (np.ndarray): Normalized 3D direction vector from sphere center
    """

    # Get viewport dimensions
    viewport_width = screen_width
    viewport_height = screen_height

    # Define camera and projection settings
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    # Camera position is fixed at z = 3
    camera_position = np.array([0.0, 0.0, 3.0])

    # Compute size of far plane in world units
    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    # Convert screen (x, y) to normalized device coordinates [-1, 1]
    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    # Project pupil center to far plane coordinates in world space
    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    # Compute ray direction from camera to far plane point
    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    # Sphere radius and center offset
    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    # Compute intersection with sphere
    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # Compute the closest point to the sphere (tangent point approximation)
        t = -np.dot(direction, L) / np.dot(direction, direction)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        positive_intersections = [value for value in (t1, t2) if value > 0]
        if positive_intersections:
            t = min(positive_intersections)
        else:
            t = -np.dot(direction, L) / np.dot(direction, direction)

    # Final intersection point
    intersection_point = origin + t * direction

    # Convert to local space relative to sphere center
    intersection_local = intersection_point - sphere_center
    intersection_norm = np.linalg.norm(intersection_local)
    if intersection_norm < 1e-6:
        target_direction = np.array([0.0, 0.0, 1.0])
    else:
        target_direction = intersection_local / intersection_norm

    # Local green ring direction
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    # Compute rotation to align local +Z to target
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        rotation_axis = np.array([1.0, 0.0, 0.0])
        rotation_axis_norm = 1.0

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # Rotation matrix from axis-angle
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
        [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
        [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
    ])

    # Rotate +Z vector to get gaze direction
    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    return sphere_center, gaze_rotated


@dataclass
class CenterOverrideDone:
    """The O override set the eyeball center and affixed the sphere.

    The gaze calibration was built from the old center, so the caller must start it over.
    """

    eye_id: str
    center: tuple


@dataclass
class EyeResult:
    """One eye frame's outcome. The tracker's state after the frame is on the EyeTracker."""

    observation: PupilObservation
    status: str  # "ok", "no_pupil" (no ellipse: gaze lost), "no_dark_area" (frame skipped, state untouched)
    model_center: tuple  # Projected eyeball center used this frame (eye frame pixels)
    sphere_radius: float  # Eye sphere radius drawn on the eye view (pixels)
    gaze_dir: np.ndarray  # Raw gaze after this frame, as fed to the gaze calibration (None when lost)
    gaze_output: list  # Sphere center + gaze for gaze_vector.txt (6 floats); None when not computed
    gaze_vector_text: tuple  # (origin, direction) lines of the eye view; kept through dropouts
    locked: bool
    adjusting: bool  # The eyeball center and radius still adapt
    override_samples: int  # Pupil centers collected by a running O override, else None
    events: list = field(default_factory=list)
    quality: float = 0.0  # pupil_quality: what the app judges a fit by; the model still uses confidence

    @property
    def frame(self):
        return self.observation.frame

    @property
    def ellipse(self):
        return self.observation.ellipse

    @property
    def pupil_center(self):
        return self.observation.center

    @property
    def confidence(self):
        return self.observation.confidence


class EyeTracker:
    """Eye model of one eye camera; feed it every frame of that camera with process()."""

    def __init__(self, eye_id, rng=random):
        self.eye_id = eye_id
        # Both eyes share the global generator by default, like the original tracker.
        self.rng = rng
        # Off: the original tracker. On: once the eyeball center is fixed, the darkest-area search
        # stays inside the eye circle (config, not tracking state, so not in STATE_NAMES).
        self.search_in_eye_circle = False
        # 1: the original single darkest-area seed. More: detect_pupil's multi-seed search, which
        # keeps a lash clump darker than the pupil from taking the fit (and the eye model).
        self.pupil_seeds = 1
        self.reset()

    def reset(self):
        self.ray_lines = []  # Store recent pupil ellipse rays
        self.model_centers = []  # Store recent estimated eye centers
        self.prev_model_center_avg = INITIAL_MODEL_CENTER
        self.max_observed_distance = 0  # Initialize adaptive radius
        self.last_sphere_radius_ellipse = None  # Last pupil ellipse used to expand the eye sphere radius
        self.last_gaze_vector_text = None  # Preserve vector text during temporary detection dropouts
        self.last_tracking_result = None  # Store the latest tracker output
        self.stored_intersections = []  # Stores all past intersections
        self.last_sphere_center = None
        self.last_gaze_dir = None
        self.calibrated_sphere_center = None
        self.sphere_center_locked_2d = False
        self.locked_model_center_avg = INITIAL_MODEL_CENTER
        self.eye_sphere_adjustment_enabled = True  # Automatic eye center/radius updates (F)
        self.center_override_collecting = False
        self.center_override_samples = []

    def get_state(self):
        """The state under the original tracker's global names (shared, not copied)."""
        return {name: getattr(self, name) for name in STATE_NAMES}

    def set_state(self, state):
        for name in STATE_NAMES:
            setattr(self, name, state[name])

    @property
    def adjusting(self):
        return self.eye_sphere_adjustment_enabled and not self.sphere_center_locked_2d

    @property
    def fixed(self):
        """Affixed (F, override) or locked: the eyeball center no longer moves."""
        return self.sphere_center_locked_2d or not self.eye_sphere_adjustment_enabled

    # --- Commands ---

    def set_adjustment_enabled(self, enabled):
        self.eye_sphere_adjustment_enabled = enabled

    def lock_sphere(self):
        """Freeze the eyeball center (and its 3D sphere center) for a gaze calibration."""
        if self.last_sphere_center is not None:
            self.calibrated_sphere_center = self.last_sphere_center.copy()
        self.locked_model_center_avg = self.prev_model_center_avg
        self.sphere_center_locked_2d = True

    def unlock_sphere(self):
        self.calibrated_sphere_center = None
        self.last_sphere_center = None
        self.last_gaze_dir = None
        self.sphere_center_locked_2d = False
        self.locked_model_center_avg = self.prev_model_center_avg

    def toggle_center_override(self):
        """Start or cancel the O override: look straight into the eye camera's lens.

        The pupil center then projects onto the eyeball center. Returns True when started.
        """
        self.center_override_samples = []
        self.center_override_collecting = not self.center_override_collecting
        return self.center_override_collecting

    def locked_state(self):
        """The eye-model state that raw gaze depends on (JSON-ready), as saved with samples."""
        sphere_center = self.calibrated_sphere_center
        return {
            "model_center": [int(v) for v in self.locked_model_center_avg],
            "sphere_radius": float(self.max_observed_distance),
            "sphere_center": None if sphere_center is None else [float(v) for v in sphere_center],
            "adjustment_enabled": bool(self.eye_sphere_adjustment_enabled),
        }

    def apply_locked_state(self, state):
        """Lock the model to a saved locked_state(), so restored samples match it again."""
        self.prev_model_center_avg = self.locked_model_center_avg = tuple(int(v) for v in state["model_center"])
        self.max_observed_distance = state["sphere_radius"]
        sphere_center = state["sphere_center"]
        self.calibrated_sphere_center = None if sphere_center is None else np.asarray(sphere_center, dtype=float)
        self.eye_sphere_adjustment_enabled = state["adjustment_enabled"]
        self.sphere_center_locked_2d = True

    # --- Per frame ---

    def process(self, frame):
        """Track one BGR frame of this eye's camera (any size)."""
        return self.update(detect_pupil(frame, self.search_region(), self.pupil_seeds))

    def search_region(self):
        """(cx, cy, radius) of the eye circle when search_in_eye_circle applies, else None."""
        if not (self.search_in_eye_circle and self.fixed and self.max_observed_distance > 0):
            return None
        center = self.locked_model_center_avg if self.sphere_center_locked_2d else self.prev_model_center_avg
        return center[0], center[1], float(self.max_observed_distance)

    def update(self, observation):
        if observation.darkest_point is None:
            return self._result(observation, "no_dark_area", self.prev_model_center_avg, None, [])

        events = []
        final_rotated_rect = observation.ellipse
        center_x, center_y = observation.center
        best_ratio_under_ellipse = observation.confidence

        if final_rotated_rect is not None:
            if self.center_override_collecting and best_ratio_under_ellipse >= PUPIL_CONFIDENCE_THRESHOLD:
                self._collect_center_override_sample(center_x, center_y, events)

            if (
                self.eye_sphere_adjustment_enabled
                and not self.sphere_center_locked_2d
                and best_ratio_under_ellipse >= PUPIL_CONFIDENCE_THRESHOLD
            ):
                self.ray_lines.append(final_rotated_rect)
                if len(self.ray_lines) > MAX_RAYS:
                    num_to_remove = len(self.ray_lines) - MAX_RAYS
                    self.ray_lines = self.ray_lines[num_to_remove:]  # Keep only the last MAX_RAYS elements

        model_center_average = (self.locked_model_center_avg if self.sphere_center_locked_2d
                                else self.prev_model_center_avg)

        if self.eye_sphere_adjustment_enabled and not self.sphere_center_locked_2d:
            model_center = self._compute_average_intersection(
                observation.frame.shape,
                self.ray_lines,
                INTERSECTION_RAY_COUNT,
                MAX_STORED_INTERSECTIONS,
                MINIMUM_INTERSECTION_ANGLE_DEGREES,
            )
            if model_center is not None and model_center != (0, 0):
                model_center_average = update_and_average_point(
                    self.model_centers, model_center, MODEL_CENTER_AVERAGE_COUNT)

        if model_center_average[0] == 320:
            model_center_average = self.prev_model_center_avg
        if model_center_average[0] != 0:
            self.prev_model_center_avg = model_center_average
            if not self.sphere_center_locked_2d:
                self.locked_model_center_avg = model_center_average

        if (final_rotated_rect is None or center_x is None or center_y is None
                or model_center_average[0] is None or model_center_average[1] is None):
            self.last_tracking_result = None
            self.last_gaze_dir = None
            return self._result(observation, "no_pupil", model_center_average, None, events)

        if self.eye_sphere_adjustment_enabled and not self.sphere_center_locked_2d:
            self._update_eye_sphere_radius(model_center_average, final_rotated_rect, best_ratio_under_ellipse)

        self.last_tracking_result = {
            "pupil_ellipse": {
                "center": [float(final_rotated_rect[0][0]), float(final_rotated_rect[0][1])],
                "axes": [float(final_rotated_rect[1][0]), float(final_rotated_rect[1][1])],
                "angle_degrees": float(final_rotated_rect[2]),
            },
            "eye_center": [int(model_center_average[0]), int(model_center_average[1])],
            "sphere_radius": float(self.max_observed_distance),
        }

        sphere_center, gaze_rotated = gaze_vector_from_pixels(
            center_x, center_y, model_center_average[0], model_center_average[1])
        self.last_sphere_center = sphere_center.copy()
        self.last_gaze_dir = gaze_rotated.copy()
        sphere_center_out = (self.calibrated_sphere_center if self.calibrated_sphere_center is not None
                             else sphere_center)
        gaze_output = [float(v) for v in np.concatenate((sphere_center_out, gaze_rotated))]

        center, direction = sphere_center_out, gaze_rotated
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
        self.last_gaze_vector_text = (origin_text, dir_text)
        return self._result(observation, "ok", model_center_average, gaze_output, events)

    def _result(self, observation, status, model_center, gaze_output, events):
        return EyeResult(
            observation, status, model_center, float(self.max_observed_distance), self.last_gaze_dir,
            gaze_output, self.last_gaze_vector_text, bool(self.sphere_center_locked_2d), bool(self.adjusting),
            len(self.center_override_samples) if self.center_override_collecting else None, events,
            pupil_quality(observation),  # Now: the eye overlay later draws on the frame
        )

    def _collect_center_override_sample(self, center_x, center_y, events):
        self.center_override_samples.append((center_x, center_y))
        if len(self.center_override_samples) < CENTER_OVERRIDE_FRAMES_NEEDED:
            return

        # Median rejects the odd blink or lash-line fit.
        median_center = np.median(np.array(self.center_override_samples), axis=0)
        self.prev_model_center_avg = (int(median_center[0]), int(median_center[1]))
        self.center_override_collecting = False
        self.eye_sphere_adjustment_enabled = False
        self.unlock_sphere()
        events.append(CenterOverrideDone(self.eye_id, self.prev_model_center_avg))
        log.info("%s eye center overridden to %s; sphere affixed.", self.eye_id.title(), self.prev_model_center_avg)

    def _update_eye_sphere_radius(self, eye_center, current_pupil_ellipse, current_pupil_confidence):
        if self.last_sphere_radius_ellipse is not None:
            anchored_distance = distance_to_pupil_outer_edge(
                eye_center,
                self.last_sphere_radius_ellipse,
            )
            if anchored_distance is not None:
                self.max_observed_distance = anchored_distance

        if (
            current_pupil_ellipse is not None
            and current_pupil_confidence >= PUPIL_CONFIDENCE_THRESHOLD_SPHERE
            and len(self.model_centers) >= MIN_MODEL_CENTERS
        ):
            current_distance = distance_to_pupil_outer_edge(
                eye_center,
                current_pupil_ellipse,
            )
            if (
                current_distance is not None
                and (
                    self.last_sphere_radius_ellipse is None
                    or current_distance > self.max_observed_distance
                )
            ):
                self.max_observed_distance = current_distance
                self.last_sphere_radius_ellipse = current_pupil_ellipse
                log.info("Radius-increasing ellipse confidence: %.2f%%", current_pupil_confidence * 100)

    def _compute_average_intersection(self, frame_shape, ray_lines, number_lines, total_lines, minimum_angle_degrees):
        """
        Selects `number_lines` random lines from the list, computes their intersections,
        conditionally stores them (only if they agree within `pixel_limit`), and prunes
        stored intersections when exceeding `total_lines`.
        """
        pixel_limit = 30
        angle_threshold = 5  # degrees

        if len(ray_lines) < 2 or number_lines < 2:
            return (0, 0)

        height, width = frame_shape[:2]

        selected_lines = self.rng.sample(ray_lines, min(number_lines, len(ray_lines)))

        intersections = []
        for i in range(len(selected_lines) - 1):
            line1 = selected_lines[i]
            line2 = selected_lines[i + 1]

            angle1 = line1[2]
            angle2 = line2[2]

            if abs(angle1 - angle2) >= minimum_angle_degrees:
                intersection = find_line_intersection(line1, line2)
                if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                    intersections.append(intersection)

        # Nothing usable this frame
        if not intersections:
            return (0, 0)

        # Check mutual agreement within pixel_limit
        accept = True
        if len(intersections) >= 2:
            for i in range(len(intersections)):
                for j in range(i + 1, len(intersections)):
                    # --- distance check ---
                    dx = intersections[i][0] - intersections[j][0]
                    dy = intersections[i][1] - intersections[j][1]
                    if (dx * dx + dy * dy) ** 0.5 > pixel_limit:
                        accept = False
                        break

                    # --- angle check ---
                    angle_i = selected_lines[i][2]
                    angle_j = selected_lines[j][2]
                    if abs(angle_i - angle_j) < angle_threshold:
                        accept = False
                        break
                if not accept:
                    break

        if accept:
            self.stored_intersections.extend(intersections)

        # Prune stored intersections
        if len(self.stored_intersections) > total_lines:
            self.stored_intersections = prune_intersections(self.stored_intersections, total_lines)

        if not self.stored_intersections:
            return (0, 0)

        avg_x = np.mean([pt[0] for pt in self.stored_intersections])
        avg_y = np.mean([pt[1] for pt in self.stored_intersections])

        if np.isnan(avg_x) or np.isnan(avg_y):
            return (0, 0)

        return (int(avg_x), int(avg_y))
