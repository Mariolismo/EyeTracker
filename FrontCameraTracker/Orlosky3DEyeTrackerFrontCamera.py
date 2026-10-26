"""Front-camera tracker: binocular gaze mapped into a forward-facing camera, free of parallax.

Select left eye, right eye and forward cameras. Calibrate by looking at the center of a
printed ArUco marker (DICT_4X4_50, id 0) and pressing C, with the marker both NEAR
(~40 cm) and FAR (~2 m). The two depths let the fit locate each eye relative to the
front camera; at runtime the vergence of the two eyes gives the gaze depth.
F affixes/resumes sphere localization (resuming clears calibration).
"""

import cv2
import random
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time

# Allow launching this script directly from its tracker subfolder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from gazecore import front_calibration as fc
from gazecore import gaze_geometry as gg
from gazecore import paths
from gazecore.eye_tracker import (
    distance_to_pupil_outer_edge, find_line_intersection, gaze_vector_from_pixels, prune_intersections,
    update_and_average_point,
)
from gazecore.pupil_detection import detect_pupil

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

ray_lines = []  # Store recent pupil ellipse rays
model_centers = []  # Store recent estimated eye centers
min_model_centers = 100  # Minimum model centers required before updating sphere radius
max_rays = 100  # Limit the number of stored pupil rays
prev_model_center_avg = (320,240)  # Preserve the last valid eye center
max_observed_distance = 0  # Initialize adaptive radius
last_sphere_radius_ellipse = None  # Last pupil ellipse used to expand the eye sphere radius
last_gaze_vector_text = None  # Preserve vector text during temporary detection dropouts
eye_sphere_adjustment_enabled = True  # Toggle automatic eye center/radius updates with F
render_intersections = False  # Release view omits intersection debugging overlays
pupil_confidence_threshold = 0.85  # Minimum pupil confidence for storing a ray
pupil_confidence_threshold_sphere = 0.90  # Minimum pupil confidence for determining eye sphere radius
intersection_ray_count = 4  # Rays sampled for each intersection estimate
minimum_intersection_angle_degrees = 8  # Minimum angle between sampled rays
last_tracking_result = None  # Store the latest tracker output

# TEMPORARY: manual eye center override (O). Look straight into the eye camera lens;
# the pupil center then projects onto the eyeball center.
center_override_collecting = False
center_override_samples = []
center_override_frames_needed = 30  # High-confidence pupil centers averaged per override

capture_stuck_ellipses = False     # toggled with 'e'
stuck_ellipses = []                # saved pupil ellipses
capture_frame_counter = 0          # counts processed frames while capture is enabled
ellipse_capture_interval = 5      # save one ellipse every 10 frames
max_stuck_ellipses = 20           # safety cap

CAMERA_CAPTURE_MODES = (
    (
        ("MSMF", cv2.CAP_MSMF, None),
        ("DirectShow + mp4v", cv2.CAP_DSHOW, "XVID"),
    )
    if sys.platform == "win32"
    else (("Auto", cv2.CAP_ANY, None),)
)



# --- Per-eye gaze state (swapped in and out by load/save_eye_tracking_state) ---
last_sphere_center = None
last_gaze_dir = None
calibrated_sphere_center = None
sphere_center_locked_2d = False
locked_model_center_avg = prev_model_center_avg

EYE_IDS = ("left", "right")
current_eye_id = "left"
PER_EYE_STATE_NAMES = (
    "ray_lines", "model_centers", "prev_model_center_avg", "max_observed_distance",
    "last_sphere_radius_ellipse", "last_gaze_vector_text", "last_tracking_result",
    "stored_intersections", "last_sphere_center", "last_gaze_dir", "calibrated_sphere_center",
    "sphere_center_locked_2d", "locked_model_center_avg", "eye_sphere_adjustment_enabled",
    "center_override_collecting", "center_override_samples",
)
eye_gaze_outputs = {eye_id: [0.0] * 6 for eye_id in EYE_IDS}

# External camera / screen params (for 640x480)
EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2
FRONT_DIAGONAL_FOV_DEGREES = 60.0  # Fallback pinhole model until intrinsics are calibrated

# Front camera model and fitted eye offsets persist per headset (see front_calibration).
CALIBRATION_FILE = paths.CALIBRATION_FILE
# The last successful gaze calibration, restored when the cameras start (or with L).
GAZE_CALIBRATION_FILE = paths.GAZE_CALIBRATION_FILE
# Every capture session, successful or not, for analysis and recovery.
GAZE_SESSIONS_DIR = paths.GAZE_SESSIONS_DIR
gaze_session_file = None  # Timestamped file of the session being captured
front_camera = fc.CameraModel.from_fov((EXT_WIDTH, EXT_HEIGHT), FRONT_DIAGONAL_FOV_DEGREES)
saved_eye_origins = {}
marker_target = fc.MarkerTarget(length_m=0.10)  # Large enough to locate precisely at 2 m
intrinsics_calibrator = None  # Set while ChArUco views are being captured
last_marker = None  # Latest MarkerDetection in the front frame

CALIBRATION_PROMPT = "Look at the marker center (NEAR ~40 cm); press C."
gaze_calibration = None  # GazeCalibration being collected
calibration_result = None
gaze_mapper = None
baseline_mapper = None  # Same samples with the eyes at the camera, to show the parallax removed
gaze_estimate = None
calibration_message = CALIBRATION_PROMPT

ACCURACY_PROMPT = "Hold the marker at any depth, look at its center, press T."
accuracy_test = fc.AccuracyTest()
accuracy_test_visible = False
accuracy_message = ACCURACY_PROMPT


def open_camera_capture(cam_index, start_mode=0):
    for mode_index in range(start_mode, len(CAMERA_CAPTURE_MODES)):
        mode_name, backend, fourcc = CAMERA_CAPTURE_MODES[mode_index]
        cap = cv2.VideoCapture(cam_index, backend)
        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read() if cap.isOpened() else (False, None)

        if ret:
            return cap, frame, mode_index

        cap.release()
        print(f"Could not grab a frame from camera {cam_index} with {mode_name}.")

    return None, None, None


def create_eye_tracking_state():
    return {
        "ray_lines": [],
        "model_centers": [],
        "prev_model_center_avg": (320, 240),
        "max_observed_distance": 0,
        "last_sphere_radius_ellipse": None,
        "last_gaze_vector_text": None,
        "last_tracking_result": None,
        "stored_intersections": [],
        "last_sphere_center": None,
        "last_gaze_dir": None,
        "calibrated_sphere_center": None,
        "sphere_center_locked_2d": False,
        "locked_model_center_avg": (320, 240),
        "eye_sphere_adjustment_enabled": True,
        "center_override_collecting": False,
        "center_override_samples": [],
    }


eye_tracking_states = {eye_id: create_eye_tracking_state() for eye_id in EYE_IDS}


# The tracking code works on module globals; these swap one eye's state in and out.
# Between frames the dicts are authoritative, so change them only with an eye loaded.
def load_eye_tracking_state(eye_id):
    global current_eye_id
    current_eye_id = eye_id
    globals().update(eye_tracking_states[eye_id])


def save_eye_tracking_state(eye_id):
    eye_tracking_states[eye_id] = {name: globals()[name] for name in PER_EYE_STATE_NAMES}


def for_all_eyes(action, eye_ids=EYE_IDS):
    """Run action(eye_id) with each eye's state loaded, then restore the loaded eye."""
    loaded_eye = current_eye_id
    save_eye_tracking_state(loaded_eye)
    for eye_id in eye_ids:
        load_eye_tracking_state(eye_id)
        action(eye_id)
        save_eye_tracking_state(eye_id)
    load_eye_tracking_state(loaded_eye)


def eye_window_name(base_name):
    return f"{current_eye_id.title()} Eye {base_name}"


def update_eye_sphere_radius(eye_center, current_pupil_ellipse, current_pupil_confidence):
    global max_observed_distance
    global last_sphere_radius_ellipse

    if last_sphere_radius_ellipse is not None:
        anchored_distance = distance_to_pupil_outer_edge(
            eye_center,
            last_sphere_radius_ellipse,
        )
        if anchored_distance is not None:
            max_observed_distance = anchored_distance

    if (
        current_pupil_ellipse is not None
        and current_pupil_confidence >= pupil_confidence_threshold_sphere
        and len(model_centers) >= min_model_centers
    ):
        current_distance = distance_to_pupil_outer_edge(
            eye_center,
            current_pupil_ellipse,
        )
        if (
            current_distance is not None
            and (
                last_sphere_radius_ellipse is None
                or current_distance > max_observed_distance
            )
        ):
            max_observed_distance = current_distance
            last_sphere_radius_ellipse = current_pupil_ellipse
            print(f"Radius-increasing ellipse confidence: {current_pupil_confidence * 100:.2f}%")


def toggle_center_override(eye_id):
    def toggle(_):
        global center_override_collecting
        global center_override_samples
        center_override_samples = []
        center_override_collecting = not center_override_collecting
        if center_override_collecting:
            print(f"{eye_id.title()} eye center override: look straight into that eye camera's lens.")
        else:
            print(f"{eye_id.title()} eye center override cancelled.")

    for_all_eyes(toggle, (eye_id,))


def collect_center_override_sample(center_x, center_y):
    global center_override_collecting
    global prev_model_center_avg
    global eye_sphere_adjustment_enabled

    center_override_samples.append((center_x, center_y))
    if len(center_override_samples) < center_override_frames_needed:
        return

    # Median rejects the odd blink or lash-line fit.
    median_center = np.median(np.array(center_override_samples), axis=0)
    prev_model_center_avg = (int(median_center[0]), int(median_center[1]))
    center_override_collecting = False
    eye_sphere_adjustment_enabled = False
    # The gaze calibration was built from the old center, so start it over.
    reset_front_calibration()
    print(f"{current_eye_id.title()} eye center overridden to {prev_model_center_avg}; "
          "sphere affixed. F resumes automatic.")


def get_eye_sphere_adjustment_prompt():
    if center_override_collecting:
        cancel_key = "o" if current_eye_id == "left" else "Shift+O"
        return (f"Look into the eye camera... {len(center_override_samples)}/"
                f"{center_override_frames_needed} ({cancel_key} cancels)")
    if eye_sphere_adjustment_enabled and not sphere_center_locked_2d:
        return "Press F to affix eyeball sphere"
    return "Press F again to engage automatic eye sphere adjustment"


def eyes_fixed():
    return any(
        state["sphere_center_locked_2d"] or not state["eye_sphere_adjustment_enabled"]
        for state in eye_tracking_states.values()
    )


def toggle_eye_sphere_adjustment():
    """Applies to both eyes so they stay consistent with one gaze calibration."""
    save_eye_tracking_state(current_eye_id)
    if any(state["sphere_center_locked_2d"] for state in eye_tracking_states.values()):
        reset_front_calibration()
        enabled = True
        print("Calibration cleared; automatic eye sphere adjustment resumed.")
    else:
        enabled = eyes_fixed()  # Any affixed eye resumes both; otherwise affix both.
        print(f"Eye sphere adjustment: {'automatic' if enabled else 'affixed'}")

    def apply(_):
        global eye_sphere_adjustment_enabled
        eye_sphere_adjustment_enabled = enabled

    for_all_eyes(apply)


def reset_eye_tracking_state(eye_id):
    eye_tracking_states[eye_id] = create_eye_tracking_state()
    eye_gaze_outputs[eye_id] = [0.0] * 6
    if current_eye_id == eye_id:
        load_eye_tracking_state(eye_id)
    reset_front_calibration()


def reset_tracking_state():
    global capture_frame_counter
    global stuck_ellipses

    for eye_id in EYE_IDS:
        eye_tracking_states[eye_id] = create_eye_tracking_state()
        eye_gaze_outputs[eye_id] = [0.0] * 6
    load_eye_tracking_state("left")
    capture_frame_counter = 0
    stuck_ellipses = []
    reset_front_calibration()


def draw_persistent_overlays(frame):
    prompt_text = get_eye_sphere_adjustment_prompt()
    prompt_shadow = (12, frame.shape[0] - 63)
    prompt_origin = (10, frame.shape[0] - 65)
    cv2.putText(frame, prompt_text, prompt_shadow, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, prompt_text, prompt_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if last_gaze_vector_text is None:
        return

    origin_text, dir_text = last_gaze_vector_text
    text_origin = (12, frame.shape[0] - 38)
    text_dir = (12, frame.shape[0] - 13)
    text_origin_green = (10, frame.shape[0] - 40)
    text_dir_green = (10, frame.shape[0] - 15)

    cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, origin_text, text_origin_green, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, dir_text, text_dir_green, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def get_last_tracking_result():
    return last_tracking_result


def draw_stuck_ellipses(frame):
    global stuck_ellipses

    for ellipse in stuck_ellipses:
        if ellipse is not None:
            cv2.ellipse(frame, ellipse,(0, 255, 255), 2)

# Function to detect available cameras
def detect_cameras(max_cams=10):
    available_cameras = []

    for i in range(max_cams):
        cap, _, _ = open_camera_capture(i)
        if cap is not None:
            available_cameras.append(i)
            cap.release()

    return available_cameras

# Process frames for pupil detection
def process_frames(observation):
    """Update the loaded eye's model from one pupil observation, then draw and show the eye view."""
    global ray_lines
    global max_rays
    global prev_model_center_avg
    global max_observed_distance
    global last_sphere_radius_ellipse
    global last_tracking_result
    global last_gaze_vector_text
    global last_gaze_dir
    global capture_stuck_ellipses
    global stuck_ellipses
    global capture_frame_counter
    global ellipse_capture_interval
    global max_stuck_ellipses

    frame = observation.frame
    final_rotated_rect = observation.ellipse
    center_x, center_y = observation.center
    best_ratio_under_ellipse = observation.confidence

    if final_rotated_rect is not None:
        if center_override_collecting and best_ratio_under_ellipse >= pupil_confidence_threshold:
            collect_center_override_sample(center_x, center_y)

        if (
            eye_sphere_adjustment_enabled
            and not sphere_center_locked_2d
            and best_ratio_under_ellipse >= pupil_confidence_threshold
        ):
            ray_lines.append(final_rotated_rect)
            if len(ray_lines) > max_rays:
                num_to_remove = len(ray_lines) - max_rays
                ray_lines = ray_lines[num_to_remove:]  # Keep only the last `max_rays` elements

    global locked_model_center_avg
    model_center_average = locked_model_center_avg if sphere_center_locked_2d else prev_model_center_avg

    if eye_sphere_adjustment_enabled and not sphere_center_locked_2d:
        model_center = compute_average_intersection(
            frame,
            ray_lines,
            intersection_ray_count,
            1500,
            minimum_intersection_angle_degrees,
        )
        if model_center is not None and model_center != (0, 0):
            model_center_average = update_and_average_point(model_centers, model_center, 200)

    if model_center_average[0] == 320:
        model_center_average = prev_model_center_avg
    if model_center_average[0] != 0:
        prev_model_center_avg = model_center_average
        if not sphere_center_locked_2d:
            locked_model_center_avg = model_center_average
    
    # Example safety check
    if final_rotated_rect is None or center_x is None or center_y is None or model_center_average[0] is None or model_center_average[1] is None:
        last_tracking_result = None
        last_gaze_dir = None
        draw_persistent_overlays(frame)
        cv2.imshow(eye_window_name("Frame with Ellipse and Rays"), frame)
        return  # or skip this frame

    if eye_sphere_adjustment_enabled and not sphere_center_locked_2d:
        update_eye_sphere_radius(
            model_center_average,
            final_rotated_rect,
            best_ratio_under_ellipse,
        )

    last_tracking_result = {
        "pupil_ellipse": {
            "center": [float(final_rotated_rect[0][0]), float(final_rotated_rect[0][1])] if final_rotated_rect is not None else None,
            "axes": [float(final_rotated_rect[1][0]), float(final_rotated_rect[1][1])] if final_rotated_rect is not None else None,
            "angle_degrees": float(final_rotated_rect[2]) if final_rotated_rect is not None else None,
        } if final_rotated_rect is not None else None,
        "eye_center": [int(model_center_average[0]), int(model_center_average[1])],
        "sphere_radius": float(max_observed_distance),
    }

    # Draw reference lines/ellipses
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)  # Draw eye sphere (circle)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)  # Draw eye center



    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  # # Draw line from eye center to ellipse center
        
    if final_rotated_rect is not None:
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)  # draw current ellipse

        # If capture mode is on, save one ellipse every N frames
        if capture_stuck_ellipses:
            capture_frame_counter += 1
            if capture_frame_counter % ellipse_capture_interval == 0:
                stuck_ellipses.append(final_rotated_rect)

                # safety cap so list does not grow forever
                if len(stuck_ellipses) > max_stuck_ellipses:
                    stuck_ellipses = stuck_ellipses[-max_stuck_ellipses:]

    # Draw previously saved ellipses so they stay "stuck" on screen
        draw_stuck_ellipses(frame)

    # Calculate the extended endpoint of gaze line
    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        # Compute the vector from model_center_average to center_x, center_y
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]

        # Scale the vector by 1.2x
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)

        # Draw the extended gaze line
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3) 




    # The single GL sphere window follows the left eye.
    gl_image = None
    if GL_SPHERE_AVAILABLE and current_eye_id == "left":
        gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, model_center_average[0], model_center_average[1])
    #cv2.circle(frame, (center_x, center_y), 22, (255, 255, 0), -1)  # Draw intersection center

    # Call the function
    center, direction = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

    if center is not None and direction is not None:
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
        last_gaze_vector_text = (origin_text, dir_text)

    ratio_text = f"{best_ratio_under_ellipse * 100:.2f}%"
    cv2.putText(frame, ratio_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, ratio_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    draw_persistent_overlays(frame)
    cv2.imshow(eye_window_name("Frame with Ellipse and Rays"), frame)

    if gl_image is not None:
        blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
        cv2.imshow(eye_window_name("Tracker + Sphere"), blended)

    return final_rotated_rect

def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    """
    Draws a ray passing through the center of an ellipse orthogonally to its major axis.
    
    Parameters:
    - image: The OpenCV image to draw on.
    - ellipse: A tuple ((cx, cy), (major_axis, minor_axis), angle) representing the fitted ellipse.
    - length: Length of the ray to draw on each side of the ellipse center.
    - color: Color of the line in BGR format (default: green).
    - thickness: Thickness of the line (default: 2).
    """

    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Compute the normal vector at the ellipse center (perpendicular to surface)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)  # Minor axis component
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)

    # Compute start and end points of the orthogonal ray
    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))

    # Draw the ray
    cv2.line(image, pt1, pt2, color, thickness)

    return image 

stored_intersections = []  # Stores all past intersections

def compute_average_intersection(frame, ray_lines, number_lines, total_lines, minimum_angle_degrees):
    """
    Selects `number_lines` random lines from the list, highlights them in red on the frame,
    computes their intersections, conditionally stores them (only if they agree within
    `pixel_limit`), and prunes stored intersections when exceeding `total_lines`.
    """
    pixel_limit = 30
    angle_threshold = 5  # degrees

    global stored_intersections

    if len(ray_lines) < 2 or number_lines < 2:
        return (0, 0)

    height, width = frame.shape[:2]

    selected_lines = random.sample(ray_lines, min(number_lines, len(ray_lines)))

    # Highlight selected rays
    if render_intersections:
        for ray in selected_lines:
            draw_orthogonal_ray(frame, ray, color=(0, 0, 255), thickness=2)

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
        stored_intersections.extend(intersections)

    # Prune stored intersections
    if len(stored_intersections) > total_lines:
        stored_intersections = prune_intersections(stored_intersections, total_lines)

    if not stored_intersections:
        return (0, 0)

    # Draw stored intersections
    if render_intersections:
        for pt in stored_intersections:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)

    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])

    if np.isnan(avg_x) or np.isnan(avg_y):
        return (0, 0)

    return (int(avg_x), int(avg_y))

def rotation_from_a_to_b(a, b):
    """
    Compute rotation matrix R such that R @ a = b
    using Rodrigues' rotation formula.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-6:
        # Vectors are parallel or nearly so
        if c > 0:
            return np.eye(3, dtype=np.float32)
        else:
            # 180-degree flip: choose any axis orthogonal to a
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, axis)
            v = v / np.linalg.norm(v)
            s = np.linalg.norm(v)
    else:
        s = np.linalg.norm(v)
        v = v / s

    # Skew-symmetric cross-product matrix
    vx, vy, vz = v
    K = np.array([
        [0,    -vz,  vy],
        [vz,    0,  -vx],
        [-vy,  vx,   0 ]
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + K * s + (K @ K) * ((1 - c) / (s ** 2))
    return R

# --- Front camera calibration: orchestration only; the workflow lives in front_calibration ---

def load_front_calibration():
    global front_camera, saved_eye_origins
    try:
        camera, saved_eye_origins = fc.load_calibration(CALIBRATION_FILE, (EXT_WIDTH, EXT_HEIGHT))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not read {CALIBRATION_FILE}: {exc}")
        return
    if camera is not None:
        front_camera = camera
    if camera is not None or saved_eye_origins:
        print(f"Loaded front camera calibration from {CALIBRATION_FILE}.")


def save_front_calibration():
    try:
        fc.save_calibration(CALIBRATION_FILE, front_camera, saved_eye_origins)
    except OSError as exc:
        print(f"Could not write {CALIBRATION_FILE}: {exc}")


def save_printable_targets(parent=None):
    folder = filedialog.askdirectory(parent=parent, title="Folder for the printable marker and board")
    if not folder:
        return
    cv2.imwrite(os.path.join(folder, "gaze_target_marker.png"), marker_target.printable_image())
    cv2.imwrite(os.path.join(folder, "charuco_board.png"),
                fc.IntrinsicsCalibrator((EXT_WIDTH, EXT_HEIGHT)).board_image())
    print("Saved gaze_target_marker.png and charuco_board.png. "
          "Measure the printed marker's black square and enter its side as the marker size.")


def toggle_intrinsics_mode():
    global intrinsics_calibrator
    if intrinsics_calibrator is None:
        intrinsics_calibrator = fc.IntrinsicsCalibrator((EXT_WIDTH, EXT_HEIGHT))
        print("Intrinsics capture: show the ChArUco board at varied angles, distances and image positions.")
    else:
        intrinsics_calibrator = None
        print("Intrinsics capture cancelled.")


def update_intrinsics_capture(frame):
    """Feed one frame to the intrinsics capture; returns a status line, or None once done."""
    global intrinsics_calibrator, front_camera, saved_eye_origins
    calibrator = intrinsics_calibrator
    views_before = len(calibrator.views)
    corners, ids = calibrator.add_frame(frame, time.time())
    if ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0, 255, 0))
    if len(calibrator.views) > views_before:
        print(f"Intrinsics view {len(calibrator.views)}/{calibrator.views_needed} captured.")
    if not calibrator.done:
        corner_count = 0 if ids is None else len(ids)
        return (f"Intrinsics: {len(calibrator.views)}/{calibrator.views_needed} views, "
                f"{corner_count} corners | move the board | I cancels")

    intrinsics_calibrator = None
    try:
        camera, rms = calibrator.calibrate()
    except (cv2.error, ValueError) as exc:
        print("Intrinsics calibration failed:", exc)
        return None
    front_camera = camera
    saved_eye_origins = {}  # Fitted with the old camera model
    save_front_calibration()
    reset_front_calibration()
    print(f"Front camera intrinsics calibrated (RMS {rms:.2f} px) and saved to {CALIBRATION_FILE}.")
    return None


def current_gaze_dirs():
    return {eye_id: state["last_gaze_dir"] for eye_id, state in eye_tracking_states.items()}


def lock_eye_spheres(eye_ids):
    def lock(_):
        global calibrated_sphere_center, locked_model_center_avg, sphere_center_locked_2d
        if last_sphere_center is not None:
            calibrated_sphere_center = last_sphere_center.copy()
        locked_model_center_avg = prev_model_center_avg
        sphere_center_locked_2d = True

    for_all_eyes(lock, eye_ids)


def clear_gaze_calibration():
    global gaze_calibration, calibration_result, gaze_mapper, baseline_mapper, gaze_estimate
    global calibration_message
    reset_accuracy_test()
    gaze_calibration = calibration_result = gaze_mapper = baseline_mapper = gaze_estimate = None
    calibration_message = CALIBRATION_PROMPT


def capture_calibration_sample(active_eyes):
    global gaze_calibration, calibration_message, gaze_session_file
    if calibration_result is not None:
        # Restart keeps the affixed eye spheres, so new samples match the same eye models.
        clear_gaze_calibration()
        print(calibration_message)
        return True
    if last_marker is None:
        calibration_message = "Marker not visible in the front camera; retry C."
        print(calibration_message)
        return False
    calibration = gaze_calibration or fc.GazeCalibration(tuple(active_eyes))
    try:
        calibration.add_sample(current_gaze_dirs(), last_marker)
    except ValueError as exc:
        calibration_message = f"{exc}; keep looking at the marker and retry C."
        print(calibration_message)
        return False
    if gaze_calibration is None:
        gaze_calibration = calibration
        lock_eye_spheres(calibration.eye_ids)
        gaze_session_file = os.path.join(GAZE_SESSIONS_DIR, time.strftime("gaze_%Y%m%d_%H%M%S.json"))
    save_gaze_session(gaze_session_file)
    if calibration.ready:
        return solve_gaze_calibration()
    calibration_message = (f"Captured at {last_marker.point[2]:.2f} m. "
                           "Move the marker across the image and depths, press C again.")
    print(calibration_message, calibration.progress_text())
    return True


def locked_eye_states(eye_ids):
    """The eye-model state the samples' raw gaze depends on (JSON-ready)."""
    save_eye_tracking_state(current_eye_id)
    states = {}
    for eye_id in eye_ids:
        state = eye_tracking_states[eye_id]
        sphere_center = state["calibrated_sphere_center"]
        states[eye_id] = {
            "model_center": [int(v) for v in state["locked_model_center_avg"]],
            "sphere_radius": float(state["max_observed_distance"]),
            "sphere_center": None if sphere_center is None else [float(v) for v in sphere_center],
            "adjustment_enabled": bool(state["eye_sphere_adjustment_enabled"]),
        }
    return states


def apply_eye_states(eye_states):
    """Lock each eye's model to a saved state, so restored samples match it again."""
    def apply(eye_id):
        global prev_model_center_avg, locked_model_center_avg, sphere_center_locked_2d
        global calibrated_sphere_center, max_observed_distance, eye_sphere_adjustment_enabled
        state = eye_states[eye_id]
        prev_model_center_avg = locked_model_center_avg = tuple(int(v) for v in state["model_center"])
        max_observed_distance = state["sphere_radius"]
        sphere_center = state["sphere_center"]
        calibrated_sphere_center = None if sphere_center is None else np.asarray(sphere_center, dtype=float)
        eye_sphere_adjustment_enabled = state["adjustment_enabled"]
        sphere_center_locked_2d = True

    for_all_eyes(apply, tuple(eye_states))


def save_gaze_session(path):
    try:
        fc.save_gaze_session(path, gaze_calibration, locked_eye_states(gaze_calibration.eye_ids))
    except OSError as exc:
        print(f"Could not write {path}: {exc}")


def restore_gaze_calibration(active_eyes, quiet=False):
    """Reload the last successful calibration (its eye centers and samples) and re-solve it."""
    global gaze_calibration
    try:
        calibration, eye_states, saved_at = fc.load_gaze_session(GAZE_CALIBRATION_FILE)
    except FileNotFoundError:
        if not quiet:
            print("No saved gaze calibration yet; calibrate with C.")
        return False
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not read {GAZE_CALIBRATION_FILE}: {exc}")
        return False
    missing = [eye_id for eye_id in calibration.eye_ids if eye_id not in active_eyes]
    if missing:
        if not quiet:
            print(f"Start the {' and '.join(missing)} eye camera to restore the saved calibration.")
        return False
    reset_front_calibration()
    apply_eye_states(eye_states)
    gaze_calibration = calibration
    print(f"Restoring the gaze calibration from {saved_at} ({len(calibration.samples)} samples). "
          "If the headset moved since, check with G/T and recalibrate with C.")
    return solve_gaze_calibration(save=False)


def solve_gaze_calibration(save=True):
    global calibration_result, gaze_mapper, baseline_mapper, calibration_message
    if gaze_calibration is None:
        calibration_message = "No calibration samples yet; press C while looking at the marker."
        print(calibration_message)
        return False
    try:
        result = gaze_calibration.solve(saved_eye_origins)
    except ValueError as exc:
        calibration_message = f"Calibration failed: {exc}"
        print(calibration_message, f"(samples saved to {gaze_session_file})" if save else "")
        return False

    calibration_result = result
    if save:
        save_gaze_session(GAZE_CALIBRATION_FILE)
    gaze_mapper = fc.BinocularGazeMapper(result.models, front_camera)
    baseline_mapper = None if result.baselines is None else fc.BinocularGazeMapper(result.baselines, front_camera)
    if result.origins_fitted:
        saved_eye_origins.update({eye_id: model["origin"].copy() for eye_id, model in result.models.items()})
        save_front_calibration()
    for eye_id, model in result.models.items():
        bands = ", ".join(f"{band} {rms:.2f}" for band, rms in model["band_rms_deg"].items())
        dropped = np.flatnonzero(~model["inliers"]) + 1
        dropped_text = f", ignored glitched sample(s) {dropped.tolist()}" if dropped.size else ""
        print(f"{eye_id.title()} eye: origin {np.round(model['origin'] * 1000, 1)} mm, RMS deg: {bands}{dropped_text}")
    separation = result.eye_separation_mm
    if separation is not None and not 50 <= separation <= 75:
        print(f"Warning: eye separation {separation:.0f} mm is implausible; "
              "check the marker size and front camera intrinsics, then recalibrate.")
    calibration_message = result.summary() + ". C restarts."
    print(calibration_message)
    return True


def write_gaze_vector_file():
    """CSV: left origin+direction, right origin+direction (eye model units), then the
    gaze point (front camera frame, metres) and its depth; zeros when unavailable."""
    file_path = paths.GAZE_VECTOR_FILE
    try:
        with open(file_path, "a"):
            pass
    except IOError:
        print("File is currently in use. Skipping write.")
        return
    if gaze_estimate is None:
        point, depth = [0.0, 0.0, 0.0], 0.0
    else:
        point, depth = list(gaze_estimate.point), gaze_estimate.depth
    values = eye_gaze_outputs["left"] + eye_gaze_outputs["right"] + point + [depth]
    try:
        with open(file_path, "w") as f:
            f.write(",".join(f"{value:.6f}" for value in values) + "\n")
    except Exception as e:
        print("Write error:", e)


def update_binocular_gaze():
    global gaze_estimate
    gaze_estimate = None if gaze_mapper is None else gaze_mapper.update(current_gaze_dirs())
    write_gaze_vector_file()


def reset_accuracy_test():
    global accuracy_test, accuracy_message
    accuracy_test = fc.AccuracyTest()
    accuracy_message = ACCURACY_PROMPT


def toggle_accuracy_test():
    global accuracy_test_visible
    accuracy_test_visible = not accuracy_test_visible
    reset_accuracy_test()
    print("Accuracy test shown." if accuracy_test_visible else "Accuracy test hidden and reset.")


def record_accuracy_sample():
    global accuracy_message
    if not accuracy_test_visible:
        print("Show the accuracy test with G before pressing T.")
        return False
    if gaze_estimate is None:
        accuracy_message = "No calibrated gaze: calibrate or reacquire the eyes, then retry T."
        return False
    if last_marker is None:
        accuracy_message = "Marker not visible; retry T."
        return False
    baseline = None if baseline_mapper is None else baseline_mapper.update(current_gaze_dirs())
    sample = accuracy_test.record(front_camera, last_marker, gaze_estimate.pixel,
                                  None if baseline is None else baseline.pixel)
    accuracy_message = f"{sample['depth']:.2f} m: {sample['error']:.2f} deg"
    if sample["baseline_error"] is not None:
        accuracy_message += f" (no-offset model {sample['baseline_error']:.2f} deg)"
    print("Accuracy sample:", accuracy_message, f"| gaze depth {gaze_estimate.depth:.2f} m")
    return True


def accuracy_summary_lines():
    lines = [accuracy_message]
    for band, (count, error, baseline_error) in accuracy_test.summary_by_band().items():
        line = f"{band.upper()} n={count}: {error:.2f} deg"
        if baseline_error is not None:
            line += f" | no-offset {baseline_error:.2f} deg"
        lines.append(line)
    return lines


def draw_text_lines(frame, lines, top=24):
    for index, line in enumerate(lines):
        origin = (10, top + 22 * index)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def to_drawable_pixel(pixel):
    if pixel is None or not np.isfinite(pixel).all():
        return None
    return (int(np.clip(round(pixel[0]), 0, EXT_WIDTH - 1)), int(np.clip(round(pixel[1]), 0, EXT_HEIGHT - 1)))


def draw_front_overlay(frame, intrinsics_text):
    band_colors = {"near": (0, 200, 0), "mid": (200, 200, 200), "far": (255, 120, 0)}
    for sample in gaze_calibration.samples if gaze_calibration is not None else []:
        point = to_drawable_pixel(sample["pixel"])
        if point is not None:
            cv2.circle(frame, point, 4, band_colors[gg.depth_band(sample["point"][2])], -1)
    if last_marker is not None:
        cv2.aruco.drawDetectedMarkers(frame, [last_marker.corners])
        center = to_drawable_pixel(last_marker.pixel)
        if center is not None:
            cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            draw_text_lines(frame, [f"{last_marker.point[2]:.2f} m"], top=min(center[1] + 28, EXT_HEIGHT - 8))
    gaze_point = to_drawable_pixel(None if gaze_estimate is None else gaze_estimate.pixel)
    if gaze_point is not None:
        cv2.circle(frame, gaze_point, 8, (0, 0, 255), -1)

    if intrinsics_text is not None:
        lines = [intrinsics_text]
    elif calibration_result is None:
        progress = (gaze_calibration or fc.GazeCalibration(EYE_IDS)).progress_text()
        lines = [progress, calibration_message]
    else:
        lines = [calibration_message]
        if gaze_estimate is not None:
            lines.append(f"Gaze depth {gaze_estimate.depth:.2f} m")
    if not front_camera.calibrated and intrinsics_text is None:
        lines.append("Approximate camera model: press I with the ChArUco board")
    if accuracy_test_visible:
        lines += accuracy_summary_lines()
    draw_text_lines(frame, lines)


def process_front_frame(frame):
    global last_marker
    capturing_intrinsics = intrinsics_calibrator is not None
    intrinsics_text = update_intrinsics_capture(frame) if capturing_intrinsics else None
    last_marker = None if capturing_intrinsics else marker_target.detect(frame, front_camera)
    draw_front_overlay(frame, intrinsics_text)
    cv2.imshow("External Camera (Gaze)", frame)


def reset_front_calibration():
    global last_marker
    clear_gaze_calibration()
    last_marker = None

    def unlock(_):
        global calibrated_sphere_center, last_sphere_center, last_gaze_dir
        global sphere_center_locked_2d, locked_model_center_avg
        calibrated_sphere_center = None
        last_sphere_center = None
        last_gaze_dir = None
        sphere_center_locked_2d = False
        locked_model_center_avg = prev_model_center_avg

    for_all_eyes(unlock)



def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Gaze of the loaded eye from pupil and eyeball center pixels; records it for the outputs.
    Returns:
        sphere_center (np.ndarray): 3D position of the sphere center in world space
        gaze_direction (np.ndarray): Normalized 3D direction vector from sphere center
    """
    global last_sphere_center, last_gaze_dir
    sphere_center, gaze_rotated = gaze_vector_from_pixels(x, y, center_x, center_y, screen_width, screen_height)
    last_sphere_center = sphere_center.copy()
    last_gaze_dir = gaze_rotated.copy()
    sphere_center_out = calibrated_sphere_center if calibrated_sphere_center is not None else sphere_center

    # gaze_vector.txt is written once per frame for both eyes by write_gaze_vector_file().
    eye_gaze_outputs[current_eye_id] = [float(v) for v in np.concatenate((sphere_center_out, gaze_rotated))]

    return sphere_center_out, gaze_rotated


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame, return_rendered_frame=False):
    observation = detect_pupil(frame)
    frame = observation.frame
    if observation.darkest_point is None:
        return (None, frame) if return_rendered_frame else None

    final_rotated_rect = process_frames(observation)
    
    if return_rendered_frame:
        return final_rotated_rect, frame

    return final_rotated_rect


# Process a selected video file
def process_video(parent=None):
    while True:
        video_path = filedialog.askopenfilename(
            parent=parent,
            filetypes=[("Video Files", "*.mp4 *.avi"), ("All files", "*.*")],
        )

        if not video_path:
            cv2.destroyAllWindows()
            return  # User canceled selection

        reset_tracking_state()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            continue

        video_stem, video_extension = os.path.splitext(video_path)
        output_path = f"{video_stem}_tracked{video_extension}"
        output_codec = "XVID" if video_extension.lower() == ".avi" else "mp4v"
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*output_codec),
            60,
            (640, 480),
        )
        if not writer.isOpened():
            print(f"Error: Could not create tracked video: {output_path}")
            cap.release()
            continue

        quit_requested = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # A single-eye video is tracked as the left eye.
                _, tracked_frame = process_frame(frame, return_rendered_frame=True)
                save_eye_tracking_state("left")
                write_gaze_vector_file()
                writer.write(tracked_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    quit_requested = True
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
                elif key in (ord('f'), ord('F')):
                    toggle_eye_sphere_adjustment()
                elif key in (ord('o'), ord('O')):
                    toggle_center_override("left")
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        print(f"Tracked video saved to: {output_path}")

        if quit_requested:
            return
    

# GUI for selecting camera or video
def selection_gui():
    global capture_stuck_ellipses
    global capture_frame_counter
    global stuck_ellipses

    camera_roles = EYE_IDS + ("front",)
    camera_labels = {"left": "Left Eye Camera", "right": "Right Eye Camera", "front": "Forward-facing Camera"}
    cameras = detect_cameras()
    active_captures = {role: None for role in camera_roles}
    active_camera_indexes = {role: None for role in camera_roles}
    active_capture_mode_indexes = {role: None for role in camera_roles}
    camera_paused = False

    # Create Tkinter window
    root = tk.Tk()
    root.title("Orlosky 3D Front Camera")
    tk.Label(root, text="Orlosky Eye Tracker 3D", font=("Arial", 12, "bold")).pack(pady=10)

    camera_dropdowns = {}
    flip_image_vars = {}

    def any_camera_active():
        return any(capture is not None for capture in active_captures.values())

    def active_eyes():
        return tuple(eye_id for eye_id in EYE_IDS if active_captures[eye_id] is not None)

    def stop_camera(role):
        if active_captures[role] is not None:
            active_captures[role].release()
            active_captures[role] = None

        reset_front_calibration()
        active_camera_indexes[role] = None
        active_capture_mode_indexes[role] = None

        if not any_camera_active():
            cv2.destroyAllWindows()

    def stop_all_cameras():
        for role in camera_roles:
            stop_camera(role)

    def start_camera(role):
        nonlocal camera_paused

        selected_camera = camera_dropdowns[role].get()
        if not selected_camera.isdigit():
            return

        cam_index = int(selected_camera)
        if any(active_camera_indexes[other] == cam_index for other in camera_roles if other != role):
            print("Each camera role must use a different camera index.")
            return
        stop_camera(role)
        new_capture, _, mode_index = open_camera_capture(cam_index)

        if new_capture is None:
            print(f"Error: Could not open camera {cam_index}.")
            return

        if role in EYE_IDS:
            reset_eye_tracking_state(role)
            cv2.namedWindow(f"{role.title()} Eye Frame with Ellipse and Rays")
        else:
            reset_front_calibration()
        sync_fixed_sphere_checkbox()
        active_captures[role] = new_capture
        active_camera_indexes[role] = cam_index
        active_capture_mode_indexes[role] = mode_index
        camera_paused = False
        print(
            f"Streaming {role} camera {cam_index} "
            f"with {CAMERA_CAPTURE_MODES[mode_index][0]}."
        )
        if all(active_captures[other] is not None for other in camera_roles):
            restore_gaze_calibration(active_eyes(), quiet=True)
            sync_fixed_sphere_checkbox()

    def restore():
        if active_captures["front"] is None or not active_eyes():
            print("Start the forward-facing and eye cameras before restoring the calibration.")
            return
        restore_gaze_calibration(active_eyes())
        sync_fixed_sphere_checkbox()

    def switch_active_camera(role):
        if active_captures[role] is not None:
            start_camera(role)

    camera_controls_frame = ttk.Frame(root)
    camera_controls_frame.pack(pady=5)

    for column_index, role in enumerate(camera_roles):
        if column_index:
            ttk.Separator(camera_controls_frame, orient="vertical").grid(
                row=0, column=column_index * 2 - 1, sticky="ns", padx=8
            )
        frame = ttk.Frame(camera_controls_frame)
        frame.grid(row=0, column=column_index * 2, padx=5, pady=5, sticky="n")

        camera_label = camera_labels[role]
        tk.Label(frame, text=f"{camera_label}:").grid(row=0, column=0, columnspan=2, pady=2)

        dropdown = ttk.Combobox(frame, values=[str(index) for index in cameras], state="readonly", width=12)
        if cameras:
            default_index = min(camera_roles.index(role), len(cameras) - 1)
            dropdown.current(default_index)
        else:
            dropdown.set("No cameras found")
        dropdown.grid(row=1, column=0, columnspan=2, pady=2)
        dropdown.bind("<<ComboboxSelected>>", lambda event, selected_role=role: switch_active_camera(selected_role))
        camera_dropdowns[role] = dropdown

        tk.Button(frame, text="Start", command=lambda selected_role=role: start_camera(selected_role)).grid(row=2, column=0, pady=2)
        tk.Button(frame, text="Stop", command=lambda selected_role=role: stop_camera(selected_role)).grid(row=2, column=1, pady=2)

        flip_var = tk.BooleanVar(value=role in EYE_IDS)
        tk.Checkbutton(frame, text="flip image", variable=flip_var, command=reset_front_calibration).grid(row=3, column=0, columnspan=2, pady=2)
        flip_image_vars[role] = flip_var

    fixed_sphere_var = tk.BooleanVar(value=eyes_fixed())

    def sync_fixed_sphere_checkbox():
        fixed_sphere_var.set(eyes_fixed())

    def set_fixed_sphere_from_checkbox():
        if fixed_sphere_var.get() != eyes_fixed():
            toggle_eye_sphere_adjustment()
        sync_fixed_sphere_checkbox()

    marker_size_var = tk.StringVar(value=f"{marker_target.length_m * 1000:g}")

    def set_marker_size(*_):
        try:
            size_mm = float(marker_size_var.get())
        except ValueError:
            return
        if size_mm > 0:
            marker_target.length_m = size_mm / 1000.0

    marker_size_var.trace_add("write", set_marker_size)

    def process_camera_frame():
        nonlocal camera_paused
        global capture_stuck_ellipses
        global capture_frame_counter
        global stuck_ellipses

        if not camera_paused and any_camera_active():
            front_frame = None
            for role in camera_roles:
                active_capture = active_captures[role]
                if active_capture is None:
                    continue

                ret, frame = active_capture.read()
                if not ret:
                    failed_capture = active_captures[role]
                    active_captures[role] = None
                    failed_capture.release()

                    retry_mode = active_capture_mode_indexes[role] + 1
                    new_capture, frame, mode_index = open_camera_capture(
                        active_camera_indexes[role],
                        retry_mode,
                    )
                    if new_capture is None:
                        print(f"Error: Could not read from {role} camera {active_camera_indexes[role]}.")
                        stop_camera(role)
                        continue

                    active_captures[role] = new_capture
                    active_capture_mode_indexes[role] = mode_index
                    print(
                        f"Switched {role} camera {active_camera_indexes[role]} to "
                        f"{CAMERA_CAPTURE_MODES[mode_index][0]}."
                    )

                if flip_image_vars[role].get():
                    frame = cv2.flip(frame, 0)

                if role in EYE_IDS:
                    load_eye_tracking_state(role)
                    process_frame(frame)
                    save_eye_tracking_state(role)
                else:
                    front_frame = cv2.resize(frame, (EXT_WIDTH, EXT_HEIGHT))

            # Both eyes are current now, so the vergence depth uses one moment in time.
            update_binocular_gaze()
            if front_frame is not None:
                process_front_frame(front_frame)

        if any_camera_active():
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_all_cameras()
            elif key == ord(' '):
                camera_paused = not camera_paused
            elif key in (ord('f'), ord('F')):
                toggle_eye_sphere_adjustment()
                sync_fixed_sphere_checkbox()
            elif key == ord('o'):
                toggle_center_override("left")
            elif key == ord('O'):
                toggle_center_override("right")
            elif key == ord('e'):
                capture_stuck_ellipses = not capture_stuck_ellipses
                capture_frame_counter = 0
                print(f"Ellipse capture mode: {'ON' if capture_stuck_ellipses else 'OFF'}")
            elif key in (ord('g'), ord('G')):
                toggle_accuracy_test()
            elif key in (ord('t'), ord('T')):
                if camera_paused or active_captures["front"] is None or not active_eyes():
                    print("Resume the front and eye camera streams before recording accuracy samples.")
                else:
                    record_accuracy_sample()
            elif key in (ord('c'), ord('C')):
                calibrate()
            elif key in (ord('k'), ord('K')):
                solve()
            elif key in (ord('l'), ord('L')):
                restore()
            elif key in (ord('i'), ord('I')):
                intrinsics()
            elif key in (ord('x'), ord('X')):
                stuck_ellipses.clear()
                print("Cleared stuck ellipses.")

        sync_fixed_sphere_checkbox()
        root.after(1, process_camera_frame)

    def calibrate():
        if camera_paused:
            print("Resume the camera streams before capturing a calibration sample.")
            return
        if active_captures["front"] is None or not active_eyes():
            print("Start the forward-facing camera and at least one eye camera before calibrating.")
            return
        if intrinsics_calibrator is not None:
            print("Finish or cancel the intrinsics capture (I) first.")
            return
        capture_calibration_sample(active_eyes())
        sync_fixed_sphere_checkbox()

    def solve():
        if calibration_result is not None:
            print("Already calibrated; C restarts calibration.")
            return
        solve_gaze_calibration()

    def intrinsics():
        if active_captures["front"] is None:
            print("Start the forward-facing camera before calibrating its intrinsics.")
            return
        toggle_intrinsics_mode()

    def close_app():
        stop_all_cameras()
        root.destroy()

    def browse_video():
        stop_all_cameras()
        try:
            process_video(parent=root)
        finally:
            sync_fixed_sphere_checkbox()
            root.lift()

    tk.Checkbutton(
        root,
        text="fixed eye spheres",
        variable=fixed_sphere_var,
        command=set_fixed_sphere_from_checkbox,
    ).pack(pady=5)

    marker_frame = ttk.Frame(root)
    marker_frame.pack(pady=5)
    tk.Label(marker_frame, text="Marker black square side (mm):").pack(side="left")
    tk.Entry(marker_frame, textvariable=marker_size_var, width=6).pack(side="left", padx=4)
    tk.Button(marker_frame, text="Save printable marker + board", command=lambda: save_printable_targets(root)).pack(side="left", padx=4)

    tk.Button(root, text="Capture Calibration Sample (C)", command=calibrate).pack(pady=5)
    tk.Button(root, text="Solve Calibration Now (K)", command=solve).pack(pady=5)
    tk.Button(root, text="Restore Last Calibration (L)", command=restore).pack(pady=5)
    tk.Button(root, text="Calibrate Front Camera Intrinsics (I)", command=intrinsics).pack(pady=5)
    tk.Button(root, text="Browse Video", command=browse_video).pack(pady=5)
    tk.Label(
        root,
        text=(
            "Q: stop | Space: pause | F: fixed/auto spheres | E/X: capture/clear ellipses\n"
            "C: capture sample looking at the marker center, NEAR (<0.6 m) and FAR (>1.5 m)\n"
            "K: solve with current samples (single depth reuses saved eye offsets)\n"
            "L: restore the last calibration (automatic once all cameras run)\n"
            "I: front camera intrinsics with the ChArUco board\n"
            "G: accuracy test | T: record accuracy at the marker\n"
            "o / Shift+O: override left / right eye center (look into that eye camera)"
        ),
    ).pack(pady=5)

    if GL_SPHERE_AVAILABLE:
        # Start GL sphere window once
        app = gl_sphere.start_gl_window()

    load_front_calibration()
    reset_tracking_state()
    sync_fixed_sphere_checkbox()
    root.protocol("WM_DELETE_WINDOW", close_app)
    root.after(1, process_camera_frame)
    root.mainloop()

# Run GUI to select camera or video
if __name__ == "__main__":
    selection_gui()
