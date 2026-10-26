"""Golden per-frame record of the eye pipeline on eye_test.mp4.

The fixture was generated from the front-camera tracker before its eye pipeline moved into
gazecore (tag golden-baseline). Every refactor of that pipeline must reproduce it exactly.

Run A tracks the left eye alone and scripts F, lock, unlock, the O override and restoring a
locked state. Run B tracks both eyes (the right one resized and flipped, so cropping and
flipping are exercised) with a gaze mapper, a lock of both eyes and an override that unlocks
the other eye mid-frame.

Regenerate only if the video decoder changed (the frame CRCs say so):
    python gazecore/tests/golden/eye_golden.py
"""

import contextlib
import io
import os
import random
import sys
import tempfile
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TRACKER_DIR = os.path.join(REPO, "FrontCameraTracker")
EYE_VIDEO = os.path.join(REPO, "eye_test.mp4")
FIXTURE = os.path.join(HERE, "eye_pipeline_v1.npz")

SEED_A, SEED_B = 20261026, 20261027
FRAMES_A, ITERATIONS_B = 400, 300
NAN5 = [np.nan] * 5


def read_frames(count):
    capture = cv2.VideoCapture(EYE_VIDEO)
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < count:
        raise FileNotFoundError(f"{EYE_VIDEO}: need {count} frames, got {len(frames)}")
    return frames


def crc(array):
    return zlib.crc32(np.ascontiguousarray(array).tobytes())


def vector(value, size):
    return [np.nan] * size if value is None else [float(v) for v in value]


def ellipse_values(ellipse):
    if ellipse is None:
        return NAN5
    (cx, cy), (a0, a1), angle = ellipse
    return [float(cx), float(cy), float(a0), float(a1), float(angle)]


def result_values(result):
    """last_tracking_result as [ellipse(5), eye center(2), sphere radius]."""
    if result is None:
        return NAN5 + [np.nan] * 3
    pupil = result["pupil_ellipse"]
    ellipse = NAN5 if pupil is None else pupil["center"] + pupil["axes"] + [pupil["angle_degrees"]]
    return ellipse + [float(v) for v in result["eye_center"]] + [result["sphere_radius"]]


class ConfidenceProbe:
    """Captures the "xx.xx%" pupil confidence the tracker draws on the eye frame."""

    def __init__(self):
        self.real_put_text = cv2.putText
        self.value = np.nan

    def __call__(self, image, text, *args, **kwargs):
        if text.endswith("%"):
            self.value = float(text[:-1])
        return self.real_put_text(image, text, *args, **kwargs)


class Rows:
    def __init__(self):
        self.columns = {}

    def add(self, prefix, **fields):
        for name, value in fields.items():
            self.columns.setdefault(prefix + name, []).append(value)

    def arrays(self):
        return {name: np.asarray(values) for name, values in self.columns.items()}


def record_eye(rows, prefix, tracker, eye_id, frame, probe):
    """Process one eye frame the way the tracker's main loop does, then record the eye state."""
    probe.value = np.nan
    tracker.load_eye_tracking_state(eye_id)
    frame_crc = crc(frame)
    _, rendered = tracker.process_frame(frame, return_rendered_frame=True)
    tracker.save_eye_tracking_state(eye_id)
    state = tracker.eye_tracking_states[eye_id]
    rows.add(
        prefix,
        frame_crc=frame_crc,
        render_crc=crc(rendered),
        rng_crc=zlib.crc32(repr(random.getstate()).encode()),
        result=result_values(state["last_tracking_result"]),
        prev_center=vector(state["prev_model_center_avg"], 2),
        locked_center=vector(state["locked_model_center_avg"], 2),
        radius=float(state["max_observed_distance"]),
        radius_ellipse=ellipse_values(state["last_sphere_radius_ellipse"]),
        gaze_dir=vector(state["last_gaze_dir"], 3),
        sphere_center=vector(state["last_sphere_center"], 3),
        calibrated_center=vector(state["calibrated_sphere_center"], 3),
        gaze_output=vector(tracker.eye_gaze_outputs[eye_id], 6),
        counts=[len(state["ray_lines"]), len(state["model_centers"]),
                len(state["stored_intersections"]), len(state["center_override_samples"])],
        flags=[bool(state["sphere_center_locked_2d"]), bool(state["eye_sphere_adjustment_enabled"]),
               bool(state["center_override_collecting"])],
        confidence=probe.value,
    )


def dump_lists(arrays, prefix, tracker, eye_id):
    state = tracker.eye_tracking_states[eye_id]
    arrays[prefix + "ray_lines"] = np.array([ellipse_values(e) for e in state["ray_lines"]]).reshape(-1, 5)
    arrays[prefix + "model_centers"] = np.array(state["model_centers"], dtype=float).reshape(-1, 2)
    arrays[prefix + "stored_intersections"] = np.array(state["stored_intersections"], dtype=float).reshape(-1, 2)


def run_a(tracker, frames, probe):
    rows, saved = Rows(), {}
    events = {  # The video's pupil is confident mostly in frames 50-250.
        60: lambda: tracker.toggle_center_override("left"),  # ends affixed
        140: tracker.toggle_eye_sphere_adjustment,  # automatic again
        170: tracker.toggle_eye_sphere_adjustment,  # affix
        200: tracker.toggle_eye_sphere_adjustment,  # automatic again
        230: lambda: tracker.lock_eye_spheres(("left",)),
        240: lambda: saved.update(tracker.locked_eye_states(("left",))),
        280: tracker.toggle_eye_sphere_adjustment,  # unlock (clears the calibration)
        330: lambda: tracker.apply_eye_states(saved),
    }
    tracker.reset_tracking_state()
    random.seed(SEED_A)
    for index, frame in enumerate(frames[:FRAMES_A]):
        if index in events:
            events[index]()
        record_eye(rows, "a_left_", tracker, "left", frame, probe)
    arrays = rows.arrays()
    dump_lists(arrays, "a_left_", tracker, "left")
    return arrays


def run_b(tracker, frames, probe, models, camera):
    rows = Rows()
    events = {
        120: lambda: tracker.lock_eye_spheres(tracker.EYE_IDS),
        160: lambda: tracker.toggle_center_override("right"),  # its end unlocks the left eye too
        240: tracker.toggle_eye_sphere_adjustment,
    }
    tracker.reset_tracking_state()
    tracker.front_camera = camera
    tracker.gaze_mapper = tracker.fc.BinocularGazeMapper(models, camera)
    random.seed(SEED_B)
    for index, frame in enumerate(frames[:ITERATIONS_B]):
        if index in events:
            events[index]()
        right_frame = cv2.flip(cv2.resize(frame, (1280, 720)), 0)
        record_eye(rows, "b_left_", tracker, "left", frame, probe)
        record_eye(rows, "b_right_", tracker, "right", right_frame, probe)
        tracker.update_binocular_gaze()
        estimate = tracker.gaze_estimate
        rows.add("b_", estimate=[np.nan] * 6 if estimate is None
                 else vector(estimate.point, 3) + [float(estimate.depth)] + vector(estimate.pixel, 2))
    arrays = rows.arrays()
    for eye_id in tracker.EYE_IDS:
        dump_lists(arrays, f"b_{eye_id}_", tracker, eye_id)
    return arrays


def synthetic_models():
    """Deterministic per-eye ray models (noise-free rendered calibration)."""
    sys.path.insert(0, REPO)
    from gazecore.tests import test_front_calibration as rendered
    _, result = rendered.calibrate_with_rendered_marker([0.4, 2.0], noise=0.0)
    return result.models, rendered.CAMERA


def record(tracker):
    """Run both scripted sessions through `tracker` (the old tracker module or an adapter)."""
    frames = read_frames(max(FRAMES_A, ITERATIONS_B))
    models, camera = synthetic_models()
    probe = ConfidenceProbe()
    saved = (cv2.imshow, cv2.putText, tracker.write_gaze_vector_file, tracker.CALIBRATION_FILE,
             tracker.GAZE_CALIBRATION_FILE, tracker.GAZE_SESSIONS_DIR, tracker.front_camera,
             tracker.saved_eye_origins, tracker.GL_SPHERE_AVAILABLE)
    cv2.imshow = lambda *args, **kwargs: None
    cv2.putText = probe
    tracker.write_gaze_vector_file = lambda: None
    tracker.GL_SPHERE_AVAILABLE = False
    try:
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            tracker.CALIBRATION_FILE = os.path.join(folder, "calibration.json")
            tracker.GAZE_CALIBRATION_FILE = os.path.join(folder, "gaze_calibration.json")
            tracker.GAZE_SESSIONS_DIR = os.path.join(folder, "calibrations")
            tracker.saved_eye_origins = {}
            arrays = run_a(tracker, frames, probe)
            arrays.update(run_b(tracker, frames, probe, models, camera))
    finally:
        (cv2.imshow, cv2.putText, tracker.write_gaze_vector_file, tracker.CALIBRATION_FILE,
         tracker.GAZE_CALIBRATION_FILE, tracker.GAZE_SESSIONS_DIR, tracker.front_camera,
         tracker.saved_eye_origins, tracker.GL_SPHERE_AVAILABLE) = saved
        tracker.reset_tracking_state()
    return arrays


def compare(actual, expected=None):
    """Returns a list of mismatch descriptions (empty when identical)."""
    if expected is None:
        with np.load(FIXTURE) as data:
            expected = {name: data[name] for name in data.files}
    problems = []
    for name in sorted(expected):
        if name.endswith("frame_crc") and not np.array_equal(actual.get(name), expected[name]):
            return [f"{name}: the decoded video frames differ (OpenCV/codec changed?); "
                    "regenerate the fixture from tag golden-baseline"]
    for name in sorted(set(expected) | set(actual)):
        if name not in actual or name not in expected:
            problems.append(f"{name}: missing in {'actual' if name not in actual else 'fixture'}")
            continue
        a, e = actual[name], expected[name]
        if a.shape != e.shape:
            problems.append(f"{name}: shape {a.shape} != {e.shape}")
        elif not np.array_equal(a, e, equal_nan=a.dtype.kind == "f"):
            rows = np.flatnonzero(~np.all((a == e) | (np.isnan(a) & np.isnan(e)) if a.dtype.kind == "f" else a == e,
                                          axis=tuple(range(1, a.ndim))))
            problems.append(f"{name}: first mismatch at row {rows[0]} of {len(a)}")
    return problems


def load_old_tracker():
    sys.path.insert(0, TRACKER_DIR)
    import Orlosky3DEyeTrackerFrontCamera as tracker
    return tracker


if __name__ == "__main__":
    arrays = record(load_old_tracker())
    if "--check" in sys.argv:
        problems = compare(arrays)
        print("\n".join(problems) or "Identical to the fixture.")
        sys.exit(1 if problems else 0)
    again = record(load_old_tracker())
    assert not compare(again, arrays), "recording is not deterministic"
    np.savez_compressed(FIXTURE, **arrays)
    print(f"Wrote {FIXTURE}: {len(arrays)} arrays")
