"""Headless integration tests of the tracker's orchestration (no cameras, no windows).

Run: python FrontCameraTracker/test_tracker_integration.py   (or pytest)
"""

import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Orlosky3DEyeTrackerFrontCamera as tracker  # noqa: E402
import test_front_calibration as rendered  # noqa: E402
import test_gaze_geometry as syn  # noqa: E402

EYE_VIDEO = os.path.join(os.path.dirname(HERE), "eye_test.mp4")


def headless(test):
    """Silence windows and file output; keep calibration files in a temp folder."""
    def wrapper():
        saved = (cv2.imshow, tracker.write_gaze_vector_file, tracker.CALIBRATION_FILE,
                 tracker.GAZE_CALIBRATION_FILE, tracker.GAZE_SESSIONS_DIR)
        cv2.imshow = lambda *args, **kwargs: None
        tracker.write_gaze_vector_file = lambda: None
        with tempfile.TemporaryDirectory() as folder:
            tracker.CALIBRATION_FILE = os.path.join(folder, "calibration.json")
            tracker.GAZE_CALIBRATION_FILE = os.path.join(folder, "gaze_calibration.json")
            tracker.GAZE_SESSIONS_DIR = os.path.join(folder, "calibrations")
            tracker.saved_eye_origins = {}
            tracker.front_camera = rendered.CAMERA
            tracker.reset_tracking_state()
            try:
                test()
            finally:
                (cv2.imshow, tracker.write_gaze_vector_file, tracker.CALIBRATION_FILE,
                 tracker.GAZE_CALIBRATION_FILE, tracker.GAZE_SESSIONS_DIR) = saved
    wrapper.__name__ = test.__name__
    return wrapper


def look_at(point, noise=0.3):
    """Simulate both eye trackers reporting a fixation on point."""
    for eye in tracker.EYE_IDS:
        tracker.eye_tracking_states[eye]["last_gaze_dir"] = syn.raw_gaze(eye, point, noise)
        tracker.eye_tracking_states[eye]["last_sphere_center"] = np.zeros(3)
    tracker.load_eye_tracking_state(tracker.current_eye_id)


@headless
def test_eye_states_are_isolated():
    tracker.load_eye_tracking_state("left")
    tracker.ray_lines.append("left ray")
    tracker.save_eye_tracking_state("left")
    tracker.load_eye_tracking_state("right")
    assert tracker.ray_lines == []
    tracker.load_eye_tracking_state("left")
    assert tracker.ray_lines == ["left ray"]

    tracker.toggle_eye_sphere_adjustment()
    assert tracker.eyes_fixed()
    assert all(not s["eye_sphere_adjustment_enabled"] for s in tracker.eye_tracking_states.values())
    tracker.toggle_eye_sphere_adjustment()
    assert not tracker.eyes_fixed()


@headless
def test_calibration_flow_through_tracker():
    for target in rendered.grid_targets([0.4, 2.0]):
        tracker.process_front_frame(rendered.render_marker(target))
        look_at(tracker.last_marker.point)
        assert tracker.capture_calibration_sample(tracker.EYE_IDS), tracker.calibration_message
        if tracker.calibration_result is not None:
            break
    print(tracker.calibration_message)
    assert tracker.calibration_result is not None
    assert all(s["sphere_center_locked_2d"] for s in tracker.eye_tracking_states.values())
    assert os.path.exists(tracker.CALIBRATION_FILE)

    tracker.toggle_accuracy_test()
    for depth in (0.3, 3.0):
        target = np.array([0.05, 0.03, 1.0]) * depth
        tracker.process_front_frame(rendered.render_marker(target))
        look_at(tracker.last_marker.point, noise=0.0)
        for _ in range(20):  # let the depth filter settle
            tracker.update_binocular_gaze()
        assert tracker.record_accuracy_sample(), tracker.accuracy_message
    print("\n".join(tracker.accuracy_summary_lines()))
    assert all(sample["error"] < 1.0 for sample in tracker.accuracy_test.samples)

    # C after calibrating restarts, but keeps the spheres affixed for the new samples.
    tracker.capture_calibration_sample(tracker.EYE_IDS)
    assert tracker.calibration_result is None and tracker.gaze_calibration is None
    assert all(s["sphere_center_locked_2d"] for s in tracker.eye_tracking_states.values())

    # The saved eye offsets now allow a quick single-depth recalibration (K).
    tracker.saved_eye_origins = {}
    tracker.load_front_calibration()
    for target in rendered.grid_targets([1.0]):
        tracker.process_front_frame(rendered.render_marker(target))
        look_at(tracker.last_marker.point)
        tracker.capture_calibration_sample(tracker.EYE_IDS)
    assert tracker.solve_gaze_calibration(), tracker.calibration_message


@headless
def test_calibration_survives_restart():
    assert not tracker.restore_gaze_calibration(tracker.EYE_IDS)  # nothing saved yet
    # A failed session is still kept (timestamped) for analysis.
    tracker.process_front_frame(rendered.render_marker(rendered.grid_targets([0.4])[0]))
    look_at(tracker.last_marker.point)
    tracker.capture_calibration_sample(tracker.EYE_IDS)
    assert len(os.listdir(tracker.GAZE_SESSIONS_DIR)) == 1
    tracker.reset_tracking_state()

    for eye in tracker.EYE_IDS:  # distinct eye centers, to check they come back
        tracker.eye_tracking_states[eye]["prev_model_center_avg"] = (300 + len(eye), 250)
    tracker.load_eye_tracking_state("left")
    for target in rendered.grid_targets([0.4, 2.0]):
        tracker.process_front_frame(rendered.render_marker(target))
        look_at(tracker.last_marker.point)
        tracker.capture_calibration_sample(tracker.EYE_IDS)
        if tracker.calibration_result is not None:
            break
    origins = {eye: model["origin"] for eye, model in tracker.calibration_result.models.items()}
    assert os.path.exists(tracker.GAZE_CALIBRATION_FILE)

    tracker.reset_tracking_state()  # what a restart (or restarting the cameras) does
    assert tracker.calibration_result is None
    assert not tracker.restore_gaze_calibration(("left",))  # the right eye camera is not running
    assert tracker.restore_gaze_calibration(tracker.EYE_IDS)
    for eye in tracker.EYE_IDS:
        state = tracker.eye_tracking_states[eye]
        assert state["sphere_center_locked_2d"] and state["locked_model_center_avg"] == (300 + len(eye), 250)
        assert np.allclose(tracker.calibration_result.models[eye]["origin"], origins[eye])
    assert tracker.gaze_mapper is not None


@headless
def test_marker_must_be_visible():
    tracker.process_front_frame(np.full((480, 640, 3), 128, np.uint8))
    look_at(np.array([0, 0, 1.0]))
    assert not tracker.capture_calibration_sample(tracker.EYE_IDS)
    assert tracker.gaze_calibration is None


@headless
def test_eye_video_still_tracks():
    if not os.path.exists(EYE_VIDEO):
        print("eye_test.mp4 not found; skipped")
        return
    capture = cv2.VideoCapture(EYE_VIDEO)
    gazes = 0
    for _ in range(40):
        ok, frame = capture.read()
        if not ok:
            break
        tracker.process_frame(frame, return_rendered_frame=True)
        tracker.save_eye_tracking_state("left")
        gazes += tracker.eye_tracking_states["left"]["last_gaze_dir"] is not None
    capture.release()
    print(f"eye video: gaze on {gazes}/40 frames")
    assert gazes > 20


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("All tracker integration tests passed.")
