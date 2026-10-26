"""EyeTracker and the eye overlay must reproduce the golden eye-pipeline record bit for bit.

The record was made with the front-camera tracker's globals-based pipeline. ClassHarness
rebuilds that tracker's per-eye rules (F, lock, O override, restore) on EyeTracker, so
eye_golden.record() can drive both the same way.

Run: python gazecore/tests/test_eye_tracker_golden.py   (or pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazecore import front_calibration as fc  # noqa: E402
from gazecore.eye_overlay import draw_eye_overlay  # noqa: E402
from gazecore.eye_tracker import CENTER_OVERRIDE_FRAMES_NEEDED, CenterOverrideDone, EyeTracker  # noqa: E402
from gazecore.tests.golden import eye_golden  # noqa: E402


class ClassHarness:
    """The tracker module's interface used by eye_golden.record(), backed by EyeTracker."""

    EYE_IDS = ("left", "right")
    fc = fc
    GL_SPHERE_AVAILABLE = False
    CALIBRATION_FILE = GAZE_CALIBRATION_FILE = GAZE_SESSIONS_DIR = None
    front_camera = None
    saved_eye_origins = {}

    def __init__(self):
        self.reset_tracking_state()

    def write_gaze_vector_file(self):
        pass

    def reset_tracking_state(self):
        self.trackers = {eye_id: EyeTracker(eye_id) for eye_id in self.EYE_IDS}
        self.eye_gaze_outputs = {eye_id: [0.0] * 6 for eye_id in self.EYE_IDS}
        self.current_eye_id = "left"
        self.clear_gaze_calibration()

    def clear_gaze_calibration(self):
        self.gaze_mapper = self.gaze_estimate = None

    @property
    def eye_tracking_states(self):
        return {eye_id: tracker.get_state() for eye_id, tracker in self.trackers.items()}

    def load_eye_tracking_state(self, eye_id):
        self.current_eye_id = eye_id

    def save_eye_tracking_state(self, eye_id):
        pass

    def prompt(self, tracker):
        if tracker.center_override_collecting:
            cancel_key = "o" if tracker.eye_id == "left" else "Shift+O"
            return (f"Look into the eye camera... {len(tracker.center_override_samples)}/"
                    f"{CENTER_OVERRIDE_FRAMES_NEEDED} ({cancel_key} cancels)")
        if tracker.adjusting:
            return "Press F to affix eyeball sphere"
        return "Press F again to engage automatic eye sphere adjustment"

    def process_frame(self, frame, return_rendered_frame=False):
        tracker = self.trackers[self.current_eye_id]
        result = tracker.process(frame)
        if result.gaze_output is not None:
            self.eye_gaze_outputs[tracker.eye_id] = result.gaze_output
        for event in result.events:
            if isinstance(event, CenterOverrideDone):  # the tracker restarted the calibration, freeing all eyes
                self.clear_gaze_calibration()
                for other in self.trackers.values():
                    if other is not tracker:
                        other.unlock_sphere()
        draw_eye_overlay(result, self.prompt(tracker))
        rect = result.ellipse if result.status == "ok" else None
        return (rect, result.frame) if return_rendered_frame else rect

    def toggle_eye_sphere_adjustment(self):
        if any(tracker.sphere_center_locked_2d for tracker in self.trackers.values()):
            self.clear_gaze_calibration()
            for tracker in self.trackers.values():
                tracker.unlock_sphere()
            enabled = True
        else:
            enabled = any(tracker.fixed for tracker in self.trackers.values())
        for tracker in self.trackers.values():
            tracker.set_adjustment_enabled(enabled)

    def lock_eye_spheres(self, eye_ids):
        for eye_id in eye_ids:
            self.trackers[eye_id].lock_sphere()

    def locked_eye_states(self, eye_ids):
        return {eye_id: self.trackers[eye_id].locked_state() for eye_id in eye_ids}

    def apply_eye_states(self, eye_states):
        for eye_id, state in eye_states.items():
            self.trackers[eye_id].apply_locked_state(state)

    def toggle_center_override(self, eye_id):
        self.trackers[eye_id].toggle_center_override()

    def update_binocular_gaze(self):
        gaze_dirs = {eye_id: tracker.last_gaze_dir for eye_id, tracker in self.trackers.items()}
        self.gaze_estimate = None if self.gaze_mapper is None else self.gaze_mapper.update(gaze_dirs)


def test_eye_tracker_matches_golden_record():
    if not os.path.exists(eye_golden.EYE_VIDEO):
        print("eye_test.mp4 not found; skipped")
        return
    problems = eye_golden.compare(eye_golden.record(ClassHarness()))
    assert not problems, "\n".join(problems[:20])


if __name__ == "__main__":
    test_eye_tracker_matches_golden_record()
    print("EyeTracker matches the golden record.")
