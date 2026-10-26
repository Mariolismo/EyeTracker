"""The app's headless pipeline and its validity rules. Run: python -m pytest FrontCameraApp"""

import os
import random
import tempfile

import cv2
import numpy as np

from frontcam_app import validity as v
from frontcam_app.pipeline import Pipeline
from gazecore import calibration_store as store
from gazecore import front_calibration as fc
from gazecore.camera.camera_reader import FramePacket
from gazecore.camera.camera_setup import RoleBinding
from gazecore.tests import test_front_calibration as rendered
from gazecore.tests.golden import eye_golden

LOCK_AT, COMPARE_UNTIL = 120, 160  # Golden run B locks both eyes at 120 and overrides the right eye at 160


def packet(frame, t, seq=0):
    return FramePacket(seq, t, frame, (frame.shape[1], frame.shape[0]))


def test_pipeline_reproduces_golden_run_b():
    if not os.path.exists(eye_golden.EYE_VIDEO):
        return
    frames = eye_golden.read_frames(COMPARE_UNTIL)
    models, camera = eye_golden.synthetic_models()
    pipeline = Pipeline(camera, {"right": RoleBinding(flip=True)}, gaze_vector_path=None)
    pipeline.mapper = fc.BinocularGazeMapper(models, camera)
    random.seed(eye_golden.SEED_B)
    rows = {"left": [], "right": [], "estimate": []}
    for index, frame in enumerate(frames):
        if index == LOCK_AT:
            for tracker in pipeline.trackers.values():
                tracker.lock_sphere()
        t = float(index)
        bundle = pipeline.step({"left": packet(frame, t, index + 1),
                                "right": packet(cv2.resize(frame, (1280, 720)), t, index + 1)}, now=t)
        for eye in ("left", "right"):
            rows[eye].append(eye_golden.vector(pipeline.trackers[eye].last_gaze_dir, 3)
                             + eye_golden.vector(pipeline.gaze_outputs[eye], 6))
        e = bundle.estimate
        rows["estimate"].append([np.nan] * 6 if e is None else
                                eye_golden.vector(e.point, 3) + [float(e.depth)] + eye_golden.vector(e.pixel, 2))
    with np.load(eye_golden.FIXTURE) as golden:
        for eye in ("left", "right"):
            expected = np.hstack([golden[f"b_{eye}_gaze_dir"], golden[f"b_{eye}_gaze_output"]])[:COMPARE_UNTIL]
            assert np.array_equal(np.array(rows[eye]), expected, equal_nan=True), eye
        assert np.array_equal(np.array(rows["estimate"]), golden["b_estimate"][:COMPARE_UNTIL], equal_nan=True)


def test_gaze_vector_file_format():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "gaze_vector.txt")
        pipeline = Pipeline(rendered.CAMERA, gaze_vector_path=path)
        pipeline.step({})
        values = open(path).read().strip().split(",")
    assert len(values) == 16 and all(value == "0.000000" for value in values)


# --- Validity ---

EYE = {"device_id": "USB\\VID_0C45&PID_6366\\A", "native_size": [640, 480], "flip": True, "mirror": False}
LIVE = {"left": EYE, "right": dict(EYE, device_id="USB\\VID_0C45&PID_6366\\B"),
        "front": {"device_id": "USB\\VID_0BDA&PID_5697\\C", "native_size": [640, 480], "flip": False, "mirror": False}}


def stored(meta=None):
    calibration, _ = rendered.calibrate_with_rendered_marker([0.4, 2.0])
    states = {eye: {"model_center": [320, 240], "sphere_radius": 100.0, "sphere_center": None,
                    "adjustment_enabled": False} for eye in ("left", "right")}
    return store.StoredGaze(calibration, states, "now", meta)


def calibrated_camera():
    return fc.CameraModel(rendered.CAMERA.K, [0, 0, 0, 0, 0], (640, 480), calibrated=True)


def test_legacy_files_are_accepted_with_a_banner():
    front = store.FrontCalibration(calibrated_camera(), {}, None)
    result = v.evaluate(LIVE, front, stored(), gaze_solves=True)
    assert result.ok and result.legacy
    flipped = dict(LIVE, right=dict(LIVE["right"], flip=False))
    assert v.evaluate(flipped, front, stored(), True).first_step == v.EYE_RIGHT


def test_first_missing_step():
    camera = calibrated_camera()
    front = store.FrontCalibration(camera, {}, None)
    assert v.evaluate(dict(LIVE, front=None), front, stored(), True).first_step == v.CAMERAS
    assert v.evaluate(LIVE, store.FrontCalibration(None, {}, None), None, False).first_step == v.INTRINSICS
    assert v.evaluate(LIVE, front, None, False).first_step == v.EYE_LEFT
    assert v.evaluate(LIVE, front, stored(), gaze_solves=False).first_step == v.GAZE


def test_app_metadata_must_match_the_cameras_and_lens():
    camera = calibrated_camera()
    front = store.FrontCalibration(camera, {}, {"front": LIVE["front"]})
    gaze = stored()
    meta = {"cameras": LIVE, "intrinsics_fingerprint": store.intrinsics_fingerprint(camera),
            "fingerprint": store.samples_fingerprint(gaze.calibration, gaze.eye_states)}
    gaze.app_meta = meta
    assert v.evaluate(LIVE, front, gaze, True) == v.Validity()
    moved = dict(LIVE, left=dict(EYE, native_size=[1280, 720]))
    assert v.evaluate(moved, front, gaze, True).first_step == v.EYE_LEFT
    gaze.app_meta = dict(meta, intrinsics_fingerprint="other")
    assert v.evaluate(LIVE, front, gaze, True).first_step == v.GAZE
    other_front = dict(LIVE, front=dict(LIVE["front"], device_id="X"))
    assert v.evaluate(other_front, front, gaze, True).first_step == v.INTRINSICS


def test_files_written_with_metadata_still_load_in_the_tracker_format():
    calibration, _ = rendered.calibrate_with_rendered_marker([0.4, 2.0])
    states = stored().eye_states
    with tempfile.TemporaryDirectory() as folder:
        plain, extended = os.path.join(folder, "a.json"), os.path.join(folder, "b.json")
        fc.save_calibration(plain, calibrated_camera(), {"left": np.zeros(3)})
        fc.save_calibration(plain + "2", calibrated_camera(), {"left": np.zeros(3)}, extra=None)
        assert open(plain).read() == open(plain + "2").read()
        store.save_gaze(calibration, states, {"version": 1}, path=extended)
        loaded, loaded_states, _ = fc.load_gaze_session(extended)  # what the tracker calls
        assert loaded_states == states and len(loaded.samples) == len(calibration.samples)
        assert store.load_gaze(extended).app_meta == {"version": 1}


def test_a_legacy_gaze_calibration_older_than_the_lens_is_refused():
    gaze = stored()
    gaze.saved_at = "2026-10-01T10:00:00"
    front = store.FrontCalibration(calibrated_camera(), {}, {"front": LIVE["front"], "calibrated_at": "2026-10-20T09:00:00"})
    assert v.evaluate(LIVE, front, gaze, True).first_step == v.GAZE
    front.app_meta["calibrated_at"] = "2026-09-01T09:00:00"
    assert v.evaluate(LIVE, front, gaze, True).ok


def test_lens_views_spread_over_the_image():
    pipeline = Pipeline(rendered.CAMERA, gaze_vector_path=None)
    pipeline.start_intrinsics(fc.IntrinsicsCalibrator((640, 480), min_interval_s=0.0))
    board = pipeline.intrinsics_calibrator.board_image((700, 500), margin=20)
    frame = rendered.render_plane(board, (0.21, 0.15), np.eye(3), np.array([0.0, 0.0, 0.35]), background=255)
    for i in range(6):  # The same view again and again: only the first ones count
        bundle = pipeline.step({"front": packet(frame, float(i), i + 1)}, now=float(i))
    assert bundle.intrinsics_views == 2 and sum(bundle.charuco_cells.values()) == 2
    assert bundle.charuco is not None
