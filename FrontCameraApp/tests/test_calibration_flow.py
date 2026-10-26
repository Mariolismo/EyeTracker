"""Capture gates, the calibration session and its review. Run: python -m pytest FrontCameraApp"""

import os
import tempfile

import numpy as np

from frontcam_app import capture_gates as gates
from frontcam_app import review
from frontcam_app.calibration_session import CalibrationSession
from frontcam_app.monitors import Monitors
from frontcam_app.pipeline import HistoryEntry
from gazecore import calibration_store as store
from gazecore import front_calibration as fc
from gazecore.tests import test_front_calibration as rendered
from gazecore.tests import test_gaze_geometry as syn

CAMERA = rendered.CAMERA
STATES = {eye: {"model_center": [320, 240], "sphere_radius": 100.0, "sphere_center": None,
                "adjustment_enabled": False} for eye in ("left", "right")}


def entries(point, count=30, noise_deg=0.0, confidence=0.95, skew=0.0, marker_every=1, generation=0, t0=0.0):
    point = np.asarray(point, float)
    pixel = CAMERA.project(point)
    corners = pixel + np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]])
    out = []
    for i in range(count):
        t = t0 + i / 30
        visible = i % marker_every == 0
        out.append(HistoryEntry(
            t, {eye: syn.raw_gaze(eye, point, noise_deg) for eye in ("left", "right")},
            {"left": confidence, "right": confidence}, point if visible else None, pixel if visible else None,
            generation, t + skew, {"left": t, "right": t}, corners if visible else None))
    return out


def target(u, v, depth):
    ray = CAMERA.ray((u, v))
    return ray / ray[2] * depth


def test_a_steady_look_passes_and_averages():
    result = gates.evaluate(entries(target(320, 240, 0.45)), gates.GateContext(band="near"), CAMERA)
    assert result.ok, result.failures
    assert np.allclose(result.sample.point, target(320, 240, 0.45))
    assert result.sample.frames == 30


def test_each_gate_explains_itself():
    near = target(320, 240, 0.45)
    cases = [
        (entries(near, count=10), gates.GateContext(), "Hold still"),
        (entries(near, marker_every=3), gates.GateContext(), "not visible"),
        (entries(near, generation=0), gates.GateContext(marker_generation=1), "not visible"),
        (entries(target(10, 240, 0.45)), gates.GateContext(), "edge"),
        (entries(near), gates.GateContext(band="far"), "Move back"),
        (entries(target(320, 240, 2.0)), gates.GateContext(band="near"), "Move closer"),
        (entries(near, confidence=0.5), gates.GateContext(), "pupil"),
        (entries(near, skew=0.1), gates.GateContext(), "out of step"),
        (entries(near, noise_deg=2.0), gates.GateContext(), "eyes on the marker"),
        (entries(near), gates.GateContext(earlier_pixels=[CAMERA.project(near) + 30]), "Already have"),
        (entries(near), gates.GateContext(prerequisites=["Lock both eyes first"]), "Lock both"),
    ]
    for history, context, expected in cases:
        result = gates.evaluate(history, context, CAMERA)
        assert not result.ok and expected in result.failures[0], (expected, result.failures)


def test_single_frame_capture_like_the_tracker():
    result = gates.evaluate(entries(target(320, 240, 0.45), count=1), gates.GateContext(window=1), CAMERA)
    assert result.ok and result.sample.frames == 1


def test_band_coverage_needs_rows_and_columns():
    in_a_row = [(100 + 80 * i, 240) for i in range(6)]
    assert not gates.band_ready(in_a_row)
    spread = [(100, 100), (320, 100), (540, 100), (100, 380), (320, 380), (540, 380)]
    assert gates.band_ready(spread) and not gates.band_ready(spread[:5])


def test_plausibility_flags_a_sample_taken_looking_elsewhere():
    first = gates.evaluate(entries(target(160, 240, 0.45)), gates.GateContext(), CAMERA).sample
    second = gates.evaluate(entries(target(480, 240, 0.45)), gates.GateContext(), CAMERA).sample
    assert gates.plausibility(first, second) is None
    stared = gates.Sample(first.gaze, second.point, second.pixel, 1.0, 30, {}, 0.0)
    assert "were you looking" in gates.plausibility(first, stared)


def capture(session, depths, noise=0.2):
    for point in rendered.grid_targets(depths):
        history = entries(point, noise_deg=0.0)
        context = gates.GateContext(earlier_pixels=session.pixels(gates.gg.depth_band(point[2])))
        result = gates.evaluate(history, context, CAMERA)
        assert result.ok, result.failures
        sample = result.sample
        sample.gaze = {eye: syn.raw_gaze(eye, sample.point, noise) for eye in sample.gaze}
        session.add(sample)


def test_session_solve_review_accept_and_reload():
    with tempfile.TemporaryDirectory() as folder:
        session = CalibrationSession(STATES, {}, settings={"capture_mode": "manual", "frames_averaged": 30},
                                     session_path=os.path.join(folder, "calibrations", "s.json"))
        capture(session, [0.45])
        assert session.band_ready("near") and not session.ready
        capture(session, [2.0])
        assert session.ready and os.path.exists(session.session_path)
        session.undo()
        assert session.counts()["far"] == 8
        solved = session.solve()
        report = review.review(solved.result, session.calibration)
        assert report.level == review.GREEN, report.lines
        front = store.FrontCalibration(fc.CameraModel(CAMERA.K, None, (640, 480), calibrated=True), {}, None)
        gaze_path, front_path = os.path.join(folder, "g.json"), os.path.join(folder, "f.json")
        cameras = {"left": {"device_id": "a", "native_size": [640, 480], "flip": True, "mirror": False}}
        session.accept(solved, front, cameras, gaze_path, front_path)
        loaded = store.load_gaze(gaze_path)
        assert loaded.app_meta["cameras"] == cameras and len(loaded.app_meta["sample_meta"]) == 17
        assert loaded.app_meta["fingerprint"] == store.samples_fingerprint(loaded.calibration, loaded.eye_states)
        assert set(store.load_front(front_path).eye_origins) == {"left", "right"}

        session.remove_band("far")
        assert session.counts()["far"] == 0 and not session.ready


def test_quick_recalibration_reuses_the_saved_origins():
    _, full = rendered.calibrate_with_rendered_marker([0.4, 2.0])
    origins = {eye: model["origin"] for eye, model in full.models.items()}
    with tempfile.TemporaryDirectory() as folder:
        session = CalibrationSession(STATES, origins, quick=True, session_path=os.path.join(folder, "s.json"))
        capture(session, [0.45])
        assert session.ready
        solved = session.solve()
    assert not solved.result.origins_fitted
    assert any("reused" in text for text, _ in review.review(solved.result).lines)


def test_review_flags_swapped_eyes_and_bad_separation():
    _, result = rendered.calibrate_with_rendered_marker([0.4, 2.0])
    result.models["left"], result.models["right"] = result.models["right"], result.models["left"]
    texts = [text for text, level in review.review(result).lines if level == review.RED]
    assert any("swapped" in text for text in texts)


def test_monitor_scale_is_remembered():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "monitors.json")
        Monitors(path).confirm("DELL 2560x1440", 0.2331, "card")
        assert Monitors(path).get("DELL 2560x1440")["mm_per_px"] == 0.2331
        assert Monitors(path).get("other") is None
