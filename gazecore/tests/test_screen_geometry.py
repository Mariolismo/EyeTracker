"""Screen pose from corner markers, eye rays on the screen, gaze smoothing and the heatmap.

Run: python -m pytest gazecore/tests/test_screen_geometry.py
"""

import os
import tempfile

import numpy as np

from gazecore import front_calibration as fc
from gazecore import screen_geometry as sg
from gazecore.filters import OneEuroFilter2D
from gazecore.heatmap import Heatmap
from gazecore.tests import test_front_calibration as rendered
from gazecore.tests import test_gaze_geometry as syn

LAYOUT = sg.ScreenLayout.corners(1280, 720, 0.46, midpoints=True)  # A 589 x 331 mm screen
CAMERA = rendered.CAMERA


def screen_image(layout):
    image = np.full((layout.height_px, layout.width_px), 128, np.uint8)
    for marker_id, (cx, cy, _) in layout.markers.items():
        marker = layout.marker_image(marker_id)
        size = marker.shape[0]
        x, y = int(round(cx - size / 2)), int(round(cy - size / 2))
        image[y:y + size, x:x + size] = marker
    return image


def pose_for(rotation, center):
    """The pose that render_plane uses for a screen centred at center (camera frame)."""
    middle = LAYOUT.to_metres(LAYOUT.width_px / 2, LAYOUT.height_px / 2)
    return rotation, np.asarray(center, float) - rotation @ middle


def render_screen(rotation, center):
    size_m = (LAYOUT.width_px * LAYOUT.mm_per_px / 1000, LAYOUT.height_px * LAYOUT.mm_per_px / 1000)
    return rendered.render_plane(screen_image(LAYOUT), size_m, rotation, np.asarray(center, float))


def test_pose_from_rendered_screens():
    tracker = sg.ScreenTracker(LAYOUT, CAMERA)
    for rotation, center in ((np.eye(3), [0, 0, 0.8]),
                             (rendered.rotation([0, 1, 0], 20), [0.05, -0.03, 0.9]),
                             (rendered.rotation([1, 0.3, 0], -15), [-0.06, 0.04, 1.1])):
        detections = tracker.detect(render_screen(rotation, center))
        assert len(detections) >= 4
        pose = sg.estimate_pose(detections, LAYOUT, CAMERA)
        assert pose is not None
        expected_rotation, expected_translation = pose_for(rotation, center)
        assert np.allclose(pose.rotation, expected_rotation, atol=0.03)
        assert np.linalg.norm(pose.translation - expected_translation) < 0.015
        assert pose.camera_position[2] < 0  # The camera is in front of the screen


def test_pose_needs_two_markers_and_rejects_noise():
    tracker = sg.ScreenTracker(LAYOUT, CAMERA)
    detections = tracker.detect(render_screen(np.eye(3), [0, 0, 0.8]))
    assert sg.estimate_pose({1: detections[1]}, LAYOUT, CAMERA) is None
    two = {i: detections[i] for i in (1, 2)}
    assert sg.estimate_pose(two, LAYOUT, CAMERA) is not None
    swapped = {1: detections[2], 2: detections[1]}  # Wrong ids: no consistent pose
    assert sg.estimate_pose(swapped, LAYOUT, CAMERA) is None


def test_ray_hits_the_screen_where_it_points():
    rotation, translation = pose_for(rendered.rotation([0, 1, 0], 25), [0.1, 0, 0.7])
    pose = sg.ScreenPose(rotation, translation, 0.0, ())
    origin = np.array([0.03, 0.04, -0.02])
    for x_px, y_px in ((0, 0), (640, 360), (1279, 719), (-200, 900)):
        target = rotation @ LAYOUT.to_metres(x_px, y_px) + translation
        hit = sg.ray_hit(pose, origin, target - origin)
        assert np.allclose(LAYOUT.to_pixels(hit), (x_px, y_px), atol=1e-6)
    away = rotation @ LAYOUT.to_metres(640, 360) + translation - origin
    assert sg.ray_hit(pose, origin, -away) is None  # Looking away from the screen


def test_end_to_end_gaze_on_a_rendered_screen():
    _, result = rendered.calibrate_with_rendered_marker([0.4, 2.0])
    mapper = fc.BinocularGazeMapper(result.models, CAMERA)
    rotation, center = rendered.rotation([0, 1, 0], 10), [0.0, 0.0, 0.8]
    frame = render_screen(rotation, center)
    rotation, translation = pose_for(rotation, center)
    tracker = sg.ScreenTracker(LAYOUT, CAMERA)
    for x_px, y_px in ((320, 200), (640, 360), (1000, 560)):
        target = rotation @ LAYOUT.to_metres(x_px, y_px) + translation
        gaze = {eye: syn.raw_gaze(eye, target) for eye in ("left", "right")}
        screen = tracker.update(frame, mapper, gaze, 0.0)
        assert screen.on_screen and screen.eyes_used == 2
        error_mm = np.linalg.norm(screen.pixel - (x_px, y_px)) * LAYOUT.mm_per_px
        assert error_mm < 6, error_mm
        assert screen.disagreement_mm < 10
    one_eye = tracker.update(frame, mapper, {"left": gaze["left"], "right": None}, 0.0)
    assert one_eye.eyes_used == 1 and one_eye.disagreement_mm is None


def run_filter(rate_hz, seconds=1.0):
    smoother = OneEuroFilter2D()
    value = None
    for i in range(int(seconds * rate_hz) + 1):
        t = i / rate_hz
        value = smoother(np.array([100.0 if t >= 0.2 else 0.0, 0.0]), t)
    return value


def test_filter_is_rate_independent_and_steady():
    assert abs(run_filter(30)[0] - run_filter(60)[0]) < 2.0
    smoother = OneEuroFilter2D()
    rng = np.random.default_rng(0)
    out = [smoother(np.array([50.0, 50.0]) + rng.normal(scale=3.0, size=2), i / 30) for i in range(90)]
    assert np.std(np.array(out[30:]), axis=0).max() < 1.5  # A fixation stays put


def test_filter_restarts_after_a_gap():
    smoother = OneEuroFilter2D()
    smoother(np.zeros(2), 0.0)
    assert np.allclose(smoother(np.array([300.0, 0.0]), 2.0), [300.0, 0.0])


def heat(rate_hz, seconds=2.0):
    heatmap = Heatmap(400, 300, 0.5)
    for i in range(int(seconds * rate_hz) + 1):
        heatmap.add((200, 150), i / rate_hz)
    return heatmap


def test_heatmap_counts_seconds_not_frames():
    slow, fast = heat(30), heat(60)
    assert np.isclose(slow.total.sum(), fast.total.sum(), rtol=0.02)
    assert np.allclose(slow.live, fast.live, rtol=0.05, atol=1e-4)
    assert slow.live.sum() < slow.total.sum()  # The live map decays


def test_heatmap_ignores_off_screen_and_saves():
    heatmap = Heatmap(400, 300, 0.5)
    heatmap.add((200, 150), 0.0)
    heatmap.add((-5000, 150), 0.1)
    assert heatmap.total.sum() == 0
    heatmap.add((100, 100), 0.2)
    assert heatmap.rgba().shape == (300, 400, 4)
    with tempfile.TemporaryDirectory() as folder:
        png, csv = os.path.join(folder, "h.png"), os.path.join(folder, "h.csv")
        heatmap.save(png, csv)
        assert os.path.getsize(png) > 0 and np.loadtxt(csv, delimiter=",").shape == heatmap.total.shape
