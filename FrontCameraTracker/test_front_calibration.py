"""Integration tests for front_calibration with rendered ArUco/ChArUco images and synthetic eyes.

Run: python FrontCameraTracker/test_front_calibration.py   (or pytest)
"""

import math
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import front_calibration as fc  # noqa: E402
import gaze_geometry as gg  # noqa: E402
import test_gaze_geometry as syn  # noqa: E402

IMAGE_SIZE = (640, 480)
CAMERA = fc.CameraModel.from_fov(IMAGE_SIZE, 60.0)
TARGET = fc.MarkerTarget(length_m=0.10)


def rotation(axis, degrees):
    return cv2.Rodrigues(np.asarray(axis, dtype=float) / np.linalg.norm(axis) * math.radians(degrees))[0]


def render_plane(image, plane_size_m, rotation_matrix, center, background=128):
    """Render a planar image (x right, y down on the plane) centered at `center` (camera frame)."""
    height, width = image.shape[:2]
    scale = np.diag([plane_size_m[0] / width, plane_size_m[1] / height, 1.0])
    to_center = np.array([[1, 0, -plane_size_m[0] / 2], [0, 1, -plane_size_m[1] / 2], [0, 0, 1]])
    pose = np.column_stack([rotation_matrix[:, 0], rotation_matrix[:, 1], center])
    homography = CAMERA.K @ pose @ to_center @ scale
    warped = cv2.warpPerspective(image, homography, IMAGE_SIZE, borderValue=background)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def render_marker(center, rotation_matrix=np.eye(3)):
    image = TARGET.printable_image(side_pixels=400, border_pixels=50)
    side = TARGET.length_m * image.shape[0] / 400  # the white border extends the printed square
    return render_plane(image, (side, side), rotation_matrix, np.asarray(center, dtype=float))


def grid_targets(depths, columns=(140, 320, 500), rows=(120, 240, 360)):
    return [CAMERA.ray((u, v)) / CAMERA.ray((u, v))[2] * depth for depth in depths for u in columns for v in rows]


def calibrate_with_rendered_marker(depths, noise=0.3, saved_origins=None):
    calibration = fc.GazeCalibration(("left", "right"))
    for target in grid_targets(depths):
        detection = TARGET.detect(render_marker(target), CAMERA)
        assert detection is not None, target
        calibration.add_sample({eye: syn.raw_gaze(eye, detection.point, noise) for eye in calibration.eye_ids},
                               detection)
    return calibration, calibration.solve(saved_origins)


def test_marker_pose_is_metric():
    for center in ([0.0, 0.0, 0.4], [0.15, -0.08, 0.6], [-0.4, 0.2, 2.0]):
        detection = TARGET.detect(render_marker(center, rotation([0, 1, 0], 20)), CAMERA)
        # Parallax depends on 1/Z (GAZE_GEOMETRY.md eq. 1), so judge depth in dioptres.
        inverse_depth_error = abs(1 / detection.point[2] - 1 / center[2])
        direction_error = gg.angular_errors_degrees([detection.point], [center])[0]
        print(f"marker at {center}: detected {np.round(detection.point, 3)}, "
              f"{inverse_depth_error:.3f} D, {direction_error:.2f} deg")
        assert inverse_depth_error < 0.03 and direction_error < 0.2
    assert TARGET.detect(np.full((480, 640, 3), 128, np.uint8), CAMERA) is None


def test_intrinsics_from_rendered_charuco_views():
    calibrator = fc.IntrinsicsCalibrator(IMAGE_SIZE)
    board = calibrator.board_image(size=(1400, 1000), margin=0)
    rng = np.random.default_rng(3)
    for index in range(40):
        if calibrator.done:
            break
        tilt = rotation(rng.normal(size=3), rng.uniform(10, 35))
        center = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.04, 0.04), rng.uniform(0.3, 0.45)])
        calibrator.add_frame(render_plane(board, (0.21, 0.15), tilt, center, background=255), now=index)
    camera, rms = calibrator.calibrate()
    focal_error = abs(camera.K[0, 0] / CAMERA.K[0, 0] - 1)
    print(f"intrinsics: f = {camera.K[0, 0]:.1f} px (true {CAMERA.K[0, 0]:.1f}), RMS {rms:.2f} px")
    assert camera.calibrated and focal_error < 0.02 and rms < 1.0


def test_intrinsics_capture_respects_interval():
    calibrator = fc.IntrinsicsCalibrator(IMAGE_SIZE, min_interval_s=1.0)
    frame = render_plane(calibrator.board_image(size=(1400, 1000), margin=0), (0.21, 0.15),
                         rotation([1, 0, 0], 15), np.array([0, 0, 0.4]), background=255)
    corners, ids = calibrator.add_frame(frame, now=0.0)
    cv2.aruco.drawDetectedCornersCharuco(frame.copy(), corners, ids, (0, 255, 0))  # what the tracker draws
    calibrator.add_frame(frame, now=0.5)
    calibrator.add_frame(frame, now=1.0)
    assert len(calibrator.views) == 2
    for empty in (np.full((480, 640, 3), 128, np.uint8), render_marker([0, 0, 0.5])):
        assert calibrator.add_frame(empty, now=5.0) == (None, None)
    try:
        calibrator.calibrate()
    except ValueError:
        return
    raise AssertionError("calibrate() must refuse before enough views")


def test_two_depth_calibration_and_binocular_mapping():
    calibration, result = calibrate_with_rendered_marker([0.4, 2.0])
    assert calibration.ready and result.origins_fitted
    print(result.summary())
    assert abs(result.eye_separation_mm - 64.0) < 3.0
    for eye, model in result.models.items():
        assert np.abs(model["origin"] - syn.TRUE_ORIGINS[eye])[:2].max() < 0.005

    mapper = fc.BinocularGazeMapper(result.models, CAMERA, alpha=1.0)
    baseline_mapper = fc.BinocularGazeMapper(result.baselines, CAMERA)
    accuracy = fc.AccuracyTest()
    for depth in (0.3, 0.8, 3.0):
        detection = TARGET.detect(render_marker(np.array([0.05, 0.03, 1.0]) * depth), CAMERA)
        gaze = {eye: syn.raw_gaze(eye, detection.point) for eye in result.models}
        estimate, baseline = mapper.update(gaze), baseline_mapper.update(gaze)
        sample = accuracy.record(CAMERA, detection, estimate.pixel, baseline.pixel)
        print(f"{depth} m: {sample['error']:.2f} deg, no-offset {sample['baseline_error']:.2f} deg, "
              f"vergence depth {estimate.depth:.2f} m")
        assert sample["error"] < 1.0 and sample["baseline_error"] > sample["error"]
    assert set(accuracy.summary_by_band()) == {"near", "mid", "far"}


def test_single_eye_mapping_uses_last_depth():
    _, result = calibrate_with_rendered_marker([0.4, 2.0], noise=0.0)
    mapper = fc.BinocularGazeMapper(result.models, CAMERA, alpha=1.0)
    target = np.array([0.1, 0.0, 1.0])
    mapper.update({eye: syn.raw_gaze(eye, target) for eye in result.models})
    estimate = mapper.update({"left": syn.raw_gaze("left", target), "right": None})
    error = gg.angular_errors_degrees([CAMERA.ray(estimate.pixel)], [target])[0]
    assert abs(estimate.depth - 1.0) < 0.1 and error < 0.5
    assert mapper.update({"left": None, "right": None}) is None


def test_single_depth_needs_saved_origins():
    try:
        calibrate_with_rendered_marker([0.4])
    except ValueError as exc:
        print("single depth without saved origins rejected:", exc)
    else:
        raise AssertionError("single-depth calibration without saved origins must fail")
    _, full = calibrate_with_rendered_marker([0.4, 2.0])
    saved = {eye: model["origin"] for eye, model in full.models.items()}
    calibration, result = calibrate_with_rendered_marker([1.0], saved_origins=saved)
    assert not calibration.ready and not result.origins_fitted
    assert np.allclose(result.models["left"]["origin"], saved["left"])


def test_missing_gaze_is_rejected():
    calibration = fc.GazeCalibration(("left", "right"))
    detection = TARGET.detect(render_marker([0, 0, 0.5]), CAMERA)
    try:
        calibration.add_sample({"left": np.array([0, 0, 1.0]), "right": None}, detection)
    except ValueError:
        assert calibration.samples == []
        return
    raise AssertionError("a sample without both eyes' gaze must be rejected")


def test_calibration_file_round_trip():
    camera = fc.CameraModel(CAMERA.K * [[1.01], [1.01], [1]], [0.1, -0.05, 0, 0, 0], IMAGE_SIZE, calibrated=True)
    origins = {"left": np.array([-0.03, 0.03, -0.01]), "right": np.array([0.03, 0.03, -0.01])}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "calibration.json")
        assert fc.load_calibration(path, IMAGE_SIZE) == (None, {})
        fc.save_calibration(path, camera, origins)
        loaded, loaded_origins = fc.load_calibration(path, IMAGE_SIZE)
        assert loaded.calibrated and np.allclose(loaded.K, camera.K) and np.allclose(loaded.dist, camera.dist)
        assert all(np.allclose(loaded_origins[eye], origins[eye]) for eye in origins)
        other_size, kept_origins = fc.load_calibration(path, (1280, 720))
        assert other_size is None and set(kept_origins) == {"left", "right"}

    calibration, _ = calibrate_with_rendered_marker([0.4, 2.0])
    eye_states = {"left": {"center": [300, 250]}, "right": {"center": [310, 240]}}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sessions", "gaze.json")
        fc.save_gaze_session(path, calibration, eye_states)
        restored, restored_states, saved_at = fc.load_gaze_session(path)
    assert restored_states == eye_states and saved_at
    assert restored.band_counts() == calibration.band_counts()
    assert np.allclose(restored.solve().models["right"]["origin"], calibration.solve().models["right"]["origin"])


def test_distorted_camera_round_trip():
    camera = fc.CameraModel(CAMERA.K, [-0.2, 0.05, 0.001, -0.001, 0], IMAGE_SIZE, calibrated=True)
    point = np.array([0.3, -0.2, 1.2])
    assert camera.angular_error_degrees(camera.project(point), CAMERA.project(point)) > 0.1
    assert gg.angular_errors_degrees([camera.ray(camera.project(point))], [point])[0] < 1e-3


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("All front_calibration tests passed.")
