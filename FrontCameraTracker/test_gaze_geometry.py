"""Synthetic checks for gaze_geometry. Run: python FrontCameraTracker/test_gaze_geometry.py"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gaze_geometry as gg  # noqa: E402

RNG = np.random.default_rng(7)
K = gg.default_intrinsics(640, 480, 60.0)
# Eyes sit below (+y) and behind (-z) a forehead camera, 64 mm apart.
TRUE_ORIGINS = {"left": np.array([-0.032, 0.035, -0.015]), "right": np.array([0.032, 0.035, -0.015])}


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    t = 1 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


# Raw eye-tracker frame: rotated, with gain errors (the model's M must undo this).
TRUE_DISTORTION = {
    "left": rotation([0.3, 1.0, 0.2], 25) @ np.diag([1.15, 0.9, 1.0]),
    "right": rotation([-0.2, 1.0, 0.4], -20) @ np.diag([0.95, 1.1, 1.0]),
}


def raw_gaze(eye, target, noise_degrees=0.0):
    true_direction = gg.unit(target - TRUE_ORIGINS[eye])
    gaze = np.linalg.solve(TRUE_DISTORTION[eye], true_direction)
    if noise_degrees:
        gaze = rotation(RNG.normal(size=3), RNG.normal(scale=noise_degrees)) @ gaze
    return gg.unit(gaze)


def targets_at(depths, grid=4):
    points = []
    for depth in depths:
        for u in np.linspace(80, 560, grid):
            for v in np.linspace(60, 420, grid):
                points.append(gg.pixel_ray(K, None, (u, v)) * depth / gg.pixel_ray(K, None, (u, v))[2])
    return np.array(points)


def fit(eye, targets, noise, fixed_origin=None):
    gazes = np.array([raw_gaze(eye, p, noise) for p in targets])
    return gg.fit_eye_model(gazes, targets, fixed_origin=fixed_origin)


def camera_error_degrees(model, eye, depth):
    """Front-camera angular error of a single-eye prediction at a known depth."""
    targets = targets_at([depth], grid=5)
    predicted = []
    for target in targets:
        origin, direction = gg.eye_ray(model, raw_gaze(eye, target))
        predicted.append(gg.point_at_depth(origin, direction, depth))
    return float(np.max(gg.angular_errors_degrees(predicted, targets)))


def test_recovers_origin_and_removes_parallax():
    targets = targets_at([0.4, 2.0])
    for eye in TRUE_ORIGINS:
        model = fit(eye, targets, noise=0.5)
        offset_error = np.abs(model["origin"] - TRUE_ORIGINS[eye])
        print(f"{eye}: origin {np.round(model['origin'] * 1000, 1)} mm, error {np.round(offset_error * 1000, 1)} mm, "
              f"fit RMS {model['rms_deg']:.2f} deg")
        assert offset_error[:2].max() < 0.006, offset_error
        assert offset_error[2] < 0.03, offset_error
        for depth in (0.25, 1.0, 5.0):
            error = camera_error_degrees(model, eye, depth)
            print(f"  max error at {depth} m: {error:.2f} deg")
            assert error < 1.0, (eye, depth, error)


def test_mirrored_eye_camera():
    """A mirrored eye image gives left-handed raw gaze; the fit must still work."""
    targets = targets_at([0.4, 2.0])
    mirror = np.diag([-1.0, 1.0, 1.0])
    gazes = np.array([mirror @ raw_gaze("left", p, 0.3) for p in targets])
    model = gg.fit_eye_model(gazes, targets)
    assert np.abs(model["origin"] - TRUE_ORIGINS["left"])[:2].max() < 0.006, model["origin"]
    target = np.array([0.2, -0.1, 1.0])
    origin, direction = gg.eye_ray(model, mirror @ raw_gaze("left", target))
    assert gg.angular_errors_degrees([gg.point_at_depth(origin, direction, 1.0)], [target])[0] < 0.5


def test_glitched_sample_is_ignored():
    """A pupil-tracking flicker during capture must not ruin the eye's fit."""
    targets = targets_at([0.4, 2.0])
    gazes = np.array([raw_gaze("right", p, 0.5) for p in targets])
    gazes[3] = rotation([0, 1, 0], 60) @ gazes[3]
    model = gg.fit_eye_model(gazes, targets)
    assert not model["inliers"][3] and model["inliers"].sum() == len(targets) - 1
    assert np.abs(model["origin"] - TRUE_ORIGINS["right"])[:2].max() < 0.006, model["origin"]
    assert model["rms_deg"] < 1.0


def test_no_offset_model_shows_parallax():
    targets = targets_at([0.4])
    model = fit("left", targets, noise=0.0, fixed_origin=np.zeros(3))
    near = camera_error_degrees(model, "left", 0.4)
    far = camera_error_degrees(model, "left", 5.0)
    expected = math.degrees(np.linalg.norm(TRUE_ORIGINS["left"][:2]) * (1 / 0.4 - 1 / 5.0))
    print(f"no-offset model: {near:.2f} deg at 0.4 m, {far:.2f} deg at 5 m (|o|*d(1/Z) ~ {expected:.2f})")
    assert near < 1.5 and far > 0.6 * expected


def test_single_depth_is_rejected():
    try:
        fit("left", targets_at([0.4]), noise=0.0)
    except ValueError as exc:
        print("single depth rejected:", exc)
        return
    raise AssertionError("single-depth calibration should be rejected")


def test_saved_origin_allows_single_depth_recalibration():
    saved_origin = fit("left", targets_at([0.4, 2.0]), noise=0.5)["origin"]
    model = fit("left", targets_at([1.0]), noise=0.5, fixed_origin=saved_origin)
    for depth in (0.25, 5.0):
        error = camera_error_degrees(model, "left", depth)
        print(f"single-depth recalibration with saved origin: {error:.2f} deg at {depth} m")
        assert error < 1.5


def test_binocular_depth():
    targets = targets_at([0.4, 2.0])
    models = {eye: fit(eye, targets, noise=0.0) for eye in TRUE_ORIGINS}
    for depth in (0.3, 1.0, 3.0):
        target = np.array([0.1, -0.05, depth])
        depth_filter = gg.InverseDepthFilter(alpha=1.0)
        point, estimated = gg.binocular_gaze_point(models, {e: raw_gaze(e, target) for e in models}, depth_filter)
        error = gg.angular_errors_degrees([point], [target])[0]
        print(f"binocular target at {depth} m: depth {estimated:.3f} m, camera error {error:.3f} deg")
        # The synthetic distortion is not exactly representable, so allow model mismatch.
        # Parallax depends on 1/Z, so judge depth in dioptres rather than metres.
        assert abs(1 / estimated - 1 / depth) < 0.1 and error < 0.5


def test_parallel_rays_clamp_far():
    depth_filter = gg.InverseDepthFilter(alpha=1.0)
    d = np.array([0.0, 0.0, 1.0])
    point, ok = gg.triangulate_rays(np.array([-0.03, 0, 0]), d, np.array([0.03, 0, 0]), d)
    assert point is None and not ok
    assert depth_filter.update(None) == gg.DEPTH_LIMITS_M[1]
    assert depth_filter.update(0.05) == gg.DEPTH_LIMITS_M[0]


def test_model_json_round_trip_and_range():
    model = fit("left", targets_at([0.4, 2.0]), noise=0.0)
    restored = gg.model_from_json(gg.model_to_json(model))
    gaze = raw_gaze("left", np.array([0.1, 0.1, 1.0]))
    for a, b in zip(gg.eye_ray(model, gaze), gg.eye_ray(restored, gaze)):
        assert np.allclose(a, b)
    behind = -np.asarray(model["basis"])[2]  # opposite to the calibrated gaze range
    try:
        gg.eye_ray(model, behind)
    except ValueError:
        return
    raise AssertionError("gaze outside the calibrated range must raise")


def test_invalid_inputs_are_rejected():
    targets = targets_at([0.4, 2.0])
    gazes = np.array([raw_gaze("left", p) for p in targets])
    # Targets along one horizontal line (two depths): the gaze spans only one axis.
    line_targets = np.array([[x, 0.0, z] for z in (0.4, 2.0) for x in np.linspace(-0.3, 0.3, 8) * z])
    line_gazes = np.array([raw_gaze("left", p) for p in line_targets])
    bad_inputs = [
        (gazes[:5], targets[:5]),  # too few samples
        (gazes, targets[:-1]),  # mismatched shapes
        (np.where(np.arange(len(gazes))[:, None] == 0, np.nan, gazes), targets),  # non-finite
        (gazes, targets * np.array([1, 1, -1])),  # behind the camera
        (line_gazes, line_targets),  # one axis only
    ]
    for gaze_input, target_input in bad_inputs:
        try:
            gg.fit_eye_model(gaze_input, target_input)
        except ValueError:
            continue
        raise AssertionError("invalid calibration input was accepted")


def test_projection_round_trip():
    point = np.array([0.2, -0.1, 1.5])
    pixel = gg.project_point(K, None, point)
    assert gg.angular_errors_degrees([gg.pixel_ray(K, None, pixel)], [point])[0] < 1e-6


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("All gaze_geometry tests passed.")
