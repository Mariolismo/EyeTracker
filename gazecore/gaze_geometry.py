"""Parallax-aware gaze geometry for the front-camera tracker.

All 3D quantities are in the front camera frame (OpenCV convention: x right, y down,
z forward, metres). Each eye is modelled as a ray:

    u = tangent coordinates of the raw gaze in a basis around the mean gaze
    direction = normalize(R @ [A @ u, 1]),  origin = o

R (rotation) absorbs the eye-camera/front-camera alignment and kappa, A (symmetric
2x2) absorbs gain errors of the eye model, and o is the eye's rotation center.
A plain direction-to-pixel map ignores o, which is why it is only exact at its
calibration depth. o is observable only when calibration targets span several depths.
A full projective map is deliberately avoided: its perspective terms trade off
against o along the viewing axis and make the fit degenerate.
"""

import math

import numpy as np

NEAR_BAND_MAX_M = 0.6
FAR_BAND_MIN_M = 1.5  # NEAR and FAR samples always span MIN_INVERSE_DEPTH_SPAN
MIN_INVERSE_DEPTH_SPAN = 1.0  # dioptres; 0.4 m and 2 m give 2.0
MIN_SAMPLES = 6
DEPTH_LIMITS_M = (0.2, 5.0)
OUTLIER_MIN_DEG = 5.0  # a sample is dropped only if its error exceeds this...
OUTLIER_MEDIAN_FACTOR = 4.0  # ...and this multiple of the median error
MAX_EYE_OFFSET_M = 0.2  # eye rotation centers lie within this distance of the front camera


def unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def angular_errors_degrees(a, b):
    """Angle between matching rows of a and b; atan2 stays accurate near zero."""
    a, b = unit(a), unit(b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def depth_band(depth):
    if depth < NEAR_BAND_MAX_M:
        return "near"
    if depth > FAR_BAND_MIN_M:
        return "far"
    return "mid"


def _rotation_from_vector(w):
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def _tangent_basis(gaze_dirs):
    """Rows: right, up, forward, with forward along the mean gaze."""
    forward = unit(np.mean(gaze_dirs, axis=0))
    helper = np.eye(3)[np.argmin(np.abs(forward))]
    right = unit(np.cross(helper, forward))
    up = np.cross(forward, right)
    return np.array([right, up, forward])


def _tangent_coords(basis, gaze_dirs):
    local = np.atleast_2d(gaze_dirs) @ basis.T
    if np.any(local[:, 2] <= 1e-3):
        raise ValueError("Gaze points too far from the calibrated range.")
    return local[:, :2] / local[:, 2:3]


def _kabsch(source, target):
    """Rotation R minimizing sum |target - R source|^2 for unit vectors."""
    u, _, vt = np.linalg.svd(source.T @ target)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, sign]) @ u.T


def _unpack(params, base_rotation, fixed_origin):
    rotation = base_rotation @ _rotation_from_vector(params[0:3])
    gain = np.array([[params[3], params[4]], [params[4], params[5]]])
    origin = params[6:9] if fixed_origin is None else fixed_origin
    return rotation, gain, origin


def _directions(rotation, gain, tangent):
    local = np.column_stack([tangent @ gain.T, np.ones(len(tangent))])
    return unit(local @ rotation.T)


def _residuals(params, base_rotation, fixed_origin, tangent, targets):
    rotation, gain, origin = _unpack(params, base_rotation, fixed_origin)
    # Cross product length is the sine of the angular error.
    return np.cross(unit(targets - origin), _directions(rotation, gain, tangent)).ravel()


def _levenberg_marquardt(params, base_rotation, fixed_origin, tangent, targets, iterations=100):
    args = (base_rotation, fixed_origin, tangent, targets)
    damping = 1e-3
    residual = _residuals(params, *args)
    cost = residual @ residual
    for _ in range(iterations):
        jacobian = np.empty((residual.size, params.size))
        for index in range(params.size):
            step = np.zeros(params.size)
            step[index] = 1e-7
            jacobian[:, index] = (_residuals(params + step, *args) - residual) / 1e-7
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        while True:
            delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            candidate = params + delta
            candidate_residual = _residuals(candidate, *args)
            candidate_cost = candidate_residual @ candidate_residual
            if candidate_cost < cost:
                params, residual, cost = candidate, candidate_residual, candidate_cost
                damping = max(damping / 10.0, 1e-12)
                break
            damping *= 10.0
            if damping > 1e8:
                return params, normal
        if np.linalg.norm(delta) < 1e-10:
            break
    return params, normal


def fit_eye_model(gaze_dirs, targets, fixed_origin=None):
    """Fit one eye's ray model from raw gaze directions and 3D target points.

    Returns a model dict (basis, rotation, gain, origin) plus per-sample errors, the
    inlier mask (outliers are dropped from the fit) and inlier RMS overall/per band in degrees. Raises ValueError when the samples cannot
    determine the model. With fixed_origin the eye position is not fitted, so a single
    depth suffices: pass a previously fitted origin for a quick recalibration, or
    zeros for the classic no-offset model (for comparison).
    """
    fit_origin = fixed_origin is None
    if not fit_origin:
        fixed_origin = np.asarray(fixed_origin, dtype=float)
    gaze_dirs = np.asarray(gaze_dirs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if gaze_dirs.ndim != 2 or gaze_dirs.shape[1] != 3 or targets.shape != gaze_dirs.shape:
        raise ValueError("Gaze directions and targets must be matching (N, 3) arrays.")
    if len(gaze_dirs) < MIN_SAMPLES:
        raise ValueError(f"At least {MIN_SAMPLES} samples are required.")
    if not (np.isfinite(gaze_dirs).all() and np.isfinite(targets).all()):
        raise ValueError("Samples contain non-finite values.")
    if np.any(targets[:, 2] <= 0.05):
        raise ValueError("Targets must lie in front of the front camera.")
    gaze_dirs = unit(gaze_dirs)
    if fit_origin and np.ptp(1.0 / targets[:, 2]) < MIN_INVERSE_DEPTH_SPAN:
        raise ValueError(
            "Targets do not span enough depth to locate the eye; collect both NEAR and FAR samples."
        )

    # Glitched samples (e.g. a pupil-tracking flicker) can pull a free origin to a far-away
    # false minimum, where all rays are parallel and every residual vanishes. So screen them
    # first with the origin held fixed, then fit the origin on the clean samples.
    inliers = np.ones(len(targets), dtype=bool)
    screening_origin = np.zeros(3) if fit_origin else fixed_origin
    _, _, inliers = _reject_outliers(gaze_dirs, targets, screening_origin, inliers, fit_origin)
    model, normal, inliers = _reject_outliers(gaze_dirs, targets, fixed_origin, inliers, fit_origin)
    errors = _sample_errors(model, gaze_dirs, targets)

    if np.linalg.cond(normal) > 1e12:
        raise ValueError("Calibration is degenerate; spread targets across the image and depths.")
    if np.linalg.norm(model["origin"]) > MAX_EYE_OFFSET_M:
        raise ValueError("Fitted eye position is implausible; repeat calibration looking at the marker center.")
    gain = model["gain"]
    if np.linalg.det(gain) <= 0 or np.linalg.cond(gain) > 10:
        eigenvalues = np.round(np.linalg.eigvalsh(gain), 2)
        raise ValueError(f"Fitted gaze gain {eigenvalues} is implausible; repeat calibration.")

    bands = {}
    for band in ("near", "mid", "far"):
        selected = errors[inliers & np.array([depth_band(z) == band for z in targets[:, 2]])]
        if selected.size:
            bands[band] = float(np.sqrt(np.mean(np.square(selected))))
    model.update(
        errors_deg=errors,
        inliers=inliers,
        rms_deg=float(np.sqrt(np.mean(errors[inliers] ** 2))),
        band_rms_deg=bands,
    )
    return model


def _reject_outliers(gaze_dirs, targets, fixed_origin, inliers, keep_depth_span):
    """Refit, dropping the worst sample while it is an outlier and enough samples
    (and, if keep_depth_span, enough depth span) remain. Returns (model, normal, inliers)."""
    while True:
        model, normal = _fit_once(gaze_dirs[inliers], targets[inliers], fixed_origin)
        errors = _sample_errors(model, gaze_dirs, targets)
        worst = np.argmax(np.where(inliers, errors, -1.0))
        threshold = max(OUTLIER_MIN_DEG, OUTLIER_MEDIAN_FACTOR * np.median(errors[inliers]))
        candidate = inliers.copy()
        candidate[worst] = False
        if (errors[worst] <= threshold or candidate.sum() < MIN_SAMPLES
                or (keep_depth_span and np.ptp(1.0 / targets[candidate, 2]) < MIN_INVERSE_DEPTH_SPAN)):
            return model, normal, inliers
        inliers = candidate


def _fit_once(gaze_dirs, targets, fixed_origin):
    """One least-squares fit on all given samples; returns (model, normal matrix)."""
    basis = _tangent_basis(gaze_dirs)
    tangent = _tangent_coords(basis, gaze_dirs)
    # A mirrored eye camera gives left-handed gaze; R and a positive gain cannot undo that,
    # so mirror the tangent x axis (stored in the basis, so eye_ray applies it too).
    local = np.column_stack([tangent, np.ones(len(tangent))])
    if np.linalg.det(np.linalg.lstsq(local, unit(targets), rcond=None)[0]) < 0:
        basis = basis * np.array([[-1.0], [1.0], [1.0]])
        tangent = _tangent_coords(basis, gaze_dirs)
    if np.linalg.svd(tangent - tangent.mean(axis=0), compute_uv=False)[-1] < 1e-3:
        raise ValueError("Gaze samples do not span two axes; spread targets across the image.")
    local = unit(np.column_stack([tangent, np.ones(len(tangent))]))
    base_rotation = _kabsch(local, unit(targets))
    params = np.array([0, 0, 0, 1, 0, 1] + ([0, 0, 0] if fixed_origin is None else []), dtype=float)
    params, normal = _levenberg_marquardt(params, base_rotation, fixed_origin, tangent, targets)
    rotation, gain, origin = _unpack(params, base_rotation, fixed_origin)
    return {"basis": basis, "rotation": rotation, "gain": gain, "origin": origin.copy()}, normal


def _sample_errors(model, gaze_dirs, targets):
    """Angular error per sample; 180 deg for gaze outside the model's range."""
    local = gaze_dirs @ model["basis"].T
    in_range = local[:, 2] > 1e-3
    errors = np.full(len(gaze_dirs), 180.0)
    tangent = local[in_range, :2] / local[in_range, 2:3]
    errors[in_range] = angular_errors_degrees(
        _directions(model["rotation"], model["gain"], tangent), targets[in_range] - model["origin"]
    )
    return errors


def eye_ray(model, gaze_dir):
    """Raises ValueError if the gaze is outside the model's range."""
    tangent = _tangent_coords(np.asarray(model["basis"]), np.asarray(gaze_dir, dtype=float))
    direction = _directions(np.asarray(model["rotation"]), np.asarray(model["gain"]), tangent)[0]
    return np.asarray(model["origin"], dtype=float), direction


def model_to_json(model):
    return {key: np.asarray(model[key]).tolist() for key in ("basis", "rotation", "gain", "origin")}


def model_from_json(data):
    return {key: np.asarray(data[key], dtype=float) for key in ("basis", "rotation", "gain", "origin")}


def triangulate_rays(o1, d1, o2, d2):
    """Midpoint of the closest points of two rays; ok is False if they do not converge ahead."""
    w0 = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    denominator = a * c - b * b
    if denominator < 1e-12:
        return None, False
    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator
    if s <= 0 or t <= 0:
        return None, False
    return 0.5 * ((o1 + s * d1) + (o2 + t * d2)), True


def point_at_depth(origin, direction, depth):
    if direction[2] <= 1e-6:
        return None
    return origin + direction * ((depth - origin[2]) / direction[2])


class InverseDepthFilter:
    """Exponential smoothing in 1/Z, where vergence noise and parallax error are uniform."""

    def __init__(self, alpha=0.2, default_depth=DEPTH_LIMITS_M[1]):
        self.alpha = alpha
        self.default_depth = default_depth
        self.inverse_depth = None

    def reset(self):
        self.inverse_depth = None

    @property
    def depth(self):
        return self.default_depth if self.inverse_depth is None else 1.0 / self.inverse_depth

    def update(self, depth):
        """depth=None means the rays diverged or were parallel: treat as far."""
        near, far = DEPTH_LIMITS_M
        depth = far if depth is None else float(np.clip(depth, near, far))
        if self.inverse_depth is None:
            self.inverse_depth = 1.0 / depth
        else:
            self.inverse_depth += self.alpha * (1.0 / depth - self.inverse_depth)
        return self.depth


def binocular_gaze_point(models, gaze_dirs, depth_filter):
    """Combine available eye rays into one 3D gaze point.

    models, gaze_dirs: dicts keyed by eye id; missing or None gaze entries are skipped.
    With two eyes the vergence depth updates depth_filter; with one eye the last
    smoothed depth is reused. Returns (point, depth) or (None, None).
    """
    rays = []
    for eye, model in models.items():
        if model is None or gaze_dirs.get(eye) is None:
            continue
        try:
            rays.append(eye_ray(model, gaze_dirs[eye]))
        except ValueError:
            continue
    if not rays:
        return None, None
    if len(rays) == 2:
        point, ok = triangulate_rays(rays[0][0], rays[0][1], rays[1][0], rays[1][1])
        depth_filter.update(point[2] if ok else None)
    depth = depth_filter.depth
    points = [point_at_depth(origin, direction, depth) for origin, direction in rays]
    points = [point for point in points if point is not None]
    if not points:
        return None, None
    return np.mean(points, axis=0), depth


def default_intrinsics(width, height, diagonal_fov_degrees):
    """Pinhole K with square pixels and centered principal point."""
    focal = math.hypot(width, height) / (2.0 * math.tan(math.radians(diagonal_fov_degrees) / 2.0))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def project_point(K, dist, point):
    point = np.asarray(point, dtype=float)
    if point[2] <= 1e-6:
        return None
    if dist is not None and np.any(np.asarray(dist) != 0):
        import cv2
        pixels, _ = cv2.projectPoints(point.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        return pixels.reshape(2)
    homogeneous = K @ (point / point[2])
    return homogeneous[:2]


def pixel_ray(K, dist, pixel):
    if dist is not None and np.any(np.asarray(dist) != 0):
        import cv2
        normalized = cv2.undistortPoints(np.array([[pixel]], dtype=float), K, dist).reshape(2)
    else:
        normalized = (np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0]))[:2]
    ray = np.array([normalized[0], normalized[1], 1.0])
    return ray / np.linalg.norm(ray)
