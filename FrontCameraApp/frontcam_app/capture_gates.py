"""When may a calibration sample be taken? Pure checks over the pipeline's recent history.

Both capture modes use the same gates: manual capture refuses with the first failure's text,
dwell capture fills its ring only while everything passes. A sample is the average of the
window (frames_averaged entries; 1 restores the tracker's single-frame capture).
"""

from dataclasses import dataclass, field

import numpy as np

from gazecore import gaze_geometry as gg

MIN_CONFIDENCE = 0.85
CONFIDENT_SHARE = 0.9  # Of the window
MARKER_SHARE = 0.9
MAX_GAZE_STD_DEG = 0.75
MAX_MARKER_STD_PX = 3.0
MAX_FRAME_SKEW_S = 0.06  # Eye and front frames of one step
EDGE_MARGIN_PX = 8
MIN_SAMPLE_SPACING_PX = 80
GRID = 3  # Coverage grid over the front image
MIN_PER_BAND, TARGET_PER_BAND = gg.MIN_SAMPLES, 9
EYE_IDS = ("left", "right")


@dataclass
class Sample:
    gaze: dict  # {eye: unit raw gaze}, averaged over the window
    point: np.ndarray  # Marker center (front camera frame, m)
    pixel: np.ndarray
    t: float
    frames: int
    gaze_std_deg: dict
    marker_std_px: float

    @property
    def depth(self):
        return float(self.point[2])

    def meta(self):
        return {"t": round(self.t, 3), "frames": self.frames, "depth_m": round(self.depth, 3),
                "gaze_std_deg": {eye: round(v, 3) for eye, v in self.gaze_std_deg.items()},
                "marker_std_px": round(self.marker_std_px, 2)}


@dataclass
class GateResult:
    failures: list = field(default_factory=list)  # Plain language, most fundamental first
    sample: Sample = None
    window_fill: float = 0.0  # 0..1: how much of the window is collected (for "hold still")

    @property
    def ok(self):
        return not self.failures and self.sample is not None


@dataclass
class GateContext:
    prerequisites: list = field(default_factory=list)  # Failures the caller found (lens, scale, eyes)
    band: str = None  # "near" / "far", or None for any depth (verification)
    marker_generation: int = 0
    earlier_pixels: list = field(default_factory=list)  # Samples already taken in this band
    image_size: tuple = (640, 480)
    window: int = 30


def cell_of(pixel, image_size=(640, 480)):
    """(row, column) of the 3x3 coverage grid."""
    width, height = image_size
    column = int(np.clip(pixel[0] / width * GRID, 0, GRID - 1))
    row = int(np.clip(pixel[1] / height * GRID, 0, GRID - 1))
    return row, column


def band_ready(pixels, image_size=(640, 480)):
    """Enough samples in a band, spread over at least 2 rows and 2 columns."""
    cells = [cell_of(p, image_size) for p in pixels]
    return (len(pixels) >= MIN_PER_BAND and len({r for r, _ in cells}) >= 2
            and len({c for _, c in cells}) >= 2)


def _angular_std_deg(directions):
    mean = gg.unit(np.mean(directions, axis=0))
    errors = gg.angular_errors_degrees(directions, np.repeat(mean[None], len(directions), axis=0))
    return mean, float(np.sqrt(np.mean(errors ** 2)))


def _clear_of_edges(corners, image_size):
    if corners is None:
        return False
    corners = np.asarray(corners).reshape(-1, 2)
    width, height = image_size
    return bool(np.all(corners >= EDGE_MARGIN_PX) and np.all(corners[:, 0] <= width - 1 - EDGE_MARGIN_PX)
                and np.all(corners[:, 1] <= height - 1 - EDGE_MARGIN_PX))


def evaluate(history, context, camera=None):
    """history: the pipeline's HistoryEntry sequence (oldest first)."""
    result = GateResult(list(context.prerequisites))
    if result.failures:
        return result
    window = list(history)[-max(1, context.window):]
    result.window_fill = min(1.0, len(window) / max(1, context.window))
    if len(window) < context.window:
        result.failures.append("Hold still: collecting frames")
        return result

    with_marker = [e for e in window if e.marker_point is not None and e.marker_generation == context.marker_generation]
    if len(with_marker) < MARKER_SHARE * len(window):
        result.failures.append("The marker is not visible to the front camera")
        return result
    latest = with_marker[-1]
    if not _clear_of_edges(latest.marker_corners, context.image_size):
        result.failures.append("The marker is too close to the edge of the front camera's view")
        return result
    points = np.array([e.marker_point for e in with_marker])
    depth = float(np.median(points[:, 2]))
    if context.band is not None and gg.depth_band(depth) != context.band:
        if context.band == "near":
            result.failures.append(f"Move closer: {depth:.2f} m, need under {gg.NEAR_BAND_MAX_M} m")
        else:
            result.failures.append(f"Move back: {depth:.2f} m, need over {gg.FAR_BAND_MIN_M} m")
        return result

    for eye in EYE_IDS:
        confident = sum(1 for e in window if e.confidence.get(eye, 0.0) >= MIN_CONFIDENCE and e.gaze.get(eye) is not None)
        if confident < CONFIDENT_SHARE * len(window):
            result.failures.append(f"The {eye} pupil is not tracked steadily (blinking? lighting?)")
            return result

    skewed = 0
    for e in with_marker:
        for eye in EYE_IDS:
            eye_t = e.eye_t.get(eye)
            if e.front_t is None or eye_t is None or abs(e.front_t - eye_t) > MAX_FRAME_SKEW_S:
                skewed += 1
                break
    if skewed > (1 - MARKER_SHARE) * len(with_marker):
        result.failures.append("The cameras' frames are out of step (a camera is lagging)")
        return result

    usable = [e for e in with_marker if all(e.gaze.get(eye) is not None for eye in EYE_IDS)]
    if not usable:
        result.failures.append("No gaze from both eyes")
        return result
    gaze, gaze_std = {}, {}
    for eye in EYE_IDS:
        gaze[eye], gaze_std[eye] = _angular_std_deg(np.array([e.gaze[eye] for e in usable]))
    worst = max(gaze_std, key=gaze_std.get)
    if gaze_std[worst] > MAX_GAZE_STD_DEG:
        result.failures.append(f"Keep your eyes on the marker's center ({worst} eye wanders {gaze_std[worst]:.1f}°)")
        return result
    pixels = np.array([e.marker_pixel for e in with_marker])
    mean_pixel = pixels.mean(axis=0)
    marker_std = float(np.sqrt(np.mean(np.sum((pixels - mean_pixel) ** 2, axis=1))))
    if marker_std > MAX_MARKER_STD_PX:
        result.failures.append("Hold your head still")
        return result

    point = points.mean(axis=0)
    pixel = mean_pixel if camera is None else np.asarray(camera.project(point), dtype=float)
    for earlier in context.earlier_pixels:
        if np.linalg.norm(np.asarray(earlier) - pixel) < MIN_SAMPLE_SPACING_PX:
            result.failures.append("Already have a sample here: put the marker somewhere new in the view")
            return result
    result.sample = Sample(gaze, point, pixel, window[-1].t, len(usable), gaze_std, marker_std)
    return result


def plausibility(previous, sample, min_marker_move_deg=4.0, ratio_range=(0.4, 2.5)):
    """Did the gaze move like the marker did since the previous sample? Text warning or None.

    Raw gaze is in each eye camera's frame, the marker in the front camera's: the directions
    differ, but the angles moved should be similar.
    """
    if previous is None:
        return None
    marker_move = float(gg.angular_errors_degrees([previous.point], [sample.point])[0])
    if marker_move < min_marker_move_deg:
        return None
    for eye in EYE_IDS:
        gaze_move = float(gg.angular_errors_degrees([previous.gaze[eye]], [sample.gaze[eye]])[0])
        ratio = gaze_move / marker_move
        if not ratio_range[0] <= ratio <= ratio_range[1]:
            return (f"The {eye} eye moved {gaze_move:.0f}° while the marker moved {marker_move:.0f}°: "
                    "were you looking at the marker? Backspace removes this sample.")
    return None
