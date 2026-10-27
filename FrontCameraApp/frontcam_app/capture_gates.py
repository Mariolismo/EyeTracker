"""When may a calibration sample be taken? Pure checks over the pipeline's recent history.

Both capture modes use the same gates: manual capture refuses with the first failure's text,
dwell capture fills its ring only while everything passes. A sample is the average of the
window (frames_averaged entries; 1 restores the tracker's single-frame capture).

Noisy frames are read and discarded rather than failing the whole window: a frame is kept when
both pupils are confident, then frames whose gaze strays from the window's median are dropped.
The window passes when enough frames survive; the sample averages only those.
"""

from dataclasses import dataclass, field

import numpy as np

from gazecore import gaze_geometry as gg

MIN_CONFIDENCE = 0.75  # Pupil quality (gazecore pupil_quality): lash fits stay under about 0.65
MIN_KEPT_SHARE = 0.5  # Of the window: frames with both pupils confident (blinks are dropped)
OUTLIER_DEG = 1.5  # Frames whose gaze is this far from the window's median direction are dropped
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
    dropped: int = 0  # Window frames discarded (low confidence, no gaze, outliers)

    @property
    def depth(self):
        return float(self.point[2])

    def meta(self):
        return {"t": round(self.t, 3), "frames": self.frames, "dropped": self.dropped, "depth_m": round(self.depth, 3),
                "gaze_std_deg": {eye: round(v, 3) for eye, v in self.gaze_std_deg.items()},
                "marker_std_px": round(self.marker_std_px, 2)}


@dataclass
class GateResult:
    failures: list = field(default_factory=list)  # Plain language, most fundamental first
    sample: Sample = None
    window_fill: float = 0.0  # 0..1: how much of the window is collected (for "hold still")
    stats: dict = field(default_factory=dict)  # Per-window numbers for the debug log

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


def _drop_gaze_outliers(entries):
    """Keep the entries whose gaze is within OUTLIER_DEG of the median direction, for both eyes."""
    keep = np.ones(len(entries), dtype=bool)
    for eye in EYE_IDS:
        directions = np.array([e.gaze[eye] for e in entries])
        median = gg.unit(np.median(directions, axis=0))
        errors = gg.angular_errors_degrees(directions, np.repeat(median[None], len(directions), axis=0))
        keep &= np.asarray(errors) <= OUTLIER_DEG
    return [e for e, k in zip(entries, keep) if k]


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

    need = max(1, int(np.ceil(MIN_KEPT_SHARE * len(window))))
    confident = {eye: [e for e in with_marker if e.confidence.get(eye, 0.0) >= MIN_CONFIDENCE
                       and e.gaze.get(eye) is not None] for eye in EYE_IDS}
    kept = [e for e in with_marker if all(e in confident[eye] for eye in EYE_IDS)]
    result.stats = {"window": len(window), "marker": len(with_marker), "kept": len(kept),
                    **{f"{eye}_confident": len(confident[eye]) for eye in EYE_IDS},
                    **{f"{eye}_median_conf": round(float(np.median([e.confidence.get(eye, 0.0) for e in window])), 2)
                       for eye in EYE_IDS}}
    if len(kept) < need:
        worst = min(EYE_IDS, key=lambda eye: len(confident[eye]))
        result.failures.append(
            f"The {worst} pupil is not tracked steadily: {len(confident[worst])}/{len(window)} frames with pupil quality over "
            f"{MIN_CONFIDENCE:.2f} (median {result.stats[f'{worst}_median_conf']:.2f}), need {need} with both eyes")
        return result

    kept = [e for e in kept if e.front_t is not None and all(
        e.eye_t.get(eye) is not None and abs(e.front_t - e.eye_t[eye]) <= MAX_FRAME_SKEW_S for eye in EYE_IDS)]
    if len(kept) < need:
        result.failures.append("The cameras' frames are out of step (a camera is lagging)")
        return result

    kept = _drop_gaze_outliers(kept)
    result.stats["after_outliers"] = len(kept)
    if len(kept) < need:
        result.failures.append(f"Keep your eyes on the marker's center (gaze jumps: {len(kept)}/{len(window)} frames steady)")
        return result
    gaze, gaze_std = {}, {}
    for eye in EYE_IDS:
        gaze[eye], gaze_std[eye] = _angular_std_deg(np.array([e.gaze[eye] for e in kept]))
    result.stats["gaze_std_deg"] = {eye: round(v, 3) for eye, v in gaze_std.items()}
    worst = max(gaze_std, key=gaze_std.get)
    if gaze_std[worst] > MAX_GAZE_STD_DEG:
        result.failures.append(f"Keep your eyes on the marker's center ({worst} eye wanders {gaze_std[worst]:.1f}°)")
        return result
    pixels = np.array([e.marker_pixel for e in kept])
    mean_pixel = pixels.mean(axis=0)
    marker_std = float(np.sqrt(np.mean(np.sum((pixels - mean_pixel) ** 2, axis=1))))
    if marker_std > MAX_MARKER_STD_PX:
        result.failures.append("Hold your head still")
        return result

    point = np.mean([e.marker_point for e in kept], axis=0)
    pixel = mean_pixel if camera is None else np.asarray(camera.project(point), dtype=float)
    for earlier in context.earlier_pixels:
        if np.linalg.norm(np.asarray(earlier) - pixel) < MIN_SAMPLE_SPACING_PX:
            result.failures.append("Already have a sample here: put the marker somewhere new in the view")
            return result
    result.sample = Sample(gaze, point, pixel, window[-1].t, len(kept), gaze_std, marker_std,
                           dropped=len(window) - len(kept))
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
