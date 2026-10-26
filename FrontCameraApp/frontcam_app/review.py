"""Is a solved gaze calibration good enough to keep? Plain-language lines with a traffic light."""

from dataclasses import dataclass, field

import numpy as np

GREEN, AMBER, RED = "green", "amber", "red"
GOOD_RMS_DEG, FAIR_RMS_DEG = 1.0, 1.5
SEPARATION_RANGE_MM = (50, 75)


def rms_level(rms_deg):
    if rms_deg < GOOD_RMS_DEG:
        return GREEN
    return AMBER if rms_deg < FAIR_RMS_DEG else RED


@dataclass
class Review:
    lines: list = field(default_factory=list)  # [(text, level)]

    @property
    def level(self):
        levels = [level for _, level in self.lines]
        return RED if RED in levels else AMBER if AMBER in levels else GREEN

    def add(self, text, level=GREEN):
        self.lines.append((text, level))


def review(result, calibration=None):
    """result: fc.CalibrationResult; calibration: the GazeCalibration it was solved from."""
    report = Review()
    for eye_id, model in result.models.items():
        bands = ", ".join(f"{band} {rms:.2f}°" for band, rms in model["band_rms_deg"].items())
        report.add(f"{eye_id.title()} eye: {model['rms_deg']:.2f}° RMS ({bands})", rms_level(model["rms_deg"]))
        dropped = int(np.count_nonzero(~model["inliers"]))
        if dropped:
            numbers = (np.flatnonzero(~model["inliers"]) + 1).tolist()
            report.add(f"{eye_id.title()} eye ignored {dropped} glitched sample(s): {numbers}",
                       AMBER if dropped > 2 else GREEN)
        if result.baselines is not None and eye_id in result.baselines:
            baseline = result.baselines[eye_id]["rms_deg"]
            report.add(f"{eye_id.title()} eye without the eye-offset model: {baseline:.2f}°",
                       GREEN if model["rms_deg"] <= baseline + 0.05 else AMBER)
    if not result.origins_fitted:
        report.add("One depth only: the saved eye positions were reused", GREEN)
    separation = result.eye_separation_mm
    if separation is not None:
        low, high = SEPARATION_RANGE_MM
        ok = low <= separation <= high
        report.add(f"Eye separation {separation:.0f} mm" + ("" if ok else f" (expected {low}-{high} mm: "
                   "check the screen scale and the lens calibration)"), GREEN if ok else RED)
    if {"left", "right"} <= set(result.models):
        left_x, right_x = result.models["left"]["origin"][0], result.models["right"]["origin"][0]
        if left_x >= right_x:
            report.add("The left eye came out to the right of the right eye: are the eye cameras swapped?", RED)
    if calibration is not None:
        counts = calibration.band_counts()
        report.add(f"Samples: near {counts['near']}, far {counts['far']}, mid {counts['mid']}", GREEN)
    return report
