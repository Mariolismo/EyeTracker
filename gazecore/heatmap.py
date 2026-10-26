"""Gaze heatmap on a monitor: a decaying live map plus the session total.

Each gaze point adds a Gaussian (sigma in mm, so the blob is the same physical size on any
monitor) weighted by the time since the previous point: the map counts seconds of looking,
independent of the frame rate. The live map halves every half_life_s. Buffers are at a
quarter of the screen resolution.
"""

import cv2
import numpy as np

MAX_STEP_S = 0.2  # A gap longer than this (gaze lost) adds no more than this much time


class Heatmap:
    def __init__(self, width_px, height_px, mm_per_px, sigma_mm=12.0, half_life_s=8.0, downsample=4):
        self.width_px, self.height_px = width_px, height_px
        self.downsample = downsample
        self.sigma_cells = sigma_mm / mm_per_px / downsample
        self.half_life_s = half_life_s
        shape = (int(np.ceil(height_px / downsample)), int(np.ceil(width_px / downsample)))
        self.live = np.zeros(shape, np.float32)
        self.total = np.zeros(shape, np.float32)
        self.t = None
        self.decayed_to = None
        radius = int(np.ceil(3 * self.sigma_cells))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / max(self.sigma_cells, 1e-6)) ** 2)
        self.kernel = np.outer(kernel, kernel).astype(np.float32)
        self.radius = radius

    def clear(self):
        self.live[:] = 0
        self.total[:] = 0
        self.t = self.decayed_to = None

    def decay(self, t):
        if self.decayed_to is not None and t > self.decayed_to:
            self.live *= np.float32(0.5 ** ((t - self.decayed_to) / self.half_life_s))
        self.decayed_to = t if self.decayed_to is None else max(t, self.decayed_to)

    def add(self, pixel, t):
        """pixel: screen-local (x, y), or None when there is no gaze (only time passes)."""
        dt = 0.0 if self.t is None else min(MAX_STEP_S, max(0.0, t - self.t))
        self.t = t
        self.decay(t)
        if pixel is None or dt <= 0:
            return
        cx, cy = int(round(pixel[0] / self.downsample)), int(round(pixel[1] / self.downsample))
        height, width = self.live.shape
        r = self.radius
        x0, x1, y0, y1 = max(0, cx - r), min(width, cx + r + 1), max(0, cy - r), min(height, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return  # Off-screen
        patch = self.kernel[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)] * np.float32(dt)
        self.live[y0:y1, x0:x1] += patch
        self.total[y0:y1, x0:x1] += patch

    def rgba(self, which="live", max_alpha=170):
        """Full-resolution BGRA image (premultiplied-ready colours, alpha by intensity)."""
        values = self.live if which == "live" else self.total
        peak = float(values.max())
        if peak <= 0:
            return np.zeros((self.height_px, self.width_px, 4), np.uint8)
        normalized = np.clip(values / peak, 0, 1)
        small = (normalized * 255).astype(np.uint8)
        colours = cv2.applyColorMap(small, cv2.COLORMAP_JET)
        alpha = (np.sqrt(normalized) * max_alpha).astype(np.uint8)
        alpha[normalized < 0.03] = 0
        image = np.dstack([colours, alpha])
        return cv2.resize(image, (self.width_px, self.height_px), interpolation=cv2.INTER_LINEAR)

    def save(self, png_path, csv_path=None, background=None):
        """The session total as a PNG (over background, a BGR screenshot, when given) and a CSV grid
        of seconds per cell."""
        image = self.rgba("total", max_alpha=200)
        if background is not None:
            base = cv2.resize(background, (self.width_px, self.height_px)).astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            out = (image[:, :, :3] * alpha + base * (1 - alpha)).astype(np.uint8)
        else:
            out = image
        cv2.imwrite(png_path, out)
        if csv_path is not None:
            np.savetxt(csv_path, self.total, delimiter=",", fmt="%.4f",
                       header=f"seconds per {self.downsample}x{self.downsample} px cell", comments="# ")
