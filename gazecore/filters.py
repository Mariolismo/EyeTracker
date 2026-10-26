"""Smoothing for the on-screen gaze point.

The One Euro filter (Casiez et al. 2012): a low-pass whose cutoff rises with speed, so a
fixation is steady and a saccade still lands without lag. Time-based, so it behaves the same
at 30 and 60 Hz. Units are the caller's (the screen tracker uses mm on the screen).
"""

import math

import numpy as np


def _alpha(cutoff_hz, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    def __init__(self, min_cutoff_hz=0.8, beta=0.005, derivative_cutoff_hz=1.0, reset_after_s=0.5):
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta  # Cutoff added per unit/s of speed
        self.derivative_cutoff_hz = derivative_cutoff_hz
        self.reset_after_s = reset_after_s  # A longer gap (gaze lost) starts over
        self.reset()

    def reset(self):
        self.value = None
        self.speed = None
        self.t = None

    def __call__(self, value, t):
        value = np.asarray(value, dtype=float)
        if self.value is None or t - self.t > self.reset_after_s:
            self.value, self.speed, self.t = value.copy(), np.zeros_like(value), t
            return self.value.copy()
        if t <= self.t:  # Same timestamp again: nothing new to filter
            return self.value.copy()
        dt = t - self.t
        raw_speed = (value - self.value) / dt
        self.speed = self.speed + _alpha(self.derivative_cutoff_hz, dt) * (raw_speed - self.speed)
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self.speed))
        self.value = self.value + _alpha(cutoff, dt) * (value - self.value)
        self.t = t
        return self.value.copy()
