"""Millimetres per pixel of each monitor (monitors.json), confirmed once with a card or a ruler.

The OS reports a physical size (EDID), often rounded or wrong; the marker sizes, and so every
calibration depth, depend on this value, so the wizard asks to confirm it once per monitor.
"""

from datetime import datetime

from .settings import MONITORS_FILE, read_json, write_json

CARD_WIDTH_MM = 85.6  # ISO/IEC 7810 ID-1 (credit card)
CARD_HEIGHT_MM = 53.98


def monitor_key(name, width_px, height_px, model=""):
    return f"{model or name} {width_px}x{height_px}".strip()


def os_mm_per_px(physical_width_mm, width_px):
    """The OS's guess, or None when it reports no physical size."""
    if not physical_width_mm or physical_width_mm < 50 or width_px <= 0:
        return None
    return physical_width_mm / width_px


class Monitors:
    def __init__(self, path=MONITORS_FILE):
        self.path = path
        self.entries = read_json(path, {}) or {}

    def get(self, key):
        """{mm_per_px, method, date} once confirmed, else None."""
        return self.entries.get(key)

    def confirm(self, key, mm_per_px, method):
        self.entries[key] = {"mm_per_px": float(mm_per_px), "method": method,
                             "date": datetime.now().isoformat(timespec="seconds")}
        write_json(self.path, self.entries)
        return self.entries[key]
