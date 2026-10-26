"""Which camera plays which role, remembered by stable device ID (camera_setup.json).

Re-matching at startup follows GazeScreen3D's resolve_saved_role: a saved device that is
present is used at whatever index it has now; one that is missing leaves its role unbound,
because its old index now belongs to another camera. Indices are trusted only when devices
cannot be enumerated (not Windows), or for video files.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import win_cameras

SETUP_VERSION = 1
ROLES = ("left", "right", "front")
ROLE_LABELS = {"left": "Left eye", "right": "Right eye", "front": "Front"}
# The front-camera tracker's defaults: eye cameras see the eye upside down.
DEFAULT_FLIP = {"left": True, "right": True, "front": False}


@dataclass
class RoleBinding:
    device_id: str = None  # Stable PnP ID; "index:N" when devices cannot be enumerated
    name: str = ""
    index: int = None  # Capture index last time (only a hint for real device IDs)
    flip: bool = False  # cv2.flip(frame, 0): upside down
    mirror: bool = False  # cv2.flip(frame, 1): left-right
    native_size: list = None  # [w, h] the camera delivered last time
    backend: str = None
    video_file: str = None  # Play this file instead of a camera (development, demos)

    @property
    def geometry(self):
        """What the frames' geometry depends on; a calibration is only valid for the same values."""
        return {"device_id": self.device_id, "native_size": list(self.native_size or []) or None,
                "flip": bool(self.flip), "mirror": bool(self.mirror)}


@dataclass
class CameraSetup:
    roles: dict = field(default_factory=lambda: {role: None for role in ROLES})
    saved_at: str = None

    def bound(self):
        return {role: binding for role, binding in self.roles.items() if binding is not None}


def load_setup(path):
    """A missing or unreadable file gives an empty setup."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        roles = {}
        for role in ROLES:
            entry = (data.get("roles") or {}).get(role)
            if entry:
                known = {k: v for k, v in entry.items() if k in RoleBinding.__dataclass_fields__}
                roles[role] = RoleBinding(**known)
            else:
                roles[role] = None
        return CameraSetup(roles, data.get("saved_at"))
    except (OSError, ValueError, TypeError):
        return CameraSetup()


def save_setup(path, setup):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    setup.saved_at = datetime.now().isoformat(timespec="seconds")
    data = {"version": SETUP_VERSION, "saved_at": setup.saved_at,
            "roles": {role: None if b is None else asdict(b) for role, b in setup.roles.items()}}
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary, path)


def is_synthetic(device_id):
    return not device_id or device_id.startswith("index:")


def resolve(binding, devices):
    """The capture index to open for binding now, or None if its device is absent.

    devices: win_cameras.list_capture_devices() (None when enumeration is unavailable).
    """
    if binding is None:
        return None
    if binding.video_file:
        return binding.video_file
    if devices is None or is_synthetic(binding.device_id):
        return binding.index
    return win_cameras.find_index_for_device_id(devices, binding.device_id)


def vid_pid(device_id):
    match = re.search(r"VID_([0-9A-F]{4})&PID_([0-9A-F]{4})", device_id or "", re.IGNORECASE)
    return None if match is None else (match.group(1).upper(), match.group(2).upper())


def moved_port_candidates(binding, devices, claimed_ids):
    """Present devices of the same model that no role uses: the camera on another USB port?

    The instance part of the ID changes with the port for cameras without a serial number.
    """
    model = vid_pid(binding.device_id) if binding else None
    if model is None or not devices:
        return []
    claimed = {d.upper() for d in claimed_ids if d}
    return [d for d in devices if vid_pid(d["device_id"]) == model and d["device_id"].upper() not in claimed]


def default_setup(devices):
    """First run: the front-camera tracker's defaults (left 0, right 1, front 2)."""
    setup = CameraSetup()
    for role, device in zip(ROLES, devices or []):
        setup.roles[role] = RoleBinding(device["device_id"], device["name"], device["index"], DEFAULT_FLIP[role])
    return setup
