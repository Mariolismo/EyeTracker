"""Can the app go straight to tracking, or which wizard step comes first?

A calibration is only valid for the frames it was made from: the same cameras, with the same
frame size and orientation (raw gaze depends on the eye frame's pixels, the marker positions
on the front camera's lens model). Files written by the front-camera tracker carry none of
this ("legacy"): they are accepted if the setup still matches the tracker's defaults, with a
banner asking to verify.
"""

from dataclasses import dataclass, field

from gazecore.calibration_store import intrinsics_fingerprint, samples_fingerprint
from gazecore.camera.camera_setup import DEFAULT_FLIP, ROLES

EYE_IDS = ("left", "right")

# Wizard steps, in order
CAMERAS, SCREEN, INTRINSICS, EYE_LEFT, EYE_RIGHT, GAZE = (
    "cameras", "screen", "intrinsics", "eye:left", "eye:right", "gaze")
STEPS = (CAMERAS, SCREEN, INTRINSICS, EYE_LEFT, EYE_RIGHT, GAZE)


@dataclass
class Validity:
    first_step: str = None  # None: everything is valid, go to tracking
    legacy: bool = False  # Valid only by assumption (files without app metadata)
    reasons: list = field(default_factory=list)  # Plain-language, for the banner

    @property
    def ok(self):
        return self.first_step is None


def geometry_matches(saved, live):
    """saved/live: RoleBinding.geometry dicts. A saved native size of None matches any."""
    if saved is None or live is None:
        return False
    if (saved.get("device_id") or "").upper() != (live.get("device_id") or "").upper():
        return False
    if saved.get("native_size") and live.get("native_size") and list(saved["native_size"]) != list(live["native_size"]):
        return False
    return bool(saved.get("flip")) == bool(live.get("flip")) and bool(saved.get("mirror")) == bool(live.get("mirror"))


def legacy_orientation_ok(role, live):
    return bool(live.get("flip")) == DEFAULT_FLIP[role] and not live.get("mirror")


def evaluate(live, front, gaze, gaze_solves):
    """live: {role: geometry dict, or None if that camera delivers no frames}.
    front: calibration_store.FrontCalibration; gaze: StoredGaze or None; gaze_solves: bool.
    """
    validity = Validity()
    missing = [role for role in ROLES if live.get(role) is None]
    if missing:
        validity.first_step = CAMERAS
        validity.reasons.append("No frames from the " + ", ".join(missing) + " camera" + "s" * (len(missing) > 1))
        return validity

    if front.camera is None:
        validity.first_step = INTRINSICS
        validity.reasons.append("The front camera's lens is not calibrated yet")
        return validity
    front_meta = (front.app_meta or {}).get("front")
    if front_meta is None:
        if not legacy_orientation_ok("front", live["front"]):
            validity.first_step = INTRINSICS
            validity.reasons.append("The front camera's orientation changed since its lens calibration")
            return validity
        validity.legacy = True
    elif not geometry_matches(front_meta, live["front"]):
        validity.first_step = INTRINSICS
        validity.reasons.append("The front camera changed since its lens calibration")
        return validity

    if gaze is None:
        validity.first_step = EYE_LEFT
        validity.reasons.append("No gaze calibration yet")
        return validity
    if tuple(gaze.calibration.eye_ids) != EYE_IDS or set(gaze.eye_states) != set(EYE_IDS):
        validity.first_step = EYE_LEFT
        validity.reasons.append("The saved gaze calibration is not for both eyes")
        return validity
    meta = gaze.app_meta
    if meta is None:
        changed = [eye for eye in EYE_IDS if not legacy_orientation_ok(eye, live[eye])]
        if changed:
            validity.first_step = "eye:" + changed[0]
            validity.reasons.append(f"The {changed[0]} eye camera's orientation changed since calibrating")
            return validity
        validity.legacy = True
    else:
        for eye in EYE_IDS:
            if not geometry_matches(meta.get("cameras", {}).get(eye), live[eye]):
                validity.first_step = "eye:" + eye
                validity.reasons.append(f"The {eye} eye camera changed since calibrating")
                return validity
        if meta.get("intrinsics_fingerprint") != intrinsics_fingerprint(front.camera):
            validity.first_step = GAZE
            validity.reasons.append("The front camera's lens was recalibrated since the gaze calibration")
            return validity
        if meta.get("fingerprint") != samples_fingerprint(gaze.calibration, gaze.eye_states):
            validity.legacy = True
            validity.reasons.append("The calibration file was edited outside the app")
    if not gaze_solves:
        validity.first_step = GAZE
        validity.reasons.append("The saved gaze calibration no longer solves")
        return validity
    if validity.legacy:
        validity.reasons.append("Calibration from the front-camera tracker: press V to check its accuracy")
    return validity
