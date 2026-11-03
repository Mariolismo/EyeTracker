"""The app's state machine: Connecting -> Evaluate -> Tracking, or the wizard from its first
missing step. It owns the cameras, the worker and every file write.

Invalidation is explicit, replacing the tracker's hidden resets:
- a camera dropping out and coming back with the same frames keeps everything;
- a camera now delivering another frame size stops the gaze mapping, with a banner;
- choosing another camera or orientation happens only on the Cameras page, after confirming,
  and leads to the steps it invalidates;
- new lens intrinsics clear the eye origins and the gaze calibration (in memory and in
  front_camera_calibration.json; gaze_calibration.json is then refused as older than the lens).
"""

import copy
import logging
import os
import re
import time
from datetime import datetime

from PyQt5.QtCore import QObject, Qt, QTimer
from PyQt5.QtWidgets import QApplication

from gazecore import calibration_store as store
from gazecore import front_calibration as fc
from gazecore.camera import camera_reader as cr
from gazecore.camera import win_cameras
from gazecore.camera.camera_setup import (DEFAULT_FLIP, ROLE_LABELS, ROLES, RoleBinding, default_setup, load_setup,
                                          moved_port_candidates, resolve, save_setup)
from gazecore.filters import OneEuroFilter2D
from gazecore.screen_geometry import ScreenLayout, ScreenTracker

from . import review, sounds
from . import validity as v
from .calibration_session import CalibrationSession
from .eye_capture import EyeCapture
from .monitors import Monitors, monitor_key, os_mm_per_px
from .pipeline import FRONT_SIZE, Pipeline
from .settings import CAMERA_SETUP_FILE, SESSIONS_DIR, AppSettings
from .ui.camera_page import CameraPage
from .ui.common import STATUS_COLORS, assert_unscaled, confirm, run_in_pool
from .ui.main_window import MainWindow
from .ui.overlay import OverlayTray, OverlayWindow
from .ui.tracking_view import TrackingView
from .ui.wizard_pages import (ConnectingPage, EyePage, LensPage, ReviewPage, ScreenScalePage, TargetPage)
from .worker import Worker

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 30.0  # The rig's front camera takes ~12 s to open over MSMF
FRONT_DIAGONAL_FOV_DEGREES = 60.0  # The tracker's fallback until the lens is calibrated
DEFAULT_MM_PER_PX = 0.25
PENDING = (cr.OFFLINE, cr.OPENING, cr.RECONNECTING, cr.STALLED)
STATUS_KEYS = {cr.OK: "running", cr.OPENING: "opening", cr.RECONNECTING: "reconnecting", cr.STALLED: "stalled",
               cr.FROZEN: "stalled", cr.ABSENT: "failed", cr.FAILED: "failed", cr.OFFLINE: "stopped"}
WIZARD_TAIL = ("review",)


class Controller(QObject):
    def __init__(self, app, video_overrides=None, settings=None):
        super().__init__()
        self.app = app
        self.settings = settings or AppSettings()
        self.monitors = Monitors()
        self.devices = win_cameras.list_capture_devices()
        self.setup = load_setup(CAMERA_SETUP_FILE)
        self.setup_saved = bool(self.setup.bound())
        if not self.setup_saved:
            self.setup = default_setup(self.devices)
        for role, path in (video_overrides or {}).items():  # Development: play files instead (not saved)
            self.setup.roles[role] = RoleBinding(name="video", video_file=path, flip=DEFAULT_FLIP[role])
            self.setup_saved = True
        self.video_overrides = dict(video_overrides or {})

        self.front = store.load_front(image_size=FRONT_SIZE)
        self.camera = self.front.camera or fc.CameraModel.from_fov(FRONT_SIZE, FRONT_DIAGONAL_FOV_DEGREES)
        self.gaze = None  # StoredGaze restored from the files
        self.solved = None  # store.Solved the pipeline maps with, or None
        self.validity = None
        self.calibrated_geometry = {}  # {role: geometry} the current calibration was validated with
        self.session = None
        self.pending = None  # Solved, awaiting review
        self.bundle = None
        self.mode = "connecting"
        self.wizard_steps, self.wizard_index = [], 0
        self.overlay, self.tray = None, None
        self.eye_capture = None  # EyeCapture while E saves raw eye frames
        self.last_status = {}
        self.other_tracker_warning = self._other_tracker_running()

        self.manager = cr.CameraManager()
        gaze_vector = self.settings["gaze_vector_path"] or None
        self.pipeline = Pipeline(self.camera, dict(self.setup.roles), gaze_vector_path=gaze_vector)
        for tracker in self.pipeline.trackers.values():
            tracker.search_in_eye_circle = bool(self.settings["pupil_search_in_eye_circle"])
            tracker.pupil_seeds = max(1, int(self.settings["pupil_seeds"]))
        self.worker = Worker(self.pipeline, self.manager)
        self.worker.bundle_ready.connect(self._on_bundle)

        self.screen = self._target_screen()
        self.window = MainWindow(self)
        self.pages = {
            "connecting": ConnectingPage(self), "cameras": CameraPage(self), "tracking": TrackingView(self),
            "screen": ScreenScalePage(self), "intrinsics": LensPage(self), "eye:left": EyePage(self, "left"),
            "eye:right": EyePage(self, "right"), "gaze": TargetPage(self), "review": ReviewPage(self),
        }
        for page in self.pages.values():
            self.window.add_page(page)
        self.page = None
        self.timer = QTimer(self, interval=250, timeout=self._tick)

    # --- Startup ---

    def start(self):
        for role in ROLES:
            self._bind(role, self.setup.roles.get(role))
        self.worker.start()
        self.window.setGeometry(self.screen.geometry())
        self.window.showFullScreen()
        try:
            assert_unscaled(self.window)
        except RuntimeError as exc:
            self.banner(f"{exc}: on-screen marker sizes will be wrong", "error")
        self.connect_started = time.perf_counter()
        self._show("connecting")
        self.timer.start()
        if self.other_tracker_warning:
            self.banner("Another tracker seems to be running (gaze_vector.txt is being written): close it, "
                        "or the cameras may not open", "warning", 12)

    def _other_tracker_running(self):
        path = self.settings["gaze_vector_path"]
        try:
            return bool(path) and time.time() - os.path.getmtime(path) < 2.0
        except OSError:
            return False

    def _target_screen(self):
        wanted = self.settings["target_screen"]
        for screen in self.app.screens():
            if screen.name() == wanted:
                return screen
        return self.app.primaryScreen()

    def _bind(self, role, binding):
        if binding is None:
            self.manager.unbind(role)
            return
        if binding.video_file:
            self.manager.bind(role, binding.video_file)
            return
        index = resolve(binding, self.devices)
        # An absent device still gets a reader: it waits (ABSENT) and opens when plugged back in.
        self.manager.bind(role, binding.index if index is None else index, binding.device_id)

    def refresh_devices(self):
        self.devices = win_cameras.list_capture_devices()

    def live_geometry(self):
        """{role: geometry} of the cameras delivering frames now (None for the others)."""
        live = {}
        for role in ROLES:
            reader, binding = self.manager.get(role), self.setup.roles.get(role)
            if reader is None or binding is None or reader.status != cr.OK:
                live[role] = None
                continue
            geometry = binding.geometry
            geometry["native_size"] = list(reader.native_size) if reader.native_size else None
            live[role] = geometry
        return live

    def evaluate(self):
        self.mode = "evaluating"
        live = self.live_geometry()
        reasons = []
        try:
            gaze = store.load_gaze()
        except FileNotFoundError:
            gaze = None
        except (OSError, ValueError, KeyError) as exc:
            gaze = None
            reasons.append(f"The saved gaze calibration could not be read ({exc})")

        def solve():
            return None if gaze is None else store.solve(gaze.calibration, self.front.eye_origins)

        def solved(result, error):
            validity = v.evaluate(live, self.front, gaze, result is not None)
            validity.reasons = reasons + validity.reasons
            self.validity = validity
            if validity.first_step != v.CAMERAS:
                self._remember_native_sizes()
            log.info("Validity: %s", validity)
            self._moved_port_banner()
            if validity.ok:
                self.restore(gaze, result, live)
                self.show_tracking()
                self._legacy_banner()
            else:
                self.start_wizard(validity.first_step)
                self.banner(". ".join(validity.reasons), "warning", 10)

        run_in_pool(solve, solved)

    def _moved_port_banner(self):
        claimed = [b.device_id for b in self.setup.bound().values()]
        for role, binding in self.setup.bound().items():
            reader = self.manager.get(role)
            if reader is not None and reader.status == cr.ABSENT and moved_port_candidates(binding, self.devices, claimed):
                self.banner(f"The {ROLE_LABELS[role].lower()} camera is missing, but a camera of the same model is "
                            "plugged in elsewhere: did it move to another USB port? Press C to choose it.", "warning")

    def _remember_native_sizes(self):
        if self.video_overrides:
            return
        for role, binding in self.setup.bound().items():
            reader = self.manager.get(role)
            if reader is not None and reader.native_size:
                binding.native_size = list(reader.native_size)
                binding.backend = reader.backend_name
        save_setup(CAMERA_SETUP_FILE, self.setup)
        self.setup_saved = True

    def restore(self, gaze, solved, live):
        """Lock the eye models to the saved states and map with the re-solved models; writes nothing."""
        self.gaze, self.solved, self.calibrated_geometry = gaze, solved, live
        states, result = gaze.eye_states, solved.result

        def apply(pipeline):
            pipeline.apply_eye_states(states)
            pipeline.set_mapping(result)

        self.worker.call(apply)

    def _legacy_banner(self):
        verify = self.settings["last_verify"]
        verified = verify and self.gaze is not None and verify.get("saved_at") == self.gaze.saved_at
        if self.validity is not None and self.validity.legacy and not verified:
            self.banner("Calibration from the front-camera tracker, not checked here yet: press V to verify it",
                        "info", 15)

    # --- Pages ---

    def _show(self, name, steps=None):
        page = self.pages[name]
        if self.page is not None and self.page is not page:
            self.page.leave()
        self.page = page
        self.worker.paused = name == "cameras"
        self.window.show_page(page, steps, name if steps else None)
        page.enter()

    def show_tracking(self):
        self.mode = "tracking"
        self.session = None
        self._show("tracking")

    def open_cameras(self):
        self.camera_return = self.mode
        if self.mode != "wizard":
            self.mode = "cameras"
        self._show("cameras", self.wizard_steps if self.camera_return == "wizard" else None)

    # --- Wizard ---

    def start_wizard(self, first_step=v.CAMERAS):
        steps = list(v.STEPS[v.STEPS.index(first_step):])
        if self.monitor_entry() is None and v.SCREEN not in steps:
            steps.insert(0, v.SCREEN)
        self.mode = "wizard"
        self.wizard_steps, self.wizard_index = steps + list(WIZARD_TAIL), 0
        self._show_step()

    def _show_step(self):
        step = self.wizard_steps[self.wizard_index]
        if step == v.CAMERAS:
            self.camera_return = "wizard"
            self._show("cameras", self.wizard_steps)
        elif step == v.GAZE:
            self.start_session(quick=False)
        elif step == "review":
            self._show("review", self.wizard_steps)
        else:
            self._show(step, self.wizard_steps)

    def wizard_next(self):
        if self.mode != "wizard":
            self.show_tracking()
            return
        if self.wizard_index + 1 < len(self.wizard_steps) and self.wizard_steps[self.wizard_index + 1] != "review":
            self.wizard_index += 1
            self._show_step()

    def wizard_back(self):
        if self.mode in ("quick", "verify") or self.mode != "wizard":
            self.show_tracking()
            return
        if self.wizard_index == 0:
            self.show_tracking()
            return
        self.wizard_index -= 1
        self._show_step()

    # --- Cameras ---

    def preview_bind(self, role, binding):
        self._bind(role, binding)

    def cancel_camera_setup(self, original):
        previewed = self.pages["cameras"].bindings()
        for role in ROLES:
            if _source(previewed.get(role)) != _source(original.roles.get(role)):
                self._bind(role, original.roles.get(role))
        self.setup = original
        self._leave_cameras(changed=())

    def apply_camera_setup(self, bindings):
        changed = [role for role in ROLES if _identity(bindings.get(role)) != _identity(self.setup.roles.get(role))]
        if changed and (self.solved is not None or self.lens_calibrated()):
            affected = []
            if "front" in changed:
                affected.append("the front camera's lens calibration")
            affected += [f"the {role} eye model" for role in ("left", "right") if role in changed]
            affected.append("the gaze calibration")
            if not confirm(self.window, "Change cameras", "This invalidates " + ", ".join(affected)
                           + ". The steps that depend on it come next. Continue?"):
                return
        for role in ROLES:
            if _source(bindings.get(role)) != _source(self.setup.roles.get(role)):
                self._bind(role, bindings.get(role))
        self.setup.roles = {role: copy.deepcopy(bindings.get(role)) for role in ROLES}
        roles = dict(self.setup.roles)
        self.worker.call(lambda p: setattr(p, "bindings", roles))
        if not self.video_overrides:
            save_setup(CAMERA_SETUP_FILE, self.setup)
        self.setup_saved = True
        self._leave_cameras(changed)

    def _leave_cameras(self, changed):
        came_from = getattr(self, "camera_return", "tracking")
        if changed:
            self.invalidate_gaze("the cameras changed")
            if "front" in changed:
                first = v.INTRINSICS
            else:
                first = "eye:left" if "left" in changed else "eye:right"
            self.start_wizard(first)
            return
        if came_from == "wizard":
            self.mode = "wizard"
            if self.wizard_steps[self.wizard_index] == v.CAMERAS:
                self.wizard_next()
            else:
                self._show_step()
        elif came_from in ("connecting", "evaluating"):
            self.connect_started = time.perf_counter()
            self.mode = "connecting"
            self._show("connecting")
        else:
            self.show_tracking()

    # --- Screen ---

    def monitor_key(self):
        size = self.screen.geometry()
        return monitor_key(self.screen.name(), size.width(), size.height(), self.screen.model())

    def monitor_entry(self):
        return self.monitors.get(self.monitor_key())

    def mm_per_px_guess(self):
        entry = self.monitor_entry()
        if entry:
            return entry["mm_per_px"]
        return os_mm_per_px(self.screen.physicalSize().width(), self.screen.geometry().width()) or DEFAULT_MM_PER_PX

    def mm_per_px(self):
        return self.mm_per_px_guess()

    def screen_size_text(self, mm_per_px):
        size = self.screen.geometry()
        return f"screen {size.width() * mm_per_px / 10:.1f} x {size.height() * mm_per_px / 10:.1f} cm"

    def confirm_screen_scale(self, mm_per_px, method):
        self.monitors.confirm(self.monitor_key(), mm_per_px, method)
        sounds.play("done")
        self.wizard_next()

    def shift_held(self):
        return bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)

    # --- Lens ---

    def lens_calibrated(self):
        return bool(self.camera.calibrated)

    def on_intrinsics(self, camera, rms):
        live = self.live_geometry()
        self.camera = camera
        self.front = store.FrontCalibration(camera, {}, {
            "version": store.APP_META_VERSION, "front": live.get("front"), "rms_px": round(rms, 3),
            "calibrated_at": datetime.now().isoformat(timespec="seconds")})
        store.save_front(self.front)
        self.worker.call(lambda p: p.set_camera(camera))
        self.invalidate_gaze("the lens was recalibrated")
        self.banner(f"Lens calibrated (reprojection {rms:.2f} px) and saved", "info", 6)
        self.wizard_next()

    # --- Gaze calibration ---

    def invalidate_gaze(self, reason):
        if self.solved is not None:
            self.banner(f"Gaze calibration cleared: {reason}", "warning", 6)
        self.solved = None
        self.worker.call(lambda p: p.set_mapping(None))

    def prerequisites(self, verify=False):
        failures = []
        if verify:
            if self.solved is None:
                failures.append("Not calibrated: press Esc, then W")
            return failures
        if not self.lens_calibrated():
            failures.append("Calibrate the front camera's lens first (W)")
        if self.monitor_entry() is None:
            failures.append("Confirm the screen scale first")
        bundle = self.bundle
        for eye in ("left", "right"):
            result = None if bundle is None else bundle.eyes.get(eye)
            if result is None:
                failures.append(f"No frames from the {eye} eye camera")
            elif not result.locked:
                failures.append(f"Lock the {eye} eye model first")
        return failures

    def start_session(self, quick):
        if quick and not all(eye in self.front.eye_origins for eye in ("left", "right")):
            self.banner("A quick recalibration needs the eye positions of a full calibration: press W", "warning", 8)
            return

        def got_states(states):
            self.session = CalibrationSession(states, self.front.eye_origins, quick=quick, settings={
                "capture_mode": self.settings["capture_mode"], "frames_averaged": self.settings["frames_averaged"],
                "monitor": {"key": self.monitor_key(), "mm_per_px": self.mm_per_px()}})
            page = self.pages["gaze"]
            page.start(TargetPage.QUICK if quick else TargetPage.CALIBRATE)
            if quick:
                self.mode = "quick"
            self._show("gaze")

        self.worker.call(lambda p: p.locked_eye_states(), got_states)

    def start_quick(self):
        self.start_session(quick=True)

    def solve_session(self):
        session = self.session

        def solved(result, error):
            if error is not None:
                sounds.play("error")
                self.pages["gaze"]._flash(f"Solving failed: {error}", 6.0)
                return
            self.pending = result
            self.pages["review"].show_review(review.review(result.result, session.calibration), session.quick)
            if self.mode == "wizard":
                self.wizard_index = self.wizard_steps.index("review")
                self._show_step()
            else:
                self._show("review")

        run_in_pool(session.solve, solved)

    def accept_calibration(self):
        solved, session = self.pending, self.session
        live = self.live_geometry()
        cameras = {role: live.get(role) or self.setup.roles[role].geometry for role in ROLES if self.setup.roles.get(role)}
        try:
            session.accept(solved, self.front, cameras)
            self.gaze = store.load_gaze()
        except (OSError, ValueError, KeyError) as exc:
            self.banner(f"Could not save the calibration: {exc}", "error")
            return
        self.solved, self.calibrated_geometry, self.pending = solved, live, None
        self.validity = v.Validity()
        result = solved.result
        self.worker.call(lambda p: p.set_mapping(result))
        sounds.play("done")
        self.show_tracking()
        self.banner("Calibration saved. " + "; ".join(store.summary_lines(result)), "info", 8)

    def redo(self, band):
        page = self.pages["gaze"]
        quick = self.session.quick
        if band == "all":
            self.session.clear()
            band = "near"
        elif band is not None:
            self.session.remove_band(band)
        else:
            band = page.band or "near"
        page.start(TargetPage.QUICK if quick else TargetPage.CALIBRATE, band)
        if self.mode == "wizard":
            self.wizard_index = self.wizard_steps.index(v.GAZE)
        self._show("gaze", self.wizard_steps if self.mode == "wizard" else None)

    # --- Verify ---

    def start_verify(self):
        if self.solved is None:
            self.banner("Not calibrated yet: press W", "warning", 5)
            return
        self.mode = "verify"
        self.pages["gaze"].start(TargetPage.VERIFY)
        self._show("gaze")

    def estimate_for(self, gaze):
        """(GazeEstimate, baseline GazeEstimate) for one averaged look, without disturbing tracking."""
        if self.solved is None:
            return None, None
        result = self.solved.result
        estimate = fc.BinocularGazeMapper(result.models, self.camera).update(gaze)
        baseline = None if result.baselines is None else fc.BinocularGazeMapper(result.baselines, self.camera).update(gaze)
        return estimate, baseline

    def finish_verify(self, accuracy):
        summary = accuracy.summary_by_band()
        if summary and self.gaze is not None:
            self.settings.set("last_verify", {"saved_at": self.gaze.saved_at, "date": datetime.now().isoformat(
                timespec="seconds"), "summary": {band: list(values) for band, values in summary.items()}})
            text = "; ".join(f"{band} {error:.2f}° (n={n})" for band, (n, error, _) in summary.items())
            self.banner(f"Accuracy: {text}", "info", 10)
        self.show_tracking()

    # --- Overlay ---

    def start_overlay(self):
        if self.solved is None:
            self.banner("Calibrate first (W): the overlay needs the gaze calibration", "warning", 5)
            return
        if self.monitor_entry() is None:
            self.banner("Confirm the screen scale first (W)", "warning", 5)
            return
        geometry, available = self.screen.geometry(), self.screen.availableGeometry()
        area = (available.x() - geometry.x(), available.y() - geometry.y(), available.width(), available.height())
        layout = ScreenLayout.corners(geometry.width(), geometry.height(), self.mm_per_px(), area,
                                      midpoints=bool(self.settings["overlay_midpoints"]))
        camera = self.camera
        self.worker.call(lambda p: setattr(p, "screen_tracker", ScreenTracker(layout, camera, OneEuroFilter2D())))
        self.overlay = OverlayWindow(layout, self.screen, bool(self.settings["overlay_heatmap"]))
        self.tray = OverlayTray(self)
        self.tray.show()
        self.overlay.start()
        self.mode = "overlay"
        self.window.showMinimized()
        self.tray.showMessage("Gaze overlay", "Running. Use this tray icon to stop it or save the heatmap.")

    def stop_overlay(self):
        if self.overlay is None:
            return
        self.worker.call(lambda p: setattr(p, "screen_tracker", None))
        self.overlay.stop()
        self.overlay.deleteLater()
        self.tray.hide()
        self.overlay = self.tray = None
        self.window.showFullScreen()
        self.window.activateWindow()
        self.show_tracking()

    def toggle_heatmap(self):
        visible = not self.settings["overlay_heatmap"]
        self.settings.set("overlay_heatmap", visible)
        if self.overlay is not None:
            self.overlay.set_heatmap_visible(visible)
        self.banner(f"Heatmap {'on' if visible else 'off'}", "info", 2)

    # --- Eye data capture (PupilNet training frames) ---

    def toggle_eye_capture(self):
        capture = self.eye_capture
        if capture is not None:  # Closed once the worker no longer adds to it
            self.worker.call(lambda p: setattr(p, "eye_capture", None), lambda _: self._close_eye_capture(capture))
            return
        # A debug recording first: no prompt. PupilNet's training reads the subject from meta.json;
        # set capture_subject in app_settings.json when collecting training data.
        subject = re.sub(r"[^A-Za-z0-9_-]", "", self.settings["capture_subject"]) or "unknown"
        roles = {role: {"device_id": b.device_id, "name": b.name, "video_file": b.video_file, "flip": b.flip,
                        "mirror": b.mirror, "native_size": b.native_size}
                 for role, b in self.setup.roles.items() if b is not None and role in ("left", "right")}
        settings = {key: self.settings[key] for key in ("pupil_seeds", "pupil_search_in_eye_circle")}
        directory = os.path.join(SESSIONS_DIR, f"eye_capture_{time.strftime('%Y%m%d_%H%M%S')}_{subject}")
        capture = EyeCapture(directory, subject, {"cameras": roles, "settings": settings},
                             every=self.settings["capture_every"])
        self.eye_capture = capture
        self.worker.call(lambda p: setattr(p, "eye_capture", capture))
        log.info("Capturing eye frames to %s", directory)
        self.banner(f"Recording eye frames (E stops): {directory}", "info", 4)

    def _close_eye_capture(self, capture):
        if self.eye_capture is not capture:
            return
        self.eye_capture = None
        capture.close()
        log.info("Saved %d eye frames (%d dropped) to %s", capture.saved, capture.dropped, capture.directory)
        self.banner(f"Saved {capture.saved} eye frames: {capture.directory}", "info", 5)

    def save_heatmap(self):
        if self.overlay is None:
            return
        path = self.overlay.save_heatmap(SESSIONS_DIR)
        self.tray.showMessage("Heatmap saved", path)

    # --- Running ---

    def _on_bundle(self):
        bundle = self.worker.take_bundle()
        if bundle is None:
            return
        self.bundle = bundle
        if self.overlay is not None:
            self.overlay.update_gaze(bundle.screen, bundle.t)
        if self.page is not None and self.window.isVisible() and not self.window.isMinimized():
            self.page.on_bundle(bundle)

    def _tick(self):
        if self.mode == "connecting":
            readers = [self.manager.get(role) for role in self.setup.bound()]
            pending = [r for r in readers if r is not None and r.status in PENDING]
            if not pending or time.perf_counter() - self.connect_started > CONNECT_TIMEOUT_S:
                self.evaluate()
        self._watch_cameras()
        if self.page is not None:
            self.page.on_tick()

    def _watch_cameras(self):
        for role in ROLES:
            reader = self.manager.get(role)
            status = None if reader is None else reader.status
            before = self.last_status.get(role)
            self.last_status[role] = status
            if self.mode in ("connecting", "evaluating", "cameras") or before is None or status == before:
                continue
            if status == cr.OK and before != cr.FROZEN:
                self.banner(f"{ROLE_LABELS[role]} camera is back", "info", 3)
                expected = (self.calibrated_geometry.get(role) or {}).get("native_size")
                if self.solved is not None and expected and reader.native_size and list(reader.native_size) != expected:
                    self.invalidate_gaze(f"the {ROLE_LABELS[role].lower()} camera now delivers "
                                         f"{reader.native_size[0]}x{reader.native_size[1]} frames, not "
                                         f"{expected[0]}x{expected[1]}: recalibrate (W)")
            elif status != cr.OK:
                message = f" ({reader.message})" if reader.message else ""
                self.banner(f"{ROLE_LABELS[role]} camera: {status.lower()}{message}", "warning",
                            None if status in (cr.ABSENT, cr.FAILED) else 8)

    def status_html(self):
        parts = []
        for role in ROLES:
            reader = self.manager.get(role)
            status = cr.OFFLINE if reader is None else reader.status
            colour = STATUS_COLORS[STATUS_KEYS.get(status, "stopped")].name()
            fps = "" if reader is None else f" {reader.fps:.0f}"
            parts.append(f"<span style='color:{colour}'>&#9679;</span> {ROLE_LABELS[role]}{fps}")
        if self.bundle is not None:
            parts.append(f"processed {self.bundle.processing_fps:.0f} fps")
        if self.solved is None:
            parts.append("<b>not calibrated: W</b>")
        else:
            models = self.solved.result.models
            rms = " ".join(f"{eye[0].upper()} {m['rms_deg']:.2f}°" for eye, m in models.items())
            separation = self.solved.result.eye_separation_mm
            parts.append(f"RMS {rms}" + ("" if separation is None else f", {separation:.0f} mm"))
            if self.gaze is not None and self.gaze.saved_at:
                parts.append(f"calibrated {_age(self.gaze.saved_at)}")
            if self.validity is not None and self.validity.legacy:
                parts.append("<span style='color:#ffcc33'>legacy file, unverified (V)</span>")
        if self.overlay is not None:
            parts.append(f"overlay paint {self.overlay.paint_ms:.1f} ms")
        if self.eye_capture is not None:
            parts.append(f"<span style='color:#ff5555'>&#9679;</span> eye frames: "
                         f"{self.eye_capture.saved} frames (E stops)")
        parts.append("F1 help")
        return " &nbsp;|&nbsp; ".join(parts)

    def banner(self, text, level="info", seconds=None):
        log.info("Banner (%s): %s", level, text)
        self.window.banner.show_message(text, level, seconds)

    # --- Keys ---

    def handle_key(self, key, text):
        if self.page is not None and self.page.handle_key(key, text):
            return True
        if key == Qt.Key_E:  # Every page: the eyes are tracked on all of them, calibrated or not
            self.toggle_eye_capture()
            return True
        if self.mode != "tracking":
            if key == Qt.Key_Escape and self.mode == "wizard" and self.solved is not None:
                self.show_tracking()
                return True
            return False
        tracking = self.pages["tracking"]
        actions = {Qt.Key_R: self.start_quick, Qt.Key_V: self.start_verify, Qt.Key_W: self.start_wizard,
                   Qt.Key_C: self.open_cameras, Qt.Key_O: self.start_overlay, Qt.Key_H: self.toggle_heatmap,
                   Qt.Key_X: tracking.toggle_recording,
                   Qt.Key_F1: tracking.toggle_help, Qt.Key_Q: self.window.close}
        if key in actions:
            actions[key]()
            return True
        return False

    def close_requested(self):
        if not confirm(self.window, "Quit", "Quit the gaze tracker?"):
            return False
        self.shutdown()
        return True

    def shutdown(self):
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        self.pages["tracking"].stop_recording()
        self.timer.stop()
        if self.overlay is not None:
            self.overlay.stop()
        if self.tray is not None:
            self.tray.hide()
        self.worker.stop()
        if self.eye_capture is not None:  # The worker is stopped: nothing adds to it any more
            self._close_eye_capture(self.eye_capture)
        self.manager.stop_all()


def _source(binding):
    """What a reader opens for a binding (orientation does not matter to the reader)."""
    return None if binding is None else (binding.video_file, (binding.device_id or "").upper())


def _identity(binding):
    """What a calibration depends on: the source and the orientation."""
    return None if binding is None else _source(binding) + (bool(binding.flip), bool(binding.mirror))


def _age(saved_at):
    try:
        seconds = (datetime.now() - datetime.fromisoformat(saved_at)).total_seconds()
    except ValueError:
        return saved_at
    if seconds < 3600:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.0f} h ago"
    return f"{seconds / 86400:.0f} days ago"

