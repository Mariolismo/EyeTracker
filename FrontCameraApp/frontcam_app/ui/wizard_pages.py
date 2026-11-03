"""The wizard's pages, in order: screen scale, lens, each eye, gaze near/far, review.

Plus the connecting page shown at startup, and the target page's other two uses from the
tracking view: quick recalibration (R) and verification (V). Each page talks to the pipeline
only through controller.worker.call, and to the files only through the controller.
"""

import collections
import logging
import time

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
                             QWidget)

from gazecore import front_calibration as fc
from gazecore import gaze_geometry as gg
from gazecore.camera.camera_setup import ROLE_LABELS, ROLES
from gazecore.eye_tracker import CENTER_OVERRIDE_FRAMES_NEEDED, MIN_MODEL_CENTERS, CenterOverrideDone

from .. import capture_gates, review, sounds
from ..monitors import CARD_HEIGHT_MM, CARD_WIDTH_MM
from ..pipeline import FRONT_SIZE
from .common import FrameView, Page, run_in_pool
from .target_canvas import TargetCanvas, board_qimage

CAPTURE_KEYS = (Qt.Key_Space, Qt.Key_C, Qt.Key_PageDown)  # PageDown: a presenter clicker's "next"
UNDO_KEYS = (Qt.Key_Backspace, Qt.Key_PageUp)  # PageUp: the clicker's "back"
NEAR_MARKER_MM, FAR_MARKER_MM = 60.0, 150.0
DWELL_COOLDOWN_S = 1.0
PAGE_STYLE = "QLabel{font-size:18px;} QPushButton{font-size:17px;padding:10px 20px;}"
LEVEL_COLOURS = {review.GREEN: "#1e9e4a", review.AMBER: "#d08a00", review.RED: "#c62828"}

log = logging.getLogger(__name__)


def heading(text):
    label = QLabel(text)
    label.setStyleSheet("font-size:28px;font-weight:bold;padding:6px 0;")
    label.setWordWrap(True)
    return label


def button(text, slot):
    widget = QPushButton(text)
    widget.clicked.connect(slot)
    return widget


class StepBar(QLabel):
    """Where the wizard is: '1 Cameras > 2 Screen > ...' with the current step in bold."""

    def show_steps(self, steps, current):
        names = {"cameras": "Cameras", "screen": "Screen scale", "intrinsics": "Lens", "eye:left": "Left eye",
                 "eye:right": "Right eye", "gaze": "Gaze", "review": "Review"}
        parts = []
        for i, step in enumerate(steps, 1):
            text = f"{i} {names.get(step, step)}"
            parts.append(f"<b><u>{text}</u></b>" if step == current else f"<span style='color:#888'>{text}</span>")
        self.setText(" &nbsp;&rsaquo;&nbsp; ".join(parts))


# --- Startup ---

class ConnectingPage(Page):
    def __init__(self, controller):
        super().__init__(controller)
        self.text = QLabel()
        self.text.setStyleSheet("font-size:22px;")
        self.text.setTextFormat(Qt.RichText)
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(heading("Connecting the cameras..."), 0, Qt.AlignCenter)
        layout.addWidget(self.text, 0, Qt.AlignCenter)
        layout.addWidget(button("Camera setup (C)", controller.open_cameras), 0, Qt.AlignCenter)
        layout.addStretch(1)

    def on_tick(self):
        lines = []
        for role in ROLES:
            reader = self.controller.manager.get(role)
            state = "not set up" if reader is None else reader.status
            if reader is not None and reader.message:
                state += f" ({reader.message})"
            lines.append(f"{ROLE_LABELS[role]}: <b>{state}</b>")
        self.text.setText("<br>".join(lines))


# --- Screen scale ---

class ScaleCanvas(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.white)
        page = self.page
        w, h = self.width(), self.height()
        p.setRenderHint(QPainter.Antialiasing, False)
        if page.method == "card":
            card_w, card_h = page.card_width_px, page.card_width_px * CARD_HEIGHT_MM / CARD_WIDTH_MM
            x, y = int((w - card_w) / 2), int((h - card_h) / 2)
            p.setPen(QPen(QColor(30, 80, 220), 1))
            p.setBrush(QColor(200, 220, 255))
            p.drawRect(x, y, int(round(card_w)) - 1, int(round(card_h)) - 1)
        else:
            length = page.ruler_px
            x, y = int((w - length) / 2), h // 2
            p.setPen(QPen(Qt.black, 1))
            p.drawLine(x, y, x + length, y)
            for tick in range(0, 11):
                tx = x + round(length * tick / 10)
                p.drawLine(tx, y - (30 if tick in (0, 10) else 12), tx, y + (30 if tick in (0, 10) else 12))
        p.end()


class ScreenScalePage(Page):
    """Confirm the monitor's millimetres per pixel: every on-screen marker size depends on it."""

    def __init__(self, controller):
        super().__init__(controller)
        self.method = "card"
        self.canvas = ScaleCanvas(self)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.ruler_mm = QDoubleSpinBox()
        self.ruler_mm.setRange(10, 2000)
        self.ruler_mm.setDecimals(1)
        self.ruler_mm.setSuffix(" mm")
        self.ruler_mm.valueChanged.connect(self._ruler_changed)
        self.method_button = button("Use a ruler instead (M)", self.toggle_method)
        top = QVBoxLayout()
        top.addWidget(heading("Screen scale"))
        top.addWidget(self.info)
        row = QHBoxLayout()
        row.addWidget(self.method_button)
        row.addWidget(QLabel("Measured length:"))
        row.addWidget(self.ruler_mm)
        row.addStretch(1)
        row.addWidget(button("Back", controller.wizard_back))
        row.addWidget(button("Confirm (Enter)", self.confirm))
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(row)
        self.setStyleSheet(PAGE_STYLE)

    def enter(self):
        self.mm_per_px = self.controller.mm_per_px_guess()
        self.card_width_px = CARD_WIDTH_MM / self.mm_per_px
        self.ruler_px = int(min(self.width() * 0.7, 200 / self.mm_per_px)) or 600
        self.ruler_mm.blockSignals(True)
        self.ruler_mm.setValue(self.ruler_px * self.mm_per_px)
        self.ruler_mm.blockSignals(False)
        self._update()

    def toggle_method(self):
        self.method = "ruler" if self.method == "card" else "card"
        self.method_button.setText("Use a credit card instead (M)" if self.method == "ruler" else "Use a ruler instead (M)")
        self._update()

    def _ruler_changed(self, value):
        self.mm_per_px = value / self.ruler_px
        self.card_width_px = CARD_WIDTH_MM / self.mm_per_px
        self._update()

    def _update(self):
        confirmed = self.controller.monitor_entry()
        known = (f"Confirmed before ({confirmed['method']}, {confirmed['date'][:10]})."
                 if confirmed else "Not confirmed yet: the value below is the operating system's guess.")
        if self.method == "card":
            how = ("Hold a credit card (or any ID-1 card) flat against the screen and use the Left/Right arrow "
                   "keys (Shift: faster) until the blue rectangle is exactly the size of the card.")
        else:
            how = ("Measure the black line between the long end ticks with a ruler held against the screen, "
                   "and type its length in millimetres below.")
        self.ruler_mm.setEnabled(self.method == "ruler")
        self.info.setText(f"{how}<br><b>{self.mm_per_px:.4f} mm per pixel</b> "
                          f"({self.controller.screen_size_text(self.mm_per_px)}). {known}")
        self.canvas.update()

    def confirm(self):
        self.controller.confirm_screen_scale(self.mm_per_px, self.method)

    def handle_key(self, key, text):
        if key in (Qt.Key_Left, Qt.Key_Right) and self.method == "card":
            step = 10 if self.controller.shift_held() else 1
            self.card_width_px = max(50.0, self.card_width_px + (step if key == Qt.Key_Right else -step))
            self.mm_per_px = CARD_WIDTH_MM / self.card_width_px
            self._update()
            return True
        if key == Qt.Key_M:
            self.toggle_method()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.confirm()
            return True
        if key == Qt.Key_Escape:
            self.controller.wizard_back()
            return True
        return False


# --- Lens (front camera intrinsics) ---

class LensPage(Page):
    """The ChArUco board full-screen; the user moves the head so it appears all over the front image."""

    def __init__(self, controller):
        super().__init__(controller)
        self.canvas = TargetCanvas()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.calibrating = False

    def enter(self):
        self.calibrating = False
        self.canvas.board_image = board_qimage(fc.IntrinsicsCalibrator(FRONT_SIZE), int(self.width() * 0.7),
                                               int(self.height() * 0.62))
        self.canvas.title = "Lens calibration: move your head so the board appears in every part of the view"
        self.canvas.footer = "Esc: back    S: skip (keep the current lens calibration)"
        self.canvas.status, self.canvas.status_ok = "Look at the board from different angles and distances", True
        self.start()

    def start(self):
        self.calibrating = False
        self.controller.worker.call(lambda p: p.start_intrinsics(fc.IntrinsicsCalibrator(FRONT_SIZE)))

    def leave(self):
        self.controller.worker.call(lambda p: p.stop_intrinsics())

    def on_bundle(self, bundle):
        canvas = self.canvas
        canvas.set_front(bundle.front)
        canvas.front_marks = [] if bundle.charuco is None else [(bundle.charuco[0], QColor(0, 200, 0))]
        canvas.cells = bundle.charuco_cells or {}
        canvas.counts = f"Views {bundle.intrinsics_views}/15"
        canvas.update()
        if bundle.intrinsics_views >= 15 and not self.calibrating:
            self.calibrating = True
            canvas.status, canvas.status_ok = "Calibrating the lens...", True
            self.controller.worker.call(self._take_calibrator, self._calibrate)

    @staticmethod
    def _take_calibrator(pipeline):
        calibrator = pipeline.intrinsics_calibrator
        pipeline.stop_intrinsics()
        return calibrator

    def _calibrate(self, calibrator):
        if calibrator is None:
            return
        run_in_pool(calibrator.calibrate, self._calibrated)

    def _calibrated(self, result, error):
        if error is not None:
            sounds.play("error")
            self.canvas.status, self.canvas.status_ok = (
                f"That did not work: {error}. Starting again: cover the corners of the view too.", False)
            self.start()
            return
        sounds.play("done")
        camera, rms = result
        self.controller.on_intrinsics(camera, rms)

    def handle_key(self, key, text):
        if key == Qt.Key_Escape:
            self.controller.wizard_back()
            return True
        if key == Qt.Key_S and self.controller.lens_calibrated():
            self.controller.wizard_next()
            return True
        return False


# --- Each eye ---

class EyePage(Page):
    def __init__(self, controller, eye):
        super().__init__(controller)
        self.eye = eye
        self.view = FrameView(f"No frames from the {eye} eye camera")
        self.title = heading(f"{eye.title()} eye")
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.state = QLabel()
        self.state.setWordWrap(True)
        self.state.setTextFormat(Qt.RichText)
        self.confidence = QProgressBar()
        self.confidence.setFormat("Pupil quality %p%")
        self.convergence = QProgressBar()
        self.convergence.setFormat("Eye model %p%")
        side = QVBoxLayout()
        side.addWidget(self.title)
        side.addWidget(self.instructions)
        side.addWidget(self.confidence)
        side.addWidget(self.convergence)
        side.addWidget(self.state)
        side.addStretch(1)
        for text, slot in (("Lock eye model (L)", self.lock), ("Look into the lens (O)", self.override),
                           ("Unlock / start the model again (U)", self.unlock), ("Wink test (T)", self.start_wink)):
            side.addWidget(button(text, slot))
        # Both eyes, from any page but Cameras (tracking pauses there); the text follows E pressed anywhere
        self.capture_button = button("Record eye frames (E)", controller.toggle_eye_capture)
        self.capture_button.setToolTip("Save both eyes' raw frames and pupil fits to FrontCameraApp/sessions, "
                                       "to replay the pupil detector on them. E again stops.")
        side.addWidget(self.capture_button)
        row = QHBoxLayout()
        row.addWidget(button("Back", controller.wizard_back))
        row.addWidget(button("Next (Enter)", self.next))
        side.addLayout(row)
        layout = QHBoxLayout(self)
        layout.addWidget(self.view, 3)
        panel = QWidget()
        panel.setLayout(side)
        layout.addWidget(panel, 2)
        self.setStyleSheet(PAGE_STYLE + " QProgressBar{font-size:16px;height:28px;}")
        self.centers = collections.deque(maxlen=60)
        self.center_count = 0
        self.result = None
        self.message = ""
        self.wink = None  # None, or {"start": t, "own_low": s, "other_low": s, "last": t}
        self.wink_result = None

    def enter(self):
        self.centers.clear()
        self.message = ""
        self.wink, self.wink_result = None, None
        self.instructions.setText(
            "1. Check that the eye looks upright, eyelashes at the top (if not: C for Cameras).<br>"
            "2. Look around slowly, in all directions, until the eye model is complete and steady.<br>"
            "3. Look straight ahead and press L to lock it. If the blue circle is off-center, "
            "look straight into this eye camera's lens and press O first.<br>"
            "4. Optional: T, then close this eye for a second, to confirm it is the right camera.")

    def on_bundle(self, bundle):
        result = bundle.eyes.get(self.eye)
        self.result = result
        self.view.set_frame(None if result is None else result.frame)
        for event in bundle.events:
            if isinstance(event, CenterOverrideDone) and event.eye_id == self.eye:
                self.message = "Eye center set from the look into the lens. Look straight ahead and press L."
                sounds.play("done")
                self.controller.invalidate_gaze("an eye center changed")
        if result is not None and result.status == "ok":
            self.centers.append(result.model_center)
        self._wink_update(bundle)
        self._show()

    def on_tick(self):
        eye = self.eye
        self.controller.worker.call(lambda p: len(p.trackers[eye].model_centers), self._got_count)
        capture = self.controller.eye_capture
        self.capture_button.setText("Record eye frames (E)" if capture is None
                                    else f"Stop eye recording: {capture.saved} frames (E)")

    def _got_count(self, count):
        self.center_count = count or 0

    def stability_px(self):
        if len(self.centers) < 10:
            return None
        return float(np.max(np.std(np.array(self.centers, dtype=float), axis=0)))

    def _show(self):
        result = self.result
        confidence = 0.0 if result is None or result.status != "ok" else result.quality
        self.confidence.setValue(int(confidence * 100))
        stability = self.stability_px()
        progress = min(1.0, self.center_count / MIN_MODEL_CENTERS)
        if stability is not None and progress >= 1:
            progress = 1.0 if stability < 3 else 0.9
        self.convergence.setValue(int(progress * 100))
        lines = []
        if result is None:
            lines.append("<b style='color:#c62828'>No frames from this camera.</b>")
        elif result.override_samples is not None:
            lines.append(f"Look into the lens... {result.override_samples}/{CENTER_OVERRIDE_FRAMES_NEEDED} (O cancels)")
        elif result.locked:
            lines.append("<b style='color:#1e9e4a'>Locked.</b> Press Enter for the next step, or U to start over.")
        elif not result.adjusting:
            lines.append("Eye center fixed: look straight ahead and press L to lock.")
        else:
            steady = "" if stability is None else f", center moving {stability:.1f} px"
            lines.append(f"Adapting: {self.center_count} model points{steady}.")
        if self.message:
            lines.append(self.message)
        if self.wink is not None:
            lines.append(f"<b>Close your {self.eye.upper()} eye now (keep the other open)...</b>")
        if self.wink_result:
            lines.append(self.wink_result)
        self.state.setText("<br>".join(lines))

    def lock(self):
        if self.result is None or self.result.status != "ok":
            self.message = "Need a tracked pupil to lock: look straight ahead."
            sounds.play("error")
            return
        eye = self.eye

        def lock(pipeline):
            tracker = pipeline.trackers[eye]
            tracker.set_adjustment_enabled(False)
            tracker.lock_sphere()

        self.controller.worker.call(lock)
        self.controller.invalidate_gaze(f"the {eye} eye model was locked again")
        self.message = ""
        sounds.play("capture")

    def unlock(self):
        eye = self.eye

        def unlock(pipeline):
            # A fresh model: the radius only ever grows, so one built on lash fits never recovers
            pipeline.trackers[eye].reset()

        self.controller.worker.call(unlock)
        self.controller.invalidate_gaze(f"the {eye} eye model was unlocked")
        self.centers.clear()
        self.message = ""

    def override(self):
        eye = self.eye
        self.controller.worker.call(lambda p: p.trackers[eye].toggle_center_override())

    # Wink test: this eye's pupil must vanish while the other eye's stays.

    def start_wink(self):
        self.wink = {"start": time.perf_counter(), "own": 0.0, "other": 0.0, "last": None}
        self.wink_result = None

    def _wink_update(self, bundle):
        if self.wink is None:
            return
        now = bundle.t
        other = "right" if self.eye == "left" else "left"

        def tracked(eye):
            result = bundle.eyes.get(eye)
            return result is not None and result.status == "ok" and result.quality >= 0.7

        dt = 0.0 if self.wink["last"] is None else min(0.2, now - self.wink["last"])
        self.wink["last"] = now
        if not tracked(self.eye) and tracked(other):
            self.wink["own"] += dt
        elif tracked(self.eye) and not tracked(other):
            self.wink["other"] += dt
        if self.wink["own"] >= 0.4:
            self.wink, self.wink_result = None, "<b style='color:#1e9e4a'>Wink test passed: this is your " \
                                                f"{self.eye} eye.</b>"
            sounds.play("done")
        elif self.wink["other"] >= 0.4:
            self.wink = None
            self.wink_result = (f"<b style='color:#c62828'>The {other} eye camera lost its pupil instead: the eye "
                                "cameras look swapped. Press C and swap them.</b>")
            sounds.play("error")
        elif time.perf_counter() - self.wink["start"] > 6:
            self.wink, self.wink_result = None, "Wink test timed out: press T to try again."

    def next(self):
        if self.result is None or not self.result.locked:
            self.message = "Lock the eye model first (L)."
            sounds.play("error")
            return
        self.controller.wizard_next()

    def handle_key(self, key, text):
        actions = {Qt.Key_L: self.lock, Qt.Key_O: self.override, Qt.Key_U: self.unlock, Qt.Key_T: self.start_wink,
                   Qt.Key_Return: self.next, Qt.Key_Enter: self.next, Qt.Key_Escape: self.controller.wizard_back,
                   Qt.Key_C: self.controller.open_cameras}
        if key in actions:
            actions[key]()
            return True
        return False


# --- Gaze targets: calibration (near, far), quick recalibration, verification ---

class TargetPage(Page):
    CALIBRATE, QUICK, VERIFY = "calibrate", "quick", "verify"

    def __init__(self, controller):
        super().__init__(controller)
        self.canvas = TargetCanvas()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.mode = self.CALIBRATE
        self.band = "near"
        self.generation = None
        self.position = 4
        self.ok_since = None
        self.cooldown_until = 0.0
        self.last_result = None
        self.flash = ""  # Shown instead of the gate text for a moment after a capture
        self.flash_until = 0.0
        self.accuracy = None
        self.last_errors = []

    # --- Setup ---

    def start(self, mode, band="near"):
        """Called by the controller before showing the page."""
        self.mode, self.band = mode, (None if mode == self.VERIFY else band)
        self.position = 4
        self.ok_since, self.last_result = None, None
        self.flash, self.flash_until = "", 0.0
        if mode == self.VERIFY:
            self.accuracy = fc.AccuracyTest()
            self.last_errors = []

    def enter(self):
        self._set_marker()

    def marker_side_px(self):
        mm = FAR_MARKER_MM if self.band == "far" else NEAR_MARKER_MM
        return int(min(mm / self.controller.mm_per_px(), self.height() * 0.3))

    def positions(self):
        area = self.canvas.free_area()
        side = self.marker_side_px()
        xs = np.linspace(area.left() + side, area.right() - side, 3)
        ys = np.linspace(area.top() + side, area.bottom() - side, 3)
        return [(float(x), float(y)) for y in ys for x in xs]

    def _set_marker(self):
        side = self.marker_side_px()
        center = self.positions()[self.position if self.band != "far" else 4]
        self.canvas.set_marker(center, side)
        length_m = side * self.controller.mm_per_px() / 1000.0
        self.generation = None  # Gates wait until the pipeline uses the new size

        def set_side(pipeline):
            pipeline.set_marker_side(length_m)
            return pipeline.marker_generation

        self.controller.worker.call(set_side, self._got_generation)

    def _got_generation(self, generation):
        self.generation = generation

    def move_marker(self, step=1):
        self.position = (self.position + step) % 9
        self._set_marker()

    # --- Per bundle ---

    def context(self):
        session = self.controller.session
        earlier = [] if self.mode == self.VERIFY or session is None else session.pixels(self.band)
        return capture_gates.GateContext(
            prerequisites=self.controller.prerequisites(verify=self.mode == self.VERIFY),
            band=self.band, marker_generation=-1 if self.generation is None else self.generation,
            earlier_pixels=earlier, image_size=FRONT_SIZE, window=int(self.controller.settings["frames_averaged"]))

    def on_bundle(self, bundle):
        canvas = self.canvas
        result = capture_gates.evaluate(bundle.history, self.context(), self.controller.camera)
        self.last_result = result
        reason = result.failures[0] if result.failures else "ok"
        if reason != getattr(self, "_last_gate_reason", None):
            self._last_gate_reason = reason
            log.info("Gate %s: %s %s", self.band, reason, result.stats)
        now = time.perf_counter()
        dwell = self.controller.settings["capture_mode"] == "dwell"
        band_done = self._band_done_text()
        if result.ok and now >= self.cooldown_until and band_done is None:
            self.ok_since = self.ok_since or now
        else:
            self.ok_since = None
        canvas.dwell = 0.0
        if dwell and self.ok_since is not None:
            canvas.dwell = (now - self.ok_since) / float(self.controller.settings["dwell_seconds"])
            if canvas.dwell >= 1.0:
                self.capture()
        if now < self.flash_until:
            canvas.status, canvas.status_ok = self.flash, True
        elif band_done is not None:
            canvas.status, canvas.status_ok = band_done, True
        elif result.ok:
            canvas.status = "Ready: hold your gaze" if dwell else "Ready: press Space (or the clicker)"
            canvas.status_ok = True
        else:
            canvas.status, canvas.status_ok = result.failures[0], False
        marker = bundle.marker
        canvas.depth = None if marker is None else float(marker.point[2])
        canvas.band = self.band
        canvas.current_cell = None if marker is None else capture_gates.cell_of(marker.pixel, FRONT_SIZE)
        canvas.confidence = {eye: (0.0 if r is None or r.status != "ok" else r.quality)
                             for eye, r in bundle.eyes.items()}
        canvas.set_front(bundle.front)
        canvas.set_eyes({eye: None if r is None else r.frame for eye, r in bundle.eyes.items()})
        canvas.front_marks = [] if marker is None else [(marker.corners, QColor(0, 160, 255))]
        if bundle.estimate is not None and self.mode == self.VERIFY:
            canvas.front_marks.append((np.asarray(bundle.estimate.pixel).reshape(1, 2), QColor(255, 40, 40)))
        self._texts()
        canvas.update()

    def _band_done_text(self):
        """The next step once this band has its target samples (dwell stops; Space still adds more), else None."""
        session = self.controller.session
        if self.mode == self.VERIFY or session is None:
            return None
        count = len(session.pixels(self.band))
        if count < capture_gates.TARGET_PER_BAND:
            return None
        label = self.band.upper()
        if self.mode == self.CALIBRATE and self.band == "near" and not session.band_ready("far"):
            return f"{label} done ({count}): press N for the FAR samples"
        if session.ready:
            return f"{label} done ({count}): press S to solve"
        return f"{label} done ({count}): press B and finish NEAR"

    def _texts(self):
        canvas = self.canvas
        mode_text = "Dwell" if self.controller.settings["capture_mode"] == "dwell" else "Manual"
        if self.mode == self.VERIFY:
            canvas.title = "Verify: look at the marker's center, at several places and distances"
            lines = [f"{band.upper()} n={n}: {error:.2f}°" for band, (n, error, _) in
                     self.accuracy.summary_by_band().items()]
            canvas.counts = "\n".join(lines + [f"last {e:.2f}°" for e in self.last_errors[-1:]])
            canvas.cells = {}
            canvas.footer = f"{mode_text} (M)    Space/clicker: measure    Tab: move marker    Esc: done"
            return
        session = self.controller.session
        counts = session.counts()
        canvas.cells = collections.Counter(capture_gates.cell_of(p, FRONT_SIZE) for p in session.pixels(self.band))
        target = capture_gates.TARGET_PER_BAND
        if self.mode == self.QUICK:
            canvas.title = f"Quick recalibration: stand under {gg.NEAR_BAND_MAX_M} m and look at the marker's center"
            canvas.counts = f"Near {counts['near']}/{target}"
        elif self.band == "near":
            canvas.title = (f"Gaze NEAR: stand under {gg.NEAR_BAND_MAX_M} m; turn your head to put the "
                            "marker in each square")
            canvas.counts = f"Near {counts['near']}/{target}\nFar {counts['far']}/{target}"
        else:
            canvas.title = (f"Gaze FAR: stand over {gg.FAR_BAND_MIN_M} m; turn your head to put the "
                            "marker in each square")
            canvas.counts = f"Near {counts['near']}/{target}\nFar {counts['far']}/{target}"
        keys = [f"{mode_text} (M)", "Space/clicker: capture", "Backspace: undo", "Tab: move marker"]
        if self.mode == self.CALIBRATE and self.band == "near":
            keys.append("N: on to FAR")
        if self.mode == self.CALIBRATE and self.band == "far":
            keys.append("B: back to NEAR")
        if session.ready:
            keys.append("S: solve")
        keys.append("Esc: back")
        canvas.footer = "    ".join(keys)

    # --- Actions ---

    def capture(self):
        result = self.last_result
        if result is None or not result.ok:
            sounds.play("error")
            return
        self.ok_since = None
        self.cooldown_until = time.perf_counter() + DWELL_COOLDOWN_S
        sample = result.sample
        if self.mode == self.VERIFY:
            self._measure(sample)
            return
        warning = self.controller.session.add(sample)
        sounds.play("error" if warning else "capture")
        count = len(self.controller.session.pixels(self.band))
        text = warning or f"Captured at {sample.depth:.2f} m ({count} in this band)"
        if count == capture_gates.TARGET_PER_BAND:
            sounds.play("done")
            text = warning or ("Near done: press N for the far samples" if self.mode == self.CALIBRATE and
                               self.band == "near" else "Done: press S to solve")
        self._flash(text, 3.0 if warning else 1.5)  # The marker stays: head turns cover the grid (Tab moves it)

    def _measure(self, sample):
        estimate, baseline = self.controller.estimate_for(sample.gaze)
        if estimate is None:
            self._flash("No calibrated gaze for this look", 1.5)
            sounds.play("error")
            return
        detection = fc.MarkerDetection(None, sample.point, sample.pixel)
        record = self.accuracy.record(self.controller.camera, detection, estimate.pixel,
                                      None if baseline is None else baseline.pixel)
        self.last_errors.append(record["error"])
        sounds.play("capture")
        self._flash(f"{record['depth']:.2f} m: {record['error']:.2f}°", 2.0)
        self.move_marker()

    def _flash(self, text, seconds):
        self.flash, self.flash_until = text, time.perf_counter() + seconds

    def undo(self):
        if self.mode != self.VERIFY and self.controller.session.undo():
            self._flash("Removed the last sample", 1.5)

    def solve(self):
        if self.mode == self.VERIFY:
            return
        if not self.controller.session.ready:
            need = "both bands" if self.mode == self.CALIBRATE else "the near band"
            self._flash(f"Not yet: {need} need {capture_gates.MIN_PER_BAND}+ samples over 2 rows and 2 columns", 3.0)
            sounds.play("error")
            return
        self._flash("Solving...", 10.0)
        self.controller.solve_session()

    def set_band(self, band):
        if self.mode == self.CALIBRATE and band != self.band:
            self.band = band
            self.position = 4
            self._set_marker()

    def toggle_mode(self):
        mode = "manual" if self.controller.settings["capture_mode"] == "dwell" else "dwell"
        self.controller.settings.set("capture_mode", mode)
        self._flash(f"{mode.title()} capture", 1.5)

    def handle_key(self, key, text):
        if key in CAPTURE_KEYS:
            self.capture()
        elif key in UNDO_KEYS:
            self.undo()
        elif key == Qt.Key_Tab:
            self.move_marker()
        elif key == Qt.Key_M:
            self.toggle_mode()
        elif key == Qt.Key_N:
            self.set_band("far")
        elif key == Qt.Key_B:
            self.set_band("near")
        elif key in (Qt.Key_S, Qt.Key_Return, Qt.Key_Enter):
            self.solve()
        elif key == Qt.Key_Escape:
            if self.mode == self.VERIFY:
                self.controller.finish_verify(self.accuracy)
            elif self.mode == self.CALIBRATE and self.band == "far":
                self.set_band("near")
            else:
                self.controller.wizard_back()
        else:
            return False
        return True


# --- Review ---

class ReviewPage(Page):
    def __init__(self, controller):
        super().__init__(controller)
        self.lines = QLabel()
        self.lines.setTextFormat(Qt.RichText)
        self.lines.setWordWrap(True)
        self.lines.setStyleSheet("font-size:22px;")
        self.verdict = heading("")
        self.buttons = QHBoxLayout()
        layout = QVBoxLayout(self)
        layout.addWidget(heading("Calibration result"))
        layout.addWidget(self.verdict)
        layout.addWidget(self.lines)
        layout.addStretch(1)
        layout.addLayout(self.buttons)
        self.setStyleSheet(PAGE_STYLE)
        self.accept_button = button("Accept and track (Enter)", controller.accept_calibration)
        self.redo_near = button("Redo near (N)", lambda: controller.redo("near"))
        self.redo_far = button("Redo far (F)", lambda: controller.redo("far"))
        self.redo_all = button("Redo all (A)", lambda: controller.redo("all"))
        self.back = button("Back to capturing (Esc)", lambda: controller.redo(None))
        for widget in (self.back, self.redo_all, self.redo_near, self.redo_far):
            self.buttons.addWidget(widget)
        self.buttons.addStretch(1)
        self.buttons.addWidget(self.accept_button)

    def show_review(self, report, quick):
        verdicts = {review.GREEN: "Good: accept it.", review.AMBER: "Usable, but check the amber lines.",
                    review.RED: "Problems found: redo, or accept only if you know why."}
        self.verdict.setText(f"<span style='color:{LEVEL_COLOURS[report.level]}'>{verdicts[report.level]}</span>")
        self.lines.setText("<br>".join(f"<span style='color:{LEVEL_COLOURS[level]}'>&#9679;</span> {text}"
                                       for text, level in report.lines))
        self.redo_far.setVisible(not quick)
        self.redo_all.setVisible(not quick)

    def handle_key(self, key, text):
        controller = self.controller
        actions = {Qt.Key_Return: controller.accept_calibration, Qt.Key_Enter: controller.accept_calibration,
                   Qt.Key_N: lambda: controller.redo("near"), Qt.Key_Escape: lambda: controller.redo(None)}
        if self.redo_far.isVisible():
            actions.update({Qt.Key_F: lambda: controller.redo("far"), Qt.Key_A: lambda: controller.redo("all")})
        if key in actions:
            actions[key]()
            return True
        return False
