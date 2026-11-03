"""Tracking: the front camera full-screen with the gaze dot and depth, the eye views as insets
and a status bar. The keys (R V W O H C X E F1 Q) are the controller's, so they work from any page
where they make sense."""

import logging

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPen
from PyQt5.QtWidgets import QLabel, QPushButton

from ..recorder import Recorder
from ..settings import SESSIONS_DIR
from .common import FrameView, Page

log = logging.getLogger(__name__)

HELP = """<h2>Keys</h2>
<table cellpadding=4>
<tr><td><b>R</b></td><td>Quick recalibration (near only, keeps the eye positions)</td></tr>
<tr><td><b>V</b></td><td>Verify the accuracy on the marker</td></tr>
<tr><td><b>W</b></td><td>Full calibration wizard</td></tr>
<tr><td><b>C</b></td><td>Cameras</td></tr>
<tr><td><b>O</b></td><td>Desktop overlay: gaze ring on the monitor</td></tr>
<tr><td><b>H</b></td><td>Heatmap on the overlay on/off</td></tr>
<tr><td><b>X</b></td><td>Record this view to a video (again to stop)</td></tr>
<tr><td><b>E</b></td><td>Capture raw eye frames for PupilNet training (again to stop)</td></tr>
<tr><td><b>F1</b></td><td>This help</td></tr>
<tr><td><b>Q</b></td><td>Quit</td></tr>
</table>"""


class FrontView(FrameView):
    def __init__(self):
        super().__init__("Waiting for the front camera")
        self.bundle = None

    def paint_overlay(self, painter):
        bundle = self.bundle
        if bundle is None:
            return
        painter.setRenderHint(painter.Antialiasing, True)
        if bundle.marker is not None:
            corners = [self.to_widget(x, y) for x, y in bundle.marker.corners.reshape(4, 2)]
            if None not in corners:
                painter.setPen(QPen(QColor(0, 200, 255), 2))
                painter.drawPolygon(*[QPointF(*c) for c in corners])
        estimate = bundle.estimate
        if estimate is None:
            return
        point = self.to_widget(*estimate.pixel)
        if point is None:
            return
        x, y = point
        painter.setPen(QPen(QColor(0, 0, 0), 6))
        painter.setBrush(QColor(255, 60, 60, 210))
        painter.drawEllipse(QPointF(x, y), 14, 14)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPointF(x, y), 14, 14)
        font = QFont("Segoe UI")
        font.setPixelSize(22)
        font.setBold(True)
        painter.setFont(font)
        text_rect = QRectF(x + 20, y - 16, 160, 32)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(text_rect.translated(2, 2), Qt.AlignLeft | Qt.AlignVCenter, f"{estimate.depth:.2f} m")
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, f"{estimate.depth:.2f} m")


class TrackingView(Page):
    def __init__(self, controller):
        super().__init__(controller)
        self.front = FrontView()
        self.front.setParent(self)
        self.eyes = {eye: FrameView(f"{eye.title()} eye offline") for eye in ("left", "right")}
        for view in self.eyes.values():
            view.setParent(self)
        self.status = QLabel(self)
        self.status.setStyleSheet("background:rgba(0,0,0,170);color:white;font-size:15px;padding:4px 10px;")
        self.status.setTextFormat(Qt.RichText)
        self.help = QLabel(HELP, self)
        self.help.setStyleSheet("background:rgba(20,20,20,225);color:white;font-size:17px;padding:20px;"
                                "border-radius:10px;")
        self.help.hide()
        self.recorder = None
        self.record_button = QPushButton(self)
        self.record_button.setFocusPolicy(Qt.NoFocus)  # Keys stay with the app-wide filter
        self.record_button.clicked.connect(self.toggle_recording)
        self._style_record_button()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        self.front.setGeometry(0, 0, w, h)
        inset_w = int(w * 0.2)
        inset_h = inset_w * 3 // 4
        self.eyes["left"].setGeometry(10, h - inset_h - 50, inset_w, inset_h)  # Front camera: the user's own view
        self.eyes["right"].setGeometry(w - inset_w - 10, h - inset_h - 50, inset_w, inset_h)
        self.status.setGeometry(0, h - 40, w, 40)
        self.record_button.setGeometry(w - 170, 10, 160, 40)
        self.help.adjustSize()
        self.help.move((w - self.help.width()) // 2, (h - self.help.height()) // 2)

    def toggle_help(self):
        self.help.setVisible(not self.help.isVisible())
        self.help.raise_()

    def on_bundle(self, bundle):
        self.front.bundle = bundle
        self.front.set_frame(bundle.front)
        for eye, view in self.eyes.items():
            result = bundle.eyes.get(eye)
            view.set_frame(None if result is None else result.frame)
        if self.recorder is not None:
            self.recorder.write(self._grab_bgr(), bundle)

    def toggle_recording(self):
        if self.recorder is None:
            self.recorder = Recorder(SESSIONS_DIR)
            log.info("Recording to %s", self.recorder.video_path)
            self.controller.banner(f"Recording: {self.recorder.video_path}", "info", 3)
        else:
            recorder, self.recorder = self.recorder, None
            recorder.close()
            log.info("Recorded %d frames to %s", recorder.frames, recorder.video_path)
            self.controller.banner(f"Saved {recorder.frames} frames: {recorder.video_path}", "info", 5)
        self._style_record_button()

    def stop_recording(self):
        if self.recorder is not None:
            self.toggle_recording()

    def _style_record_button(self):
        recording = self.recorder is not None
        self.record_button.setText("■ Stop (X)" if recording else "● Record (X)")
        self.record_button.setStyleSheet(
            "QPushButton{background:%s;color:white;font-size:16px;font-weight:bold;border-radius:6px;}"
            % ("rgba(200,30,30,230)" if recording else "rgba(0,0,0,170)"))

    def _grab_bgr(self):
        """The page as drawn now (front view, gaze dot, eye insets, status), as a BGR array."""
        self.front.repaint()  # The new bundle's dot, before the grab
        image = self.grab().toImage().convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        data = np.frombuffer(image.constBits().asstring(image.bytesPerLine() * height), np.uint8)
        rgb = data.reshape(height, image.bytesPerLine())[:, :width * 3].reshape(height, width, 3)
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def on_tick(self):
        self.status.setText(self.controller.status_html())
