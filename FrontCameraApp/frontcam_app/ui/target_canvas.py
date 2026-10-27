"""The full-screen target: the ArUco marker (or the ChArUco board) drawn 1:1 in device pixels,
with a heads-up display big enough to read from 2 m.

Layout: instructions across the top, the depth gauge on the left, coverage grid and pupil
bars bottom-left, the front camera bottom-right with the two eye views above it. The target sits in the middle region, so
the HUD never covers its white quiet zone.
"""

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gazecore import front_calibration as fc
from gazecore import gaze_geometry as gg

from .. import capture_gates
from .common import fit_rect, to_qimage

GAUGE_RANGE_M = (0.2, 3.0)
TEXT = QColor(20, 20, 20)
OK_COLOUR, BAD_COLOUR, BAND_COLOUR = QColor(20, 150, 60), QColor(200, 40, 40), QColor(120, 170, 255, 90)


def marker_qimage(side_px, marker_id=0):
    dictionary = cv2.aruco.getPredefinedDictionary(fc.TARGET_DICTIONARY)
    return to_qimage(cv2.aruco.generateImageMarker(dictionary, marker_id, int(side_px)))


class TargetCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.title = ""
        self.status = ""  # The first failing gate, or what just happened
        self.status_ok = False
        self.footer = ""  # Keys
        self.counts = ""  # "Near 3/9   Far 0/9"
        self.marker_center = None  # (x, y) widget px; None hides the marker
        self.marker_side = 0
        self._marker_image = None
        self.board_image = None  # QImage drawn 1:1 centred instead of the marker
        self.depth = None
        self.band = None  # "near"/"far": highlights the target zone on the gauge
        self.cells = {}  # {(row, col): count} of the coverage grid
        self.current_cell = None
        self.dwell = 0.0  # 0..1 ring around the marker
        self.confidence = {}  # {eye: 0..1}
        self.front = None  # QImage of the front camera
        self.front_marks = []  # [(points (N, 2) in front pixels, QColor)] drawn on the inset
        self.eyes = {}  # {eye: QImage of the drawn eye view, or None}

    # --- State ---

    def set_marker(self, center, side_px):
        if side_px != self.marker_side or self._marker_image is None:
            self._marker_image = marker_qimage(side_px)
        self.marker_center, self.marker_side = center, int(side_px)
        self.update()

    def set_front(self, frame):
        self.front = to_qimage(frame)

    def set_eyes(self, frames):
        self.eyes = {eye: to_qimage(frame) for eye, frame in frames.items()}

    def free_area(self):
        """Where targets may go (widget px): clear of the HUD."""
        w, h = self.width(), self.height()
        return QRectF(w * 0.12, h * 0.17, w * 0.62, h * 0.66)

    # --- Painting ---

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.white)
        if self.board_image is not None:
            x = (self.width() - self.board_image.width()) // 2
            y = (self.height() - self.board_image.height()) // 2
            p.drawImage(x, y, self.board_image)  # 1:1, no scaling
        if self.marker_center is not None and self._marker_image is not None:
            half = self.marker_side // 2
            x, y = int(round(self.marker_center[0])) - half, int(round(self.marker_center[1])) - half
            p.drawImage(x, y, self._marker_image)  # 1:1, no scaling
            if self.dwell > 0:
                p.setRenderHint(QPainter.Antialiasing, True)
                radius = self.marker_side * 0.95
                p.setPen(QPen(OK_COLOUR, max(8, self.marker_side // 14), Qt.SolidLine, Qt.RoundCap))
                p.drawArc(QRectF(x + half - radius, y + half - radius, 2 * radius, 2 * radius),
                          90 * 16, -int(360 * 16 * min(1.0, self.dwell)))
        p.setRenderHint(QPainter.Antialiasing, True)
        self._paint_text(p)
        self._paint_gauge(p)
        self._paint_grid(p)
        self._paint_confidence(p)
        self._paint_front(p)
        self._paint_eyes(p)
        p.end()

    def _font(self, fraction, bold=False):
        font = QFont("Segoe UI")
        font.setPixelSize(max(14, int(self.height() * fraction)))
        font.setBold(bold)
        return font

    def _paint_text(self, p):
        w, h = self.width(), self.height()
        p.setPen(TEXT)
        p.setFont(self._font(0.045, bold=True))
        p.drawText(QRectF(0, h * 0.01, w, h * 0.07), Qt.AlignCenter, self.title)
        p.setPen(OK_COLOUR if self.status_ok else BAD_COLOUR)
        p.setFont(self._font(0.036, bold=True))
        p.drawText(QRectF(0, h * 0.08, w, h * 0.07), Qt.AlignCenter | Qt.TextWordWrap, self.status)
        p.setPen(TEXT)
        p.setFont(self._font(0.03, bold=True))
        p.drawText(QRectF(w * 0.75, h * 0.17, w * 0.24, h * 0.1), Qt.AlignRight | Qt.AlignTop, self.counts)
        p.setFont(self._font(0.02))
        p.drawText(QRectF(0, h * 0.955, w, h * 0.04), Qt.AlignCenter, self.footer)

    def _paint_gauge(self, p):
        h = self.height()
        top, bottom, x, width = h * 0.2, h * 0.8, self.width() * 0.02, self.width() * 0.035
        low, high = GAUGE_RANGE_M

        def y_of(depth):  # Nearer is lower, like walking towards the screen
            return bottom - (np.clip(depth, low, high) - low) / (high - low) * (bottom - top)

        p.setPen(QPen(TEXT, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(x, top, width, bottom - top))
        for band, (a, b) in (("near", (low, gg.NEAR_BAND_MAX_M)), ("far", (gg.FAR_BAND_MIN_M, high))):
            colour = QColor(BAND_COLOUR)
            if band == self.band:
                colour.setAlpha(200)
            p.fillRect(QRectF(x, y_of(b), width, y_of(a) - y_of(b)), colour)
            p.setFont(self._font(0.022, bold=True))
            p.drawText(QRectF(x + width + 6, (y_of(a) + y_of(b)) / 2 - 15, 120, 30), Qt.AlignVCenter, band.upper())
        if self.depth is not None:
            y = y_of(self.depth)
            in_band = self.band is None or gg.depth_band(self.depth) == self.band
            p.setPen(QPen(OK_COLOUR if in_band else BAD_COLOUR, 6))
            p.drawLine(QPointF(x - 8, y), QPointF(x + width + 8, y))
            p.setFont(self._font(0.03, bold=True))
            p.drawText(QRectF(x - 4, bottom + 8, 200, 50), Qt.AlignLeft, f"{self.depth:.2f} m")

    def _paint_grid(self, p):
        h = self.height()
        size = h * 0.18
        x0, y0 = self.width() * 0.02, h * 0.95 - size * 0.75 - 8
        cell_w, cell_h = size / 3, size * 0.75 / 3
        p.setFont(self._font(0.022, bold=True))
        for row in range(3):
            for col in range(3):
                rect = QRectF(x0 + col * cell_w, y0 + row * cell_h, cell_w - 3, cell_h - 3)
                count = self.cells.get((row, col), 0)
                p.fillRect(rect, QColor(60, 170, 90) if count else QColor(225, 225, 225))
                if (row, col) == self.current_cell:
                    p.setPen(QPen(QColor(30, 80, 220), 4))
                    p.drawRect(rect)
                p.setPen(Qt.white if count else QColor(140, 140, 140))
                p.drawText(rect, Qt.AlignCenter, str(count))

    def _paint_confidence(self, p):
        h, w = self.height(), self.width()
        x0, y0, bar_w, bar_h = w * 0.13, h * 0.95 - h * 0.135 - 8, w * 0.012, h * 0.135
        p.setFont(self._font(0.018, bold=True))
        for i, eye in enumerate(("left", "right")):
            value = float(self.confidence.get(eye, 0.0))
            x = x0 + i * bar_w * 2.4
            p.fillRect(QRectF(x, y0, bar_w, bar_h), QColor(225, 225, 225))
            colour = OK_COLOUR if value >= capture_gates.MIN_CONFIDENCE else BAD_COLOUR
            p.fillRect(QRectF(x, y0 + bar_h * (1 - value), bar_w, bar_h * value), colour)
            p.setPen(TEXT)
            p.drawText(QRectF(x - 10, y0 + bar_h + 2, bar_w + 20, 22), Qt.AlignCenter, eye[0].upper())

    def _paint_front(self, p):
        if self.front is None:
            return
        area = QRectF(self.width() * 0.76, self.height() * 0.62, self.width() * 0.23, self.height() * 0.33)
        rect = fit_rect(self.front.width(), self.front.height(), area)
        p.drawImage(rect, self.front)
        scale = rect.width() / self.front.width()
        for points, colour in self.front_marks:
            p.setPen(QPen(colour, 3))
            pts = [QPointF(rect.x() + x * scale, rect.y() + y * scale) for x, y in np.asarray(points).reshape(-1, 2)]
            if len(pts) == 4:
                p.drawPolygon(*pts)
            else:
                for pt in pts:
                    p.drawEllipse(pt, 3, 3)
        p.setPen(QPen(TEXT, 2))
        p.drawRect(rect)


    def _paint_eyes(self, p):
        """Left and right eye views side by side above the front camera: what the pupil fit sees."""
        w, h = self.width(), self.height()
        top, height, width = h * 0.29, h * 0.31, w * 0.113
        p.setFont(self._font(0.018, bold=True))
        for i, eye in enumerate(("left", "right")):
            area = QRectF(w * 0.76 + i * (width + w * 0.004), top, width, height)
            image = self.eyes.get(eye)
            rect = fit_rect(4, 3, area) if image is None else fit_rect(image.width(), image.height(), area)
            value = float(self.confidence.get(eye, 0.0))
            if image is None:
                p.fillRect(rect, QColor(60, 60, 60))
            else:
                p.drawImage(rect, image)
            p.setPen(QPen(OK_COLOUR if value >= capture_gates.MIN_CONFIDENCE else BAD_COLOUR, 4))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)
            p.setPen(TEXT)
            p.drawText(QRectF(rect.x(), rect.bottom() + 2, rect.width(), 24), Qt.AlignCenter,
                       f"{eye.title()} {value:.0%}" if image is not None else f"{eye.title()}: no frame")


def board_qimage(calibrator, max_width, max_height):
    """The ChArUco board at the largest exact size that fits (7 x 5 squares plus margin)."""
    columns, rows = calibrator.board.getChessboardSize()
    square = int(min((max_width - 40) / columns, (max_height - 40) / rows))
    size = (square * columns + 40, square * rows + 40)
    return to_qimage(calibrator.board.generateImage(size, marginSize=20))

