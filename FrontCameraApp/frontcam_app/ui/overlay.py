"""The desktop overlay: corner markers, the gaze ring and a live heatmap over everything else.

A frameless, transparent, click-through, always-on-top tool window on the target monitor: the
desktop stays usable underneath, and no OS mouse control is involved. The ring repaints only
its own region at the bundle rate; the heatmap (a full-screen repaint) at about 5 Hz.
A tray icon controls it, because the overlay itself never takes focus or keys.
"""

import os
import time

import numpy as np
from PyQt5.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QAction, QMenu, QSystemTrayIcon, QWidget

from gazecore.heatmap import Heatmap

from .common import to_qimage

HEATMAP_PERIOD_MS = 200
RING_RADIUS = 28


class OverlayWindow(QWidget):
    def __init__(self, layout, screen, show_heatmap=True):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.layout_ = layout
        self.setGeometry(screen.geometry())
        self.markers = {i: to_qimage(layout.marker_image(i)) for i in layout.markers}
        self.heatmap = Heatmap(layout.width_px, layout.height_px, layout.mm_per_px)
        self.show_heatmap = show_heatmap
        self._heat_image = None
        self.screen_gaze = None
        self._ring_rect = None
        self.timer = QTimer(self, interval=HEATMAP_PERIOD_MS, timeout=self._refresh_heatmap)
        self.paint_ms = 0.0  # Last full repaint cost, for the status bar

    def start(self):
        self.show()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.hide()

    def update_gaze(self, screen_gaze, t):
        """screen_gaze: the Bundle's ScreenGaze (or None)."""
        self.screen_gaze = screen_gaze
        pixel = None if screen_gaze is None or not screen_gaze.on_screen else screen_gaze.pixel
        self.heatmap.add(pixel, t)
        rect = self._ring_bounds()
        dirty = rect if self._ring_rect is None else rect.united(self._ring_rect) if rect else self._ring_rect
        self._ring_rect = rect
        if dirty is not None:
            self.update(dirty)

    def _ring_bounds(self):
        gaze = self.screen_gaze
        if gaze is None or gaze.pixel is None:
            return None
        x, y = self._clamped(gaze.pixel)
        r = RING_RADIUS + 8
        return QRect(int(x - r), int(y - r), 2 * r, 2 * r)

    def _clamped(self, pixel):
        """Off-screen gaze is drawn pinned to the nearest edge."""
        return (min(max(pixel[0], RING_RADIUS), self.width() - RING_RADIUS),
                min(max(pixel[1], RING_RADIUS), self.height() - RING_RADIUS))

    def set_heatmap_visible(self, visible):
        self.show_heatmap = visible
        self._heat_image = None
        self.update()

    def _refresh_heatmap(self):
        self.heatmap.decay(time.perf_counter())
        if self.show_heatmap:
            self._heat_image = to_qimage_rgba(self.heatmap.rgba())
            self.update()

    def paintEvent(self, event):
        started = time.perf_counter()
        p = QPainter(self)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(event.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self.show_heatmap and self._heat_image is not None:
            p.drawImage(0, 0, self._heat_image)
        for marker_id, image in self.markers.items():
            cx, cy, _ = self.layout_.markers[marker_id]
            p.drawImage(int(round(cx - image.width() / 2)), int(round(cy - image.height() / 2)), image)
        self._paint_ring(p)
        p.end()
        if event.rect().width() >= self.width():
            self.paint_ms = (time.perf_counter() - started) * 1000

    def _paint_ring(self, p):
        gaze = self.screen_gaze
        if gaze is None or gaze.pixel is None:
            return
        x, y = self._clamped(gaze.pixel)
        p.setRenderHint(QPainter.Antialiasing, True)
        if not gaze.on_screen:
            colour, style = QColor(255, 150, 0, 230), Qt.DotLine
        elif gaze.eyes_used < 2:
            colour, style = QColor(255, 220, 0, 230), Qt.DashLine
        else:
            colour, style = QColor(255, 40, 40, 230), Qt.SolidLine
        p.setPen(QPen(QColor(0, 0, 0, 160), 7))
        p.drawEllipse(QPointF(x, y), RING_RADIUS, RING_RADIUS)
        p.setPen(QPen(colour, 4, style))
        p.drawEllipse(QPointF(x, y), RING_RADIUS, RING_RADIUS)

    def save_heatmap(self, folder):
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, time.strftime("heatmap_%Y%m%d_%H%M%S"))
        background = None
        screen = self.screen()
        if screen is not None:
            self.hide()  # The screenshot shows the desktop, not the overlay
            shot = screen.grabWindow(0)
            self.show()
            background = qimage_to_bgr(shot.toImage())
        self.heatmap.save(stem + ".png", stem + ".csv", background)
        return stem + ".png"


def to_qimage_rgba(bgra):
    bgra = np.ascontiguousarray(bgra)
    height, width = bgra.shape[:2]
    return QImage(bgra.data, width, height, bgra.strides[0], QImage.Format_ARGB32).copy()


def qimage_to_bgr(image):
    """BGR array, or None for an empty image (a screenshot can fail, e.g. on a locked desktop)."""
    if image.isNull() or image.width() == 0:
        return None
    image = image.convertToFormat(QImage.Format_RGB32)
    width, height = image.width(), image.height()
    pointer = image.constBits()
    pointer.setsize(image.byteCount())
    return np.array(pointer, dtype=np.uint8).reshape(height, image.bytesPerLine() // 4, 4)[:, :width, :3].copy()


def tray_icon_pixmap():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(255, 40, 40), 4))
    p.drawEllipse(QRectF(5, 5, 22, 22))
    p.setBrush(QColor(255, 40, 40))
    p.drawEllipse(QRectF(13, 13, 6, 6))
    p.end()
    return pixmap


class OverlayTray(QSystemTrayIcon):
    def __init__(self, controller):
        super().__init__(QIcon(tray_icon_pixmap()))
        self.setToolTip("Front camera gaze overlay")
        menu = QMenu()
        for text, slot in (("Heatmap on/off", controller.toggle_heatmap), ("Save heatmap", controller.save_heatmap),
                           ("Stop the overlay and show the app", controller.stop_overlay)):
            action = QAction(text, menu)
            action.triggered.connect(slot)
            menu.addAction(action)
        self.setContextMenu(menu)
        self.menu = menu
        self.activated.connect(lambda reason: controller.stop_overlay() if reason == QSystemTrayIcon.DoubleClick else None)
