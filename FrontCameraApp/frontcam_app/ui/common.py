"""Small pieces every page uses: DPI setup, frame conversion, the frame view, banners.

Frames arrive as BGR numpy arrays from the worker; the UI never draws on them (the eye views
come already drawn by gazecore.eye_overlay), it only paints them and adds Qt overlays on top.
"""

import os

import numpy as np
from PyQt5.QtCore import QObject, QRectF, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

STATUS_COLORS = {  # CameraReader status -> colour for dots and labels
    "running": QColor(60, 200, 90),
    "opening": QColor(230, 170, 40),
    "reconnecting": QColor(230, 170, 40),
    "stalled": QColor(230, 120, 40),
    "failed": QColor(220, 60, 60),
    "stopped": QColor(130, 130, 130),
}


def setup_high_dpi():
    """Before QApplication: no scaling, so a marker of N pixels is N device pixels."""
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ.pop("QT_SCALE_FACTOR", None)
    QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)


def assert_unscaled(widget):
    """Markers are drawn 1:1, so a scaled screen would make every length wrong."""
    ratio = widget.devicePixelRatioF()
    if abs(ratio - 1.0) > 1e-6:
        raise RuntimeError(f"Device pixel ratio is {ratio}, expected 1 (high-DPI scaling must be off)")


def to_qimage(frame):
    """BGR or grey uint8 array -> QImage that owns a copy (safe after the array changes)."""
    if frame is None:
        return None
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    if frame.ndim == 2:
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_Grayscale8)
    else:
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_BGR888)
    return image.copy()


def fit_rect(image_width, image_height, area):
    """The largest rectangle with the image's aspect ratio centred in area (QRectF)."""
    if image_width <= 0 or image_height <= 0:
        return QRectF(area)
    scale = min(area.width() / image_width, area.height() / image_height)
    width, height = image_width * scale, image_height * scale
    return QRectF(area.x() + (area.width() - width) / 2, area.y() + (area.height() - height) / 2,
                  width, height)


class FrameView(QWidget):
    """Paints the latest frame aspect-fitted on black; subclasses add overlays in paint_overlay."""

    def __init__(self, placeholder="No signal", parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self._image = None
        self.image_rect = QRectF()
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def set_frame(self, frame):
        self._image = to_qimage(frame)
        self.update()

    def to_widget(self, x, y):
        """Image pixel -> widget coordinates (for overlays drawn over the frame)."""
        if self._image is None or self.image_rect.isEmpty():
            return None
        return (self.image_rect.x() + x * self.image_rect.width() / self._image.width(),
                self.image_rect.y() + y * self.image_rect.height() / self._image.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._image is None:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            self.image_rect = QRectF()
        else:
            self.image_rect = fit_rect(self._image.width(), self._image.height(), QRectF(self.rect()))
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.image_rect, self._image)
        self.paint_overlay(painter)
        painter.end()

    def paint_overlay(self, painter):
        pass


class Banner(QLabel):
    """A coloured line across the top of a page; show_message(..., seconds=None) stays up."""

    STYLES = {
        "info": "background:#1f4e79;color:white;",
        "warning": "background:#8a5a00;color:white;",
        "error": "background:#8b1e1e;color:white;",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.hide()
        self._timer = QTimer(self, singleShot=True, timeout=self.hide)

    def show_message(self, text, level="info", seconds=None):
        self.setText(text)
        self.setStyleSheet(self.STYLES.get(level, self.STYLES["info"]) + "padding:6px;font-size:15px;")
        self.show()
        self.raise_()
        self._timer.stop()
        if seconds is not None:
            self._timer.start(int(seconds * 1000))


def confirm(parent, title, text):
    return QMessageBox.question(parent, title, text, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes


class _TaskSignals(QObject):
    done = pyqtSignal(object, object)  # (result, exception)


class _Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _TaskSignals()

    def run(self):
        try:
            result, error = self.fn(), None
        except Exception as exc:  # Handed to on_done: a failed solve is a normal outcome
            result, error = None, exc
        self.signals.done.emit(result, error)


_running = set()  # Keeps each task's signals alive until it reports


def run_in_pool(fn, on_done):
    """Run fn() on the thread pool; on_done(result, exception) runs back on the UI thread."""
    task = _Task(fn)
    task.setAutoDelete(False)

    def finished(result, error):
        _running.discard(task)
        on_done(result, error)

    task.signals.done.connect(finished)
    _running.add(task)
    QThreadPool.globalInstance().start(task)


class Page(QWidget):
    """A screen of the main window; the controller calls these hooks on the visible page."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

    def enter(self):
        pass

    def leave(self):
        pass

    def on_bundle(self, bundle):
        pass

    def on_tick(self):
        """About 4 times a second: camera states and anything slow."""

    def handle_key(self, key, text):
        """True if the page used the key."""
        return False
