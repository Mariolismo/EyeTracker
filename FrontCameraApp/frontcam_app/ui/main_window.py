"""The one full-screen window: a stack of pages, the wizard's step bar and a banner.

Keys go through an application-wide event filter to the controller, so they work whatever
widget has focus (the old tracker only saw keys while a cv2 window had it). Text fields keep
their typing keys.
"""

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit, QStackedWidget, QVBoxLayout, QWidget

from .common import Banner
from .wizard_pages import StepBar

PASS_TO_TEXT_FIELDS = (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_F1)


class KeyFilter(QObject):
    def __init__(self, controller, window):
        super().__init__(window)
        self.controller, self.window = controller, window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress or not self.window.isActiveWindow():
            return False
        if obj is not QApplication.focusWidget() and obj is not self.window:
            return False  # The same event comes again for each parent widget: handle it once
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)) and event.key() not in PASS_TO_TEXT_FIELDS:
            return False
        if event.isAutoRepeat() and event.key() not in (Qt.Key_Left, Qt.Key_Right):
            return True
        return self.controller.handle_key(event.key(), event.text())


class MainWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Front camera gaze")
        self.setStyleSheet("background:#f4f4f4;")
        self.steps = StepBar()
        self.steps.setStyleSheet("font-size:16px;padding:6px 12px;background:#e6e6e6;")
        self.steps.hide()
        self.stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.steps)
        layout.addWidget(self.stack, 1)
        self.banner = Banner(self)
        self.key_filter = KeyFilter(controller, self)
        QApplication.instance().installEventFilter(self.key_filter)

    def add_page(self, page):
        self.stack.addWidget(page)

    def show_page(self, page, steps=None, current=None):
        """steps: the wizard's step list to show above the page, or None to hide the bar."""
        if steps:
            self.steps.show_steps(steps, current)
            self.steps.show()
        else:
            self.steps.hide()
        self.stack.setCurrentWidget(page)
        page.setFocus()

    def resizeEvent(self, event):
        self.banner.setGeometry(0, 0, self.width(), 44)

    def closeEvent(self, event):
        if self.controller.close_requested():
            event.accept()
        else:
            event.ignore()
