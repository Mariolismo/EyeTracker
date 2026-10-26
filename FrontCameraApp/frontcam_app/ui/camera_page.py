"""Which camera is which: three role panels with live previews, flip/mirror and status.

Choosing a device previews it at once (the reader opens in the background), but nothing is
kept until Apply: only then are the setup saved and the pipeline's orientation changed, after
confirming what that invalidates. Cancel puts the previous cameras back.
"""

import copy

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from gazecore.camera.camera_setup import ROLE_LABELS, ROLES, RoleBinding, moved_port_candidates

from ..pipeline import orient
from .common import FrameView, Page

VIDEO_FILE, NONE = "video", "none"


class RolePanel(QWidget):
    def __init__(self, page, role):
        super().__init__()
        self.page, self.role = page, role
        self.binding = None
        title = QLabel(ROLE_LABELS[role])
        title.setStyleSheet("font-size:20px;font-weight:bold;")
        self.combo = QComboBox()
        self.combo.activated.connect(self._chosen)
        self.preview = FrameView("No camera")
        self.preview.setMinimumSize(320, 240)
        self.flip = QCheckBox("Upside down (flip)")
        self.mirror = QCheckBox("Mirror left-right")
        for box in (self.flip, self.mirror):
            box.toggled.connect(self._orientation_changed)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        for widget in (title, self.combo, self.preview, self.flip, self.mirror, self.status):
            layout.addWidget(widget)
        layout.setStretchFactor(self.preview, 1)

    def load(self, binding, devices, claimed):
        self.binding = copy.deepcopy(binding)
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("(none)", NONE)
        candidates = {d["device_id"].upper() for d in moved_port_candidates(binding, devices, claimed)}
        for device in devices or []:
            label = f"[{device['index']}] {device['name']}"
            if device["device_id"].upper() in candidates:
                label += "  (same model as the saved camera: moved USB port?)"
            self.combo.addItem(label, device)
        if devices is None:
            for index in range(4):
                self.combo.addItem(f"Camera index {index}", {"index": index, "device_id": f"index:{index}",
                                                            "name": f"Camera {index}"})
        self.combo.addItem("Video file...", VIDEO_FILE)
        self.combo.setCurrentIndex(self._index_of(binding))
        self.combo.blockSignals(False)
        self.flip.blockSignals(True)
        self.mirror.blockSignals(True)
        self.flip.setChecked(bool(binding and binding.flip))
        self.mirror.setChecked(bool(binding and binding.mirror))
        self.flip.blockSignals(False)
        self.mirror.blockSignals(False)

    def _index_of(self, binding):
        if binding is None:
            return 0
        if binding.video_file:
            self.combo.insertItem(self.combo.count() - 1, f"File: {binding.video_file}", {"video_file": binding.video_file})
            return self.combo.count() - 2
        for i in range(self.combo.count()):
            data = self.combo.itemData(i)
            if isinstance(data, dict) and (data.get("device_id") or "").upper() == (binding.device_id or "").upper():
                return i
        return 0

    def _chosen(self, index):
        data = self.combo.itemData(index)
        if data == NONE:
            new = None
        elif data == VIDEO_FILE:
            path, _ = QFileDialog.getOpenFileName(self, "Video file", "", "Videos (*.mp4 *.avi *.mkv);;All files (*)")
            if not path:
                self.combo.setCurrentIndex(self._index_of(self.binding))
                return
            new = RoleBinding(name="video", video_file=path, flip=self.flip.isChecked(), mirror=self.mirror.isChecked())
        elif "video_file" in data:
            new = RoleBinding(name="video", video_file=data["video_file"], flip=self.flip.isChecked(),
                              mirror=self.mirror.isChecked())
        else:
            other = self.page.role_using(data["device_id"], self.role)
            if other is not None:
                QMessageBox.warning(self, "Camera in use", f"That camera is already the {ROLE_LABELS[other]} camera. "
                                    "Choose another one, or first set that role to (none).")
                self.combo.setCurrentIndex(self._index_of(self.binding))
                return
            new = RoleBinding(data["device_id"], data["name"], data["index"], self.flip.isChecked(),
                              self.mirror.isChecked())
        self.binding = new
        self.page.controller.preview_bind(self.role, new)

    def _orientation_changed(self):
        if self.binding is not None:
            self.binding.flip, self.binding.mirror = self.flip.isChecked(), self.mirror.isChecked()

    def refresh(self, reader):
        packet = None if reader is None else reader.latest()
        self.preview.set_frame(None if packet is None else orient(packet.frame, self.binding))
        if reader is None:
            self.status.setText("Not connected")
            return
        info = reader.snapshot()
        size = "x".join(map(str, info["native_size"])) if info["native_size"] else "?"
        text = f"{info['status']}  {info['fps']:.0f} fps  {size}  {info['backend'] or ''}"
        if info["message"]:
            text += f"\n{info['message']}"
        self.status.setText(text)


class CameraPage(Page):
    def __init__(self, controller):
        super().__init__(controller)
        heading = QLabel("Cameras: check that each view shows the right camera, upright")
        heading.setStyleSheet("font-size:24px;font-weight:bold;padding:8px;")
        hint = QLabel("The eye views should show the eye with the eyelashes at the top. Your left eye is the "
                      "Left eye camera. Changing a camera or its orientation invalidates the calibrations "
                      "made with it.")
        hint.setWordWrap(True)
        self.panels = {role: RolePanel(self, role) for role in ROLES}
        grid = QGridLayout()
        for column, role in enumerate(ROLES):
            grid.addWidget(self.panels[role], 0, column)
        self.refresh_button = QPushButton("Rescan devices")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setDefault(True)
        self.refresh_button.clicked.connect(self.rescan)
        self.cancel_button.clicked.connect(self.cancel)
        self.apply_button.clicked.connect(self.apply)
        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.apply_button)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(hint)
        layout.addLayout(grid, 1)
        layout.addLayout(buttons)
        self.setStyleSheet("QLabel{font-size:15px;} QPushButton{font-size:16px;padding:8px 18px;}")
        self.timer = QTimer(self, interval=66, timeout=self.refresh)  # Previews at ~15 fps

    def leave(self):
        self.timer.stop()

    def enter(self):
        self.original = copy.deepcopy(self.controller.setup)
        self._load()
        self.timer.start()

    def _load(self):
        setup = self.controller.setup
        claimed = [b.device_id for b in setup.bound().values()]
        for role, panel in self.panels.items():
            panel.load(setup.roles.get(role), self.controller.devices, claimed)

    def rescan(self):
        self.controller.refresh_devices()
        self._load()

    def role_using(self, device_id, except_role):
        for role, panel in self.panels.items():
            binding = panel.binding
            if role != except_role and binding is not None and (binding.device_id or "").upper() == device_id.upper():
                return role
        return None

    def bindings(self):
        return {role: panel.binding for role, panel in self.panels.items()}

    def cancel(self):
        self.controller.cancel_camera_setup(self.original)

    def apply(self):
        self.controller.apply_camera_setup(self.bindings())

    def refresh(self):
        for role, panel in self.panels.items():
            panel.refresh(self.controller.manager.get(role))

    def handle_key(self, key, text):
        if key == Qt.Key_Escape:
            self.cancel()
            return True
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.apply()
            return True
        return False
