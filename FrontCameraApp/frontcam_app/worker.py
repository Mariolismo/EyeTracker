"""The processing thread: pulls the newest camera frames, steps the Pipeline, publishes Bundles.

Only this thread touches the Pipeline (eye trackers, gaze mapper). The UI sends commands with
call(fn, on_done): fn(pipeline) runs between two steps and on_done(result) back on the UI thread.
Bundles never queue up: a new one is announced only once the UI took the previous one.
"""

import logging
import queue
import threading
import time

from PyQt5.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

EYE_WAIT_S = 0.06  # Wait this long for the next eye frame before stepping without it
IDLE_S = 0.03  # Without eye cameras, step at about the front camera's rate


class Worker(QThread):
    bundle_ready = pyqtSignal()
    command_done = pyqtSignal(object, object)  # (on_done, result)

    def __init__(self, pipeline, cameras):
        super().__init__()
        self.pipeline = pipeline
        self.cameras = cameras  # CameraManager
        self._commands = queue.Queue()
        self._stop = threading.Event()
        self._latest = None
        self._announced = False
        self._lock = threading.Lock()
        self.paused = False  # While choosing cameras: preview frames must not reach the eye trackers
        self.command_done.connect(lambda on_done, result: on_done(result))

    def call(self, fn, on_done=None):
        self._commands.put((fn, on_done))

    def take_bundle(self):
        """The newest Bundle (or None); lets the next one be announced."""
        with self._lock:
            bundle, self._announced = self._latest, False
            return bundle

    def stop(self):
        self._stop.set()
        self.wait(3000)

    def _run_commands(self):
        while True:
            try:
                fn, on_done = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                result = fn(self.pipeline)
            except Exception:  # A failing command must not kill tracking
                log.exception("Command failed")
                result = None
            if on_done is not None:
                self.command_done.emit(on_done, result)

    def run(self):
        seen = {}
        while not self._stop.is_set():
            self._run_commands()
            if self.paused:
                time.sleep(IDLE_S)
                continue
            # The eye cameras set the pace: wait for a new frame from each, briefly.
            deadline = time.perf_counter() + EYE_WAIT_S
            packets, waited = {}, False
            for role in ("left", "right"):
                reader = self.cameras.get(role)
                if reader is None:
                    continue
                waited = True
                packet = reader.wait_new(seen.get(role, 0), max(0.0, deadline - time.perf_counter()))
                packet = packet or reader.latest()
                if packet is not None:
                    packets[role], seen[role] = packet, packet.seq
            front = self.cameras.get("front")
            packets["front"] = None if front is None else front.latest()
            if not waited:
                time.sleep(IDLE_S)
            try:
                bundle = self.pipeline.step(packets)
            except Exception:
                log.exception("Processing step failed")
                time.sleep(0.1)
                continue
            with self._lock:
                self._latest = bundle
                announce = not self._announced
                self._announced = True
            if announce:
                self.bundle_ready.emit()
