"""Threaded camera capture that survives unplugging, driver hangs and frozen streams.

Ported from GazeScreen3D's camera_io (via MultiCamGaze3D), with these changes:
- One thread owns each capture: open, read and release never run on the UI thread.
- Opens go one at a time through an OpenGate, 0.8 s apart: back-to-back MSMF opens on a
  shared USB 2.0 link fail to grab frames. A lease keeps a hung open from blocking the others.
- A watchdog abandons a read or open stuck in the driver and starts a fresh thread.
- Identical frames for seconds (a stuck driver, a covered lens) are withheld, then reconnected.
- The capture size is NOT forced: like the front-camera tracker, only 30 fps is requested, so
  frames (and every calibration made from them) stay what they always were. The native size
  is recorded at each open.
- MSMF only on Windows: with 3 cameras on one USB 2.0 link, DirectShow starves to ~1 fps and
  serves other cameras' buffers (GazeScreen3D diag_multicam.py).
"""

import logging
import sys
import threading
import time
import zlib
from dataclasses import dataclass

import cv2

from . import win_cameras

log = logging.getLogger(__name__)

CAPTURE_TARGET_FPS = 30
CAMERA_CAPTURE_MODES = (("MSMF", cv2.CAP_MSMF),) if sys.platform == "win32" else (("Auto", cv2.CAP_ANY),)

# Status values
OPENING, OK, FROZEN, STALLED, RECONNECTING, ABSENT, FAILED, OFFLINE = (
    "OPENING", "OK", "FROZEN", "STALLED", "RECONNECTING", "ABSENT", "FAILED", "OFFLINE")


@dataclass
class CameraTimings:
    open_delay: float = 0.8  # Pause between two opens on the shared USB link
    open_timeout: float = 20.0  # The rig's front camera takes ~12 s to open over MSMF
    read_stall: float = 2.0  # A read blocking this long is a hung driver
    fail_threshold: int = 20  # Failed reads in a row before reconnecting (~1 s)
    reconnect_cooldown: float = 1.5  # Also rate-limits device enumeration while unplugged
    frozen_after: float = 2.0  # Identical frames this long: withheld
    frozen_reconnect: float = 6.0  # ... and this long: reconnect
    watchdog_period: float = 0.25


@dataclass(frozen=True)
class FramePacket:
    seq: int  # Increases by one per frame of this reader
    t: float  # time.perf_counter() when the frame arrived
    frame: object  # BGR image, read-only: copy before drawing on it
    native_size: tuple  # (width, height) as delivered


class OpenGate:
    """Lets one camera open at a time, in request order, with a pause after each."""

    def __init__(self, delay=0.8, lease_timeout=20.0):
        self.delay = delay
        self.lease_timeout = lease_timeout
        self._cond = threading.Condition()
        self._queue = []
        self._holder = None
        self._lease_until = 0.0
        self._free_at = 0.0

    def acquire(self, owner, stop_event):
        """Blocks until owner may open; False if stop_event was set first."""
        with self._cond:
            self._queue.append(owner)
            try:
                while not stop_event.is_set():
                    now = time.perf_counter()
                    if self._holder is not None and now > self._lease_until:
                        log.warning("Camera open by %s timed out; letting the next camera open.", self._holder)
                        self._holder, self._free_at = None, now
                    if self._holder is None and self._queue[0] is owner and now >= self._free_at:
                        self._holder, self._lease_until = owner, now + self.lease_timeout
                        return True
                    self._cond.wait(0.05)
                return False
            finally:
                self._queue.remove(owner)

    def release(self, owner):
        with self._cond:
            if self._holder is owner:
                self._holder = None
                self._free_at = time.perf_counter() + self.delay
            self._cond.notify_all()


def _open_capture(source, backend):
    return cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)


class CameraReader:
    """One camera (or a looping video file) read on its own thread.

    source: a capture index, or a video file path (played in a loop at its own frame rate).
    device_id: the stable Windows device ID; the index is re-resolved from it at every open,
    and the reader stays ABSENT, never grabbing another camera, while that device is gone.
    """

    def __init__(self, role, source, device_id=None, gate=None, timings=None, fps_request=CAPTURE_TARGET_FPS,
                 size=None, capture_factory=_open_capture, enumerate_devices=win_cameras.list_capture_devices):
        self.role = role
        self.source = source
        self.is_file = isinstance(source, str)
        self.device_id = device_id
        self.timings = timings or CameraTimings()
        self.gate = gate or OpenGate(self.timings.open_delay, self.timings.open_timeout)
        self.fps_request = fps_request
        self.size = size  # (w, h) to request; None keeps the camera's own
        self.capture_factory = capture_factory
        self.enumerate_devices = enumerate_devices
        self.capture_index = None if self.is_file else source
        self.backend_name = None
        self.native_size = None
        self.fps = 0.0
        self.message = ""
        self._status = OFFLINE
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._generation = 0
        self._in_call_since = None  # perf_counter() when a driver call started, else None
        self._in_call_limit = 0.0  # How long that call may take
        self._latest = None
        self._seq = 0
        self._last_crc = None
        self._last_change = 0.0
        self._reconnect_requested = False
        self._thread = None

    def __repr__(self):
        return f"{self.role} camera {self.source}"

    # --- Public ---

    @property
    def status(self):
        return self._status

    def start(self):
        self._stop.clear()
        with self._lock:
            self._generation += 1
            generation = self._generation
        self._thread = threading.Thread(target=self._run, args=(generation, 0.0), daemon=True,
                                        name=f"camera-{self.role}")
        self._thread.start()

    def stop(self):
        """Non-blocking: the thread releases its capture when its current call returns."""
        self._stop.set()
        with self._lock:
            self._generation += 1
            self._set_status(OFFLINE)
            self._latest = None
            self._lock.notify_all()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def latest(self):
        """The newest FramePacket, or None while the camera is not delivering."""
        with self._lock:
            return self._latest if self._status == OK else None

    def wait_new(self, after_seq, timeout):
        """The newest packet with seq > after_seq, waiting up to timeout; None if none arrived."""
        deadline = time.perf_counter() + timeout
        with self._lock:
            while True:
                packet = self._latest if self._status == OK else None
                if packet is not None and packet.seq > after_seq:
                    return packet
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._lock.wait(remaining)

    def snapshot(self):
        """Status line data for the UI."""
        return {"status": self._status, "fps": self.fps, "native_size": self.native_size,
                "backend": self.backend_name, "index": self.capture_index, "message": self.message}

    def check(self, now=None):
        """Watchdog: called periodically from another thread."""
        now = time.perf_counter() if now is None else now
        with self._lock:
            since, limit = self._in_call_since, self._in_call_limit
            if since is not None and now - since > limit and not self._stop.is_set():
                # The driver call hangs: abandon that thread (it releases its capture if the
                # call ever returns) and reopen on a fresh one after a cooldown.
                self._generation += 1
                generation = self._generation
                self._in_call_since = None
                self._latest = None
                self._set_status(STALLED, "driver not responding; if it persists, replug the camera")
                self._thread = threading.Thread(target=self._run, args=(generation, self.timings.reconnect_cooldown),
                                                daemon=True, name=f"camera-{self.role}")
                self._thread.start()
                return
            if self._status in (OK, FROZEN) and not self.is_file:
                unchanged = now - self._last_change
                if unchanged > self.timings.frozen_reconnect:
                    self._reconnect_requested = True
                elif unchanged > self.timings.frozen_after and self._status == OK:
                    self._set_status(FROZEN, "image not changing (lens covered?)")

    # --- Thread ---

    def _set_status(self, status, message=""):
        if status != self._status:
            log.info("%s: %s%s", self, status, f" ({message})" if message else "")
        self._status, self.message = status, message
        self._lock.notify_all()

    def _current(self, generation):
        return generation == self._generation and not self._stop.is_set()

    def _resolve_index(self):
        """The capture index to open now, or None if the device is absent."""
        if self.is_file or not self.device_id or self.device_id.startswith("index:"):
            return self.source if self.is_file else self.capture_index
        devices = self.enumerate_devices()
        if devices is None:  # Cannot enumerate: trust the index
            return self.capture_index
        return win_cameras.find_index_for_device_id(devices, self.device_id)

    def _open(self, generation):
        index = self._resolve_index()
        if index is None:
            with self._lock:
                if self._current(generation):
                    self._set_status(ABSENT, "device unplugged")
            return None
        if not self.is_file and index != self.capture_index:
            log.info("%s: index moved %s -> %s", self, self.capture_index, index)
            self.capture_index = index
        if not self.gate.acquire(self, self._stop):
            return None
        try:
            with self._lock:
                if not self._current(generation):
                    return None
                if self._status not in (STALLED, RECONNECTING):
                    self._set_status(OPENING)
                self._in_call_since, self._in_call_limit = time.perf_counter(), self.timings.open_timeout
            for mode_name, backend in ((("File", None),) if self.is_file else CAMERA_CAPTURE_MODES):
                cap = self.capture_factory(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                if self.size is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                if not self.is_file:
                    cap.set(cv2.CAP_PROP_FPS, self.fps_request)
                ok, frame = cap.read()
                with self._lock:
                    self._in_call_since = None
                    if not self._current(generation):
                        cap.release()
                        return None
                    if ok and frame is not None:
                        self.backend_name = mode_name
                        self.native_size = (frame.shape[1], frame.shape[0])
                        self._last_crc = None
                        self._publish(frame)
                        self._set_status(OK)
                        return cap
                cap.release()
            with self._lock:
                self._in_call_since = None
                if self._current(generation):
                    self._set_status(FAILED, "could not grab a frame")
            return None
        finally:
            self.gate.release(self)

    def _publish(self, frame):
        """With self._lock held."""
        now = time.perf_counter()
        crc = zlib.crc32(frame[::8, ::8].tobytes())
        if crc != self._last_crc:
            self._last_crc, self._last_change = crc, now
            if self._status == FROZEN:
                self._set_status(OK)
        frame.flags.writeable = False
        self._seq += 1
        self._latest = FramePacket(self._seq, now, frame, (frame.shape[1], frame.shape[0]))
        self._lock.notify_all()

    def _run(self, generation, initial_delay):
        cap = None
        fail_count = 0
        frames, window_start = 0, time.perf_counter()
        interval = 1.0 / self.fps_request
        if initial_delay:
            self._stop.wait(initial_delay)
        try:
            while self._current(generation):
                if cap is None:
                    cap = self._open(generation)
                    if cap is None:
                        self._stop.wait(self.timings.reconnect_cooldown)
                        continue
                    fail_count = 0
                    if self.is_file:
                        interval = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or self.fps_request)
                if self._reconnect_requested:
                    self._reconnect_requested = False
                    cap.release()
                    cap = None
                    with self._lock:
                        self._set_status(RECONNECTING, "image not changing")
                        self._latest = None
                    self._stop.wait(self.timings.reconnect_cooldown)
                    continue
                loop_start = time.perf_counter()
                with self._lock:
                    self._in_call_since, self._in_call_limit = loop_start, self.timings.read_stall
                ok, frame = cap.read()
                with self._lock:
                    self._in_call_since = None
                    if not self._current(generation):
                        break
                    if ok and frame is not None:
                        fail_count = 0
                        self._publish(frame)
                if not ok or frame is None:
                    if self.is_file:  # Loop the video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        fail_count += 1
                        if fail_count < 3:
                            continue
                    fail_count += 1
                    if fail_count > 5:
                        with self._lock:
                            self._latest = None  # Stop serving stale frames
                    if fail_count >= self.timings.fail_threshold:
                        cap.release()
                        cap = None
                        with self._lock:
                            self._set_status(RECONNECTING, "no frames")
                        self._stop.wait(self.timings.reconnect_cooldown)
                    else:
                        time.sleep(0.05)
                    continue
                frames += 1
                now = time.perf_counter()
                if now - window_start >= 1.0:
                    self.fps = frames / (now - window_start)
                    frames, window_start = 0, now
                if self.is_file:
                    sleep = interval - (time.perf_counter() - loop_start)
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            if cap is not None:
                cap.release()


class CameraManager:
    """The cameras by role, sharing one OpenGate, supervised by one watchdog thread."""

    def __init__(self, timings=None, **reader_kwargs):
        self.timings = timings or CameraTimings()
        self.gate = OpenGate(self.timings.open_delay, self.timings.open_timeout)
        self.reader_kwargs = reader_kwargs
        self.readers = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="camera-watchdog")
        self._watchdog.start()

    def bind(self, role, source, device_id=None, **kwargs):
        """(Re)start the role's reader on source; the previous one is stopped without waiting."""
        reader = CameraReader(role, source, device_id, gate=self.gate, timings=self.timings,
                              **{**self.reader_kwargs, **kwargs})
        with self._lock:
            old = self.readers.get(role)
            self.readers[role] = reader
        if old is not None:
            old.stop()
        reader.start()
        return reader

    def unbind(self, role):
        with self._lock:
            reader = self.readers.pop(role, None)
        if reader is not None:
            reader.stop()

    def get(self, role):
        return self.readers.get(role)

    def stop_all(self):
        self._stop.set()
        with self._lock:
            readers, self.readers = list(self.readers.values()), {}
        for reader in readers:
            reader.stop()

    def _watch(self):
        while not self._stop.wait(self.timings.watchdog_period):
            with self._lock:
                readers = list(self.readers.values())
            for reader in readers:
                reader.check()
