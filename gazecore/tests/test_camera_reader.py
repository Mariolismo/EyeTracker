"""CameraReader against scripted fake captures: hangs, failure bursts, frozen frames, replugging.

Run: python gazecore/tests/test_camera_reader.py   (or pytest)
"""

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gazecore.camera import camera_reader as cr  # noqa: E402

FAST = cr.CameraTimings(open_delay=0.05, open_timeout=0.6, read_stall=0.3, fail_threshold=5,
                        reconnect_cooldown=0.1, frozen_after=0.3, frozen_reconnect=0.8, watchdog_period=0.05)
DEVICE = "USB\\VID_0C45&PID_6366&MI_00\\8&183AF011&0&0000"


class FakeDevice:
    """What the fake 'driver' does; tests change it while readers run."""

    def __init__(self):
        self.present = True
        self.index = 1
        self.frozen = False
        self.fail_reads = 0  # Next reads that fail
        self.hang_next_read = False
        self.hang_next_open = False
        self.opens = 0
        self.release_event = threading.Event()

    def devices(self):
        return [{"index": self.index, "device_id": DEVICE, "name": "USB Camera"}] if self.present else []

    def factory(self, index, backend):
        return FakeCapture(self, index)


class FakeCapture:
    def __init__(self, device, index):
        self.device = device
        self.opened = device.present and index == device.index
        device.opens += 1
        if device.hang_next_open:
            device.hang_next_open = False
            device.release_event.wait(5)
        self.count = 0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        time.sleep(0.005)
        device = self.device
        if device.hang_next_read:
            device.hang_next_read = False
            device.release_event.wait(5)
        if not device.present or not self.opened:
            return False, None
        if device.fail_reads:
            device.fail_reads -= 1
            return False, None
        if not device.frozen:
            self.count += 1
        return True, np.full((480, 640, 3), self.count % 256, np.uint8)

    def release(self):
        self.opened = False


def wait_for(condition, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def make_reader(device):
    manager = cr.CameraManager(FAST, capture_factory=device.factory, enumerate_devices=device.devices)
    return manager, manager.bind("left", 1, DEVICE)


def test_frames_arrive_in_sequence():
    device = FakeDevice()
    manager, reader = make_reader(device)
    try:
        first = reader.wait_new(0, 2.0)
        second = reader.wait_new(first.seq, 1.0)
        assert first is not None and second.seq > first.seq
        assert reader.native_size == (640, 480) and not second.frame.flags.writeable
    finally:
        manager.stop_all()


def test_failure_burst_reconnects():
    device = FakeDevice()
    manager, reader = make_reader(device)
    try:
        assert wait_for(lambda: reader.status == cr.OK)
        device.fail_reads = 10
        assert wait_for(lambda: reader.status == cr.RECONNECTING)
        assert wait_for(lambda: reader.status == cr.OK)
        assert device.opens >= 2
    finally:
        manager.stop_all()


def test_unplugged_device_stays_absent_then_comes_back_at_a_new_index():
    device = FakeDevice()
    manager, reader = make_reader(device)
    try:
        assert wait_for(lambda: reader.status == cr.OK)
        device.present = False
        assert wait_for(lambda: reader.status == cr.ABSENT)
        assert reader.latest() is None
        device.index, device.present = 3, True
        assert wait_for(lambda: reader.status == cr.OK)
        assert reader.capture_index == 3
    finally:
        manager.stop_all()


def test_hung_read_is_abandoned():
    device = FakeDevice()
    manager, reader = make_reader(device)
    try:
        assert wait_for(lambda: reader.status == cr.OK)
        device.hang_next_read = True
        assert wait_for(lambda: reader.status == cr.STALLED)
        assert wait_for(lambda: reader.status == cr.OK)
    finally:
        device.release_event.set()
        manager.stop_all()


def test_hung_open_does_not_block_other_cameras():
    device, other = FakeDevice(), FakeDevice()
    device.hang_next_open = True
    manager = cr.CameraManager(FAST, enumerate_devices=lambda: None)
    try:
        stuck = manager.bind("left", 1, capture_factory=device.factory)
        assert wait_for(lambda: stuck.status == cr.OPENING)
        fine = manager.bind("right", 1, capture_factory=other.factory)
        assert wait_for(lambda: fine.status == cr.OK, timeout=4.0)
        assert wait_for(lambda: stuck.status == cr.OK, timeout=4.0)  # Reopened on a fresh thread
    finally:
        device.release_event.set()
        manager.stop_all()


def test_frozen_frames_are_withheld_then_reconnected():
    device = FakeDevice()
    manager, reader = make_reader(device)
    try:
        assert wait_for(lambda: reader.status == cr.OK)
        device.frozen = True
        assert wait_for(lambda: reader.status == cr.FROZEN)
        assert reader.latest() is None
        opens = device.opens
        assert wait_for(lambda: device.opens > opens)
        device.frozen = False
        assert wait_for(lambda: reader.status == cr.OK)
    finally:
        manager.stop_all()


def test_stop_does_not_wait_for_the_driver():
    device = FakeDevice()
    manager, reader = make_reader(device)
    assert wait_for(lambda: reader.status == cr.OK)
    device.hang_next_read = True
    time.sleep(0.05)
    start = time.perf_counter()
    manager.stop_all()
    assert time.perf_counter() - start < 0.1 and reader.status == cr.OFFLINE
    device.release_event.set()


def test_gate_opens_one_at_a_time():
    gate = cr.OpenGate(delay=0.05, lease_timeout=1.0)
    stop = threading.Event()
    active, peak = [0], [0]

    def worker(name):
        assert gate.acquire(name, stop)
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        active[0] -= 1
        gate.release(name)

    threads = [threading.Thread(target=worker, args=(f"cam{i}",)) for i in range(3)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1 and time.perf_counter() - start >= 2 * 0.05


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
    print("All camera reader tests passed.")
