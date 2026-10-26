"""List the cameras and stream them all through CameraManager, printing each one's status.

    python -m gazecore.camera.probe [seconds]

Unplug and replug a camera while it runs to see the reconnection.
"""

import logging
import sys
import time

from . import camera_reader, win_cameras


def main(seconds=30.0):
    logging.basicConfig(level=logging.INFO, format="%(relativeCreated)7.0f ms  %(message)s")
    devices = win_cameras.list_capture_devices()
    if devices is None:
        print("Device enumeration is unavailable here; probing indices 0-2.")
        devices = [{"index": i, "device_id": f"index:{i}", "name": f"Camera {i}"} for i in range(3)]
    for device in devices:
        print(f"[{device['index']}] {device['name']}: {device['device_id']}")
    manager = camera_reader.CameraManager()
    for device in devices:
        manager.bind(f"cam{device['index']}", device["index"], device["device_id"])
    try:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            time.sleep(2.0)
            print("  ".join(f"{role}: {r.status} {r.fps:4.1f} fps {r.native_size or ''}"
                            for role, r in manager.readers.items()))
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
