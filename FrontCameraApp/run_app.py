"""Start the front-camera gaze app.

    python FrontCameraApp/run_app.py                      the cameras saved in settings/camera_setup.json
    python FrontCameraApp/run_app.py --video left=eye.mp4 --video right=eye.mp4 --video front=front.mp4
    python FrontCameraApp/run_app.py --smoke 20           quit after 20 s and print where it got to

Environment: GAZECORE_DATA_DIR (calibration files, default FrontCameraTracker/) and
FRONTCAM_SETTINGS_DIR (app settings, default FrontCameraApp/settings/).
"""

import argparse
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from frontcam_app.settings import LOG_DIR  # noqa: E402
from frontcam_app.ui.common import setup_high_dpi  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", action="append", default=[], metavar="ROLE=PATH",
                        help="play a video file as the left, right or front camera (not saved)")
    parser.add_argument("--smoke", type=float, metavar="SECONDS", help="quit after SECONDS and print the state")
    args = parser.parse_args(argv)
    videos = {}
    for item in args.video:
        role, _, path = item.partition("=")
        if role not in ("left", "right", "front") or not path:
            parser.error(f"--video {item}: expected left=PATH, right=PATH or front=PATH")
        videos[role] = os.path.abspath(path)
    return args, videos


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, time.strftime("app_%Y%m%d.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()])
    return path


def main(argv=None):
    args, videos = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = setup_logging()
    setup_high_dpi()
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    from frontcam_app.controller import Controller

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Front camera gaze")
    app.setQuitOnLastWindowClosed(False)  # The overlay hides the main window; quitting is explicit
    controller = Controller(app, videos)
    logging.getLogger(__name__).info("Log file: %s", log_path)
    controller.window.destroyed.connect(app.quit)
    app.aboutToQuit.connect(controller.shutdown)
    original_close = controller.close_requested

    def close_requested():
        if original_close():
            app.quit()
            return True
        return False

    controller.close_requested = close_requested
    if args.smoke:
        def report():
            print(f"SMOKE mode={controller.mode} page={type(controller.page).__name__} "
                  f"calibrated={controller.solved is not None} "
                  f"validity={controller.validity} "
                  f"fps={0 if controller.bundle is None else round(controller.bundle.processing_fps)}", flush=True)
            app.quit()

        QTimer.singleShot(int(args.smoke * 1000), report)
    controller.start()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
