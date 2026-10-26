"""Drive the whole app offscreen on video files: every page is shown, fed bundles and painted.

Run by test_app_smoke.py in a subprocess (its own data and settings folders). Prints one
"TOUR ..." line per check and "TOUR OK" at the end; any exception exits non-zero.

    python FrontCameraApp/tests/app_tour.py LEFT.mp4 RIGHT.mp4 FRONT.mp4 EXPECT
EXPECT: "tracking" (valid calibration) or the first wizard steps, such as "screen,intrinsics".
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from frontcam_app.ui.common import setup_high_dpi  # noqa: E402

setup_high_dpi()

from PyQt5.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from frontcam_app import controller as controller_module  # noqa: E402
from frontcam_app.calibration_session import CalibrationSession  # noqa: E402
from frontcam_app.ui.target_canvas import TargetCanvas  # noqa: E402
from frontcam_app.ui.wizard_pages import TargetPage  # noqa: E402

controller_module.confirm = lambda *args: True  # No modal dialogs offscreen


def wait(app, seconds):
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec_()


def main(left, right, front, expect):
    app = QApplication(sys.argv[:1])
    controller = controller_module.Controller(app, {"left": left, "right": right, "front": front})
    controller.start()
    for _ in range(80):
        wait(app, 0.25)
        if controller.mode in ("tracking", "wizard"):
            break
    # In the wizard: its first steps, e.g. "screen,intrinsics" (an unconfirmed screen scale comes first).
    reached = "tracking" if controller.mode == "tracking" else ",".join(
        controller.wizard_steps[controller.wizard_index:][:len(expect.split(","))])
    print(f"TOUR reached {reached}", flush=True)
    assert reached == expect, (reached, expect)

    solved = controller.solved
    states = {eye: {"model_center": [320, 240], "sphere_radius": 150.0, "sphere_center": None,
                    "adjustment_enabled": False} for eye in ("left", "right")}
    controller.session = CalibrationSession(states, {}, session_path=os.path.join(os.environ["GAZECORE_DATA_DIR"],
                                                                                    "tour_session.json"))
    for name in ("tracking", "cameras", "screen", "intrinsics", "eye:left", "eye:right", "gaze", "review"):
        if name == "gaze":
            controller.monitors.confirm(controller.monitor_key(), 0.25, "tour")  # Past the screen-scale gate
            controller.pages["gaze"].start(TargetPage.CALIBRATE)
        controller._show(name)
        wait(app, 1.5)
        page = controller.pages[name]
        canvas = getattr(page, "canvas", None)
        if isinstance(canvas, TargetCanvas):
            print(f"TOUR {name}: {canvas.title} | {canvas.status}", flush=True)
        for key in {"eye:left": "LT", "gaze": [" ", "\t", "N", "M", "M"], "screen": "M"}.get(name, ""):
            code = {" ": Qt.Key_Space, "\t": Qt.Key_Tab}.get(key) or getattr(Qt, f"Key_{key}")
            page.handle_key(code, key)
            wait(app, 0.3)
        controller.window.grab()
        print(f"TOUR page {name} ok", flush=True)
    controller._show("tracking")
    controller.mode = "tracking"
    if expect == "tracking":  # Calibrated: the desktop overlay can run
        assert controller.solved is None  # Locking an eye again (L above) cleared the calibration
        controller.restore(controller.gaze, solved, controller.calibrated_geometry)
        controller.start_overlay()
        assert controller.overlay is not None and controller.mode == "overlay"
        wait(app, 1.5)
        controller.overlay.grab()
        png = controller.overlay.save_heatmap(os.environ["GAZECORE_DATA_DIR"])
        assert os.path.exists(png) and os.path.exists(png[:-4] + ".csv")
        controller.stop_overlay()
        assert controller.overlay is None and controller.mode == "tracking"
        print("TOUR overlay ok", flush=True)
    controller.pages["tracking"].toggle_help()
    wait(app, 0.5)
    controller.window.grab()
    fps = 0 if controller.bundle is None else controller.bundle.processing_fps
    print(f"TOUR fps {fps:.0f}", flush=True)
    controller.shutdown()
    print("TOUR OK", flush=True)


if __name__ == "__main__":
    try:
        main(*sys.argv[1:5])
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    os._exit(0)  # Skip Qt's teardown of the abandoned reader threads
