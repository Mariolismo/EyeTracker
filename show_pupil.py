"""Play an eye video (or camera) with gazecore's pupil fit drawn on it, without the app.

    python show_pupil.py eye_test.mp4
    python show_pupil.py 0                 # Camera index
    python show_pupil.py eye.mp4 --seeds 1  # The original single darkest-area search

OrloskyPupilDetector.py keeps its own copy of the original detector; this one runs
gazecore.pupil_detection, the detector the FrontCameraApp uses (4 seeds by default, as the app).
Space pauses, q or Esc quits. Green ellipse: quality 0.7+, orange: below.
"""
import argparse
import time

import cv2

from gazecore.pupil_detection import detect_pupil, pupil_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="eye_test.mp4", help="video file or camera index (default eye_test.mp4)")
    parser.add_argument("--seeds", type=int, default=4, help="dark areas tried per frame (default 4, as the app)")
    args = parser.parse_args()

    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    paused = False
    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            observation = detect_pupil(frame, seeds=args.seeds)
            elapsed_ms = (time.perf_counter() - started) * 1000
            view = observation.frame.copy()
            if observation.ellipse is not None:
                quality = pupil_quality(observation)
                cv2.ellipse(view, observation.ellipse, (0, 255, 0) if quality >= 0.7 else (0, 165, 255), 2)
                cv2.circle(view, tuple(int(v) for v in observation.ellipse[0]), 3, (0, 0, 255), -1)
                text = f"conf {observation.confidence:.2f}  quality {quality:.2f}  {elapsed_ms:.0f} ms"
            else:
                text = f"no pupil  {elapsed_ms:.0f} ms"
            cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Pupil (gazecore)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
