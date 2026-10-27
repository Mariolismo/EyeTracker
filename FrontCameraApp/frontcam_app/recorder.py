"""Records the tracking view to an mp4, with a CSV of each frame's gaze beside it. No Qt."""

import csv
import os
import time

import cv2

MAX_WIDTH = 1280  # Frames are scaled down to this to keep the writer cheap
FPS = 30.0


class Recorder:
    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(directory, f"tracking_{stamp}.mp4")
        self.csv_path = os.path.join(directory, f"tracking_{stamp}.csv")
        self.writer = None
        self.size = None
        self.frames = 0
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame", "t", "gaze_u", "gaze_v", "depth_m", "left_conf", "right_conf", "left_quality", "right_quality"])

    def write(self, image_bgr, bundle):
        height, width = image_bgr.shape[:2]
        if self.writer is None:
            scale = min(1.0, MAX_WIDTH / width)
            self.size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
            self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, self.size)
        if (width, height) != self.size:
            image_bgr = cv2.resize(image_bgr, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(image_bgr)
        estimate = bundle.estimate
        results = {eye: bundle.eyes.get(eye) for eye in ("left", "right")}
        confidence = {eye: ("" if r is None or r.status != "ok" else round(r.confidence, 3)) for eye, r in results.items()}
        quality = {eye: ("" if r is None or r.status != "ok" else round(r.quality, 3)) for eye, r in results.items()}
        self._csv.writerow([self.frames, round(bundle.t, 4),
                            "" if estimate is None else round(float(estimate.pixel[0]), 1),
                            "" if estimate is None else round(float(estimate.pixel[1]), 1),
                            "" if estimate is None else round(float(estimate.depth), 3),
                            confidence["left"], confidence["right"], quality["left"], quality["right"]])
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self._csv_file.close()
