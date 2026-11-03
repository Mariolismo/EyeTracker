"""Raw eye frames for training a pupil detector (PupilNet/), with the classic fit of each. No Qt.

The Pipeline hands every tracked eye frame to add() before the eye overlay draws on it. Every
`every`-th frame per eye is saved as a grey PNG (the eye cameras are IR: the channels are
equal); frames.csv gets a row for every frame, saved or not, so a labeler can check each fit
against its neighbours. PNGs are written on a thread of their own: the worker never waits on
the disk, and a frame the queue has no room for is listed with no image.

    <directory>/meta.json   subject, cameras, settings, git commit, counts
    <directory>/frames.csv  eye, seq, t, image, status, ellipse, confidence, quality, circle, fixed
    <directory>/<eye>/<seq>.png
"""

import csv
import json
import os
import queue
import subprocess
import threading
import time

import cv2

from gazecore.paths import REPO_ROOT

COLUMNS = ["eye", "seq", "t", "image", "status", "cx", "cy", "width", "height", "angle", "confidence", "quality",
           "region_x", "region_y", "region_r", "fixed"]
QUEUE_FRAMES = 120  # ~2 s of both eyes at every=2: a slow disk drops frames instead of stalling tracking
PNG_COMPRESSION = 1  # Fast; the frames are noisy, harder compression saves little


def git_commit():
    """HEAD, with "-dirty" when the tree has changes; None outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return (commit + "-dirty" if dirty else commit) or None


def _rounded(value, digits=2):
    return "" if value is None else round(float(value), digits)


class EyeCapture:
    def __init__(self, directory, subject, meta=None, every=2):
        self.directory = directory
        self.every = max(1, int(every))
        self.counts = {}  # {eye: frames seen}
        self.saved = 0
        self.dropped = 0
        for eye in ("left", "right"):
            os.makedirs(os.path.join(directory, eye), exist_ok=True)
        self.meta = {"subject": subject, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "every": self.every,
                     "frame_size": [640, 480], "git_commit": git_commit(), **(meta or {})}
        self._write_meta()
        self._csv_file = open(os.path.join(directory, "frames.csv"), "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(COLUMNS)
        self._queue = queue.Queue(QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._write_images, name="eye-capture", daemon=True)
        self._thread.start()

    def add(self, eye, seq, t, result, region, fixed):
        """One tracked frame: result is the EyeResult, region the search circle it was tracked with."""
        count = self.counts.get(eye, 0)
        self.counts[eye] = count + 1
        image = ""
        if count % self.every == 0:
            name = f"{eye}/{seq:07d}.png"
            try:
                self._queue.put_nowait((name, cv2.cvtColor(result.frame, cv2.COLOR_BGR2GRAY)))
                image = name
                self.saved += 1
            except queue.Full:
                self.dropped += 1
        ellipse = result.ellipse
        (cx, cy), (width, height), angle = ellipse if ellipse is not None else ((None, None), (None, None), None)
        region = region or (None, None, None)
        self._csv.writerow([eye, seq, round(t, 4), image, result.status, _rounded(cx), _rounded(cy), _rounded(width),
                            _rounded(height), _rounded(angle), _rounded(result.confidence, 4),
                            _rounded(result.quality, 4), _rounded(region[0], 1), _rounded(region[1], 1),
                            _rounded(region[2], 1), int(bool(fixed))])

    def close(self):
        """Waits for the queued images; call it once nothing calls add() any more."""
        self._queue.put(None)
        self._thread.join()
        self._csv_file.close()
        self.meta.update(stopped=time.strftime("%Y-%m-%dT%H:%M:%S"), frames=self.counts, saved=self.saved,
                         dropped=self.dropped)
        self._write_meta()

    def _write_images(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            name, gray = item
            cv2.imwrite(os.path.join(self.directory, name), gray, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])

    def _write_meta(self):
        with open(os.path.join(self.directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
