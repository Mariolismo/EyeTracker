"""The tracker must reproduce the golden eye-pipeline record bit for bit (see gazecore/tests/golden).

Run: python FrontCameraTracker/test_golden_old_tracker.py   (or pytest)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import Orlosky3DEyeTrackerFrontCamera as tracker  # noqa: E402
from gazecore.tests.golden import eye_golden  # noqa: E402


def test_tracker_matches_golden_record():
    if not os.path.exists(eye_golden.EYE_VIDEO):
        print("eye_test.mp4 not found; skipped")
        return
    problems = eye_golden.compare(eye_golden.record(tracker))
    assert not problems, "\n".join(problems[:20])


if __name__ == "__main__":
    test_tracker_matches_golden_record()
    print("The tracker matches the golden record.")
