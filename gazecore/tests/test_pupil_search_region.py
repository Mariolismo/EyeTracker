"""The eye-circle search: a darker blob outside the circle (eyelashes, a corner shadow) must not
seed the thresholds once the eyeball center is fixed; off by default, as in the original tracker."""

import cv2
import numpy as np

from gazecore import pupil_detection as pd
from gazecore.eye_tracker import EyeTracker


def eye_image():
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.ellipse(image, (320, 240), (22, 20), 0, 0, 360, (40, 40, 40), -1)  # Pupil
    cv2.rectangle(image, (540, 60), (600, 110), (5, 5, 5), -1)  # Darker lash clump, far from the eye
    return image


def test_the_region_keeps_the_search_on_the_pupil():
    image = eye_image()
    assert np.hypot(*np.subtract(pd.get_darkest_area(image), (570, 85))) < 40  # The original picks the lashes
    observation = pd.detect_pupil(image, search_region=(320, 240, 120))
    assert np.hypot(*np.subtract(observation.darkest_point, (320, 240))) < 25
    assert np.hypot(*np.subtract(observation.ellipse[0], (320, 240))) < 3


def test_a_bigger_blob_outside_the_circle_cannot_take_the_ellipse():
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.ellipse(image, (320, 240), (22, 20), 0, 0, 360, (40, 40, 40), -1)  # Pupil
    cv2.ellipse(image, (380, 150), (45, 24), 0, 0, 360, (45, 45, 45), -1)  # Lash clump, outside
    original = pd.detect_pupil(image)
    assert original.darkest_point == (320, 240)  # Seeded on the pupil, yet...
    assert np.hypot(*np.subtract(original.ellipse[0], (380, 150))) < 5  # ...fitted on the lashes
    observation = pd.detect_pupil(image, search_region=(320, 240, 40))
    assert np.hypot(*np.subtract(observation.ellipse[0], (320, 240))) < 3


def test_a_circle_that_misses_the_pupil_loses_to_the_whole_frame():
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.ellipse(image, (320, 240), (22, 20), 0, 0, 360, (40, 40, 40), -1)  # Pupil
    cv2.rectangle(image, (100, 330), (200, 345), (45, 45, 45), -1)  # Lash line: a poor ellipse
    observation = pd.detect_pupil(image, search_region=(150, 340, 40))  # A wrong eye model
    assert np.hypot(*np.subtract(observation.ellipse[0], (320, 240))) < 3


def test_seeds_find_a_pupil_greyer_than_a_lash_clump():
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.ellipse(image, (300, 240), (22, 20), 0, 0, 360, (45, 45, 45), -1)  # Pupil
    for x in range(470, 560, 7):  # Lash clump: black strokes with skin between them, near the pupil
        cv2.line(image, (x, 200), (x + 25, 300), (0, 0, 0), 5)
    original = pd.detect_pupil(image)
    assert np.hypot(*np.subtract(original.darkest_point, (300, 240))) > 100  # Seeded on the lashes
    assert original.ellipse is None or np.hypot(*np.subtract(original.ellipse[0], (300, 240))) > 50
    observation = pd.detect_pupil(image, seeds=4)
    assert np.hypot(*np.subtract(observation.ellipse[0], (300, 240))) < 3


def test_seeds_never_fit_outside_the_circle():
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.ellipse(image, (300, 240), (22, 20), 0, 0, 360, (45, 45, 45), -1)  # Pupil
    cv2.ellipse(image, (480, 120), (45, 24), 0, 0, 360, (20, 20, 20), -1)  # Lash clump, outside
    assert np.hypot(*np.subtract(pd.detect_pupil(image, (300, 240, 120), seeds=4).ellipse[0], (300, 240))) < 3
    cv2.ellipse(image, (300, 240), (22, 20), 0, 0, 360, (170, 170, 170), -1)  # Pupil hidden (blink)
    assert pd.detect_pupil(image, seeds=4).ellipse is not None  # The whole frame fits the lashes...
    assert pd.detect_pupil(image, (300, 240, 120), seeds=4).ellipse is None  # ...the circle doesn't


def test_quality_marks_the_lash_fit_down_and_the_pupil_fit_up():
    image = np.full((480, 640, 3), 170, np.uint8)
    cv2.ellipse(image, (300, 240), (22, 20), 0, 0, 360, (45, 45, 45), -1)  # Pupil
    for x in range(470, 560, 7):  # Lash clump
        cv2.line(image, (x, 200), (x + 25, 300), (0, 0, 0), 5)
    lash_fit, pupil_fit = pd.detect_pupil(image), pd.detect_pupil(image, seeds=4)
    assert pd.pupil_quality(lash_fit) < 0.5 < lash_fit.confidence  # The edge test catches it
    assert pd.pupil_quality(pupil_fit) > 0.95
    assert pd.pupil_quality(pd.PupilObservation(image)) == 0.0  # No ellipse


def test_the_first_seed_is_the_original_darkest_area():
    rng = np.random.default_rng(1)
    for image in [eye_image()] + [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(5)]:
        assert pd.get_darkest_areas(image, 4)[0] == pd.get_darkest_area(image)
        assert pd.get_darkest_areas(image, 4, (320, 240, 120))[0] == pd.get_darkest_area(image, (320, 240, 120))
    flat = np.full((480, 640, 3), 90, np.uint8)  # Ties: the first block in scan order wins in both
    assert pd.get_darkest_areas(flat, 1)[0] == pd.get_darkest_area(flat)


def test_a_region_with_nothing_inside_falls_back_to_the_whole_frame():
    image = eye_image()
    assert pd.detect_pupil(image, search_region=(-500, -500, 10)).darkest_point == pd.get_darkest_area(image)


def test_the_tracker_uses_the_circle_only_when_enabled_and_fixed():
    tracker = EyeTracker("left")
    tracker.apply_locked_state({"model_center": [320, 240], "sphere_radius": 120.0,
                                "sphere_center": None, "adjustment_enabled": True})
    assert tracker.search_region() is None  # Off by default: the original tracker
    tracker.search_in_eye_circle = True
    assert tracker.search_region() == (320, 240, 120.0)
    tracker.unlock_sphere()
    assert tracker.search_region() is None  # Still adapting: the whole frame
