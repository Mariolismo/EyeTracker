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


def bright_room_pupil(lit_grey=45):
    """A pupil in a lit room: its left half lifted by ambient light, a window reflection across it."""
    image = np.full((480, 640, 3), 100, np.uint8)  # Iris
    cv2.ellipse(image, (320, 240), (26, 26), 0, 0, 360, (lit_grey,) * 3, -1)
    cv2.ellipse(image, (320, 240), (26, 26), 0, -90, 90, (5, 5, 5), -1)  # The right half stays black
    cv2.rectangle(image, (314, 216), (330, 266), (190, 190, 190), -1)
    return image


def test_seeds_fit_the_whole_of_a_lit_pupil():
    image = bright_room_pupil()
    original = pd.detect_pupil(image)  # Darkest pixel + 25 stops short of the lit half...
    assert original.ellipse[0][0] > 326 and min(original.ellipse[1]) < 46  # ...and fits the black crescent
    observation = pd.detect_pupil(image, seeds=4)
    assert np.hypot(*np.subtract(observation.ellipse[0], (320, 240))) < 2
    assert min(observation.ellipse[1]) > 52  # The whole pupil (the fit runs on the dilated blob)


def test_a_reflection_on_the_pupil_does_not_lower_its_contrast():
    image = np.full((480, 640), 100, np.uint8)  # Iris
    cv2.ellipse(image, (320, 240), (26, 26), 0, 0, 360, 5, -1)  # Pupil
    cv2.rectangle(image, (302, 216), (338, 266), 190, -1)  # A reflection over most of it
    assert pd.pupil_contrast(image, ((320, 240), (52, 52), 0)) > 0.9  # Was 0: the median was the reflection's


def test_a_fit_reaching_over_the_lit_pupil_counts_as_the_same_pupil():
    image = np.full((480, 640), 137, np.uint8)  # Iris
    cv2.ellipse(image, (320, 240), (34, 34), 0, 0, 360, 27, -1)  # Pupil, lit by the room...
    cv2.ellipse(image, (320, 252), (34, 22), 0, 0, 360, 1, -1)  # ...but for a black crescent
    crescent, whole = ((320, 252), (68, 44), 0), ((320, 240), (68, 68), 0)
    assert pd.extension_is_pupil(image, crescent, whole)
    image[(image == 27)] = 120  # The same fits on a dark iris: the wider one leaked
    assert not pd.extension_is_pupil(image, crescent, whole)
    assert not pd.extension_is_pupil(image, ((420, 240), (40, 40), 0), whole)  # Its center is outside


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
