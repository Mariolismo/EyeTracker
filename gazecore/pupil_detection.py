"""Pupil detection in one eye frame (Orlosky's three-threshold ellipse fit).

Pure image processing, moved unchanged from the front-camera tracker: no state, no windows.
detect_pupil() is the entry point; EyeTracker (eye_tracker.py) turns its observations into
an eye model and a gaze direction.
"""

from dataclasses import dataclass

import cv2
import numpy as np

EYE_FRAME_SIZE = (640, 480)  # Every eye frame is cropped to 4:3 and resized to this
REGION_MARGIN_PX = 25  # Past the search circle: the circle bounds pupil centers, not pupil edges
REGION_TRUST_CONFIDENCE = 0.85  # A fit inside the circle this good is kept without a whole-frame check
SEED_SEPARATION_PX = 80  # get_darkest_areas: seeds closer than this would fit the same blob
SEED_BLUR_PX = 31  # get_darkest_areas: about a pupil's width
SEED_BLOB_PX = 10  # A seeded fit keeps only contours within this of its seed
FIT_CROP_MARGIN_PX = 60  # _fit_cropped: room past the threshold square for the ellipse masks
EDGE_OFFSET_PX = 6  # edge_support: sampled this far inside and outside the rim
EDGE_STEP = 8  # edge_support: grey levels brighter outside than inside to count as pupil edge
THRESHOLD_OFFSETS = (5, 15, 25)  # The original three thresholds, grey levels above the darkest pixel
THRESHOLD_SHARES = (0.15, 0.3, 0.45)  # adaptive_thresholds: shares of the pupil-to-surround gap
DARK_PATCH_PX = 41  # adaptive_thresholds: the pupil level is a low percentile of this patch at the seed
SURROUND_PX = 161  # adaptive_thresholds: the surround level is a percentile of this window
SURROUND_PERCENTILE = 25  # Below the median: skin and sclera fill the window, but the gap that matters is to the iris
EXTENSION_DARKNESS = 0.5  # extension_is_pupil: the added area's median grey below this share of the iris ring's


# Crop the image to maintain a specific aspect ratio (width:height) before resizing.
def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

# Apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Cast to int so uint8 pixel values do not wrap when adding the offset.
    threshold = int(darkestPixelValue) + int(addedThreshold)
    threshold = max(0, min(255, threshold))
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# Finds a square area of dark pixels in the image
def get_darkest_area(image, region=None):
    """region: optional (cx, cy, radius); only blocks centered inside that circle are searched."""
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            if region is not None and ((x + searchArea // 2 - region[0]) ** 2
                                       + (y + searchArea // 2 - region[1]) ** 2 > region[2] ** 2):
                continue
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += int(gray[y + dy, x + dx])
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

def _darkest_area_fast(image, region=None):
    """get_darkest_area vectorised: the same sampled sums, and argmin keeps its first-wins order."""
    ignoreBounds, imageSkipSize, searchArea, internalSkipSize = 20, 10, 20, 5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int64)
    ys = np.arange(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize)
    xs = np.arange(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize)
    if len(ys) == 0 or len(xs) == 0:
        return None
    sums = np.zeros((len(ys), len(xs)), np.int64)
    for dy in range(0, searchArea, internalSkipSize):
        for dx in range(0, searchArea, internalSkipSize):
            sums += gray[np.ix_(np.minimum(ys + dy, gray.shape[0] - 1), np.minimum(xs + dx, gray.shape[1] - 1))]
    centers_y, centers_x = np.meshgrid(ys + searchArea // 2, xs + searchArea // 2, indexing="ij")
    if region is not None:
        outside = (centers_x - region[0]) ** 2 + (centers_y - region[1]) ** 2 > region[2] ** 2
        if outside.all():
            return None
        sums[outside] = np.iinfo(np.int64).max
    index = int(np.argmin(sums))
    return int(centers_x.flat[index]), int(centers_y.flat[index])


def get_darkest_areas(image, count, region=None):
    """Up to `count` seeds, SEED_SEPARATION_PX apart: get_darkest_area's first, then the darkest
    spots after a SEED_BLUR_PX box blur. A solid pupil stays dark under the blur; lashes, with
    skin between them, do not, so the pupil gets a seed even when a lash clump is darker."""
    first = _darkest_area_fast(image, region)
    if first is None:
        return []
    gray = cv2.blur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (SEED_BLUR_PX, SEED_BLUR_PX))
    margin, step = SEED_BLUR_PX, 5
    sub = gray[margin:-margin:step, margin:-margin:step].astype(np.int32)
    centers_y, centers_x = np.mgrid[margin:gray.shape[0] - margin:step, margin:gray.shape[1] - margin:step]
    if region is not None:
        sub[(centers_x - region[0]) ** 2 + (centers_y - region[1]) ** 2 > region[2] ** 2] = 256
    points = [first]
    for index in np.argsort(sub, axis=None, kind="stable"):
        if len(points) == count or sub.flat[index] > 255:
            break
        point = (int(centers_x.flat[index]), int(centers_y.flat[index]))
        if all((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 >= SEED_SEPARATION_PX ** 2 for p in points):
            points.append(point)
    return points


def adaptive_thresholds(gray, seed):
    """(base, offsets) for _fit_at, from the seed's surroundings rather than its one pixel.

    In a bright room the pupil is no longer uniformly black: ambient light lifts parts of it
    to 30-40 grey while a noise pixel still reads 0, and a corneal reflection of the room can
    split it in two. darkest + 5/15/25 then keeps only the darkest crescent and the ellipse
    fits that. Here the base is the 5th percentile of a DARK_PATCH_PX patch, and each offset is
    at least the original one and at least its THRESHOLD_SHARES of the gap up to the
    SURROUND_PERCENTILE of a SURROUND_PX window, so the thresholds scale with the pupil's contrast.
    On a pupil that is already black the wider thresholds can leak into a dark iris, so
    _detect_in fits each seed both ways and lets the score pick.
    """
    def window(size):
        half = size // 2
        return gray[max(seed[1] - half, 0):seed[1] + half + 1, max(seed[0] - half, 0):seed[0] + half + 1]

    base = int(np.percentile(window(DARK_PATCH_PX), 5))
    gap = max(float(np.percentile(window(SURROUND_PX), SURROUND_PERCENTILE)) - base, 0.0)
    return base, tuple(max(fixed, int(share * gap)) for fixed, share in zip(THRESHOLD_OFFSETS, THRESHOLD_SHARES))


def pupil_contrast(gray, ellipse):
    """(ring - inside) / ring of the median grey levels: about 0.5+ for a pupil against its
    iris, lower for a lash clump with skin showing through it. Inside pixels brighter than the
    ring's 90th percentile are left out: a corneal reflection over the pupil would otherwise
    lift its median and let a lid or iris fit outrank it."""
    outer_ellipse = (ellipse[0], (ellipse[1][0] * 1.8, ellipse[1][1] * 1.8), ellipse[2])
    x, y, width, height = cv2.boundingRect(cv2.boxPoints(outer_ellipse).astype(np.int32))
    x0, y0 = max(x, 0), max(y, 0)
    gray = gray[y0:max(y + height, y0), x0:max(x + width, x0)]  # Only the ring's box: cheap
    shifted = lambda e: ((e[0][0] - x0, e[0][1] - y0), e[1], e[2])
    inside = np.zeros_like(gray)
    cv2.ellipse(inside, shifted(ellipse), 255, -1)
    ring = np.zeros_like(gray)
    cv2.ellipse(ring, shifted(outer_ellipse), 255, -1)
    ring[inside > 0] = 0
    if not inside.any() or not ring.any():
        return 0.0
    inner_pixels, ring_pixels = gray[inside > 0], gray[ring > 0]
    unlit = inner_pixels[inner_pixels <= np.percentile(ring_pixels, 90)]
    inner = np.median(unlit) if unlit.size else np.median(inner_pixels)
    outer = np.median(ring_pixels)
    return max(0.0, float(outer - inner) / max(float(outer), 1.0))


def extension_is_pupil(gray, inner, outer):
    """True when outer, a fit from the same seed, contains inner's center and the area it adds
    is pupil-dark: a median grey under EXTENSION_DARKNESS of the ring around outer. With part of
    the pupil lit by the room (~30 grey against a ~130 iris) the fixed thresholds fit the black
    crescent and the adaptive ones the whole pupil, and the crescent's blacker inside gave it the
    higher contrast, so the fit flipped between the two from frame to frame. When the adaptive
    thresholds leak into a dark iris instead, the added area is iris-grey and this is False."""
    outer_ring = (outer[0], (outer[1][0] * 1.8, outer[1][1] * 1.8), outer[2])
    x, y, width, height = cv2.boundingRect(cv2.boxPoints(outer_ring).astype(np.int32))
    x0, y0 = max(x, 0), max(y, 0)
    gray = gray[y0:max(y + height, y0), x0:max(x + width, x0)]  # Only the ring's box, as in pupil_contrast
    shifted = lambda e: ((e[0][0] - x0, e[0][1] - y0), e[1], e[2])
    masks = {}
    for name, ellipse in (("outer", outer), ("inner", inner), ("ring", outer_ring)):
        masks[name] = np.zeros_like(gray)
        cv2.ellipse(masks[name], shifted(ellipse), 255, -1)
    cx, cy = int(inner[0][0]) - x0, int(inner[0][1]) - y0
    if not (0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]) or not masks["outer"][cy, cx]:
        return False
    added = (masks["outer"] > 0) & (masks["inner"] == 0)
    ring = (masks["ring"] > 0) & (masks["outer"] == 0)
    if added.sum() < 0.1 * np.count_nonzero(masks["outer"]) or not ring.any():
        return False  # Hardly bigger: nothing to choose between
    return float(np.median(gray[added])) < EXTENSION_DARKNESS * float(np.median(gray[ring]))


def edge_support(gray, ellipse, samples=72):
    """Share of the ellipse rim where the image steps from dark inside to bright outside
    (EDGE_STEP grey levels, EDGE_OFFSET_PX either side). Glints inside the pupil and lashes
    touching it, which notch the thresholded blob, don't move it."""
    (cx, cy), (width, height), angle = ellipse
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    a = np.radians(angle)
    ex, ey = width / 2 * np.cos(t), height / 2 * np.sin(t)
    x, y = cx + ex * np.cos(a) - ey * np.sin(a), cy + ex * np.sin(a) + ey * np.cos(a)
    nx, ny = x - cx, y - cy
    norm = np.hypot(nx, ny) + 1e-9
    nx, ny = nx / norm, ny / norm
    pixel = lambda px, py: (np.clip(px, 0, gray.shape[1] - 1).astype(int), np.clip(py, 0, gray.shape[0] - 1).astype(int))
    outside = pixel(x + EDGE_OFFSET_PX * nx, y + EDGE_OFFSET_PX * ny)
    inside = pixel(x - EDGE_OFFSET_PX * nx, y - EDGE_OFFSET_PX * ny)
    # Blur only the samples' box (ranking seeds calls this per fit): 2 px past it, the 5x5 kernel's
    # reach, every sample sees the pixels a whole-frame blur would
    xs, ys = np.concatenate([outside[0], inside[0]]), np.concatenate([outside[1], inside[1]])
    x0, y0 = max(int(xs.min()) - 2, 0), max(int(ys.min()) - 2, 0)
    smooth = cv2.GaussianBlur(gray[y0:int(ys.max()) + 3, x0:int(xs.max()) + 3], (5, 5), 0).astype(np.float32)
    step = smooth[outside[1] - y0, outside[0] - x0] - smooth[inside[1] - y0, inside[0] - x0]
    return float(np.mean(step > EDGE_STEP))


def pupil_quality(observation):
    """(confidence + edge_support) / 2, 0 without an ellipse: steadier than the confidence alone
    on a well-tracked pupil, and still low on lash fits (at most 0.64 on recorded frames, where
    pupils mostly score 0.75+). The confidence itself stays as it is: the eye model uses it."""
    if observation.ellipse is None:
        return 0.0
    gray = cv2.cvtColor(observation.frame, cv2.COLOR_BGR2GRAY)
    return (float(observation.confidence) + edge_support(gray, observation.ellipse)) / 2


def touches_frame_edge(ellipse, shape, pad=20):
    """A pupil cut by the frame edge can't be fitted honestly; it's usually a shadow there."""
    x, y, width, height = cv2.boundingRect(cv2.boxPoints(ellipse).astype(np.int32))
    return x <= pad or y <= pad or x + width >= shape[1] - pad or y + height >= shape[0] - pad

# Mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

# Returns the largest contour that is not extremely long or tall
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    return [largest_contour] if largest_contour is not None else []

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap (countNonZero: the same counts as np.sum, faster)
    absolute_pixel_total_thick = cv2.countNonZero(overlap_thick)#compute with thicker border
    absolute_pixel_total_thin = cv2.countNonZero(overlap_thin)#compute with thicker border

    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = cv2.countNonZero(contour_mask)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = cv2.countNonZero(mask)  # The masks hold only 0 and 255

    # Calculate the number of white pixels within the ellipse
    covered_pixels = cv2.countNonZero(cv2.bitwise_and(binary_image, mask))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness


@dataclass
class PupilObservation:
    """What detect_pupil found in one eye frame."""

    frame: np.ndarray  # The frame cropped to 4:3 and resized to EYE_FRAME_SIZE (the overlay draws on it)
    darkest_point: tuple = None  # None: no dark area searched, so there is nothing else
    ellipse: tuple = None  # Pupil ellipse ((cx, cy), (axis, axis), angle), or None
    center: tuple = (None, None)  # Pupil center in integer pixels
    confidence: float = 0.0  # Share of the contour under the ellipse (the "%" on the eye view)


def detect_pupil(frame, search_region=None, seeds=1):
    """Find the pupil ellipse in one BGR eye frame of any size.

    search_region: optional (cx, cy, radius) in the cropped frame's pixels. The darkest-area
    search, which seeds the thresholds, stays inside it (falling back to the whole frame when
    nothing is inside), and the thresholded images are clipped to it plus REGION_MARGIN_PX, so
    eyelashes and eye-corner shadows can't win either step. The circle is only as good as the
    eye model, so a fit inside it under REGION_TRUST_CONFIDENCE is checked against the whole
    frame and the better-fitting ellipse kept.
    seeds: how many separate dark areas to fit from (get_darkest_areas). A lash clump darker
    than the pupil wins the single darkest area, and the original then fits the largest blob
    near it, wherever it starts. With seeds > 1 each fit uses only the blob around its own
    seed and is made twice, with the original thresholds and with adaptive_thresholds (a pupil
    lit unevenly or split by a reflection); fits touching the frame edge are dropped, and the
    best confidence x pupil_contrast x edge_support is kept (a lash clump can match a pupil on
    the first two, but its rim is not a dark-to-bright edge), the adaptive fit taking over its
    seed's fixed one when it is the same pupil with more of it (extension_is_pupil). None and
    1: the original.
    With seeds > 1 and a search_region there is no whole-frame check: one pass inside the
    circle, and fits centered outside it are dropped (no pupil rather than a lash fit). A wrong
    circle is fixed by resetting the eye model, not by searching past it.
    """
    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)
    if search_region is None or seeds > 1:
        return _detect_in(frame, search_region, seeds)
    inside = _detect_in(frame, search_region, seeds)
    if inside.confidence >= REGION_TRUST_CONFIDENCE:
        return inside
    anywhere = _detect_in(frame, None, seeds)
    return anywhere if anywhere.confidence > inside.confidence else inside


def _detect_in(frame, search_region, seeds):
    if seeds > 1:
        points = get_darkest_areas(frame, seeds, search_region)
        if not points:
            return PupilObservation(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = [(point, adaptive) for point in points for adaptive in (False, True)]
        fits = [(_fit_cropped(frame, point, search_region, refine=False, adaptive=adaptive), adaptive)
                for point, adaptive in candidates]
        fits = [(fit, adaptive) for fit, adaptive in fits if fit.ellipse is not None
                and not touches_frame_edge(fit.ellipse, gray.shape) and _inside_circle(fit.ellipse, search_region)]
        if not fits:
            return PupilObservation(frame, points[0])
        score = {id(fit): fit.confidence * pupil_contrast(gray, fit.ellipse) * edge_support(gray, fit.ellipse)
                 for fit, _ in fits}
        for crescent, adaptive in fits:
            for whole, whole_adaptive in fits:
                if (not adaptive and whole_adaptive and whole.darkest_point == crescent.darkest_point
                        and extension_is_pupil(gray, crescent.ellipse, whole.ellipse)):
                    score[id(whole)] = max(score[id(whole)], score[id(crescent)])  # Same pupil, all of it
        best, adaptive = max(fits, key=lambda pair: (score[id(pair[0])], pair[1]))  # Ties: the whole pupil
        final = _fit_cropped(frame, best.darkest_point, search_region, refine=True, adaptive=adaptive)
        if final.ellipse is not None and not _inside_circle(final.ellipse, search_region):
            return PupilObservation(frame, best.darkest_point)
        return final

    #find the darkest point
    darkest_point = None if search_region is None else get_darkest_area(frame, search_region)
    if darkest_point is None:
        search_region = None  # Fell back: the whole frame, unclipped
        darkest_point = get_darkest_area(frame)
    if darkest_point is None:
        return PupilObservation(frame)
    return _fit_at(frame, darkest_point, search_region)


def _inside_circle(ellipse, search_region):
    """A pupil is centered inside the eyeball's circle and smaller than it."""
    if search_region is None:
        return True
    return (np.hypot(ellipse[0][0] - search_region[0], ellipse[0][1] - search_region[1]) <= search_region[2]
            and max(ellipse[1]) / 2 <= search_region[2])


def _fit_cropped(frame, seed, search_region, refine, adaptive=False):
    """_fit_at on its own blob, run on the box its 250 px threshold square leaves unmasked (plus
    a margin for the ellipse masks): the full-frame mask sums made each seed cost a whole fit.
    adaptive: adaptive_thresholds instead of the seed pixel + THRESHOLD_OFFSETS."""
    half = 125 + FIT_CROP_MARGIN_PX
    x0, y0 = max(seed[0] - half, 0), max(seed[1] - half, 0)
    crop = frame[y0:seed[1] + half, x0:seed[0] + half]
    region = None if search_region is None else (search_region[0] - x0, search_region[1] - y0, search_region[2])
    local_seed = (seed[0] - x0, seed[1] - y0)
    thresholds = adaptive_thresholds(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), local_seed) if adaptive else None
    fit = _fit_at(crop, local_seed, region, own_blob=True, refine=refine, thresholds=thresholds)
    ellipse, center = fit.ellipse, fit.center
    if ellipse is not None:
        ellipse = ((ellipse[0][0] + x0, ellipse[0][1] + y0), ellipse[1], ellipse[2])
    if center[0] is not None:
        center = (center[0] + x0, center[1] + y0)
    return PupilObservation(frame, seed, ellipse, center, fit.confidence)


def _fit_at(frame, darkest_point, search_region, own_blob=False, refine=True, thresholds=None):
    """refine=False skips optimize_contours_by_angle (slow): enough to rank seeds.
    thresholds: (base, (strict, medium, relaxed) offsets); None is the darkest pixel and THRESHOLD_OFFSETS."""
    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if thresholds is None:
        thresholds = (int(gray_frame[darkest_point[1], darkest_point[0]]), THRESHOLD_OFFSETS)
    darkest_pixel_value, (strict, medium, relaxed) = thresholds

    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, strict)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, medium)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)

    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, relaxed)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    if search_region is not None:
        # Clip the thresholds to the circle too: seeding inside it alone still lets lashes just
        # outside it join the binary images and win the largest-contour pick
        circle = np.zeros_like(gray_frame)
        cv2.circle(circle, (int(search_region[0]), int(search_region[1])),
                   int(search_region[2]) + REGION_MARGIN_PX, 255, -1)
        image_array = [cv2.bitwise_and(image, circle) for image in image_array]
    final_contours = [] #holds final contours
    goodness = 0 #goodness value for best ellipse
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    best_ratio_under_ellipse = 0
    best_center_x, best_center_y = None, None

    # iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            dilated_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if own_blob:
            contours = [c for c in contours
                        if cv2.pointPolygonTest(c, (float(darkest_point[0]), float(darkest_point[1])), True) >= -SEED_BLOB_PX]

        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        center_x, center_y = None, None

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(
                dilated_image,
                reduced_contours[0],
                False
            )

            ellipse = cv2.fitEllipse(reduced_contours[0])
            center_x, center_y = map(int, ellipse[0])

            total_pixels = check_contour_pixels(
                reduced_contours[0],
                dilated_image.shape,
                False
            )

            final_goodness = (
                current_goodness[0]
                * total_pixels[0]
                * total_pixels[0]
                * total_pixels[1]
            )

            if final_goodness > 0 and final_goodness > goodness:
                goodness = final_goodness
                best_ratio_under_ellipse = total_pixels[1]
                final_contours = reduced_contours

                # Critical fix:
                # Keep the pupil center associated with the chosen contour.
                best_center_x = center_x
                best_center_y = center_y

    # After threshold selection, use the center from the chosen/best contour.
    center_x = best_center_x
    center_y = best_center_y

    if refine:
        final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]

    final_rotated_rect = None

    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) >= 5:
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        center_x, center_y = map(int, ellipse[0])

    return PupilObservation(frame, darkest_point, final_rotated_rect, (center_x, center_y),
                            best_ratio_under_ellipse)
