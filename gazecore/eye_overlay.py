"""The eye view: pupil ellipse, eyeball sphere and gaze line drawn on an EyeResult's frame.

Draws exactly what the front-camera tracker drew, in the same order, so its rendered eye
frames are unchanged (gazecore/tests/golden checks their CRCs).
"""

import cv2


def draw_eye_overlay(result, prompt=None, stuck_ellipses=(), show_quality=False):
    """Draw on result.frame (the 640x480 eye frame) in place and return it.

    prompt: the instruction line at the bottom (None draws none). stuck_ellipses: earlier pupil
    ellipses kept on screen (the tracker's E capture). show_quality: the corner percentage is
    result.quality instead of the original confidence.
    """
    frame = result.frame
    if result.status == "no_dark_area":
        return frame
    if result.status != "ok":
        draw_persistent_overlays(frame, prompt, result.gaze_vector_text)
        return frame

    model_center_average = result.model_center
    final_rotated_rect = result.ellipse
    center_x, center_y = result.pupil_center

    # Draw reference lines/ellipses
    cv2.circle(frame, model_center_average, int(result.sphere_radius), (255, 50, 50), 2)  # Draw eye sphere (circle)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)  # Draw eye center
    cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  # # Draw line from eye center to ellipse center
    cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)  # draw current ellipse

    # Draw previously saved ellipses so they stay "stuck" on screen
    for ellipse in stuck_ellipses:
        if ellipse is not None:
            cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

    # Calculate the extended endpoint of gaze line
    dx = center_x - model_center_average[0]
    dy = center_y - model_center_average[1]
    extended_x = int(model_center_average[0] + 2 * dx)
    extended_y = int(model_center_average[1] + 2 * dy)
    cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3)

    ratio_text = f"{(result.quality if show_quality else result.confidence) * 100:.2f}%"
    cv2.putText(frame, ratio_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, ratio_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    draw_persistent_overlays(frame, prompt, result.gaze_vector_text)
    return frame


def draw_persistent_overlays(frame, prompt_text, gaze_vector_text):
    if prompt_text is not None:
        prompt_shadow = (12, frame.shape[0] - 63)
        prompt_origin = (10, frame.shape[0] - 65)
        cv2.putText(frame, prompt_text, prompt_shadow, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, prompt_text, prompt_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if gaze_vector_text is None:
        return

    origin_text, dir_text = gaze_vector_text
    text_origin = (12, frame.shape[0] - 38)
    text_dir = (12, frame.shape[0] - 13)
    text_origin_green = (10, frame.shape[0] - 40)
    text_dir_green = (10, frame.shape[0] - 15)

    cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, origin_text, text_origin_green, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, dir_text, text_dir_green, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
