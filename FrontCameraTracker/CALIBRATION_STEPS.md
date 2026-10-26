# Calibration steps (using a monitor, no printing)

What you need: the headset (left eye, right eye and front cameras), a monitor and a ruler.

> **When do I measure?** Only the size of the black square on the screen, with a ruler,
> in Part C and again if you resize it. You do **not** need to measure distances: once the
> marker is detected, the front camera window shows its distance (e.g. `0.42 m`) under it.

> **Where do I press keys?** Click one of the camera video windows first.
> Keys do nothing while the control panel has focus.

---

## Part A: Make the images (once)

1. From the repo folder, run:
   ```
   python FrontCameraTracker/Orlosky3DEyeTrackerFrontCamera.py
   ```
2. Click **Save printable marker + board** and choose a folder. You get:
   - `gaze_target_marker.png`: one black square (for the gaze calibration)
   - `charuco_board.png`: a chessboard-like grid (for the lens calibration)

## Part B: Lens calibration of the front camera (once per camera)

3. Put on the headset. Under **Forward-facing**, choose the front camera and click **Start**.
   - If the image is upside down or mirrored, fix it with **flip image** now.
     A mirrored image can't detect the board.
4. Open `charuco_board.png` **fullscreen** on the monitor. No measuring needed here.
5. Click the front camera window and press **I**.
6. From about **30–60 cm**, slowly move your head around:
   - tilt it left, right, up and down;
   - go closer and farther;
   - put the board in each corner of the camera image.

   Green dots on the board mean it is detected. The terminal counts
   `Intrinsics view 1/15 captured.` … up to 15, about one per second.
7. Done when the terminal prints `Front camera intrinsics calibrated (RMS … px) and saved …`.
   An RMS below about 1 px is good. If it says `lens distortion is implausible`, press **I** and
   redo step 6, putting the board closer to the image corners and edges.

## Part C: Gaze calibration

8. Under **Left Eye** and **Right Eye**, choose the eye cameras and click **Start**.
   Wait until both pupils are tracked steadily.
9. Open `gaze_target_marker.png` on the monitor, so the black square is about **10 cm** wide.
10. **📏 MEASURE:** measure the black square on the screen with a ruler. Measure only the black
    part, not the white border. Type the value in mm into **Marker black square side**.
11. **Near samples.** Sit about **40 cm** from the screen. The front camera window shows the
    distance under the marker; it must read **below 0.60 m**. Then repeat 6 times:
    - look at the **center** of the black square and press **C**;
    - turn your head a little, so the square lands in a different part of the front
      camera image (left, right, top, bottom).

    The first **C** also locks both eye spheres, so keep the headset still on your head from now on.
12. **Far samples.** Step back to about **2 m**. The distance shown must read **above 1.50 m**.
    Do the same as in step 11: look at the center, press **C**, move your head, 6 times.
13. It solves automatically once both groups have 6 samples. The terminal prints the result.
    **Check the eye separation: it should be about 50–75 mm.**
    - If it is, the calibration is good. A red dot in the front camera window now shows where you look.
    - If it is not, press **C** to start Part C over (from step 11).

## Part D: Accuracy check

14. Press **G** to show the accuracy table.
15. At each of these distances (read from the front camera window), look at the square's center
    and press **T**:
    - about **0.3 m**
    - about **0.7 m**
    - about **1.5 m**
    - about **3 m**: first make the square about **15 cm** wide, **📏 measure it again** and
      update **Marker black square side**.
16. Copy the terminal output (the eye separation from step 13 and the accuracy lines) and send it.
    A good result shows similar errors at every distance, smaller than the "no-offset model" numbers.

---

## Next time

- Part A and Part B don't need repeating; the lens calibration is saved.
- Part C is saved too. Each successful calibration goes to `gaze_calibration.json`, and it is
  restored automatically once all three cameras are running ("Restoring the gaze calibration
  from …" in the terminal). **L** restores it by hand, e.g. after pressing F.
  - Glitched samples are ignored again on restore.
  - If you took the headset off since, check with **G**/**T**; if the error is high, recalibrate with **C**.
- Every capture session, also failed ones, is kept in `calibrations/gaze_<date>_<time>.json`
  for analysis and recovery.
- For a quick recalibration, pressing **C** at one distance and then **K** also works,
  because the eye offsets from the last full calibration are reused.

## Troubleshooting

| Problem | Fix |
|---|---|
| No green dots on the board (Part B) | Get closer, avoid glare, check **flip image** |
| `Marker not visible` | The whole square plus its white border must be inside the front camera image |
| Distance under the marker looks wrong | Re-measure the square (step 10); redo Part B if it is still off |
| Red gaze dot jumps around | Check that both pupils are tracked; repeat Part C |
| Keys do nothing | Click a camera video window first |
