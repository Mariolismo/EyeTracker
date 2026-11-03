# Front camera gaze app: user guide

The app tracks both eyes, maps the gaze into the front camera (with depth), and can show where
you look on the monitor. It uses the same calibration files as the front-camera tracker
(`FrontCameraTracker/`), so either program can restore what the other saved.

## Start

```
pip install -r requirements.txt
python FrontCameraApp/run_app.py
```

The app opens full-screen and connects the three cameras one after another (the front camera
can take about 12 s). Then:

- **A valid calibration exists:** it goes straight to tracking.
  A calibration from the old tracker shows the banner *"not checked here yet: press V"*:
  press **V** once to verify its accuracy.
- **Something is missing or changed:** the wizard opens at the first step that needs
  doing, and the banner says why (for example *"The left eye camera changed since calibrating"*).

## Tracking view

The front camera fills the screen. The red dot is where you look, with its distance. The eye
views are in the bottom corners, and the status bar shows each camera (green = running),
the processed frame rate, and the calibration's accuracy, eye separation and age.

| Key | |
|---|---|
| **R** | Quick recalibration: near samples only, keeps the eye positions (after the headset moved a little) |
| **V** | Verify: measure the accuracy on the marker |
| **W** | Full calibration wizard |
| **C** | Cameras |
| **O** | Desktop overlay (gaze ring and heatmap on the monitor) |
| **H** | Heatmap on/off |
| **X** | Record the tracking view to `sessions/tracking_<time>.mp4`, with a CSV of gaze pixel, depth, pupil confidence and pupil quality per frame (X again stops) |
| **E** | Capture raw eye frames for testing or training the pupil detector, on any page but Cameras (E again stops). See *Eye data capture* below |
| **F1** | Help |
| **Q** | Quit |

Keys work whatever has focus (no need to click a video window).

## The wizard

A bar at the top shows the steps. **Esc** goes back one step.

1. **Cameras.** Check each preview: the right camera in each role, eyes upright (eyelashes at
   the top), your left eye in *Left eye*. Fix with *Upside down* / *Mirror*. **Apply** saves
   it. The app remembers cameras by their device ID, so plugging them in a different order
   doesn't matter. If a camera moved to another USB port, the app says so and lets you
   choose it here.
2. **Screen scale** (once per monitor). Hold a credit card against the screen and use the
   **Left/Right** arrows (Shift: faster) until the blue rectangle matches it. Or press **M**,
   measure the black line with a ruler and type its length. **Enter** confirms. All
   marker sizes come from this, so be careful.
3. **Lens** (only when the front camera has no lens calibration). The board appears on
   the screen. Move your head so the board shows up in every part of the front camera's
   view, at different angles. The 3×3 grid fills as views are kept, and it finishes by
   itself at 15 views. **S** skips it if a lens calibration already exists.
4. **Left eye, then right eye.**
   - Look around slowly in all directions until the *Eye model* bar is full and steady.
   - Look straight ahead and press **L** to lock it.
   - If the blue circle is off-center, press **O** first and look straight into that eye
     camera's lens until it finishes, then **L**.
   - Optional: **T**, then close that eye for a second. It confirms the cameras aren't swapped.
5. **Gaze, near then far.** A marker appears on the screen.
   - **Near:** stand closer than 0.6 m. Look at the marker's center and capture. Turn your
     head between samples so the marker lands in each square of the grid (Tab moves the
     marker if you need it elsewhere). Aim for 9 samples (at least 6, over 2 rows and 2 columns of
     the grid).
   - **N** switches to **far:** stand farther than 1.5 m. The marker gets bigger. Turn your
     head so it lands in each square of the grid.
   - The line under the title says what is missing, for example *"Move closer"*,
     *"Hold your head still"*, or *"The left pupil is not tracked steadily"*. A sample is
     only taken when everything is fine.
   - **S** solves when both bands have enough samples.
6. **Review.** Accuracy per eye and per band (green < 1.0°, amber < 1.5°), eye separation
   (expected 50–75 mm) and other checks. **Enter** accepts and saves. You can also redo the
   near (**N**) or far (**F**) samples, or everything (**A**). Nothing is saved before you
   accept.

### Capturing samples

| | |
|---|---|
| **Space**, **C**, clicker *next* (PageDown) | Capture |
| **Backspace**, clicker *back* (PageUp) | Undo the last sample |
| **M** | Switch between Manual and Dwell capture |
| **Tab** | Move the marker |

- **Manual:** capture with a key or a presenter clicker, which is handy from 2 m away.
- **Dwell:** hold your gaze on the marker. A ring fills in about 1 s, then it beeps and
  captures.

Each sample averages about 1 s of frames. To capture single frames like the old tracker,
set `frames_averaged` to 1 in `FrontCameraApp/settings/app_settings.json`.

After a capture, the app warns if your gaze didn't move like the marker did (*"were you
looking at the marker?"*). Backspace removes that sample.

## Desktop overlay (O)

The app minimizes, and the monitor shows four small markers in its corners, a ring where you
look and a heatmap. Clicks go through to the desktop underneath. The front camera must
see at least two of the corner markers.

- **Ring styles:** a solid ring means both eyes are tracked, a dashed ring one eye, a dotted
  ring that your gaze is off the screen.
- **Tray icon:** *Heatmap on/off*, *Save heatmap* (PNG with a screenshot behind it, plus a
  CSV of seconds per cell, in `FrontCameraApp/sessions/`), and *Stop the overlay and show
  the app*.
- **Monitors wider than the front camera's view:** set `overlay_midpoints` to `true` in
  `app_settings.json` to add markers at the edge midpoints.

The overlay can't draw over exclusive full-screen games or Windows security prompts.

## Eye data capture (E)

Records what the pupil detector sees, to replay a detector change on real frames or to train
PupilNet (`PupilNet/README.md`). **E** starts and stops it on any page, calibrated or not, except
Cameras (eye tracking pauses there), and the Left/Right eye pages of the wizard have a *Record
eye frames* button; it records both eyes. Frames captured
before the eye models are locked have no eye circle, so they test the whole-frame search; to
test tracking as it runs, capture from the Tracking view. Each capture goes to
`sessions/eye_capture_<time>_<subject>/` (the subject is `unknown` unless set):
- every 2nd frame of each eye as a grey PNG (`capture_every` in `app_settings.json`), taken
  before anything is drawn on it;
- `frames.csv`: the current detector's ellipse, confidence and quality for every frame;
- `meta.json`: the subject, cameras and settings.

For PupilNet training, set `capture_subject` in `app_settings.json` to an anonymous subject ID (`P01`, `P02`, …) and the same one for every session of the
same person: the training keeps whole people out of training so they can be used for testing.

**Protocol, about 3 minutes per session.** Calibrate the eye models first (Tracking view, eye
bars full), press **E**, then:
1. Look around slowly in every direction, out to the extremes, for 30 s.
2. Fixate 9 points (corners, edge midpoints, center) for 2 s each.
3. Blink normally 10 times, then squint and half-close the eyes a few times.
4. Look down at the floor (the upper lid covers the pupil) and up at the ceiling.
5. If the person wears glasses: repeat 1–2 with glasses on.
6. Take the headset off, put it back on, and repeat 1.

Press **E** to stop. Aim for 6 or more people, 2 sessions each (another day or other room
lighting if possible).

## Files

| File | |
|---|---|
| `FrontCameraTracker/front_camera_calibration.json` | Lens calibration and eye positions (shared with the old tracker) |
| `FrontCameraTracker/gaze_calibration.json` | The last accepted gaze calibration (shared) |
| `FrontCameraTracker/calibrations/` | Every capture session, saved after each sample |
| `FrontCameraApp/settings/camera_setup.json` | Which camera is which |
| `FrontCameraApp/settings/app_settings.json` | Capture mode, dwell time, frames averaged, overlay options, `gaze_vector_path` |
| `FrontCameraApp/settings/monitors.json` | Screen scale per monitor |
| `FrontCameraApp/logs/` | Log file, one per day |
| `FrontCameraApp/sessions/` | Recordings (X) and eye data captures (E) |
| `gaze_vector.txt` (working directory) | Live gaze for Unity, same format as the old tracker |

The app's extra information (which cameras, frame sizes, settings) is stored under an
`app_meta` key that the old tracker ignores.

## Troubleshooting

- **A camera stays "opening" or "absent":** check the cable. A camera that comes back is
  reconnected by itself, even at a different index. If the old tracker is still running, close
  it: the app warns when `gaze_vector.txt` is being written by someone else.
- **"Gaze calibration cleared":** something the calibration depends on changed (an eye
  model was locked again, a camera now delivers a different frame size). Press **R** or **W**.
- **Eye separation shown red:** check the screen scale first, then the lens calibration.
- **Video files instead of cameras** (testing):
  `python FrontCameraApp/run_app.py --video left=eye.mp4 --video right=eye.mp4 --video front=front.mp4`
