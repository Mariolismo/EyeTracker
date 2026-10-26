# Mariolismo Fork

Fork of [JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker), hosted at https://github.com/Mariolismo/EyeTracker.
`origin` = fork (push enabled), `upstream` = original (push disabled). Sync with `git fetch upstream && git merge upstream/main`.

## Diary

### Merged upstream PR #19: NumPy 2 pupil overflow fix
- **Problem:** under NumPy 2, `get_darkest_area()` sums `uint8` pixels without promotion, so the sum wraps at 256 and the pupil seed lands on the lashes.
- **Tried:** merged the whole PR (int casts, threshold clamps, cross-platform fixes). In the front-camera tracker, kept `main`'s camera/video code and re-applied only the casts, the "no darkest point" guard and the video filter.
- **Result:** on `eye_test.mp4`, average brightness at the picked point fell from 119 to 18 (the pupil is near 0). The ellipse stays on the pupil.

### Manual eye center override (O key, temporary)
- **Idea:** looking straight into the eye camera puts the pupil center exactly on the projected eyeball center, so the slow ray-intersection estimate can be skipped.
- **Tried:** press O while looking into the lens. The median of 30 confident pupil centers becomes the eye center, the sphere is affixed and the 5-point calibration is cleared (redo C). F returns to automatic.
- **Expected limits:** ~1 mm offset from the visual/optical axis difference (kappa angle), which C calibration should mostly absorb. The camera is hard to fixate up close, and the override goes stale if the headset slips.
- **Result:** works headless on `eye_test.mp4`; accuracy on the live rig still to test.

### Parallax-free gaze: two eyes + ArUco marker
- **Problem:** the 5-point finger calibration maps gaze direction to a front-camera pixel with a 2×2 matrix and ignores the eye→camera offset, so it is exact only at the calibration depth (~40 cm) and drifts by |offset|·|1/Z − 1/Z₀| elsewhere. One eye at one depth can't separate the offset from a small rotation.
- **Tried:** a per-eye ray model (3×3 gain + 3D origin) fitted on ArUco marker positions at two depths (NEAR < 0.6 m, FAR > 1.5 m), with binocular triangulation and an inverse-depth filter at runtime. Math in `gazecore/GAZE_GEOMETRY.md` (then in `FrontCameraTracker/`), user steps in `FrontCameraTracker/CALIBRATION_STEPS.md`. The code is split into `gaze_geometry.py` (pure math), `front_calibration.py` (camera, marker, ChArUco intrinsics, files) and the tracker (GUI and orchestration), with synthetic, rendered-image and headless tests.
- **Fixed on the live rig:**
  - OpenCV 5 returns ChArUco corners as (N, 2), which crashed the drawing; reshaped.
  - The lens calibration overfit with few views (corner distortion factor 0.37). Now k1/k2 only, plus a plausibility check.
  - Eye cameras that see the eye mirrored gave a negative gain; mirroring is detected and flipped.
  - A pupil glitch in one sample pushed one eye's fit to 16.7° RMS. Outlier samples are now dropped (screened with a fixed origin first, so the origin can't run off to infinity).
  - Samples got lost when a new session started. Each session is now saved to `calibrations/gaze_<date>_<time>.json`, the last good one to `gaze_calibration.json`, and it is restored at startup (or with L), including the locked eye centers.
- **Result:** 14 samples, one bad far sample dropped automatically. RMS 0.84° (L) and 0.95° (R), against 1.63° and 2.72° for the no-offset model on the same samples. Eye separation 66 mm. Works well in live use.
- **Still to improve:** it solves silently as soon as both groups have 6 samples, which is easy to miss, and C after that starts over.

### Implausible eye origins (parked for a CAD headset)
- **Problem:** the fitted eye origins are asymmetric (22 mm height difference, midpoint 14 mm right of the camera), although the eyes sit symmetrically around the front camera.
- **Tried:** refits with symmetric constraints on the same samples, plus a synthetic headset with asymmetric eye cameras.
- **Result:** the height split is noise. The sideways shift comes from the eye-camera non-linearity, because NEAR and FAR samples covered different gaze angles. It cancels near the calibration conditions but not when extrapolating. Plan for when we design CAD glasses: `FrontCameraTracker/FUTURE_CAD_HEADSET_PLAN.md`.

### Front camera app (PyQt5) on a shared `gazecore` package
- **Problem:** the front-camera tracker's math works, but its app doesn't. One 2010-line Tk + cv2 file, camera reads blocking the UI, a slow index probe at startup, camera choices forgotten, keys only in cv2 windows, status in the terminal, a camera stop or flip silently clearing the calibration, a silent solve, and the marker measured with a ruler.
- **Tried:** a new app beside the old tracker (`FrontCameraApp/`, guide in `FrontCameraApp/USER_GUIDE.md`), with **no tracking math changes**.
  - First a golden test recorded today's eye pipeline frame by frame (tag `golden-baseline`). Then the shared code moved into `gazecore/` and the per-eye logic became an `EyeTracker` class; both reproduce it bit for bit, and the old tracker now runs on it.
  - Cameras are matched by device ID, opened one at a time, read on their own threads, and reconnected by a watchdog.
  - The app goes straight to tracking when the calibration still matches the cameras, otherwise to a wizard at the first missing step. It covers screen scale, lens, each eye, near/far gaze with plain-language capture gates, an explicit solve and a review before anything is saved. Extra information goes under an `app_meta` key, so the old tracker still reads the files.
  - A desktop overlay shows the gaze ring and a heatmap on the monitor, from corner markers and the eye rays.
- **Fixed during development:** IPPE alone left the screen pose ~3° off when seen square-on; each candidate is now refined with LM (0.13°, ~2 mm on the screen in the synthetic test). A gaze file from the old tracker that is older than a lens calibration made by the app is now refused.
- **Result:** 65 tests pass, including both golden tests. Offscreen on video files, the existing calibration opens tracking directly (24 fps processed), and every page and the overlay run. Not run on the rig yet.
- **Still to improve:** everything on the rig. That means a full wizard run (target ~0.84°/0.95°, 66 mm), V accuracy against the old tracker, both capture modes and a clicker at 2 m. For the overlay, the corner and center error in mm at 0.6 m, pose dropouts and repaint cost.
