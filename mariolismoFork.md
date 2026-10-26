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
- **Tried:** a per-eye ray model (3×3 gain + 3D origin) fitted on ArUco marker positions at two depths (NEAR < 0.6 m, FAR > 1.5 m), with binocular triangulation and an inverse-depth filter at runtime. Math in `FrontCameraTracker/GAZE_GEOMETRY.md`, user steps in `FrontCameraTracker/CALIBRATION_STEPS.md`. The code is split into `gaze_geometry.py` (pure math), `front_calibration.py` (camera, marker, ChArUco intrinsics, files) and the tracker (GUI and orchestration), with synthetic, rendered-image and headless tests.
- **Fixed on the live rig:**
  - OpenCV 5 returns ChArUco corners as (N, 2), which crashed the drawing; reshaped.
  - The lens calibration overfit with few views (corner distortion factor 0.37). Now k1/k2 only, plus a plausibility check.
  - Eye cameras that see the eye mirrored gave a negative gain; mirroring is detected and flipped.
  - A pupil glitch in one sample pushed one eye's fit to 16.7° RMS. Outlier samples are now dropped (screened with a fixed origin first, so the origin can't run off to infinity).
  - Samples got lost when a new session started. Each session is now saved to `calibrations/gaze_<date>_<time>.json`, the last good one to `gaze_calibration.json`, and it is restored at startup (or with L), including the locked eye centers.
- **Result:** 14 samples, one bad far sample dropped automatically. RMS 0.84° (L) and 0.95° (R), against 1.63° and 2.72° for the no-offset model on the same samples. Eye separation 66 mm. Works well in live use.
- **Still to improve:** it solves silently as soon as both groups have 6 samples, which is easy to miss, and C after that starts over.
