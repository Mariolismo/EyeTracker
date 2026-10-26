# Future plan: eye positions on a CAD headset

Status: **parked** until we design CAD glasses that hold the front camera and both eye cameras in known, fixed positions. Written 2026-10-26 from an analysis of the first binocular ArUco calibration.

## Problem found

The calibration in `GAZE_GEOMETRY.md` fits a 3D origin per eye (the eye's rotation center in the front-camera frame: x right, y down, z forward). On the 2026-10-26 session (14 samples, NEAR ≈ 0.5 m, FAR 1.6–2.75 m, one sample dropped):

| | x | y | z | 1σ (x, y, z) |
|---|---|---|---|---|
| Left eye | −17.2 mm | 36.6 mm | −14.2 mm | 5.9, 6.0, 17.5 mm |
| Right eye | 44.6 mm | 58.5 mm | −18.0 mm | 6.8, 7.3, 20.6 mm |

Gaze accuracy was good (RMS 0.84° / 0.95°), but the geometry is not plausible. The eyes are symmetric about the camera, so the separation should lie almost entirely along x. Instead:

- The eyes are 22 mm apart in height, which implies a 19.5° camera roll that isn't there.
- Their midpoint is 14 mm to the right of the camera.
- Depth is effectively undetermined (±20 mm).

Refitting the same samples with constraints:

| Constraint | Combined RMS | Left (mm) | Right (mm) |
|---|---|---|---|
| None (6 position parameters) | 0.90° | (−17, 37, −14) | (45, 59, −18) |
| Shared height and depth | 0.97° (p ≈ 0.08) | (−18, 47, −17) | (45, 47, −17) |
| Shared height and depth, camera centered | 1.07° (p ≈ 0.01) | (−31, 48, −27) | (31, 48, −27) |

The height split is noise. The sideways shift is a real pull in the data, but it comes from a systematic error, not from the geometry.

## Why it happens

The eye position is observable only through the parallax between NEAR and FAR. An eye 31 mm off-axis turns 3.5° to fixate a centered target at 0.5 m and 0.9° at 2 m. Fitted offset = angle difference ÷ (1/Z_near − 1/Z_far), so with 1.5 dioptres between the depths, **every 1° of systematic NEAR-vs-FAR error moves the fitted origin about 12 mm**.

Each eye's raw gaze carries a non-linear, angle-dependent error. Its sources are where the eye camera views the eye from and the eyeball-size estimate (272 px left against 172 px right in this session). The asymmetric eye-camera mounts make this error different for each eye. Per-eye rotation and linear gain can't absorb it. When NEAR and FAR samples cover different gaze angles (in this session, NEAR about ±16° and FAR about ±24° at the edges), the fit can only explain the leftover error as eye position.

## Simulation evidence

Synthetic headset: true eyes at (±31, 48, −27) mm, asymmetric eye cameras with a few percent of non-linearity. Errors are angular errors of the binocular gaze point as seen from the front camera.

| Variant | Mean error at 0.3 / 0.5 / 2 / 5 m | Max at 5 m |
|---|---|---|
| Uneven layout (like the real session), free origins | 0.98 / 0.46 / 0.68 / 2.28° | 11.7° |
| Same 3×3 angle grid at both depths, free origins | 0.55 / 0.38 / 0.39 / 1.06° | 3.2° |
| Uneven layout, shared height and depth | 0.71 / 0.37 / 0.45 / 1.10° | 3.7° |
| Uneven layout, **origins known exactly** | 0.49 / 0.34 / 0.33 / 0.84° | 2.9° |
| Uneven layout, origins off by ~3 mm sideways/height and 8 mm in depth | 0.81 / 0.48 / 0.47 / 2.44° | 10.0° |
| Old no-offset model (for reference) | 4.4 / 1.5 / 2.2 / 3.0° | 3.5° |

What the simulation shows:

- Errors cancel inside the calibrated depths and angles, so live use near calibration conditions looks good. They come back when extrapolating, especially far away at the image edges and very close.
- A low calibration RMS can hide this. The uneven layout fits its own samples better and performs worse in use.
- Exact eye positions fix the extrapolation, but they must be accurate to about 1–2 mm sideways. A 3 mm error loses most of the benefit, because the sideways position sets the distance between the two rays and therefore the vergence depth.
- No choice of eye position removes the eye-camera non-linearity itself: about 2.7° maximum error at 0.3 m remains even with exact positions.

## What a CAD headset gives us

- The **front camera on the midline**, known to well under 1 mm, gives each eye's sideways position directly from the user's distance between pupils.
- The **front-camera-to-eye-camera transforms** are known, so each eye's rotation is fixed by design and only kappa (visual vs optical axis, about 5°) is left to calibrate.
- **Mirror-symmetric eye-camera mounts** make the non-linearity roughly symmetric between the eyes, so it no longer pushes the eye midpoint sideways.

Eye positions still depend on the user (distance between pupils, nose-bridge fit, how far the eyes sit behind the frame), so height and depth stay calibrated per user, or per user and headset.

## Plan

### Headset design requirements
1. Front camera centered on the midline, with its optical axis parallel to the frame's straight-ahead direction. Record the as-built position from CAD.
2. Eye cameras mounted mirror-symmetrically, at the same distance and viewing angle for both eyes.
3. Known front-camera-to-eye-camera extrinsics from CAD, checked once per unit (target: < 1 mm, < 0.5°).
4. A defined eye-box reference (for example, the nose pad or the frame plane) from which to measure eye height and depth.

### Calibration changes
1. **Partially fixed joint fit of both eyes.** Sideways position from distance between pupils / 2 ± the midline offset, held fixed. Height and depth shared by both eyes and fitted from NEAR plus FAR samples (2 parameters). Per eye, fit only rotation and gain. This needs a joint version of `fit_eye_model`; today `fixed_origin` is all-or-nothing per eye.
2. **Per-user distance between pupils**, entered once (pupillometer or mirror and ruler, about 1 mm), saved with the calibration.
3. **3×3 on-screen target guide**: solve only when NEAR and FAR cover the same cells, so the non-linearity cancels in the NEAR-vs-FAR difference.
4. **Plausibility warnings** after solving: height or depth difference between the eyes > 10 mm, separation outside 55–75 mm, or eyes far from the CAD eye box.
5. Once height and depth are known for a user, recalibrate at a single depth with fixed origins (already supported).

### Stretch: eye positions from the eye cameras
With metric eye-camera intrinsics, the eyeball model in millimetres instead of pixels, and the CAD extrinsics, each eye's rotation center could be computed directly from its own eye camera and transformed into the front-camera frame. Calibration would then fit only kappa and gain. A metric eyeball model would also address the non-linearity at its source.

### Validation
- Accuracy test at about 0.3 m and 3–4 m with the marker at the image edges, not only at the calibration depths.
- Acceptance: errors at the edges and at untested depths within about 1.5× the calibration RMS, and fitted eye positions plausible against CAD (distance between the eyes along x, no height or depth difference beyond noise).

## Not recommended
- **Extra curvature terms in the per-eye gaze model:** more parameters trade off against eye position and make the fit degenerate again (see `GAZE_GEOMETRY.md`).
- **Measuring eye rotation centers with a ruler alone:** depth (about 13 mm behind the cornea) can't be measured to the 1–2 mm needed.

## Reproducing
The analysis used throwaway scripts that are not in the repo:
- refit with uncertainty (covariance, bootstrap, leave-one-out)
- constrained refits
- the synthetic headset

All three only use `gaze_geometry.fit_eye_model` and `binocular_gaze_point` on `gaze_calibration.json` or synthetic samples. Re-create them if the numbers need checking.
