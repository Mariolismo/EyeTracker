# Parallax-free gaze mapping into a front camera

Mathematical documentation for the front-camera gaze mapping. Every number in the worked example comes
from the code itself and can be reproduced with the commands in §11.

| File | Responsibility | Depends on |
|---|---|---|
| [`gaze_geometry.py`](gaze_geometry.py) | The math: eye model, fitting, triangulation, depth filter, projection (§3–§9) | numpy (cv2 only for lens distortion) |
| [`front_calibration.py`](front_calibration.py) | The workflow: camera model and file, ArUco target, ChArUco intrinsics, sample collection and solve, runtime mapping, accuracy test. No UI, no globals. | `gaze_geometry`, numpy, OpenCV ≥ 4.7 |
| [`Orlosky3DEyeTrackerFrontCamera.py`](../FrontCameraTracker/Orlosky3DEyeTrackerFrontCamera.py) | Pupil tracking per eye, GUI, keys, drawing; connects the two modules above | all of the above, tkinter |

---

## 1. Summary

A head-mounted eye tracker estimates **where the eye is pointing** (a direction). Scene applications need
**which pixel of the forward camera the person looks at**. The previous tracker mapped direction to pixel
directly. That mapping silently assumes that the eye and the camera sit at the same point. They do not:
on a typical headset the eye is 3–5 cm from the camera, so the mapping is exact only at the depth used
during calibration. At other depths the error is several degrees (*parallax error*, [5]).

The new algorithm:

1. models each eye as a **3D ray**, with an origin (the eye's rotation center) and a direction, both in the
   front camera frame;
2. calibrates that ray model from **metric 3D targets**: an ArUco marker whose pose the front camera measures
   [3, 4], collected at **two or more depths**, which is what makes the eye origin identifiable;
3. at runtime, intersects the two eyes' rays (**vergence**) to estimate the gaze depth, smooths it in
   **inverse depth**, and projects the 3D gaze point into the front camera.

On the synthetic headset of §10, the worst-case error over the field of view drops from **4–7°** (no
offset) to **≈0.7°**, flat from 0.25 m to 5 m.

---

## 2. Frames and notation

| Symbol | Meaning |
|---|---|
| Front camera frame | OpenCV convention: $x$ right, $y$ down, $z$ forward, metres. All 3D quantities live here. |
| $K$, $\mathbf{k}$ | Front camera intrinsic matrix and distortion coefficients [2]. |
| $\mathbf{g}$ | Raw gaze direction from the eye tracker, in the eye tracker's own (arbitrary, unitless) frame. |
| $\mathbf{o}_e$ | Rotation center of eye $e \in \{L, R\}$ in the front camera frame (m). |
| $\mathbf{d}_e$ | Calibrated gaze direction of eye $e$ in the front camera frame (unit vector). |
| $\mathbf{P}_i$ | 3D position of calibration target $i$ (marker center), measured by the front camera (m). |
| $Z$ | Depth = $z$ coordinate in the front camera frame. $1/Z$ is in dioptres (D = m⁻¹). |
| $\pi(\cdot)$ | Pinhole projection with $K$, $\mathbf{k}$: 3D point → pixel [1, Ch. 6]. |
| $\operatorname{n}(\mathbf{v})$ | Normalization $\mathbf{v}/\lVert\mathbf{v}\rVert$. |

The raw gaze $\mathbf{g}$ comes from the tracker's eye model: an eyeball sphere fitted from pupil-ellipse
normals, in the spirit of Świrski & Dodgson [10]. Its frame, scale and small gain errors are unknown. The
calibration absorbs them.

---

## 3. The problem: parallax

Place the camera at the origin and the eye at $\mathbf{o}$. A point $\mathbf{P}$ at depth $Z$ is seen from
the camera along $\operatorname{n}(\mathbf{P})$ and from the eye along $\operatorname{n}(\mathbf{P}-\mathbf{o})$.
For $\lVert\mathbf{o}\rVert \ll Z$ the angle between the two lines of sight is, to first order,

$$
\delta(Z) \;\approx\; \frac{\lVert \mathbf{o}_\perp \rVert}{Z},
$$

where $\mathbf{o}_\perp$ is the component of the offset perpendicular to the line of sight. A direction→pixel
calibration performed at depth $Z_0$ absorbs exactly $\delta(Z_0)$. At any other depth the remaining error is

$$
\boxed{\;\varepsilon(Z) \;\approx\; \lVert \mathbf{o}_\perp \rVert \,\left| \frac{1}{Z} - \frac{1}{Z_0} \right|\;}
\tag{1}
$$

Mardanbegi & Hansen analyse this error for monocular head-mounted trackers [5]. Two consequences follow from (1):

* The error is linear in **inverse depth**, not in depth. It levels off at
  $\lVert\mathbf{o}_\perp\rVert / Z_0$ toward infinity and grows quickly up close.
* Calibrating at arm's length ($Z_0 \approx 0.4$ m) makes the far field as bad as it can be.

**Numeric example.** In the synthetic headset (§10) the left eye sits at
$\mathbf{o}_L = (-32,\ 35,\ -15)$ mm, so $\lVert \mathbf{o}_\perp \rVert = \sqrt{32^2+35^2} = 47.4$ mm.
With calibration at $Z_0 = 0.4$ m:

| Viewing depth $Z$ | 0.25 m | 0.4 m | 1 m | 2 m | 5 m | ∞ |
|---|---|---|---|---|---|---|
| $\varepsilon(Z)$ from (1) | 4.08° | 0° | 4.08° | 5.43° | 6.25° | 6.79° |

For example, at 5 m: $0.0474 \times |0.2 - 2.5| = 0.109$ rad $= 6.25°$.

---

## 4. Why one calibration depth cannot fix it

Suppose all targets lie at depth $Z_0$: $\mathbf{P}_i = Z_0 \mathbf{r}_i$, where $\mathbf{r}_i$ is the camera
ray with $r_z = 1$. Seen from the eye,

$$
\mathbf{P}_i - \mathbf{o} \;=\; Z_0\left(\mathbf{r}_i - \frac{\mathbf{o}}{Z_0}\right).
$$

* The lateral part $\mathbf{o}_\perp/Z_0$ shifts every ray by **the same** tangent-plane vector. To first order
  this is a small rotation, which the calibration rotation $R$ (§5) already models.
* The axial part $o_z$ scales the tangent coordinates by $Z_0/(Z_0 - o_z)$, which looks like a gain change.
  The gain matrix $A$ (§5) already models that.

So at a single depth, $\mathbf{o}$ is confounded with $(R, A)$ to first order. Only second-order effects
remain, of the size of the gaze noise, and the fit is ill-conditioned. Numerically, for the left eye at
$Z_0 = 0.4$ m, the eye-vs-camera angle differs by 4.71°, 4.41° and 3.79° across the image ($x = -0.2,\ 0,\ +0.2$).
That is nearly constant. A rotation absorbs the 4.4° mean, and the ±0.5° variation is comparable to
tracker noise.

**Two depths break the confusion**: the shift is $\mathbf{o}_\perp/Z_1$ at one depth and
$\mathbf{o}_\perp/Z_2$ at the other, and no single rotation explains both. The module therefore rejects
calibration data whose inverse-depth span is below 1 D (constant `MIN_INVERSE_DEPTH_SPAN`). The UI bands
NEAR < 0.6 m (> 1.67 D) and FAR > 1.5 m (< 0.67 D) always satisfy it.

**Scale needs metric targets.** If target depths were unknown (e.g. a fingertip), the constraints in §6 are
invariant under $(\mathbf{o}, \mathbf{P}_i) \mapsto (s\,\mathbf{o}, s\,\mathbf{P}_i)$. This is the same
scale ambiguity as in two-view geometry [1, Ch. 9–10]. A marker of known size fixes $s$.

**Two eyes calibrated together do not change this.** Each eye has its own $\mathbf{o}_e$, confounded in the
same way. The second eye adds information at **runtime**, where vergence measures depth (§8).

---

## 5. Eye ray model

For eye $e$, with parameters $(B_e, R_e, A_e, \mathbf{o}_e)$:

$$
\begin{aligned}
\mathbf{h} &= B_e\,\mathbf{g}, \qquad
\mathbf{u} = \begin{pmatrix} h_x/h_z \\ h_y/h_z \end{pmatrix}
&& \text{(tangent-plane coordinates)}\\[2pt]
\mathbf{d}_e &= \operatorname{n}\!\left( R_e \begin{pmatrix} A_e\,\mathbf{u} \\ 1 \end{pmatrix} \right)
&& \text{(direction in the front camera frame)}\\[2pt]
\text{ray}_e(t) &= \mathbf{o}_e + t\,\mathbf{d}_e, \quad t > 0.
\end{aligned}
\tag{2}
$$

* **$B_e$** (fixed, not fitted): an orthonormal basis whose third row is the mean calibration gaze. It
  keeps $h_z > 0$ over the working range, so the tangent coordinates are well defined (`_tangent_basis`).
* **$A_e$** (3 parameters): symmetric 2×2 gain. It absorbs scale and anisotropy errors of the eye model.
  It is symmetric because an antisymmetric part would be an in-plane rotation, which $R_e$ already covers.
* **$R_e$** (3 parameters): rotation from the eye tracker frame to the front camera frame. This includes
  the constant part of the **angle kappa** between the optical axis (what pupil-based trackers measure) and
  the visual axis (what the person actually fixates), typically about 5° [9].
* **$\mathbf{o}_e$** (3 parameters): the eye's rotation center. This is the parameter a direction→pixel map
  is missing.

That is 9 parameters per eye.

**Why not a general 3×3 (projective) map?** The first implementation used $\mathbf{d} = \operatorname{n}(M\mathbf{g})$
with a general $M$ (8 degrees of freedom), fitted by DLT [1, Ch. 4]. The perspective row of $M$ could imitate
an axial shift of $\mathbf{o}$, and on synthetic data the fit moved the eye to $o_z = +1.09$ m. Restricting
the model to rotation + symmetric gain removes that degeneracy. It costs a small amount of flexibility: on
data generated with a full 3D linear distortion, about 0.2° of model mismatch remains (§10).

---

## 6. Metric calibration targets

### 6.1 Front camera intrinsics

$K$ and the distortion $\mathbf{k}$ are estimated once per camera with Zhang's planar method [2]. The
implementation uses about 15 views of a ChArUco board, a chessboard whose squares contain ArUco markers so
that it can be detected under partial occlusion [3, 13], and passes them to `cv2.calibrateCamera`. Until
this is done, the tracker falls back to a pinhole with a 60° diagonal field of view:

$$
f = \frac{\sqrt{W^2+H^2}}{2\tan(30°)} = \frac{800}{1.1547} = 692.8\ \text{px},
\qquad
K = \begin{pmatrix} 692.8 & 0 & 320 \\ 0 & 692.8 & 240 \\ 0 & 0 & 1 \end{pmatrix}.
$$

### 6.2 Marker pose

The gaze target is an ArUco marker (dictionary `DICT_4X4_50`, id 0) [3]. Its four corners are detected,
and `cv2.solvePnP` with the `SOLVEPNP_IPPE_SQUARE` solver returns its pose. That solver uses Infinitesimal
Plane-based Pose Estimation (IPPE) [4]. The translation vector **is** the marker center $\mathbf{P}_i$ in the
front camera frame. The intrinsics board uses a different dictionary (`DICT_5X5_100`), so the two can never
be confused.

Pose accuracy scales with the marker's size in pixels. A 60 mm marker at 2 m is only about 19 px wide, and
in simulation its depth came out 4% long. The default size is therefore 100 mm, which gave 2%. Errors in
$Z$ enter the fit only through $1/Z$ (eq. 1), so a few percent at 2 m (≈0.01 D) is negligible.

---

## 7. Fitting

### 7.1 Residuals

For samples $\{(\mathbf{g}_i, \mathbf{P}_i)\}_{i=1}^N$, the model direction $\mathbf{d}_i$ (eq. 2) should
point from $\mathbf{o}$ to $\mathbf{P}_i$. The residual is the cross product of the two unit vectors:

$$
\mathbf{r}_i(\theta) = \operatorname{n}(\mathbf{P}_i - \mathbf{o}) \times \mathbf{d}_i,
\qquad
\lVert \mathbf{r}_i \rVert = \sin(\text{angular error}_i).
\tag{3}
$$

Minimizing $\sum_i \lVert\mathbf{r}_i\rVert^2$ is, for small errors, least squares on angular error seen
from the eye. That is the natural noise model for a gaze tracker. Each sample gives 2 independent
equations, so 9 parameters need at least 5 samples. The module requires 6 (`MIN_SAMPLES`), and the UI
requires 6 per depth band.

### 7.2 Parameterization and initialization

* **Rotation**: $R = R_0 \exp([\mathbf{w}]_\times)$, with $\mathbf{w}\in\mathbb{R}^3$ a rotation vector
  expanded by Rodrigues' formula,
  $\exp([\mathbf{w}]_\times) = I + \sin\phi\,[\mathbf{k}]_\times + (1-\cos\phi)[\mathbf{k}]_\times^2$ with
  $\phi = \lVert\mathbf{w}\rVert$ and $\mathbf{k} = \mathbf{w}/\phi$. Optimizing a small correction around
  $R_0$ avoids singularities.
* **Initial rotation $R_0$**: the rotation that best aligns $\operatorname{n}([\mathbf{u}_i, 1])$ with
  $\operatorname{n}(\mathbf{P}_i)$. This is the orthogonal Procrustes / Kabsch problem [7, 8]:
  $H = \sum_i \mathbf{x}_i \mathbf{y}_i^\top = U\Sigma V^\top$, and
  $R_0 = V \operatorname{diag}(1, 1, \det(VU^\top))\, U^\top$. The determinant term prevents a reflection [8].
* **Mirrored eye cameras**: an eye image that is flipped (mirror-mounted camera, or a flip setting)
  gives left-handed raw gaze. $R$ and a positive-definite $A$ cannot represent a reflection; the fit
  would end with $\det A < 0$ and be rejected. Before fitting, the sign of $\det$ of the least-squares
  linear map $[\mathbf{u}_i, 1] \mapsto \operatorname{n}(\mathbf{P}_i)$ is checked; if it is negative,
  the tangent $x$ axis is mirrored (the first row of $B$ is negated), so $\mathbf{u}$ is right-handed
  again. Because the flip is stored in $B$, runtime rays (eq. 2) apply it automatically.
* **Initial values for the rest**: $A = I$ and $\mathbf{o} = \mathbf{0}$. This starts from the classic
  no-offset solution, which is already within a few degrees, and lets the optimizer move the eye.

### 7.3 Levenberg–Marquardt

The parameters $\theta = (\mathbf{w}, a_{11}, a_{12}, a_{22}, \mathbf{o})$ are refined by
Levenberg–Marquardt [6, 11] with Marquardt's diagonal scaling:

$$
\left(J^\top J + \lambda\,\operatorname{diag}(J^\top J)\right)\Delta = -J^\top \mathbf{r},
$$

where $J$ is a forward-difference Jacobian (step $10^{-7}$). $\lambda$ is divided by 10 after an
accepted step and multiplied by 10 after a rejected one [1, App. 6].

### 7.4 Validity checks

`fit_eye_model` raises `ValueError` instead of returning a poor model when:

* fewer than 6 samples, mismatched shapes, non-finite values, or targets behind the camera;
* inverse-depth span below 1 D while the origin is being fitted (§4);
* the tangent coordinates do not span two axes (all targets on a line);
* $\operatorname{cond}(J^\top J) > 10^{12}$ (degenerate geometry);
* $\det A \le 0$ or $\operatorname{cond}(A) > 10$ (physically implausible gain).

It reports the per-sample angular errors and their RMS overall and per depth band.

### 7.5 Fixed-origin refit

$\mathbf{o}_e$ depends on the headset, not the session. Once it has been fitted, it is saved and later
calibrations fit only $(R, A)$. With $\mathbf{o}$ known, the depth confusion of §4 is gone, so a single
depth suffices (`fixed_origin=` argument, the K key in the UI). Passing `fixed_origin=0` gives the
**no-offset baseline**. The accuracy test reports it next to the full model, which shows the parallax that
was removed.

---

## 8. Runtime: binocular gaze point

### 8.1 Triangulation

The two eye rays almost never intersect exactly. The module takes the **midpoint of their common
perpendicular** [1, Ch. 12; 12]. With $\mathbf{w}_0 = \mathbf{o}_1 - \mathbf{o}_2$ and
$a = \mathbf{d}_1\!\cdot\!\mathbf{d}_1,\ b = \mathbf{d}_1\!\cdot\!\mathbf{d}_2,\ c = \mathbf{d}_2\!\cdot\!\mathbf{d}_2,\
d = \mathbf{d}_1\!\cdot\!\mathbf{w}_0,\ e = \mathbf{d}_2\!\cdot\!\mathbf{w}_0$:

$$
s = \frac{be - cd}{ac - b^2},\qquad
t = \frac{ae - bd}{ac - b^2},\qquad
\mathbf{P}^* = \tfrac12\big[(\mathbf{o}_1 + s\mathbf{d}_1) + (\mathbf{o}_2 + t\mathbf{d}_2)\big].
\tag{4}
$$

If $ac - b^2 \approx 0$ (parallel rays) or $s \le 0$ or $t \le 0$ (rays diverge or converge behind the head),
the depth is treated as "far". The midpoint method is not statistically optimal [12]. It is used here
because only the depth of $\mathbf{P}^*$ is kept (§8.3), and that depth is clamped and smoothed.

### 8.2 Clamping and smoothing in inverse depth

$Z^* = P^*_z$ is clamped to $[0.2, 5]$ m. Beyond 5 m, eq. (1) changes by less than 0.6° for a 47 mm offset.
It is then smoothed with an exponential filter **in inverse depth**:

$$
\zeta_k = \zeta_{k-1} + \alpha\left(\frac{1}{Z^*_k} - \zeta_{k-1}\right), \qquad \alpha = 0.2,\qquad Z_k = 1/\zeta_k.
$$

Inverse depth is the right variable for two reasons: the parallax error (1) is linear in $1/Z$, and
vergence noise is roughly uniform in $1/Z$ (§9). Averaging in metres lets far estimates dominate. For
example, 1 m and 50 m average to 25.5 m, whereas 1 D and 0.02 D average to 0.51 D (1.96 m). Because (1) is
linear in $1/Z$, the inverse-depth mean is also the single depth that minimizes the mean squared parallax
error over the samples.

### 8.3 Gaze point and pixel

Each available eye's ray is cut at the smoothed depth,
$\mathbf{Q}_e = \mathbf{o}_e + \frac{Z_k - o_{e,z}}{d_{e,z}}\mathbf{d}_e$. The two points are averaged and
projected: $\text{pixel} = \pi(\tfrac12(\mathbf{Q}_L + \mathbf{Q}_R))$. If one eye is lost, the other eye's
ray is cut at the last smoothed depth, so parallax stays corrected with a slightly stale depth.

---

## 9. Error budget: what vergence buys

For symmetric fixation at depth $Z$ with interpupillary distance $I$, the vergence angle is
$\theta = 2\arctan\frac{I}{2Z} \approx \frac{I}{Z}$, so

$$
\frac{1}{Z} \approx \frac{\theta}{I}
\quad\Longrightarrow\quad
\delta\!\left(\frac1Z\right) \approx \frac{\delta\theta}{I}.
\tag{5}
$$

Vergence noise $\delta\theta$ is therefore uniform in inverse depth. Duchowski et al. measure gaze depth
from vergence and observe exactly this loss of precision with distance [14]. Substituting (5) into (1)
gives the **residual parallax after vergence correction**:

$$
\boxed{\;\varepsilon_{\text{res}} \approx \lVert \mathbf{o}_\perp \rVert\, \frac{\delta\theta}{I}\;}
\tag{6}
$$

This is independent of depth.

**Numeric example.** $I = 64$ mm, so the vergence angle is 7.32° at 0.5 m, 3.67° at 1 m, 1.83° at 2 m and
0.73° at 5 m. With $\delta\theta = 0.5°$ (0.00873 rad), (5) gives $\delta(1/Z) = 0.136$ D. That is a poor
**metric** depth at 2 m (0.5 D ± 0.136 D, about ±27%). But (6) gives
$\varepsilon_{\text{res}} = 0.0474 \times 0.136 = 0.0065$ rad $= 0.37°$, below typical tracker accuracy. In
short, vergence depth is too noisy for measuring distances far away, but it is precise enough to correct
parallax. If the two eyes' errors are independent, $\delta\theta$ is about $\sqrt2$ times the per-eye noise.

---

## 10. Worked numeric example (end to end)

The synthetic headset is the one in [`test_gaze_geometry.py`](tests/test_gaze_geometry.py).

**Ground truth**

| | Left eye | Right eye |
|---|---|---|
| Origin $\mathbf{o}$ (mm) | (−32.0, 35.0, −15.0) | (32.0, 35.0, −15.0) |
| Eye tracker distortion | rotation 25° about (0.3, 1, 0.2), gains diag(1.15, 0.90, 1) | rotation −20° about (−0.2, 1, 0.4), gains diag(0.95, 1.10, 1) |
| Gaze noise | 0.5° (Gaussian, random axis) | 0.5° |

The distortion is applied as $\mathbf{g} = \operatorname{n}(D^{-1}\mathbf{d}_{\text{true}})$. It is a full 3D
linear map, deliberately not of the form (2), so the test also measures model mismatch.

**Step 1: one sample.** The target $\mathbf{P} = (0.10, -0.05, 1.00)$ m projects with $K$ (§6.1) to pixel
(389.3, 205.4). Seen from the left eye, the true direction is
$\operatorname{n}(\mathbf{P}-\mathbf{o}_L) = (0.1285, -0.0828, 0.9882)$. The tracker reports the distorted
$\mathbf{g} = (-0.2450, 0.0509, 0.9682)$, which looks nothing like it. The calibration has to learn the
mapping.

**Step 2: calibration.** 16 targets on a 4×4 grid across the image at 0.4 m and 16 at 2.0 m: $N = 32$, with
an inverse-depth span of $2.5 - 0.5 = 2.0$ D ≥ 1 D.

| | Left eye | Right eye |
|---|---|---|
| Fitted origin (mm) | (−31.9, 34.7, −17.3) | (31.5, 35.8, −18.7) |
| Origin error (mm) | (0.1, 0.3, 2.3) | (0.5, 0.8, 3.7) |
| Fitted gain $A$ | [[1.102, 0.002], [0.002, 0.872]] | [[0.956, 0.001], [0.001, 1.085]] |
| RMS near / far / all | 0.44° / 0.51° / 0.47° | 0.32° / 0.47° / 0.40° |
| Eye separation | 63.4 mm (true 64.0) | |

The lateral origin is recovered within 1 mm and the axial within 4 mm. Axial is weaker, as expected:
moving the eye along the line of sight changes the angles least. The RMS is at the 0.5° noise floor.

**Step 3: runtime, single frame.** Both eyes look at $\mathbf{P} = (0.10, -0.05, 1.00)$:

| | Left ray | Right ray |
|---|---|---|
| Origin $\mathbf{o}$ | (−0.0319, 0.0347, −0.0173) | (0.0315, 0.0358, −0.0187) |
| Direction $\mathbf{d}$ | (0.1320, −0.0798, 0.9880) | (0.0694, −0.0833, 0.9941) |

Vergence angle: 3.61°. Triangulation (4) gives $\mathbf{P}^* = (0.1008, -0.0465, 0.9753)$, a depth of
0.975 m, i.e. 0.026 D from the truth. It projects to pixel (391.6, 207.0) against the true (389.3, 205.4):
**0.23°** error.

**Step 4: accuracy across depth.** Worst case over a 5×5 grid filling the image, left eye, at a known depth:

| Depth | 0.25 m | 0.4 m | 1 m | 5 m |
|---|---|---|---|---|
| No-offset model, calibrated at 0.4 m | 4.38° | 0.95° | 5.15° | 7.37° |
| Full model (§5), calibrated at 0.4 + 2 m | 0.61° | 0.63° | 0.69° | 0.73° |

The no-offset row follows eq. (1): 6.25° predicted at 5 m, 7.37° observed. The excess, and the 0.95° at its
own calibration depth, come from the second-order effects of §4, which a rotation + gain cannot absorb.

**Step 5: binocular depth across distance** (noise-free gaze, model fitted on noise-free data):

| True depth | 0.3 m | 1 m | 3 m |
|---|---|---|---|
| Vergence depth | 0.303 m | 0.952 m | 2.544 m |
| Inverse-depth error | 0.03 D | 0.05 D | 0.06 D |
| Pixel error | 0.19° | 0.21° | 0.25° |

At 3 m the metric depth is 15% short, yet the pixel error stays at 0.25°, as eq. (6) predicts.

**Step 6: tracker pipeline (headless, rendered ArUco marker).** The marker was rendered into synthetic
640×480 frames and detected by the real tracker code, with the same synthetic eyes. The result: auto-solve
after 9 NEAR + 6 FAR samples, RMS 0.25° / 0.20°, eye separation 64 mm. The accuracy test (G/T) then gave:

| Marker depth | 0.30 m | 0.81 m | 3.15 m |
|---|---|---|---|
| Full model | 0.41° | 0.10° | 0.09° |
| No-offset baseline (same samples) | 2.79° | 1.03° | 2.81° |

---

## 11. Reproducing

```sh
python -m pytest                                      # everything (each file also runs as a plain script)
python gazecore/tests/test_gaze_geometry.py           # math, numpy only
python gazecore/tests/test_front_calibration.py       # workflow on rendered ArUco/ChArUco images
python FrontCameraTracker/test_tracker_integration.py # tracker orchestration, headless, + eye video
```

Step 6 is `test_tracker_integration.py::test_calibration_flow_through_tracker`.

Steps 1–5 above use the functions and synthetic headset in `test_gaze_geometry.py`
(`raw_gaze`, `targets_at`, `fit`, `camera_error_degrees`) with `np.random.default_rng(7)`.

---

## 12. Assumptions and limitations

* **Rigid headset.** $\mathbf{o}_e$ and $R_e$ are constant. Headset slippage invalidates the calibration.
* **Fixed eye rotation center.** The eye is modelled as rotating about a fixed point. Real eyes translate
  slightly with rotation, and this is not modelled.
* **Constant kappa.** Kappa is fixed in the eye's own frame, so its effect varies a little with gaze
  angle. Treating it as part of the constant rotation $R_e$ is a first-order approximation [9].
* **Single-frame samples.** Each calibration sample uses one frame of gaze. Averaging several frames would
  reduce noise.
* **Midpoint triangulation** is not the maximum-likelihood estimate [12]. That is acceptable because only
  its smoothed inverse depth is used.
* **Depth range** is clamped to 0.2–5 m.
* **Intrinsics matter for depth.** With the 60° fallback, marker depths, and therefore $\mathbf{o}_e$, scale
  with the error in the assumed focal length. Run the ChArUco calibration (I key) before trusting the
  fitted offsets.

---

## References

1. R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University
   Press, 2004. Ch. 4 (DLT), Ch. 6 (camera models), Ch. 9–10 (two-view geometry and its scale ambiguity),
   Ch. 12 (triangulation), App. 6 (iterative estimation, Levenberg–Marquardt).
2. Z. Zhang. "A flexible new technique for camera calibration." *IEEE Trans. Pattern Analysis and Machine
   Intelligence* 22(11):1330–1334, 2000.
3. S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, M. J. Marín-Jiménez. "Automatic generation and
   detection of highly reliable fiducial markers under occlusion." *Pattern Recognition* 47(6):2280–2292, 2014.
4. T. Collins, A. Bartoli. "Infinitesimal Plane-Based Pose Estimation." *International Journal of Computer
   Vision* 109(3):252–286, 2014.
5. D. Mardanbegi, D. W. Hansen. "Parallax error in the monocular head-mounted eye trackers." *Proc. ACM
   Conference on Ubiquitous Computing (UbiComp)*, 2012, pp. 689–694.
6. D. W. Marquardt. "An algorithm for least-squares estimation of nonlinear parameters." *SIAM Journal on
   Applied Mathematics* 11(2):431–441, 1963.
7. W. Kabsch. "A solution for the best rotation to relate two sets of vectors." *Acta Crystallographica*
   A32:922–923, 1976.
8. S. Umeyama. "Least-squares estimation of transformation parameters between two point patterns."
   *IEEE Trans. Pattern Analysis and Machine Intelligence* 13(4):376–380, 1991.
9. E. D. Guestrin, M. Eizenman. "General theory of remote gaze estimation using the pupil center and corneal
   reflections." *IEEE Trans. Biomedical Engineering* 53(6):1124–1133, 2006.
10. L. Świrski, N. Dodgson. "A fully-automatic, temporally coherent approach to robust eye model fitting
    from 2D pupil observations." *Proc. PETMEI*, 2013.
11. K. Levenberg. "A method for the solution of certain non-linear problems in least squares." *Quarterly
    of Applied Mathematics* 2(2):164–168, 1944.
12. R. Hartley, P. Sturm. "Triangulation." *Computer Vision and Image Understanding* 68(2):146–157, 1997.
13. OpenCV documentation: camera calibration and 3D reconstruction (`calib3d`), ArUco marker detection and
    ChArUco board tutorials. https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html,
    https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html,
    https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html
14. A. T. Duchowski, B. Pelfrey, D. H. House, R. Wang. "Measuring gaze depth with an eye tracker during
    stereoscopic display." *Proc. ACM SIGGRAPH Symposium on Applied Perception in Graphics and
    Visualization (APGV)*, 2011, pp. 15–22.
