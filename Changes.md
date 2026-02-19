# Pipeline Changes & Improvements

This document explains all changes made across versions of the 3D human mesh fitting pipeline, from `pipeline_v5.py` to the current `pipeline_v6.py`.

---

## Bug Fixes (v5 → v5 patches)

### 1. Invalid Path Escape Sequences
**Problem:** Windows backslash paths like `"synthetic_data\images\image_1.png"` caused `SyntaxWarning: invalid escape sequence` because Python interprets `\s`, `\i`, etc. as escape characters.

**Fix:** Switched to forward slashes (`"dataset/image6.jpg"`), which work on all platforms including Windows.

---

### 2. Images Not Loading (`cv2.imread` returning `None`)
**Problem:** `cv2.imread` silently returns `None` for invalid paths, causing all views to be skipped and the pipeline to crash with `ValueError: No valid poses detected`.

**Fix:** Corrected the image paths and added a working directory anchor (`os.path.abspath`) to ensure paths resolve correctly regardless of where the script is invoked from.

---

### 3. `procrustes_align` — `np.trace` on 1D Array
**Problem:** `np.linalg.svd` returns `S` as a 1D array of singular values, but the original code called `np.trace(S)` which requires a 2D matrix, raising:
```
ValueError: diag requires an array of at least two dimensions
```

**Fix:** Replaced `np.trace(S)` with `np.sum(S)` (sum of singular values), and computed source variance directly:
```python
source_var = np.sum(weights[:, None] * source_centered ** 2)
scale = np.sum(S) / (source_var + 1e-8)
```

---

### 4. `IndexError` After Procrustes Alignment
**Problem:** After `procrustes_align` returned an already-subsetted 13-keypoint array, the code tried to index it again with `mp_indices` (values up to 28), causing:
```
IndexError: index 13 is out of bounds for axis 0 with size 13
```

**Fix:** Restructured the view processing loop so that `kps_centered[mp_indices]` subsetting happens **after** rotation but **before** passing into `procrustes_align`. The function now always receives and returns consistent `(13, 3)` arrays.

---

### 5. `np.infty` Deprecation
**Problem:** `np.infty` is deprecated in NumPy 2.x and raises warnings or errors.

**Fix:** Replaced with Python's built-in `float('inf')`:
```python
best_loss = float('inf')
```

---

## Core Improvements (v5 → v6)

### Fix 1 — Real-World Scale Calibration
**Problem:** Height was estimated as ~145 cm instead of the real ~165 cm. The old approach re-estimated scale every iteration as a ratio of vector norms, which is unstable and drifts.

**Solution:** Height is now estimated **once before optimization** using the anatomical relationship:

> Hip-to-ankle distance ≈ 53% of total body height (standard anthropometry)

```python
def estimate_real_height_m(kps):
    avg_leg = (norm(ankle_l - hip_l) + norm(ankle_r - hip_r)) / 2.0
    return avg_leg / 0.53
```

All keypoints from all views are then rescaled to this consistent real-world height before optimization begins. A fixed `GLOBAL_SCALE` constant is computed from the SMPL-X rest pose height vs the estimated height, and used as a constant multiplier throughout optimization — eliminating per-iteration drift.

**Impact:** Height accuracy improved significantly, reducing the ~20 cm error.

---

### Fix 2 — Automatic Gender Detection
**Problem:** Using `gender='NEUTRAL'` in SMPL-X causes the shape to drift feminine when betas go slightly negative, producing a female-looking mesh regardless of the subject.

**Solution:** Gender is detected from the **shoulder-width to hip-width ratio** of the front-view keypoints:

| Ratio | Detected Gender |
|-------|----------------|
| > 1.10 | Male |
| < 1.02 | Female |
| Between | Neutral |

```python
ratio = norm(shoulder_l - shoulder_r) / norm(hip_l - hip_r)
```

The SMPL-X model is then loaded with the detected gender (`MALE`, `FEMALE`, or `NEUTRAL`), ensuring the base shape template is already anatomically appropriate.

**Impact:** Eliminates incorrect gender shape template; mesh now matches subject's body type from the start.

---

### Fix 3 — Segment Length Loss
**Problem:** Optimizing only on joint positions allows asymmetric or anatomically implausible body proportions to emerge (e.g., one leg longer than the other, or thighs much longer than shins).

**Solution:** Added a segment-aware loss applied across all three optimization stages:

- **Symmetry loss:** Penalizes left-right differences in upper arm, forearm, thigh, and shin lengths.
- **Anthropometric ratio loss:** Penalizes deviation from natural proportions (thigh ≈ shin, upper arm ≈ forearm).

```python
sym_loss  = (thigh_l - thigh_r)**2 + (shin_l - shin_r)**2 + ...
ratio_loss = (thigh_l - shin_l)**2 + (upper_arm_l - forearm_l)**2 + ...
loss_seg  = sym_loss + 0.5 * ratio_loss
```

**Impact:** Produces more anatomically consistent limb proportions and segment distances.

---

### Fix 4 — Weighted Beta Regularization & Tighter Clamping
**Problem:** Betas drifting to extreme values (especially negative) causes unrealistic body shapes. Higher beta components (indices 5–29) have increasingly abstract and unstable effects on shape.

**Solution:**
- Regularization now penalizes higher beta components progressively more using a linearly increasing weight vector (`linspace(1.0, 3.0)`).
- Clamp range is tightened: primary betas `[:5]` clamped to `±2.5`, higher betas `[5:]` clamped to `±1.5`.

```python
beta_weights = torch.linspace(1.0, 3.0, NUM_BETAS, device=DEVICE)
loss_beta = weight * torch.mean(beta_weights * betas[0] ** 2)
```

**Impact:** Prevents unrealistic shape drift; body shape stays within plausible human range.

---

### Fix 5 — Axis-Specific View Weighting
**Problem:** After rotating side views by ±90°, the X-axis information from those views is unreliable (it maps to depth in the original camera frame). Treating all axes equally causes the optimizer to get conflicting signals.

**Solution:** Each view now has per-axis confidence multipliers applied to the loss:

| View | X weight | Y weight | Z weight |
|------|----------|----------|----------|
| Front | 1.0 | 1.0 | 0.2 |
| Left / Right | 0.2 | 1.0 | 1.0 |

```python
diff = (smpl_joints * GLOBAL_SCALE - target) ** 2
diff = diff * axis_weights.unsqueeze(0)
```

**Impact:** Side views now properly contribute **depth** information (belly thickness, chest depth, `BellySide` measurement) without corrupting the lateral position estimates from the front view.

---

### Fix 6 — Fixed Global Scale (No Per-Iteration Re-estimation)
**Problem:** The old scale was computed every iteration as `target_scale / smpl_scale`, which changes as the model deforms during optimization. This creates a moving target and leads to unstable convergence.

**Solution:** Scale is computed exactly once before optimization begins, using the SMPL-X rest-pose joint span vs the real estimated height:

```python
GLOBAL_SCALE = estimated_height_m / smpl_rest_height
```

This constant is used in the loss and also applied to the exported mesh vertices, so the output `.obj` and `.ply` files are in real-world metres.

**Impact:** Stable, consistent scale throughout optimization. Exported mesh dimensions are physically meaningful.

---

## Output Changes

- `fitted_smplx_mesh.obj` and `fitted_smplx_mesh_colored.ply` are now exported in **real-world metres** (scaled by `GLOBAL_SCALE`).
- The interactive 3D pyrender viewer has been removed (was broken due to `np.infty` deprecation and platform issues). The pipeline now runs to completion without hanging.
- `keypoint_alignment.png` is still saved as a diagnostic visualization of the multi-view keypoint alignment.
- A summary block is printed at the end showing detected gender, estimated height, global scale, and shape parameters.

---

## Version Summary

| Version | Key Changes |
|---------|-------------|
| `pipeline_v5.py` | Multi-view support, Procrustes alignment, segmentation-based shape init |
| `pipeline_v6.py` | Real-world scale fix, gender detection, segment loss, axis-weighted views, fixed global scale, no viewer |