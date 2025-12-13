#!/usr/bin/env python3
"""
SMPL-X fitting with:
 - Mediapipe keypoints (world or pixel fallback)
 - Improved procrustes alignment across views
 - NUM_BETAS = 30, relaxed regularization
 - Lightweight silhouette-width loss (no PyTorch3D)
 - No unconditional 180-degree flip (fixes upside-down problem)
"""

import os
import cv2
import numpy as np
import torch
import smplx
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R
from tqdm import trange
import matplotlib.pyplot as plt

# ========== CONFIG ==========
IMAGES = {
    'front': "dataset/image12.jpg",
    'left': "dataset/image12_right.jpg",
    'right': "dataset/image12_left.jpg",
}
MODEL_PATH = "full_pipeline/models/"  # SMPL-X model folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 30
OPT_ITERS = 600
LR = 5e-2
USE_SILHOUETTE = True  # Option B silhouette width loss (lightweight)
SILHOUETTE_WEIGHT = 8.0  # weight controlling silhouette loss magnitude
SILHOUETTE_SAMPLES = 6   # number of horizontal slices to match
DEBUG = True

# ========== IMPORT MEDIAPIPE (lazy import to avoid startup cost) ==========
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

# ========== HELPERS ==========
def normalize_image_aspect(img):
    """Return img and (h,w). Keep original; we use normalized coords later."""
    h, w = img.shape[:2]
    return img, (h, w)

def safe_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# ========== PROCRUSTES (weighted) ==========
def procrustes_align(source, target, weights=None):
    """
    Weighted Procrustes: align source (N,3) to target (N,3).
    Returns rotation matrix R, scale s, source_center, target_center.
    """
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    N = source.shape[0]
    if weights is None:
        weights = np.ones(N, dtype=np.float64)
    weights = weights.astype(np.float64)
    wsum = weights.sum() + 1e-12
    w = weights / wsum
    source_center = (source.T @ w).reshape(3)
    target_center = (target.T @ w).reshape(3)
    source_c = source - source_center
    target_c = target - target_center
    W = np.diag(w)
    H = source_c.T @ W @ target_c
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    numerator = S.sum()
    denominator = np.trace(source_c.T @ W @ source_c) + 1e-12
    scale = numerator / denominator
    return R_mat, scale, source_center, target_center

# ========== SILHOUETTE WIDTH FUNCTION ==========
def compute_mask_widths(seg_mask, n_samples=SILHOUETTE_SAMPLES):
    """
    Compute normalized widths (fraction of image width) of the segmentation mask
    at a set of vertical samples. Returns:
      y_samples_norm: shape (n_samples,) (values in [0,1], 0=top,1=bottom)
      widths_norm: shape (n_samples,) fraction of width (0..1)
    """
    h, w = seg_mask.shape[:2]
    mask_bin = (seg_mask > 0.5).astype(np.uint8)
    # find top and bottom of mask
    rows = np.where(mask_bin.sum(axis=1) > 0)[0]
    if len(rows) == 0:
        # empty mask -> return zeros
        ys = np.linspace(0.25, 0.85, n_samples)
        return ys, np.zeros(n_samples, dtype=np.float32)
    top = rows.min()
    bottom = rows.max()
    # sample n_samples heights between top and bottom
    ys = np.linspace(top, bottom, n_samples).astype(int)
    widths = []
    for y in ys:
        row = mask_bin[y, :]
        xs = np.where(row > 0)[0]
        if len(xs) == 0:
            widths.append(0.0)
        else:
            width_px = xs.max() - xs.min()
            widths.append(width_px / float(w))
    y_samples_norm = (ys / float(h))
    return y_samples_norm, np.array(widths, dtype=np.float32)

def pred_mesh_widths_at_samples(verts, y_samples_norm, ref_scale, ref_root_y=0.0, band=0.03):
    """
    Predict normalized widths for a mesh verts (N,3) in model coordinates.
    We'll project verts' x,y (assume root-centered) and scale by ref_scale
    to be comparable with normalized image units (rough approx).
    ref_root_y: expected root y position in image normalized coordinates (0..1)
    band: half-height around sample to consider vertices (in normalized units)
    Returns widths array sized like y_samples_norm.
    NOTE: This is approximate orthographic projection.
    """
    # verts: numpy (V,3)
    # We assume verts in same unit system as smpl joints used earlier;
    # ref_scale should map model units -> normalized image units roughly.
    proj = verts[:, :2].copy()  # x,y
    # apply scale
    proj *= float(ref_scale)
    # now interpret proj[:,1] as "normalized" centered coordinates; shift to [0,1] by adding ref_root_y
    # Here we assume root y corresponds to ref_root_y; we center around 0 and then add ref_root_y
    ys_pred = proj[:, 1] + ref_root_y
    xs_pred = proj[:, 0]
    widths = []
    for y_s in y_samples_norm:
        mask_v = (ys_pred >= (y_s - band)) & (ys_pred <= (y_s + band))
        if mask_v.sum() < 8:
            widths.append(0.0)
        else:
            xs = xs_pred[mask_v]
            width = xs.max() - xs.min()
            widths.append(width)  # note: still in normalized units (since ref_scale included)
    return np.array(widths, dtype=np.float32)

# ========== LOAD IMAGES & EXTRACT KEYPOINTS ==========
all_keypoints = []
all_confidences = []
all_view_names = []
all_images_rgb = []
all_segmentation_masks = []  # store masks + original dims

mp_to_smpl = {
    0: 15,   # nose -> head
    11: 16,  # left shoulder
    12: 17,  # right shoulder
    13: 18,  # left elbow
    14: 19,  # right elbow
    15: 20,  # left wrist
    16: 21,  # right wrist
    23: 1,   # left hip
    24: 2,   # right hip
    25: 4,   # left knee
    26: 5,   # right knee
    27: 7,   # left ankle
    28: 8,   # right ankle
}
mp_indices = np.array(list(mp_to_smpl.keys()), dtype=int)
smpl_indices = np.array(list(mp_to_smpl.values()), dtype=int)

for view_name, img_path in IMAGES.items():
    safe_print(f"\n[INFO] Processing {view_name}: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        safe_print(f"[WARNING] Could not read {img_path}; skipping.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    results = pose.process(img_rgb)
    if results.pose_landmarks is None:
        safe_print(f"[WARNING] No pose for {view_name}; skipping.")
        continue
    # segmentation mask (float 0..1) may be None
    if results.segmentation_mask is not None:
        all_segmentation_masks.append((results.segmentation_mask.copy(), (h, w)))
    # prefer world landmarks
    if getattr(results, "pose_world_landmarks", None) is not None:
        lands = results.pose_world_landmarks.landmark
        kps = np.array([[lm.x, lm.y, lm.z] for lm in lands], dtype=np.float32)
        conf = np.array([lm.visibility if hasattr(lm, "visibility") else 1.0 for lm in lands], dtype=np.float32)
        # quick heuristic: if z values seem inverted (model flips), you can try negating z
        # We'll not negate automatically but keep an option variable:
        # kps[:,2] = -kps[:,2]  # uncomment if you observe large flips
        safe_print(f"[INFO] Using pose_world_landmarks (world coords) for {view_name}")
    else:
        # pixel fallback: use normalized x,y in [0,1], use lm.z as provided (already relative)
        lands = results.pose_landmarks.landmark
        kps = np.array([[lm.x, lm.y, lm.z] for lm in lands], dtype=np.float32)
        conf = np.array([lm.visibility for lm in lands], dtype=np.float32)
        safe_print(f"[INFO] Using pixel-normalized pose_landmarks for {view_name}")

    # simple forward-lean correction using shoulders/hips (axis: y is up in these coords)
    def correct_forward_lean(kps_local):
        Ls, Rs = 11, 12
        Lh, Rh = 23, 24
        if np.any(kps_local[Ls]) and np.any(kps_local[Rs]) and np.any(kps_local[Lh]) and np.any(kps_local[Rh]):
            shoulder_mid = (kps_local[Ls] + kps_local[Rs]) / 2.0
            hip_mid = (kps_local[Lh] + kps_local[Rh]) / 2.0
            spine = shoulder_mid - hip_mid
            spine = spine / (np.linalg.norm(spine) + 1e-8)
            forward_angle = np.arctan2(spine[2], spine[1])  # tilt around x
            if abs(forward_angle) > 0.05:
                safe_print(f"[INFO] Correcting forward lean by {np.degrees(forward_angle):.1f} deg")
                correction = R.from_euler('x', -forward_angle, degrees=False).as_matrix()
                centered = kps_local - hip_mid
                corrected = (correction @ centered.T).T + hip_mid
                return corrected
        return kps_local

    kps = correct_forward_lean(kps)

    all_keypoints.append(kps)
    all_confidences.append(conf)
    all_view_names.append(view_name)
    all_images_rgb.append(img_rgb)
    safe_print(f"[INFO] Extracted {len(kps)} keypoints, avg conf={conf.mean():.3f}")

if len(all_keypoints) == 0:
    raise ValueError("No valid views / keypoints found. Aborting.")

# ========== SMPL-X MODEL LOAD ==========
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=NUM_BETAS,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# ========== PREPARE TARGET 3D KEYPOINTS (centered per view) ==========
all_targets = []
all_weights = []
reference_idx = 0
ref_kps = all_keypoints[reference_idx].copy()
ref_conf = all_confidences[reference_idx].copy()
# center ref at hip midpoint when available
if np.any(ref_kps[23]) and np.any(ref_kps[24]):
    ref_root = (ref_kps[23] + ref_kps[24]) / 2.0
else:
    ref_root = np.median(ref_kps, axis=0)
ref_kps_centered = ref_kps - ref_root
ref_kps_for_align = ref_kps_centered[mp_indices]
ref_conf_for_align = ref_conf[mp_indices]

safe_print(f"\n[INFO] Preparing {len(all_keypoints)} view(s) for optimization...")

for view_idx, (kps, conf, view_name) in enumerate(zip(all_keypoints, all_confidences, all_view_names)):
    # center at hip midpoint
    if np.any(kps[23]) and np.any(kps[24]):
        root = (kps[23] + kps[24]) / 2.0
    else:
        root = np.median(kps, axis=0)
    kps_centered = kps - root

    # apply left/right view rotation to approximate camera pose
    if 'left' in view_name.lower():
        rot = R.from_euler('y', -90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        safe_print(f"[INFO] Applied -90° Y rotation to left view")
    elif 'right' in view_name.lower():
        rot = R.from_euler('y', 90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        safe_print(f"[INFO] Applied +90° Y rotation to right view")

    # Align this view to reference using procrustes on the selected mp indices
    if len(all_keypoints) > 1 and view_idx != reference_idx:
        R_mat, scale, src_center, tgt_center = procrustes_align(
            kps_centered[mp_indices],
            ref_kps_for_align,
            weights=(conf[mp_indices] * ref_conf_for_align + 1e-6)
        )
        # apply to all keypoints of this view
        kps_centered = scale * ((kps_centered - src_center) @ R_mat.T) + tgt_center
        safe_print(f"[INFO] Aligned view {view_name} to reference (scale={scale:.3f})")

    kps_selected = kps_centered[mp_indices]
    conf_selected = conf[mp_indices].copy()
    # Boost visible-side confidences
    if 'left' in view_name.lower():
        left_mask = np.isin(mp_indices, [11, 13, 15, 23, 25, 27])
        conf_selected[left_mask] *= 1.5
    elif 'right' in view_name.lower():
        right_mask = np.isin(mp_indices, [12, 14, 16, 24, 26, 28])
        conf_selected[right_mask] *= 1.5
    elif len(all_keypoints) == 1 and 'front' in view_name.lower():
        front_mask = np.isin(mp_indices, [11, 12, 13, 14, 25, 26])
        conf_selected[front_mask] *= 1.3

    conf_selected = np.clip(conf_selected, 0.0, 1.0)
    all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))

# ========== ESTIMATE BODY SHAPE HINTS FROM SEGMENTATION ==========
body_volume_estimate = 1.0
body_width_ratio = 1.0
torso_ratio = 1.0

if len(all_segmentation_masks) > 0:
    vols = []
    widths = []
    for mask, (h, w) in all_segmentation_masks:
        body_area = float((mask > 0.5).sum())
        total_area = float(mask.shape[0] * mask.shape[1])
        body_ratio = body_area / (total_area + 1e-12)
        ys, widths_sampled = compute_mask_widths(mask, n_samples=SILHOUETTE_SAMPLES)
        avg_torso_width = widths_sampled.mean() if widths_sampled.sum() > 0 else 0.25
        ref_body_ratio = 0.12
        ref_torso_ratio = 0.25
        vol_est = np.clip(body_ratio / ref_body_ratio, 0.5, 2.5)
        wid_est = np.clip(avg_torso_width / ref_torso_ratio, 0.6, 2.5)
        vols.append(vol_est)
        widths.append(wid_est)
        safe_print(f"[INFO] mask body ratio: {body_ratio:.4f}, avg torso width: {avg_torso_width:.4f}")
    if len(vols) > 0:
        body_volume_estimate = float(np.median(vols))
        body_width_ratio = float(np.median(widths))
    safe_print(f"[INFO] Body estimates: volume_factor={body_volume_estimate:.3f}, width_ratio={body_width_ratio:.3f}")

# ========== INIT PARAMETERS ==========
initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)
initial_betas[0, 0] = (body_volume_estimate - 1.0) * 4.0
initial_betas[0, 2] = (body_width_ratio - 1.0) * 3.0
# small nudge for mid-body if width_ratio large
if body_width_ratio > 1.05 and NUM_BETAS > 5:
    initial_betas[0, 1] = (body_width_ratio - 1.0) * 2.0

safe_print(f"[INFO] Initial betas preview: {initial_betas[0,:6].cpu().numpy()}")

betas = initial_betas.clone().requires_grad_(True)
body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# spine joint indices for smoothing (SMPL joints indices in your mapping may vary)
spine_joints = [0, 3, 6, 9, 12, 15]

# ========== OPTIMIZATION STAGES ==========
# Stage 1: betas + global_orient + transl
safe_print("\n[INFO] Stage 1: optimizing shape (betas), global_orient, transl...")
opt_stage1 = torch.optim.Adam([betas, global_orient, transl], lr=LR)
for it in trange(200, desc="Stage 1"):
    opt_stage1.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose.detach(), global_orient=global_orient, transl=transl)
    smpl_joints_full = out.joints[0].cpu()
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root  # (K,3)
    total_loss = 0.0
    for target, weight in zip(all_targets, all_weights):
        valid_mask = (weight > 0.3)
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        scale = (target_scale / (smpl_scale + 1e-8)).detach()
        diff = (smpl_joints * scale.to(smpl_joints.device) - target.cpu()).to(DEVICE) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss = total_loss + loss_joints
    loss_beta = loss_beta = 5e-3 * torch.mean(betas ** 2)
    loss = total_loss / len(all_targets) + loss_beta
    loss.backward()
    opt_stage1.step()
    with torch.no_grad():
        betas.clamp_(-8.0, 8.0)
    if (it + 1) % 50 == 0:
        safe_print(f"  [Iter {it+1}] loss={loss.item():.6e}, betas={betas[0,:6].detach().cpu().numpy()}")

# Stage 2: optimize body pose + betas + global_orient + transl
safe_print("\n[INFO] Stage 2: optimizing pose (limbs) + shape refinement...")
opt_stage2 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.5)
for it in trange(250, desc="Stage 2"):
    opt_stage2.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    smpl_joints_full = out.joints[0]
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root
    total_loss = 0.0
    silhouette_loss_val = 0.0
    for vi, (target, weight) in enumerate(zip(all_targets, all_weights)):
        valid_mask = (weight > 0.3)
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        scale = (target_scale / (smpl_scale + 1e-8))
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss = total_loss + loss_joints

        # silhouette width loss (lightweight) - only if we have masks for this view
        if USE_SILHOUETTE and vi < len(all_segmentation_masks):
            # prepare mask target widths (precomputed)
            mask, (mh, mw) = all_segmentation_masks[vi]
            y_samples_norm, widths_target = compute_mask_widths(mask, n_samples=SILHOUETTE_SAMPLES)
            # predicted mesh verts in numpy (cpu)
            verts = out.vertices[0].cpu().detach().numpy()  # (V,3)
            # convert model coords -> approx normalized image coords using scale
            # Need a root Y reference: use joint root y projected to normalized (we approximate as 0.5)
            ref_root_y = 0.5
            # compute predicted widths
            pred_widths = pred_mesh_widths_at_samples(verts, y_samples_norm, ref_scale=scale.detach().cpu().numpy(), ref_root_y=ref_root_y, band=0.03)
            # ensure both in same units: widths_target is fraction of image width; pred_widths should be similar
            # If pred_widths are too large/small, a small normalization by max observed may help, but we'll directly use L2
            silhouette_loss_val = silhouette_loss_val + torch.tensor(((pred_widths - widths_target) ** 2).mean(), device=DEVICE)
    loss_spine = 3e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 3e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 1e-3 * torch.mean(betas ** 2)
    loss = (total_loss / len(all_targets)) + loss_spine + loss_pose + loss_beta
    if USE_SILHOUETTE:
        loss = loss + SILHOUETTE_WEIGHT * silhouette_loss_val
    loss.backward()
    opt_stage2.step()
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.5, 0.5)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-8.0, 8.0)
    if (it + 1) % 50 == 0:
        safe_print(f"  [Iter {it+1}] loss={loss.item():.6e}")

# Stage 3: fine-tune
safe_print("\n[INFO] Stage 3: fine-tuning...")
opt_stage3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.2)
best_loss = 1e9
best_params = None
for it in trange(150, desc="Stage 3"):
    opt_stage3.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    smpl_joints_full = out.joints[0]
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root
    total_loss = 0.0
    silhouette_loss_val = 0.0
    for vi, (target, weight) in enumerate(zip(all_targets, all_weights)):
        valid_mask = (weight > 0.3)
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        scale = (target_scale / (smpl_scale + 1e-8))
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss = total_loss + loss_joints

        if USE_SILHOUETTE and vi < len(all_segmentation_masks):
            mask, (mh, mw) = all_segmentation_masks[vi]
            y_samples_norm, widths_target = compute_mask_widths(mask, n_samples=SILHOUETTE_SAMPLES)
            verts = out.vertices[0].cpu().detach().numpy()
            pred_widths = pred_mesh_widths_at_samples(verts, y_samples_norm, ref_scale=scale.detach().cpu().numpy(), ref_root_y=0.5, band=0.03)
            silhouette_loss_val = silhouette_loss_val + torch.tensor(((pred_widths - widths_target) ** 2).mean(), device=DEVICE)

    # spine smoothness
    spine_joints_coords = out.joints[0, spine_joints, :]
    spine_dirs = spine_joints_coords[1:] - spine_joints_coords[:-1]
    spine_dirs_norm = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    loss_spine_align = 1e-2 * torch.mean((1 - torch.sum(spine_dirs_norm[:-1] * spine_dirs_norm[1:], dim=1)) ** 2)

    loss_spine_pose = 2e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 5e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 1e-3 * torch.mean(betas ** 2)

    loss = total_loss / len(all_targets) + loss_spine_align + loss_spine_pose + loss_pose + loss_beta
    if USE_SILHOUETTE:
        loss = loss + SILHOUETTE_WEIGHT * silhouette_loss_val

    loss.backward()
    opt_stage3.step()
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.6, 0.6)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-8.0, 8.0)

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = {
            'betas': betas.detach().clone(),
            'body_pose': body_pose.detach().clone(),
            'global_orient': global_orient.detach().clone(),
            'transl': transl.detach().clone(),
        }
    if (it + 1) % 50 == 0:
        safe_print(f"  [Iter {it+1}] loss={loss.item():.6e}")

safe_print(f"\n[INFO] Optimization finished. Best loss={best_loss:.6e}")
safe_print(f"[INFO] Final betas (first 8): {best_params['betas'][0,:8].cpu().numpy()}")

# ========== EXPORT FINAL MESH (no unconditional flip) ==========
final_out = smplx_model(
    betas=best_params['betas'],
    body_pose=best_params['body_pose'],
    global_orient=best_params['global_orient'],
    transl=best_params['transl']
)
verts = final_out.vertices[0].cpu().detach().numpy()
faces = smplx_model.faces
# center for visual consistency
verts_centered = verts - verts.mean(axis=0)
mesh_export = trimesh.Trimesh(verts_centered, faces)
mesh_export.export("fitted_smplx_mesh.obj")
safe_print("[INFO] Exported fitted_smplx_mesh.obj")
mesh_export.visual.vertex_colors = [200, 200, 230, 255]
mesh_export.export("fitted_smplx_mesh_colored.ply")
safe_print("[INFO] Exported fitted_smplx_mesh_colored.ply")

# ========== VISUALIZATION (pyrender) ==========
safe_print("\n[INFO] Creating visualization scene...")
scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
mesh_node = pyrender.Mesh.from_trimesh(mesh_export, smooth=True)
scene.add(mesh_node)
# Add simple skeleton spheres (from final_out.joints)
joints = final_out.joints[0].cpu().detach().numpy()
joints_centered = joints - joints.mean(axis=0)
for p in joints_centered:
    s = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
    s.apply_translation(p)
    s.visual.vertex_colors = [100, 255, 100, 255]
    scene.add(pyrender.Mesh.from_trimesh(s))
# camera & lights
camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
cam_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2.5],[0,0,0,1]], dtype=np.float32)
scene.add(camera, pose=cam_pose)
light = pyrender.DirectionalLight(color=[1,1,1], intensity=3.0)
scene.add(light, pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], dtype=np.float32))
safe_print("[INFO] Launching viewer (or rendering to image)...")
try:
    pyrender.Viewer(scene, use_raymond_lighting=True)
except Exception as e:
    safe_print("[WARNING] Viewer failed; rendering offscreen: ", e)
    r = pyrender.OffscreenRenderer(1200, 1200)
    color, _ = r.render(scene)
    cv2.imwrite("render_output.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    r.delete()
    safe_print("[INFO] Saved render_output.png")

safe_print("\n[INFO] Done. Files: fitted_smplx_mesh.obj, fitted_smplx_mesh_colored.ply, (render_output.png if headless).")
