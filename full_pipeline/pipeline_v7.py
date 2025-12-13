import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from tqdm import trange
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== CONFIG ==========
IMAGES = {
    'front': "dataset/image7.jpg",
    'left': "dataset/image7_right.jpg",
    'right': "dataset/image7_left.jpg",
}
MODEL_PATH = "full_pipeline/models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 10
OPT_ITERS = 600
LR = 5e-2

# ========== IMPROVED ASPECT RATIO HANDLING ==========
def normalize_image_aspect(img):
    """Normalize image to handle non-square aspect ratios."""
    h, w = img.shape[:2]
    return img, (h, w)

# ========== FIX: VALIDATE POSE DETECTION ==========
def is_valid_pose(landmarks, confidences, min_confidence=0.3):
    """
    Validate if detected pose is reasonable.
    Check if key body parts are detected with sufficient confidence.
    """
    # Key landmarks that must be visible
    critical_landmarks = [11, 12, 23, 24]  # shoulders and hips
    
    for idx in critical_landmarks:
        if confidences[idx] < min_confidence:
            return False, f"Low confidence for landmark {idx}"
    
    # Check if landmarks are within reasonable bounds
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Check for scattered keypoints (variance too high)
    variance = np.var(coords[:, :2], axis=0)
    if np.any(variance > 0.5):  # Normalized coordinates
        return False, "Keypoints too scattered"
    
    # Check if body proportions are reasonable
    left_shoulder = coords[11]
    right_shoulder = coords[12]
    left_hip = coords[23]
    right_hip = coords[24]
    
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
    
    if shoulder_width < 0.05 or hip_width < 0.05 or torso_height < 0.1:
        return False, "Body proportions unrealistic"
    
    return True, "Valid pose"

# ========== FIX: HEAD ALIGNMENT CORRECTION ==========
def correct_forward_lean(kps, confidences):
    """
    Correct forward head lean by aligning spine to vertical axis.
    Uses shoulder and hip positions to determine proper alignment.
    """
    left_shoulder_idx = 11
    right_shoulder_idx = 12
    left_hip_idx = 23
    right_hip_idx = 24
    
    if not (np.any(kps[left_shoulder_idx]) and np.any(kps[right_shoulder_idx]) and
            np.any(kps[left_hip_idx]) and np.any(kps[right_hip_idx])):
        print("[WARNING] Missing shoulder or hip keypoints, skipping head alignment")
        return kps
    
    shoulder_mid = (kps[left_shoulder_idx] + kps[right_shoulder_idx]) / 2.0
    hip_mid = (kps[left_hip_idx] + kps[right_hip_idx]) / 2.0
    
    spine_vector = shoulder_mid - hip_mid
    spine_vector = spine_vector / (np.linalg.norm(spine_vector) + 1e-8)
    
    # Calculate forward lean angle
    forward_lean_angle = np.arctan2(spine_vector[2], spine_vector[1])
    
    if abs(forward_lean_angle) > 0.05:
        print(f"[INFO] Correcting forward lean: {np.degrees(forward_lean_angle):.1f}°")
        correction_rot = R.from_euler('x', -forward_lean_angle, degrees=False).as_matrix()
        kps_centered = kps - hip_mid
        kps_corrected = (correction_rot @ kps_centered.T).T
        kps = kps_corrected + hip_mid
    
    return kps

# ========== KEYPOINT EXTRACTION ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

all_keypoints = []
all_view_names = []
all_confidences = []
all_images_rgb = []
all_segmentation_masks = []

for view_name, img_path in IMAGES.items():
    print(f"\n[INFO] Processing {view_name} view: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not load {img_path}, skipping...")
        continue
    
    img_rgb, (h, w) = normalize_image_aspect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    aspect_ratio = w / h
    print(f"[INFO] Image size: {w}x{h}, aspect ratio: {aspect_ratio:.2f}")
    
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        print(f"[WARNING] No pose detected in {view_name}, skipping...")
        continue
    
    # FIX: Validate pose before accepting it
    landmarks = results.pose_landmarks.landmark
    confidences_check = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    is_valid, message = is_valid_pose(landmarks, confidences_check)
    
    if not is_valid:
        print(f"[WARNING] Invalid pose in {view_name}: {message}, skipping...")
        continue
    
    print(f"[INFO] Valid pose detected in {view_name}")
    
    # Extract segmentation mask
    if results.segmentation_mask is not None:
        all_segmentation_masks.append((results.segmentation_mask, img_rgb))
    
    # Use world landmarks for better 3D coordinates
    if getattr(results, "pose_world_landmarks", None) is not None:
        landmarks = results.pose_world_landmarks.landmark
        kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
        print(f"[INFO] Using world landmarks (natural scale)")
    else:
        landmarks = results.pose_landmarks.landmark
        kps = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
        kps[:, 0] = kps[:, 0] / w
        kps[:, 1] = kps[:, 1] / h
        kps[:, 2] = kps[:, 2] / h
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
        print(f"[INFO] Using pixel landmarks (normalized)")
    
    # Correct head forward lean
    kps = correct_forward_lean(kps, confidences)
    
    all_keypoints.append(kps)
    all_view_names.append(view_name)
    all_confidences.append(confidences)
    all_images_rgb.append(img_rgb)
    print(f"[INFO] Extracted {len(kps)} keypoints, avg confidence: {confidences.mean():.3f}")

if len(all_keypoints) == 0:
    raise ValueError("No valid poses detected in any image!")

# ========== LOAD SMPL-X MODEL ==========
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=NUM_BETAS,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# Enhanced mapping
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

# ========== PROCRUSTES ALIGNMENT ==========
def procrustes_align(source, target, weights=None):
    """Align source to target using Procrustes analysis."""
    if weights is None:
        weights = np.ones(len(source))
    
    weights = weights / (weights.sum() + 1e-8)
    source_center = (source.T @ weights).reshape(3)
    target_center = (target.T @ weights).reshape(3)
    
    source_centered = source - source_center
    target_centered = target - target_center
    
    W = np.diag(weights)
    H = source_centered.T @ W @ target_centered
    
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    
    numerator = np.sum(S)
    denominator = np.trace(source_centered.T @ W @ source_centered)
    scale = numerator / (denominator + 1e-8)
    
    return R_mat, scale, source_center, target_center

# ========== FIX: IMPROVED BODY SHAPE ESTIMATION ==========
body_measurements = {
    'volume': 1.0,
    'torso_width': 1.0,
    'torso_depth': 1.0,
    'belly_size': 1.0,
    'chest_size': 1.0,
}

if len(all_segmentation_masks) > 0:
    print("\n[INFO] Estimating body shape from segmentation masks...")
    volume_estimates = []
    torso_widths = []
    belly_sizes = []
    chest_sizes = []
    
    for mask, img_rgb in all_segmentation_masks:
        mask_binary = (mask > 0.5).astype(np.uint8)
        h, w = mask_binary.shape
        
        # Overall body volume
        body_area = np.sum(mask_binary)
        total_area = h * w
        body_ratio = body_area / total_area
        
        # Chest region (20-40% from top)
        chest_start = int(h * 0.2)
        chest_end = int(h * 0.4)
        chest_mask = mask_binary[chest_start:chest_end, :]
        chest_widths = np.sum(chest_mask, axis=1)
        avg_chest_width = np.mean(chest_widths[chest_widths > 0]) if np.any(chest_widths > 0) else w * 0.25
        
        # Torso/Belly region (40-60% from top)
        torso_start = int(h * 0.4)
        torso_end = int(h * 0.6)
        torso_mask = mask_binary[torso_start:torso_end, :]
        torso_widths_arr = np.sum(torso_mask, axis=1)
        avg_torso_width = np.mean(torso_widths_arr[torso_widths_arr > 0]) if np.any(torso_widths_arr > 0) else w * 0.25
        
        # Calculate ratios
        ref_body_ratio = 0.12
        ref_torso_ratio = 0.25
        ref_chest_ratio = 0.28
        
        vol_est = np.clip(body_ratio / ref_body_ratio, 0.5, 3.0)
        torso_ratio = np.clip((avg_torso_width / w) / ref_torso_ratio, 0.5, 3.0)
        chest_ratio = np.clip((avg_chest_width / w) / ref_chest_ratio, 0.5, 3.0)
        belly_est = np.clip(torso_ratio / chest_ratio, 0.7, 2.0)  # Belly prominence
        
        volume_estimates.append(vol_est)
        torso_widths.append(torso_ratio)
        chest_sizes.append(chest_ratio)
        belly_sizes.append(belly_est)
        
        print(f"[INFO] Volume: {vol_est:.3f}, Torso: {torso_ratio:.3f}, Chest: {chest_ratio:.3f}, Belly: {belly_est:.3f}")
    
    body_measurements['volume'] = np.median(volume_estimates)
    body_measurements['torso_width'] = np.median(torso_widths)
    body_measurements['chest_size'] = np.median(chest_sizes)
    body_measurements['belly_size'] = np.median(belly_sizes)
    
    print(f"[INFO] Final measurements: {body_measurements}")

# ========== PREPARE TARGETS WITH MULTI-VIEW ALIGNMENT ==========
all_targets = []
all_weights = []
reference_idx = 0

ref_kps = all_keypoints[reference_idx].copy()
ref_conf = all_confidences[reference_idx].copy()

if np.any(ref_kps[23]) and np.any(ref_kps[24]):
    ref_root = (ref_kps[23] + ref_kps[24]) / 2.0
else:
    ref_root = np.median(ref_kps, axis=0)

ref_kps_centered = ref_kps - ref_root
ref_kps_for_align = ref_kps_centered[mp_indices]
ref_conf_for_align = ref_conf[mp_indices]

print(f"\n[INFO] Using {len(all_keypoints)} view(s) for optimization")

for view_idx, (kps, conf, view_name) in enumerate(zip(all_keypoints, all_confidences, all_view_names)):
    if np.any(kps[23]) and np.any(kps[24]):
        root = (kps[23] + kps[24]) / 2.0
    else:
        root = np.median(kps, axis=0)
    
    kps_centered = kps - root
    
    if 'left' in view_name.lower():
        rot_angle = -90
        rot = R.from_euler('y', rot_angle, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied {rot_angle}° Y rotation for left view")
        
        if len(all_keypoints) > 1 and view_idx != reference_idx:
            R_mat, scale, src_center, tgt_center = procrustes_align(
                kps_centered[mp_indices], ref_kps_for_align,
                weights=conf[mp_indices] * ref_conf_for_align
            )
            kps_centered = scale * ((kps_centered - src_center) @ R_mat.T) + tgt_center
            print(f"[INFO] Aligned left view to reference (scale={scale:.3f})")
            
    elif 'right' in view_name.lower():
        rot_angle = 90
        rot = R.from_euler('y', rot_angle, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied {rot_angle}° Y rotation for right view")
        
        if len(all_keypoints) > 1 and view_idx != reference_idx:
            R_mat, scale, src_center, tgt_center = procrustes_align(
                kps_centered[mp_indices], ref_kps_for_align,
                weights=conf[mp_indices] * ref_conf_for_align
            )
            kps_centered = scale * ((kps_centered - src_center) @ R_mat.T) + tgt_center
            print(f"[INFO] Aligned right view to reference (scale={scale:.3f})")
    
    kps_selected = kps_centered[mp_indices]
    conf_selected = conf[mp_indices].copy()
    
    if 'left' in view_name.lower():
        left_mask = np.isin(mp_indices, [11, 13, 15, 23, 25, 27])
        conf_selected[left_mask] *= 1.5
    elif 'right' in view_name.lower():
        right_mask = np.isin(mp_indices, [12, 14, 16, 24, 26, 28])
        conf_selected[right_mask] *= 1.5
    
    conf_selected = np.clip(conf_selected, 0, 1)
    
    all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))

# ========== FIX: SMPL-X BETA INITIALIZATION FOR FULL BODY ==========
# SMPL-X beta meanings (approximate):
# 0: Overall size/height
# 1: Belly/torso thickness
# 2: Overall width/broadness
# 3: Leg thickness
# 4: Chest size
# 5: Neck thickness
# etc.

initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)

# Map measurements to betas
initial_betas[0, 0] = (body_measurements['volume'] - 1.0) * 3.0  # Overall size
initial_betas[0, 1] = (body_measurements['belly_size'] - 1.0) * 4.0  # Belly prominence
initial_betas[0, 2] = (body_measurements['torso_width'] - 1.0) * 3.5  # Width
initial_betas[0, 4] = (body_measurements['chest_size'] - 1.0) * 2.5  # Chest

print(f"[INFO] Initial betas: {initial_betas[0, :5].cpu().numpy()}")

betas = initial_betas.clone().requires_grad_(True)
body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

spine_joints = [0, 3, 6, 9, 12, 15]

# ========== OPTIMIZATION ==========
print("\n[INFO] Stage 1: Optimizing shape and global orientation...")
opt_stage1 = torch.optim.Adam([betas, global_orient, transl], lr=LR)

for it in trange(200, desc="Stage 1"):
    opt_stage1.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root
    
    total_loss = 0
    for target, weight in zip(all_targets, all_weights):
        valid_mask = weight > 0.3
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        
        scale = target_scale / (smpl_scale + 1e-8)
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # FIX: Different regularization for different betas
    loss_beta_shape = 3e-5 * (betas[0, 0] ** 2 + betas[0, 2] ** 2)  # Size/width less constrained
    loss_beta_detail = 1e-4 * torch.sum(betas[0, 1:] ** 2)  # Other betas slightly more constrained
    
    loss = total_loss / len(all_targets) + loss_beta_shape + loss_beta_detail
    loss.backward()
    opt_stage1.step()
    
    with torch.no_grad():
        betas.clamp_(-8.0, 8.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, betas={betas[0,:5].detach().cpu().numpy()}")

print("\n[INFO] Stage 2: Optimizing limb poses...")
opt_stage2 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.5)

for it in trange(250, desc="Stage 2"):
    opt_stage2.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root
    
    total_loss = 0
    for target, weight in zip(all_targets, all_weights):
        valid_mask = weight > 0.3
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        
        scale = target_scale / (smpl_scale + 1e-8)
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    loss_spine = 3e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 3e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 2e-5 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine + loss_pose + loss_beta
    loss.backward()
    opt_stage2.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.3, 0.3)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-8.0, 8.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

print("\n[INFO] Stage 3: Fine-tuning...")
opt_stage3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.2)

best_loss = 1e9
best_params = None

for it in trange(150, desc="Stage 3"):
    opt_stage3.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root
    
    total_loss = 0
    for target, weight in zip(all_targets, all_weights):
        valid_mask = weight > 0.3
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        
        scale = target_scale / (smpl_scale + 1e-8)
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    spine_joints_coords = out.joints[0, spine_joints, :]
    spine_dirs = spine_joints_coords[1:] - spine_joints_coords[:-1]
    spine_dirs_norm = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    loss_spine_align = 1e-2 * torch.mean((1 - torch.sum(spine_dirs_norm[:-1] * spine_dirs_norm[1:], dim=1)) ** 2)
    
    loss_spine_pose = 2e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 5e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 1e-5 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine_align + loss_spine_pose + loss_pose + loss_beta
    loss.backward()
    opt_stage3.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.4, 0.4)
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
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

print(f"\n[INFO] Optimization finished. Best loss: {best_loss:.6f}")
print(f"[INFO] Final betas: {best_params['betas'][0, :5].cpu().numpy()}")

# ========== GENERATE FINAL MESH ==========
final_out = smplx_model(
    betas=best_params['betas'],
    body_pose=best_params['body_pose'],
    global_orient=best_params['global_orient'],
    transl=best_params['transl']
)

verts = final_out.vertices[0].cpu().detach().numpy()
faces = smplx_model.faces

# FIX: Determine correct orientation from keypoints
# Check if model needs to be flipped based on Y-axis direction
ref_kps_check = all_keypoints[0]
head_y = ref_kps_check[0, 1]  # Nose Y
hip_y = (ref_kps_check[23, 1] + ref_kps_check[24, 1]) / 2.0

# In MediaPipe, Y increases downward, so head should have smaller Y than hips
needs_flip = head_y > hip_y

verts_centered = verts - verts.mean(axis=0)

if needs_flip:
    print("[INFO] Flipping model orientation (head was below hips)")
    # rotation_fix = R.from_euler('x', 180, degrees=True).as_matrix()
    # verts_centered = (rotation_fix @ verts_centered.T).T
else:
    print("[INFO] Model orientation correct (no flip needed)")

mesh_export = trimesh.Trimesh(verts_centered, faces)
mesh_export.export("fitted_smplx_mesh.obj")
print("[INFO] Exported fitted_smplx_mesh.obj")

mesh_export.visual.vertex_colors = [200, 200, 230, 255]
mesh_export.export("fitted_smplx_mesh_colored.ply")
print("[INFO] Exported fitted_smplx_mesh_colored.ply")

# ========== VISUALIZATION ==========
print("\n[INFO] Creating visualization...")
mesh = trimesh.Trimesh(verts_centered, faces)
mesh.visual.vertex_colors = [200, 200, 230, 255]

scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
scene.add(mesh_node)

# Add skeleton
joint_positions = final_out.joints[0].cpu().detach().numpy()
joint_positions_centered = joint_positions - joint_positions.mean(axis=0)

# if needs_flip:
#     rotation_fix = R.from_euler('x', 180, degrees=True).as_matrix()
#     joint_positions_centered = (rotation_fix @ joint_positions_centered.T).T

skeleton_pairs = [
    (0, 1), (0, 2), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (9, 14),
    (13, 16), (16, 18), (18, 20), (14, 17), (17, 19), (19, 21),
]

for j1, j2 in skeleton_pairs:
    p1, p2 = joint_positions_centered[j1], joint_positions_centered[j2]
    cylinder = trimesh.creation.cylinder(radius=0.01, height=np.linalg.norm(p2 - p1))
    direction = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-8)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, direction)
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        rot_mat = trimesh.transformations.rotation_matrix(rot_angle, rot_axis)
        cylinder.apply_transform(rot_mat)
    cylinder.apply_translation((p1 + p2) / 2)
    cylinder.visual.vertex_colors = [100, 255, 100, 255]
    scene.add(pyrender.Mesh.from_trimesh(cylinder))

# Add keypoints
if len(all_keypoints) > 0:
    kps_vis = all_keypoints[0].copy()
    if np.any(kps_vis[23]) and np.any(kps_vis[24]):
        root_vis = (kps_vis[23] + kps_vis[24]) / 2.0
    else:
        root_vis = np.median(kps_vis, axis=0)
    kps_vis = kps_vis - root_vis
    
    # if needs_flip:
    #     kps_vis = (rotation_fix @ kps_vis.T).T
    
    for idx in mp_indices:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.025)
        sphere.apply_translation(kps_vis[idx])
        sphere.visual.vertex_colors = [255, 60, 60, 255]
        scene.add(pyrender.Mesh.from_trimesh(sphere))

# Lighting
light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light1, pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], dtype=np.float32))
scene.add(light2, pose=np.array([[1,0,0,0],[0,0.707,-0.707,1],[0,0.707,0.707,1],[0,0,0,1]], dtype=np.float32))

# Camera
cam_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2.5],[0,0,0,1]], dtype=np.float32)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
scene.add(camera, pose=cam_pose)

print("[INFO] Launching viewer...")
try:
    pyrender.Viewer(scene, use_raymond_lighting=True)
except Exception as e:
    print(f"[WARNING] Viewer error: {e}")
    print("[INFO] Rendering to image instead...")
    r = pyrender.OffscreenRenderer(1200, 1200)
    color, _ = r.render(scene)
    cv2.imwrite("render_output.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print("[INFO] Saved render_output.png")
    r.delete()

print("\n[INFO] ✓ Pipeline complete!")
print(f"[INFO] Number of views used: {len(all_targets)}")
print(f"[INFO] Final body shape parameters: {best_params['betas'][0, :5].cpu().numpy()}")
print(f"[INFO] Body measurements: {body_measurements}")