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
from scipy.interpolate import interp1d

# ========== CONFIG ==========
IMAGES = {
    'front': "dataset/image1.jpg",
    'left': "dataset/image1_left.jpg",
    'right': "dataset/image1_right.jpg",
}
MODEL_PATH = "full_pipeline/models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 10
LR = 5e-2
VISUALIZE_KEYPOINTS = True  # Show keypoint extraction results

# ========== KEYPOINT VALIDATION ==========
def is_valid_pose(landmarks, confidences, min_confidence=0.3):
    critical_landmarks = [11, 12, 23, 24]
    for idx in critical_landmarks:
        if confidences[idx] < min_confidence:
            return False, f"Low confidence for landmark {idx}"
    
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    variance = np.var(coords[:, :2], axis=0)
    if np.any(variance > 0.5):
        return False, "Keypoints too scattered"
    
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

# ========== EXTRACT BODY CONTOUR POINTS ==========
def extract_body_contours(mask, num_points=50):
    """
    Extract contour points from segmentation mask to capture body shape.
    Returns 3D points with depth estimated from body region.
    """
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None
    
    # Get largest contour (main body)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Resample contour to fixed number of points
    contour_points = main_contour.squeeze()
    if len(contour_points.shape) == 1:
        return None
    
    # Sample evenly spaced points along contour
    perimeter = cv2.arcLength(main_contour, True)
    sample_distances = np.linspace(0, perimeter, num_points, endpoint=False)
    
    sampled_points = []
    cumulative_dist = 0
    point_idx = 0
    
    for i in range(len(contour_points)):
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % len(contour_points)]
        segment_length = np.linalg.norm(p2 - p1)
        
        while point_idx < num_points and sample_distances[point_idx] <= cumulative_dist + segment_length:
            t = (sample_distances[point_idx] - cumulative_dist) / (segment_length + 1e-8)
            point = p1 + t * (p2 - p1)
            sampled_points.append(point)
            point_idx += 1
        
        cumulative_dist += segment_length
    
    if len(sampled_points) == 0:
        return None
    
    sampled_points = np.array(sampled_points)
    
    # Estimate depth based on distance from body center
    h, w = mask.shape
    center = np.array([w / 2, h / 2])
    
    # Normalize to [-1, 1] range
    normalized_points = np.zeros((len(sampled_points), 3))
    normalized_points[:, 0] = (sampled_points[:, 0] - w / 2) / (w / 2)
    normalized_points[:, 1] = (sampled_points[:, 1] - h / 2) / (h / 2)
    
    # Estimate depth: points farther from center are likely more forward/backward
    distances = np.linalg.norm(sampled_points - center, axis=1)
    max_dist = np.max(distances)
    normalized_points[:, 2] = (distances / (max_dist + 1e-8) - 0.5) * 0.3
    
    return normalized_points

# ========== ENHANCED BODY MEASUREMENTS ==========
def extract_detailed_measurements(mask, landmarks, h, w):
    """Extract detailed body measurements from mask and landmarks."""
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    measurements = {
        'shoulder_width': 0,
        'chest_width': 0,
        'waist_width': 0,
        'hip_width': 0,
        'thigh_width': 0,
        'arm_thickness': 0,
        'torso_depth': 0,
    }
    
    # Get landmark positions
    left_shoulder = np.array([landmarks[11].x * w, landmarks[11].y * h])
    right_shoulder = np.array([landmarks[12].x * w, landmarks[12].y * h])
    left_hip = np.array([landmarks[23].x * w, landmarks[23].y * h])
    right_hip = np.array([landmarks[24].x * w, landmarks[24].y * h])
    
    # Shoulder width
    measurements['shoulder_width'] = np.linalg.norm(left_shoulder - right_shoulder) / w
    
    # Hip width  
    measurements['hip_width'] = np.linalg.norm(left_hip - right_hip) / w
    
    # Measure widths at different body heights
    height_ranges = {
        'chest': (0.25, 0.35),
        'waist': (0.45, 0.55),
        'hip': (0.55, 0.65),
        'thigh': (0.65, 0.75),
    }
    
    for region, (start_h, end_h) in height_ranges.items():
        start_row = int(h * start_h)
        end_row = int(h * end_h)
        region_mask = mask_binary[start_row:end_row, :]
        
        if region_mask.sum() > 0:
            row_widths = np.sum(region_mask, axis=1)
            avg_width = np.mean(row_widths[row_widths > 0])
            measurements[f'{region}_width'] = avg_width / w
    
    # Arm thickness (measure at elbow height)
    if landmarks[13].visibility > 0.5:  # Left elbow
        elbow_y = int(landmarks[13].y * h)
        if 0 < elbow_y < h:
            elbow_row = mask_binary[elbow_y, :]
            left_arm_pixels = elbow_row[:int(w * 0.3)]
            if left_arm_pixels.sum() > 0:
                measurements['arm_thickness'] = left_arm_pixels.sum() / w
    
    return measurements

# ========== KEYPOINT EXTRACTION WITH VISUALIZATION ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

all_keypoints = []
all_view_names = []
all_confidences = []
all_images_rgb = []
all_segmentation_masks = []
all_contour_points = []
all_measurements = []

for view_name, img_path in IMAGES.items():
    print(f"\n[INFO] Processing {view_name} view: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not load {img_path}, skipping...")
        continue
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"[INFO] Image size: {w}x{h}")
    
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        print(f"[WARNING] No pose detected in {view_name}, skipping...")
        continue
    
    # Validate pose
    landmarks = results.pose_landmarks.landmark
    confidences_check = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    is_valid, message = is_valid_pose(landmarks, confidences_check)
    
    if not is_valid:
        print(f"[WARNING] Invalid pose in {view_name}: {message}, skipping...")
        continue
    
    print(f"[INFO] Valid pose detected in {view_name}")
    
    # Extract keypoints
    if getattr(results, "pose_world_landmarks", None) is not None:
        world_landmarks = results.pose_world_landmarks.landmark
        kps = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)
        confidences = np.array([lm.visibility for lm in world_landmarks], dtype=np.float32)
    else:
        kps = np.array([[lm.x, lm.y, lm.z * w / h] for lm in landmarks], dtype=np.float32)
        confidences = confidences_check
    
    # Extract contour points
    contour_points = None
    if results.segmentation_mask is not None:
        all_segmentation_masks.append((results.segmentation_mask, img_rgb))
        contour_points = extract_body_contours(results.segmentation_mask, num_points=50)
        if contour_points is not None:
            print(f"[INFO] Extracted {len(contour_points)} contour points")
        
        # Extract detailed measurements
        measurements = extract_detailed_measurements(
            results.segmentation_mask, landmarks, h, w
        )
        all_measurements.append(measurements)
        print(f"[INFO] Measurements: shoulder={measurements['shoulder_width']:.3f}, "
              f"chest={measurements['chest_width']:.3f}, waist={measurements['waist_width']:.3f}, "
              f"hip={measurements['hip_width']:.3f}, thigh={measurements['thigh_width']:.3f}")
    
    all_keypoints.append(kps)
    all_view_names.append(view_name)
    all_confidences.append(confidences)
    all_images_rgb.append(img_rgb)
    all_contour_points.append(contour_points)
    
    # VISUALIZATION: Show keypoint extraction
    if VISUALIZE_KEYPOINTS:
        fig = plt.figure(figsize=(15, 5))
        
        # Original image with keypoints
        ax1 = fig.add_subplot(131)
        ax1.imshow(img_rgb)
        ax1.set_title(f'{view_name.upper()} - Keypoints')
        for idx, (kp, conf) in enumerate(zip(landmarks, confidences)):
            if conf > 0.3:
                color = 'green' if conf > 0.7 else 'yellow'
                ax1.plot(kp.x * w, kp.y * h, 'o', color=color, markersize=5)
                ax1.text(kp.x * w, kp.y * h, str(idx), fontsize=6)
        ax1.axis('off')
        
        # Segmentation mask with contours
        ax2 = fig.add_subplot(132)
        if results.segmentation_mask is not None:
            ax2.imshow(results.segmentation_mask, cmap='gray')
            ax2.set_title(f'{view_name.upper()} - Segmentation')
            if contour_points is not None:
                # Convert back to image coordinates for visualization
                contour_vis = np.copy(contour_points)
                contour_vis[:, 0] = (contour_vis[:, 0] + 1) * w / 2
                contour_vis[:, 1] = (contour_vis[:, 1] + 1) * h / 2
                ax2.plot(contour_vis[:, 0], contour_vis[:, 1], 'r.', markersize=3)
        ax2.axis('off')
        
        # Measurements visualization
        ax3 = fig.add_subplot(133)
        ax3.imshow(img_rgb)
        ax3.set_title(f'{view_name.upper()} - Measurements')
        if len(all_measurements) > 0:
            meas = all_measurements[-1]
            # Draw measurement lines
            if landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:
                ax3.plot([landmarks[11].x * w, landmarks[12].x * w],
                        [landmarks[11].y * h, landmarks[12].y * h], 'g-', linewidth=2)
                ax3.text(w/2, landmarks[11].y * h - 20, f"Shoulder: {meas['shoulder_width']:.2f}", 
                        color='green', fontsize=10, ha='center')
            
            if landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5:
                ax3.plot([landmarks[23].x * w, landmarks[24].x * w],
                        [landmarks[23].y * h, landmarks[24].y * h], 'b-', linewidth=2)
                ax3.text(w/2, landmarks[23].y * h + 20, f"Hip: {meas['hip_width']:.2f}", 
                        color='blue', fontsize=10, ha='center')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'keypoint_extraction_{view_name}.png', dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved visualization: keypoint_extraction_{view_name}.png")
        plt.close()

if len(all_keypoints) == 0:
    raise ValueError("No valid poses detected in any image!")

# ========== AGGREGATE MEASUREMENTS ==========
if len(all_measurements) > 0:
    avg_measurements = {}
    for key in all_measurements[0].keys():
        values = [m[key] for m in all_measurements if m[key] > 0]
        avg_measurements[key] = np.median(values) if len(values) > 0 else 1.0
    
    print(f"\n[INFO] Average measurements across views:")
    for key, value in avg_measurements.items():
        print(f"  {key}: {value:.3f}")
else:
    avg_measurements = {
        'shoulder_width': 1.0, 'chest_width': 1.0, 'waist_width': 1.0,
        'hip_width': 1.0, 'thigh_width': 1.0, 'arm_thickness': 1.0
    }

# ========== LOAD SMPL-X MODEL ==========
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=NUM_BETAS,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# Enhanced mapping with more joints
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
    
    scale = np.sum(S) / (np.trace(source_centered.T @ W @ source_centered) + 1e-8)
    
    return R_mat, scale, source_center, target_center

# ========== PREPARE TARGETS ==========
all_targets = []
all_weights = []
all_contour_targets = []
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
    
    # Apply view rotations
    if 'left' in view_name.lower():
        rot = R.from_euler('y', -90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        
        if len(all_keypoints) > 1 and view_idx != reference_idx:
            R_mat, scale, src_center, tgt_center = procrustes_align(
                kps_centered[mp_indices], ref_kps_for_align,
                weights=conf[mp_indices] * ref_conf_for_align
            )
            kps_centered = scale * ((kps_centered - src_center) @ R_mat.T) + tgt_center
            
    elif 'right' in view_name.lower():
        rot = R.from_euler('y', 90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        
        if len(all_keypoints) > 1 and view_idx != reference_idx:
            R_mat, scale, src_center, tgt_center = procrustes_align(
                kps_centered[mp_indices], ref_kps_for_align,
                weights=conf[mp_indices] * ref_conf_for_align
            )
            kps_centered = scale * ((kps_centered - src_center) @ R_mat.T) + tgt_center
    
    kps_selected = kps_centered[mp_indices]
    conf_selected = conf[mp_indices].copy()
    
    # Boost confidence for visible keypoints
    if 'left' in view_name.lower():
        left_mask = np.isin(mp_indices, [11, 13, 15, 23, 25, 27])
        conf_selected[left_mask] *= 1.5
    elif 'right' in view_name.lower():
        right_mask = np.isin(mp_indices, [12, 14, 16, 24, 26, 28])
        conf_selected[right_mask] *= 1.5
    
    conf_selected = np.clip(conf_selected, 0, 1)
    
    all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))
    
    # Add contour points if available
    if all_contour_points[view_idx] is not None:
        contour = all_contour_points[view_idx].copy()
        if 'left' in view_name.lower():
            contour = (rot @ contour.T).T
        elif 'right' in view_name.lower():
            contour = (rot @ contour.T).T
        all_contour_targets.append(torch.tensor(contour, dtype=torch.float32, device=DEVICE))
    else:
        all_contour_targets.append(None)

# ========== BETA INITIALIZATION BASED ON MEASUREMENTS ==========
initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)

# Reference measurements for average body
ref_meas = {
    'shoulder_width': 0.25, 'chest_width': 0.30, 'waist_width': 0.25,
    'hip_width': 0.28, 'thigh_width': 0.20, 'arm_thickness': 0.08
}

# Beta mappings (approximate for SMPL-X)
# 0: overall size, 1: belly/torso, 2: overall width, 3: leg thickness
# 4: chest, 5: neck, 6: arm thickness, 7: shoulder width

initial_betas[0, 0] = (avg_measurements['chest_width'] / ref_meas['chest_width'] - 1.0) * 2.5
initial_betas[0, 1] = (avg_measurements['waist_width'] / ref_meas['waist_width'] - 1.0) * 4.0
initial_betas[0, 2] = (avg_measurements['shoulder_width'] / ref_meas['shoulder_width'] - 1.0) * 3.0
initial_betas[0, 3] = (avg_measurements['thigh_width'] / ref_meas['thigh_width'] - 1.0) * 3.5
initial_betas[0, 6] = (avg_measurements['arm_thickness'] / ref_meas['arm_thickness'] - 1.0) * 2.0

print(f"[INFO] Initial betas: {initial_betas[0, :7].cpu().numpy()}")

betas = initial_betas.clone().requires_grad_(True)
body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# ========== OPTIMIZATION WITH CONTOUR LOSS ==========
print("\n[INFO] Stage 1: Optimizing shape with contour guidance...")
opt_stage1 = torch.optim.Adam([betas, global_orient, transl], lr=LR)

for it in trange(250, desc="Stage 1"):
    opt_stage1.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    smpl_joints = smpl_joints_full[smpl_indices] - smpl_root
    
    # Joint loss
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
    
    # NEW: Contour/silhouette loss for better body shape
    loss_contour = 0
    if any(c is not None for c in all_contour_targets):
        verts = out.vertices[0] - smpl_root
        # Sample vertices from torso region for silhouette matching
        torso_verts_indices = torch.arange(1000, 4000, device=DEVICE)  # Approximate torso region
        torso_verts = verts[torso_verts_indices]
        
        for contour in all_contour_targets:
            if contour is not None:
                # Chamfer-like distance between contour and mesh vertices
                dists = torch.cdist(contour, torso_verts)
                loss_contour += dists.min(dim=1)[0].mean() * 0.1
    
    # Beta regularization with different weights
    loss_beta = 2e-5 * (betas[0, 0] ** 2 + betas[0, 2] ** 2) + 5e-5 * torch.sum(betas[0, 1:] ** 2)
    
    loss = total_loss / len(all_targets) + loss_contour + loss_beta
    loss.backward()
    opt_stage1.step()
    
    with torch.no_grad():
        betas.clamp_(-10.0, 10.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, " 
              f"joint={total_loss.item()/len(all_targets):.6f}, contour={loss_contour:.6f}, "
              f"betas={betas[0,:5].detach().cpu().numpy()}")

print("\n[INFO] Stage 2: Optimizing pose...")
opt_stage2 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.4)

for it in trange(300, desc="Stage 2"):
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
    
    loss_spine = 2e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 2e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 1e-5 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine + loss_pose + loss_beta
    loss.backward()
    opt_stage2.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.3, 0.3)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-10.0, 10.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

print("\n[INFO] Stage 3: Fine-tuning...")
opt_stage3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.15)

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
    
    spine_joints = [0, 3, 6, 9, 12, 15]
    spine_joints_coords = out.joints[0, spine_joints, :]
    spine_dirs = spine_joints_coords[1:] - spine_joints_coords[:-1]
    spine_dirs_norm = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    loss_spine_align = 8e-3 * torch.mean((1 - torch.sum(spine_dirs_norm[:-1] * spine_dirs_norm[1:], dim=1)) ** 2)
    
    loss_spine_pose = 1e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 3e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 5e-6 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine_align + loss_spine_pose + loss_pose + loss_beta
    loss.backward()
    opt_stage3.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.4, 0.4)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-10.0, 10.0)
    
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
print(f"[INFO] Final betas: {best_params['betas'][0, :7].cpu().numpy()}")

# ========== GENERATE FINAL MESH ==========
final_out = smplx_model(
    betas=best_params['betas'],
    body_pose=best_params['body_pose'],
    global_orient=best_params['global_orient'],
    transl=best_params['transl']
)

verts = final_out.vertices[0].cpu().detach().numpy()
faces = smplx_model.faces

# Determine orientation
ref_kps_check = all_keypoints[0]
head_y = ref_kps_check[0, 1]
hip_y = (ref_kps_check[23, 1] + ref_kps_check[24, 1]) / 2.0
needs_flip = head_y > hip_y

verts_centered = verts - verts.mean(axis=0)

if needs_flip:
    print("[INFO] Flipping model orientation")
    rotation_fix = R.from_euler('x', 180, degrees=True).as_matrix()
    verts_centered = (rotation_fix @ verts_centered.T).T
else:
    print("[INFO] Model orientation correct")

mesh_export = trimesh.Trimesh(verts_centered, faces)
mesh_export.export("fitted_smplx_mesh.obj")
print("[INFO] Exported fitted_smplx_mesh.obj")

mesh_export.visual.vertex_colors = [200, 200, 230, 255]
mesh_export.export("fitted_smplx_mesh_colored.ply")
print("[INFO] Exported fitted_smplx_mesh_colored.ply")

# ========== CREATE COMPARISON VISUALIZATION ==========
if VISUALIZE_KEYPOINTS:
    print("\n[INFO] Creating final comparison visualization...")
    fig = plt.figure(figsize=(20, 8))
    
    for idx, (view_name, img_rgb) in enumerate(zip(all_view_names, all_images_rgb)):
        ax = fig.add_subplot(2, len(all_view_names), idx + 1)
        ax.imshow(img_rgb)
        ax.set_title(f'{view_name.upper()} - Original')
        ax.axis('off')
    
    # Show measurements comparison
    ax_meas = fig.add_subplot(2, len(all_view_names), len(all_view_names) + 1)
    ax_meas.axis('off')
    ax_meas.set_title('Body Measurements', fontsize=14, fontweight='bold')
    
    y_pos = 0.9
    for key, value in avg_measurements.items():
        ref_val = {'shoulder_width': 0.25, 'chest_width': 0.30, 'waist_width': 0.25,
                   'hip_width': 0.28, 'thigh_width': 0.20, 'arm_thickness': 0.08}.get(key, 1.0)
        ratio = value / ref_val
        color = 'green' if 0.8 < ratio < 1.2 else 'orange' if 0.6 < ratio < 1.4 else 'red'
        ax_meas.text(0.1, y_pos, f'{key}: {value:.3f} ({ratio:.2f}x avg)', 
                    fontsize=11, color=color, transform=ax_meas.transAxes)
        y_pos -= 0.12
    
    # Show beta parameters
    ax_beta = fig.add_subplot(2, len(all_view_names), len(all_view_names) + 2)
    ax_beta.axis('off')
    ax_beta.set_title('SMPL-X Beta Parameters', fontsize=14, fontweight='bold')
    
    beta_names = ['Size', 'Belly', 'Width', 'Legs', 'Chest', 'Neck', 'Arms']
    final_betas = best_params['betas'][0, :7].cpu().numpy()
    
    y_pos = 0.9
    for i, (name, beta_val) in enumerate(zip(beta_names, final_betas)):
        color = 'blue' if abs(beta_val) < 2 else 'orange' if abs(beta_val) < 4 else 'red'
        ax_beta.text(0.1, y_pos, f'β{i} ({name}): {beta_val:.3f}', 
                    fontsize=11, color=color, transform=ax_beta.transAxes)
        y_pos -= 0.12
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
    print("[INFO] Saved final_comparison.png")
    plt.close()

# ========== VISUALIZATION ==========
print("\n[INFO] Creating 3D visualization...")
mesh = trimesh.Trimesh(verts_centered, faces)
mesh.visual.vertex_colors = [200, 200, 230, 255]

scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
scene.add(mesh_node)

# Add skeleton
joint_positions = final_out.joints[0].cpu().detach().numpy()
joint_positions_centered = joint_positions - joint_positions.mean(axis=0)

if needs_flip:
    joint_positions_centered = (rotation_fix @ joint_positions_centered.T).T

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
    
    if needs_flip:
        kps_vis = (rotation_fix @ kps_vis.T).T
    
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
print(f"[INFO] Final body shape parameters (first 7 betas): {best_params['betas'][0, :7].cpu().numpy()}")
print(f"[INFO] Average measurements: {avg_measurements}")
print(f"\n[INFO] Visualization files created:")
print(f"  - keypoint_extraction_*.png (keypoint detection results)")
print(f"  - final_comparison.png (measurements and parameters)")
print(f"  - render_output.png (3D model render)")
print(f"  - fitted_smplx_mesh.obj (exportable 3D model)")