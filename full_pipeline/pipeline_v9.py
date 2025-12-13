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
LR = 5e-2
VISUALIZE_KEYPOINTS = True

# ========== EXTRACT 3D POINT CLOUD FROM SEGMENTATION ==========
def extract_body_point_cloud(mask, depth_estimation='edges', num_points=500):
    """
    Extract 3D point cloud from segmentation mask.
    Uses edge-based depth estimation for more accurate body shape.
    """
    mask_binary = (mask > 0.5).astype(np.uint8)
    h, w = mask.shape
    
    # Get all body pixels
    body_coords = np.argwhere(mask_binary > 0)  # [y, x]
    
    if len(body_coords) == 0:
        return None
    
    # Sample points uniformly
    if len(body_coords) > num_points:
        indices = np.random.choice(len(body_coords), num_points, replace=False)
        body_coords = body_coords[indices]
    
    # Convert to normalized coordinates
    points_3d = np.zeros((len(body_coords), 3), dtype=np.float32)
    points_3d[:, 0] = (body_coords[:, 1] - w / 2) / (w / 2)  # X: -1 to 1
    points_3d[:, 1] = (body_coords[:, 0] - h / 2) / (h / 2)  # Y: -1 to 1
    
    # Estimate depth based on distance from edges
    if depth_estimation == 'edges':
        # Distance transform: pixels closer to edge are more forward/backward
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        
        # Normalize distance transform
        max_dist = np.max(dist_transform)
        
        for i, (y, x) in enumerate(body_coords):
            # Depth based on distance from edge
            # Center of body = forward (positive Z), edges = backward
            depth_ratio = dist_transform[y, x] / (max_dist + 1e-6)
            points_3d[i, 2] = (depth_ratio - 0.5) * 0.4  # Scale depth
    else:
        # Simple center-based depth
        center = np.array([w / 2, h / 2])
        for i, (y, x) in enumerate(body_coords):
            dist = np.linalg.norm([x - center[0], y - center[1]])
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            points_3d[i, 2] = (dist / max_dist - 0.5) * 0.3
    
    return points_3d

# ========== EXTRACT DENSE BODY SURFACE POINTS ==========
def extract_torso_surface_points(mask, landmarks, num_points=200):
    """
    Extract dense surface points specifically from torso region
    for better belly/waist/chest fitting.
    """
    mask_binary = (mask > 0.5).astype(np.uint8)
    h, w = mask.shape
    
    # Define torso region based on landmarks
    if landmarks[11].visibility > 0.3 and landmarks[23].visibility > 0.3:
        shoulder_y = int(landmarks[11].y * h)
        hip_y = int(landmarks[23].y * h)
        
        # Extract torso region (shoulder to hip)
        torso_start = max(0, shoulder_y)
        torso_end = min(h, hip_y)
        
        torso_mask = np.zeros_like(mask_binary)
        torso_mask[torso_start:torso_end, :] = mask_binary[torso_start:torso_end, :]
        
        # Get horizontal slices at different heights
        num_slices = 15  # More slices for better torso detail
        slice_points = []
        
        for i in range(num_slices):
            slice_y = torso_start + int((torso_end - torso_start) * i / num_slices)
            slice_row = torso_mask[slice_y, :]
            
            # Find left and right edges
            body_pixels = np.where(slice_row > 0)[0]
            if len(body_pixels) > 0:
                left_edge = body_pixels[0]
                right_edge = body_pixels[-1]
                center_x = (left_edge + right_edge) / 2
                width = right_edge - left_edge
                
                # Add points along this slice
                for x in [left_edge, center_x, right_edge]:
                    # Normalize coordinates
                    norm_x = (x - w / 2) / (w / 2)
                    norm_y = (slice_y - h / 2) / (h / 2)
                    
                    # Estimate depth based on distance from center
                    depth = (abs(x - center_x) / (width / 2 + 1e-6)) * 0.3
                    if x == center_x:
                        depth = 0.35  # Front of torso
                    
                    slice_points.append([norm_x, norm_y, depth])
        
        if len(slice_points) > 0:
            return np.array(slice_points, dtype=np.float32)
    
    return None

# ========== ADVANCED KEYPOINT EXTRACTION ==========
def extract_enhanced_keypoints(landmarks, world_landmarks, h, w):
    """
    Extract enhanced keypoints including spine, torso midpoints.
    Combines MediaPipe landmarks with computed torso points.
    """
    all_keypoints = []
    all_confidences = []
    keypoint_names = []
    
    # Standard MediaPipe keypoints
    mp_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    mp_names = ['nose', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 
                'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee',
                'l_ankle', 'r_ankle']
    
    if world_landmarks:
        for idx, name in zip(mp_indices, mp_names):
            lm = world_landmarks[idx]
            all_keypoints.append([lm.x, lm.y, lm.z])
            all_confidences.append(lm.visibility)
            keypoint_names.append(name)
    else:
        for idx, name in zip(mp_indices, mp_names):
            lm = landmarks[idx]
            all_keypoints.append([lm.x, lm.y, lm.z])
            all_confidences.append(lm.visibility)
            keypoint_names.append(name)
    
    # ENHANCED: Add spine/torso keypoints (OpenPose-style)
    # Calculate midpoints for spine
    if landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:
        # Neck (midpoint between shoulders)
        if world_landmarks:
            neck = [(world_landmarks[11].x + world_landmarks[12].x) / 2,
                   (world_landmarks[11].y + world_landmarks[12].y) / 2,
                   (world_landmarks[11].z + world_landmarks[12].z) / 2]
        else:
            neck = [(landmarks[11].x + landmarks[12].x) / 2,
                   (landmarks[11].y + landmarks[12].y) / 2,
                   (landmarks[11].z + landmarks[12].z) / 2]
        all_keypoints.append(neck)
        all_confidences.append(0.9)
        keypoint_names.append('neck')
    
    if landmarks[23].visibility > 0.5 and landmarks[24].visibility > 0.5:
        # Hip center
        if world_landmarks:
            hip_center = [(world_landmarks[23].x + world_landmarks[24].x) / 2,
                         (world_landmarks[23].y + world_landmarks[24].y) / 2,
                         (world_landmarks[23].z + world_landmarks[24].z) / 2]
        else:
            hip_center = [(landmarks[23].x + landmarks[24].x) / 2,
                         (landmarks[23].y + landmarks[24].y) / 2,
                         (landmarks[23].z + landmarks[24].z) / 2]
        all_keypoints.append(hip_center)
        all_confidences.append(0.9)
        keypoint_names.append('hip_center')
        
        # ENHANCED: Add mid-torso points (chest, upper abdomen, lower abdomen)
        if len(all_keypoints) > 0 and 'neck' in keypoint_names:
            neck_pos = all_keypoints[keypoint_names.index('neck')]
            hip_pos = hip_center
            
            # Upper chest (25% from neck to hip)
            upper_chest = [
                neck_pos[0] * 0.75 + hip_pos[0] * 0.25,
                neck_pos[1] * 0.75 + hip_pos[1] * 0.25,
                neck_pos[2] * 0.75 + hip_pos[2] * 0.25
            ]
            all_keypoints.append(upper_chest)
            all_confidences.append(0.85)
            keypoint_names.append('upper_chest')
            
            # Mid torso / waist (50%)
            mid_torso = [
                (neck_pos[0] + hip_pos[0]) / 2,
                (neck_pos[1] + hip_pos[1]) / 2,
                (neck_pos[2] + hip_pos[2]) / 2
            ]
            all_keypoints.append(mid_torso)
            all_confidences.append(0.85)
            keypoint_names.append('waist')
            
            # Lower abdomen (75%)
            lower_abdomen = [
                neck_pos[0] * 0.25 + hip_pos[0] * 0.75,
                neck_pos[1] * 0.25 + hip_pos[1] * 0.75,
                neck_pos[2] * 0.25 + hip_pos[2] * 0.75
            ]
            all_keypoints.append(lower_abdomen)
            all_confidences.append(0.85)
            keypoint_names.append('lower_abdomen')
    
    return np.array(all_keypoints, dtype=np.float32), np.array(all_confidences, dtype=np.float32), keypoint_names

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

# ========== KEYPOINT EXTRACTION ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

all_keypoints = []
all_keypoint_names = []
all_view_names = []
all_confidences = []
all_images_rgb = []
all_point_clouds = []
all_torso_points = []

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
    
    landmarks = results.pose_landmarks.landmark
    confidences_check = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    is_valid, message = is_valid_pose(landmarks, confidences_check)
    
    if not is_valid:
        print(f"[WARNING] Invalid pose in {view_name}: {message}, skipping...")
        continue
    
    print(f"[INFO] Valid pose detected in {view_name}")
    
    # Extract enhanced keypoints
    world_landmarks = results.pose_world_landmarks.landmark if hasattr(results, 'pose_world_landmarks') else None
    kps, confs, kp_names = extract_enhanced_keypoints(landmarks, world_landmarks, h, w)
    
    print(f"[INFO] Extracted {len(kps)} keypoints (including {len(kps) - 13} enhanced torso points)")
    
    # Extract 3D point cloud from segmentation
    point_cloud = None
    torso_surface = None
    if results.segmentation_mask is not None:
        point_cloud = extract_body_point_cloud(results.segmentation_mask, num_points=500)
        torso_surface = extract_torso_surface_points(results.segmentation_mask, landmarks, num_points=200)
        
        if point_cloud is not None:
            print(f"[INFO] Extracted {len(point_cloud)} body surface points")
        if torso_surface is not None:
            print(f"[INFO] Extracted {len(torso_surface)} torso surface points")
    
    all_keypoints.append(kps)
    all_keypoint_names.append(kp_names)
    all_confidences.append(confs)
    all_view_names.append(view_name)
    all_images_rgb.append(img_rgb)
    all_point_clouds.append(point_cloud)
    all_torso_points.append(torso_surface)
    
    # VISUALIZATION
    if VISUALIZE_KEYPOINTS:
        fig = plt.figure(figsize=(18, 6))
        
        # Keypoints
        ax1 = fig.add_subplot(141)
        ax1.imshow(img_rgb)
        ax1.set_title(f'{view_name.upper()} - Enhanced Keypoints ({len(kps)})')
        for idx, (kp, conf, name) in enumerate(zip(kps, confs, kp_names)):
            if world_landmarks:
                x, y = kp[0] * 1000, kp[1] * 1000  # Approximate
            else:
                x, y = kp[0] * w, kp[1] * h
            
            color = 'red' if 'torso' in name or 'abdomen' in name or 'waist' in name or 'chest' in name else 'green' if conf > 0.7 else 'yellow'
            size = 8 if 'torso' in name or 'abdomen' in name or 'waist' in name else 5
            ax1.plot(x, y, 'o', color=color, markersize=size)
            if 'torso' in name or 'abdomen' in name or 'waist' in name or 'chest' in name:
                ax1.text(x, y, name[:4], fontsize=7, color='red')
        ax1.axis('off')
        
        # Segmentation
        ax2 = fig.add_subplot(142)
        if results.segmentation_mask is not None:
            ax2.imshow(results.segmentation_mask, cmap='gray')
            ax2.set_title(f'{view_name.upper()} - Segmentation')
        ax2.axis('off')
        
        # Point cloud (2D projection)
        ax3 = fig.add_subplot(143)
        if point_cloud is not None:
            ax3.scatter(point_cloud[:, 0], point_cloud[:, 1], c=point_cloud[:, 2], 
                       cmap='viridis', s=1, alpha=0.5)
            ax3.set_title(f'Body Point Cloud ({len(point_cloud)} pts)')
            ax3.set_aspect('equal')
            ax3.invert_yaxis()
        
        # Torso surface points
        ax4 = fig.add_subplot(144)
        if torso_surface is not None:
            ax4.scatter(torso_surface[:, 0], torso_surface[:, 1], c=torso_surface[:, 2],
                       cmap='hot', s=10, alpha=0.7)
            ax4.set_title(f'Torso Surface ({len(torso_surface)} pts)')
            ax4.set_aspect('equal')
            ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'keypoint_extraction_{view_name}.png', dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved visualization: keypoint_extraction_{view_name}.png")
        plt.close()

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

# Map enhanced keypoints to SMPL joints
enhanced_kp_to_smpl = {
    'nose': 15, 'l_shoulder': 16, 'r_shoulder': 17,
    'l_elbow': 18, 'r_elbow': 19, 'l_wrist': 20, 'r_wrist': 21,
    'l_hip': 1, 'r_hip': 2, 'l_knee': 4, 'r_knee': 5,
    'l_ankle': 7, 'r_ankle': 8,
    'neck': 12, 'hip_center': 0,  # SMPL joint indices
    # Torso points map to nearby SMPL joints for guidance
    'upper_chest': 6, 'waist': 3, 'lower_abdomen': 0
}

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
all_target_joint_indices = []
all_surface_clouds = []
reference_idx = 0

ref_kps = all_keypoints[reference_idx].copy()
ref_conf = all_confidences[reference_idx].copy()
ref_names = all_keypoint_names[reference_idx]

# Find hip center for root
hip_center_idx = ref_names.index('hip_center') if 'hip_center' in ref_names else None
if hip_center_idx is not None:
    ref_root = ref_kps[hip_center_idx]
else:
    ref_root = np.median(ref_kps, axis=0)

ref_kps_centered = ref_kps - ref_root

print(f"\n[INFO] Using {len(all_keypoints)} view(s) for optimization")

for view_idx, (kps, conf, kp_names, view_name) in enumerate(zip(all_keypoints, all_confidences, all_keypoint_names, all_view_names)):
    # Find root
    hip_center_idx = kp_names.index('hip_center') if 'hip_center' in kp_names else None
    if hip_center_idx is not None:
        root = kps[hip_center_idx]
    else:
        root = np.median(kps, axis=0)
    
    kps_centered = kps - root
    
    # Apply view rotations
    if 'left' in view_name.lower():
        rot = R.from_euler('y', -90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied -90° Y rotation for left view")
        
    elif 'right' in view_name.lower():
        rot = R.from_euler('y', 90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied 90° Y rotation for right view")
    
    # Map keypoints to SMPL joint indices
    target_kps = []
    target_weights = []
    target_indices = []
    
    for kp, c, name in zip(kps_centered, conf, kp_names):
        if name in enhanced_kp_to_smpl:
            target_kps.append(kp)
            target_weights.append(c)
            target_indices.append(enhanced_kp_to_smpl[name])
    
    all_targets.append(torch.tensor(np.array(target_kps), dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(np.array(target_weights), dtype=torch.float32, device=DEVICE))
    all_target_joint_indices.append(torch.tensor(np.array(target_indices), dtype=torch.long, device=DEVICE))
    
    # Add point cloud
    if all_point_clouds[view_idx] is not None:
        pc = all_point_clouds[view_idx].copy()
        if 'left' in view_name.lower() or 'right' in view_name.lower():
            pc = (rot @ pc.T).T
        all_surface_clouds.append(torch.tensor(pc, dtype=torch.float32, device=DEVICE))
    else:
        all_surface_clouds.append(None)
    
    print(f"[INFO] {view_name}: {len(target_kps)} keypoints mapped to SMPL joints")

# ========== BETA INITIALIZATION ==========
initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)
# Start with slightly larger values to encourage body shape fitting
initial_betas[0, 0] = 0.5  # Overall size
initial_betas[0, 1] = 1.0  # Belly/torso
initial_betas[0, 2] = 0.5  # Width

print(f"[INFO] Initial betas: {initial_betas[0, :5].cpu().numpy()}")

betas = initial_betas.clone().requires_grad_(True)
body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# ========== OPTIMIZATION WITH SURFACE FITTING ==========
print("\n[INFO] Stage 1: Shape optimization with surface fitting...")
opt_stage1 = torch.optim.Adam([betas, global_orient, transl], lr=LR)

for it in trange(300, desc="Stage 1"):
    opt_stage1.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    
    # Find root joint
    if 'hip_center' in all_keypoint_names[0]:
        smpl_root = smpl_joints_full[0]  # Hip center
    else:
        smpl_root = (smpl_joints_full[1] + smpl_joints_full[2]) / 2.0
    
    # Joint loss
    total_loss = 0
    for target, weight, joint_indices in zip(all_targets, all_weights, all_target_joint_indices):
        smpl_joints_selected = smpl_joints_full[joint_indices] - smpl_root
        
        valid_mask = weight > 0.3
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints_selected[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints_selected, dim=1).mean()
        
        scale = target_scale / (smpl_scale + 1e-8)
        diff = (smpl_joints_selected * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # CRITICAL: Surface fitting loss using point cloud
    loss_surface = 0
    verts = out.vertices[0] - smpl_root
    
    for pc in all_surface_clouds:
        if pc is not None:
            # Chamfer distance: for each point in cloud, find nearest vertex
            dists = torch.cdist(pc, verts)
            loss_surface += dists.min(dim=1)[0].mean() * 2.0  # Strong weight
    
    # Beta regularization - VERY LIGHT to allow shape changes
    loss_beta = 1e-6 * torch.sum(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_surface + loss_beta
    loss.backward()
    opt_stage1.step()
    
    with torch.no_grad():
        betas.clamp_(-12.0, 12.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, joint={total_loss.item()/len(all_targets):.6f}, "
              f"surface={loss_surface:.6f}, betas={betas[0,:5].detach().cpu().numpy()}")

print("\n[INFO] Stage 2: Pose optimization...")
opt_stage2 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.3)

for it in trange(300, desc="Stage 2"):
    opt_stage2.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    smpl_root = smpl_joints_full[0]
    
    total_loss = 0
    for target, weight, joint_indices in zip(all_targets, all_weights, all_target_joint_indices):
        smpl_joints_selected = smpl_joints_full[joint_indices] - smpl_root
        
        valid_mask = weight > 0.3
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints_selected[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints_selected, dim=1).mean()
        
        scale = target_scale / (smpl_scale + 1e-8)
        diff = (smpl_joints_selected * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # Surface loss
    loss_surface = 0
    verts = out.vertices[0] - smpl_root
    for pc in all_surface_clouds:
        if pc is not None:
            dists = torch.cdist(pc, verts)
            loss_surface += dists.min(dim=1)[0].mean() * 1.5
    
    loss_spine = 1e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 2e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 5e-7 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_surface + loss_spine + loss_pose + loss_beta
    loss.backward()
    opt_stage2.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.3, 0.3)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-12.0, 12.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, surface={loss_surface:.6f}")

print("\n[INFO] Stage 3: Fine-tuning...")
opt_stage3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.15)

best_loss = 1e9
best_params = None

for it in trange(200, desc="Stage 3"):
    opt_stage3.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_joints_full = out.joints[0]
    smpl_root = smpl_joints_full[0]
    
    total_loss = 0
    for target, weight, joint_indices in zip(all_targets, all_weights, all_target_joint_indices):
        smpl_joints_selected = smpl_joints_full[joint_indices] - smpl_root
        
        valid_mask = weight > 0.3
        if valid_mask.sum() > 3:
            target_scale = torch.norm(target[valid_mask], dim=1).mean()
            smpl_scale = torch.norm(smpl_joints_selected[valid_mask], dim=1).mean()
        else:
            target_scale = torch.norm(target, dim=1).mean()
            smpl_scale = torch.norm(smpl_joints_selected, dim=1).mean()
        
        scale = target_scale / (smpl_scale + 1e-8)
        diff = (smpl_joints_selected * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # Surface fitting - keep strong weight
    loss_surface = 0
    verts = out.vertices[0] - smpl_root
    for pc in all_surface_clouds:
        if pc is not None:
            dists = torch.cdist(pc, verts)
            loss_surface += dists.min(dim=1)[0].mean() * 1.2
    
    # Spine alignment
    spine_joints = [0, 3, 6, 9, 12, 15]
    spine_joints_coords = out.joints[0, spine_joints, :]
    spine_dirs = spine_joints_coords[1:] - spine_joints_coords[:-1]
    spine_dirs_norm = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    loss_spine_align = 5e-3 * torch.mean((1 - torch.sum(spine_dirs_norm[:-1] * spine_dirs_norm[1:], dim=1)) ** 2)
    
    loss_spine_pose = 8e-3 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 3e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 1e-7 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_surface + loss_spine_align + loss_spine_pose + loss_pose + loss_beta
    loss.backward()
    opt_stage3.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.4, 0.4)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-12.0, 12.0)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = {
            'betas': betas.detach().clone(),
            'body_pose': body_pose.detach().clone(),
            'global_orient': global_orient.detach().clone(),
            'transl': transl.detach().clone(),
        }
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, surface={loss_surface:.6f}")

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
ref_names_check = all_keypoint_names[0]

# Find head and hip positions
nose_idx = ref_names_check.index('nose') if 'nose' in ref_names_check else 0
hip_idx = ref_names_check.index('hip_center') if 'hip_center' in ref_names_check else 7

head_y = ref_kps_check[nose_idx, 1]
hip_y = ref_kps_check[hip_idx, 1]
needs_flip = head_y > hip_y

verts_centered = verts - verts.mean(axis=0)

if needs_flip:
    print("[INFO] Flipping model orientation")
    rotation_fix = R.from_euler('x', 180, degrees=True).as_matrix()
    verts_centered = (rotation_fix @ verts_centered.T).T
else:
    print("[INFO] Model orientation correct")
    rotation_fix = None

mesh_export = trimesh.Trimesh(verts_centered, faces)
mesh_export.export("fitted_smplx_mesh.obj")
print("[INFO] Exported fitted_smplx_mesh.obj")

mesh_export.visual.vertex_colors = [200, 200, 230, 255]
mesh_export.export("fitted_smplx_mesh_colored.ply")
print("[INFO] Exported fitted_smplx_mesh_colored.ply")

# ========== CREATE FINAL COMPARISON VISUALIZATION ==========
if VISUALIZE_KEYPOINTS:
    print("\n[INFO] Creating final comparison...")
    fig = plt.figure(figsize=(20, 10))
    
    # Show original images
    for idx, (view_name, img_rgb) in enumerate(zip(all_view_names, all_images_rgb)):
        ax = fig.add_subplot(3, len(all_view_names), idx + 1)
        ax.imshow(img_rgb)
        ax.set_title(f'{view_name.upper()} - Original', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Show point clouds
    for idx, (view_name, pc) in enumerate(zip(all_view_names, all_point_clouds)):
        ax = fig.add_subplot(3, len(all_view_names), len(all_view_names) + idx + 1)
        if pc is not None:
            ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], cmap='viridis', s=1, alpha=0.6)
            ax.set_title(f'{view_name.upper()} - Point Cloud', fontsize=10)
            ax.set_aspect('equal')
            ax.invert_yaxis()
        ax.axis('off')
    
    # Show statistics
    ax_stats = fig.add_subplot(3, len(all_view_names), 2 * len(all_view_names) + 1)
    ax_stats.axis('off')
    ax_stats.set_title('Statistics', fontsize=14, fontweight='bold')
    
    stats_text = f"Total Keypoints: {sum(len(kps) for kps in all_keypoints)}\n"
    stats_text += f"Enhanced Points: {sum(len(kps) - 13 for kps in all_keypoints)}\n"
    stats_text += f"Surface Points: {sum(len(pc) if pc is not None else 0 for pc in all_point_clouds)}\n"
    stats_text += f"\nBeta Parameters:\n"
    for i in range(7):
        stats_text += f"  β{i}: {best_params['betas'][0, i].item():.3f}\n"
    
    ax_stats.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
                 transform=ax_stats.transAxes, family='monospace')
    
    # Show keypoint counts per view
    ax_kp = fig.add_subplot(3, len(all_view_names), 2 * len(all_view_names) + 2)
    ax_kp.axis('off')
    ax_kp.set_title('Keypoint Breakdown', fontsize=14, fontweight='bold')
    
    for idx, (view_name, kp_names) in enumerate(zip(all_view_names, all_keypoint_names)):
        kp_text = f"{view_name.upper()}:\n"
        torso_kps = [n for n in kp_names if any(x in n for x in ['chest', 'waist', 'abdomen', 'neck', 'hip_center'])]
        kp_text += f"  Standard: {len(kp_names) - len(torso_kps)}\n"
        kp_text += f"  Torso/Spine: {len(torso_kps)}\n"
        kp_text += f"  Total: {len(kp_names)}\n\n"
        ax_kp.text(0.1, 0.9 - idx * 0.3, kp_text, fontsize=10, verticalalignment='top',
                  transform=ax_kp.transAxes, family='monospace', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
    print("[INFO] Saved final_comparison.png")
    plt.close()

# ========== 3D VISUALIZATION ==========
print("\n[INFO] Creating 3D visualization...")
mesh = trimesh.Trimesh(verts_centered, faces)
mesh.visual.vertex_colors = [200, 200, 230, 255]

scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
scene.add(mesh_node)

# Add skeleton
joint_positions = final_out.joints[0].cpu().detach().numpy()
joint_positions_centered = joint_positions - joint_positions.mean(axis=0)

if needs_flip and rotation_fix is not None:
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

# Add detected keypoints
if len(all_keypoints) > 0:
    kps_vis = all_keypoints[0].copy()
    kp_names_vis = all_keypoint_names[0]
    
    # Find root
    hip_idx = kp_names_vis.index('hip_center') if 'hip_center' in kp_names_vis else 0
    root_vis = kps_vis[hip_idx]
    kps_vis = kps_vis - root_vis
    
    if needs_flip and rotation_fix is not None:
        kps_vis = (rotation_fix @ kps_vis.T).T
    
    for idx, (kp, name) in enumerate(zip(kps_vis, kp_names_vis)):
        # Color-code keypoints
        if any(x in name for x in ['chest', 'waist', 'abdomen']):
            color = [255, 100, 100, 255]  # Red for torso
            radius = 0.03
        else:
            color = [255, 60, 60, 255]  # Pink for standard
            radius = 0.025
        
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(kp)
        sphere.visual.vertex_colors = color
        scene.add(pyrender.Mesh.from_trimesh(sphere))

# Add point cloud for comparison
if all_point_clouds[0] is not None:
    pc_vis = all_point_clouds[0].copy()
    if needs_flip and rotation_fix is not None:
        pc_vis = (rotation_fix @ pc_vis.T).T
    
    # Sample subset for visualization
    if len(pc_vis) > 200:
        indices = np.random.choice(len(pc_vis), 200, replace=False)
        pc_vis = pc_vis[indices]
    
    for point in pc_vis:
        tiny_sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
        tiny_sphere.apply_translation(point)
        tiny_sphere.visual.vertex_colors = [100, 200, 255, 200]
        scene.add(pyrender.Mesh.from_trimesh(tiny_sphere))

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
print(f"[INFO] Enhanced keypoints per view: {[len(kps) for kps in all_keypoints]}")
print(f"[INFO] Torso/spine keypoints added: {[len([n for n in names if any(x in n for x in ['chest', 'waist', 'abdomen'])]) for names in all_keypoint_names]}")
print(f"[INFO] Surface points used: {[len(pc) if pc is not None else 0 for pc in all_point_clouds]}")
print(f"[INFO] Final betas: {best_params['betas'][0, :7].cpu().numpy()}")
print(f"\n[INFO] Output files:")
print(f"  - keypoint_extraction_*.png (shows all {sum(len(kps) for kps in all_keypoints)} keypoints)")
print(f"  - final_comparison.png (statistics and point clouds)")
print(f"  - render_output.png (3D model with keypoints and surface points)")
print(f"  - fitted_smplx_mesh.obj (final 3D model)")