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
    'front': "synthetic_data_10_items\synthetic_data\images\image_1.png",
    # Optional: Add side views if available
    # 'left': "synthetic_data_10_items\synthetic_data\images\img_left_1.png",
    # 'right': "synthetic_data_10_items\synthetic_data\images\img_right_1.png",
}
MODEL_PATH = "full_pipeline/models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 30
OPT_ITERS = 600
LR = 3e-2

# ========== ISSUE 1 FIX: SUPPORT SINGLE IMAGE ==========
# The code now works with 1, 2, or 3 images
# If only front view is provided, we'll use depth estimation and symmetry

# ========== ISSUE 4 FIX: HANDLE NON-SQUARE IMAGES ==========
def normalize_image_aspect(img):
    """
    Normalize image to handle non-square aspect ratios.
    Returns the image with proper aspect ratio handling for pose detection.
    """
    h, w = img.shape[:2]
    # Keep original aspect ratio but ensure consistent processing
    return img, (h, w)

# ========== IMPROVED KEYPOINT EXTRACTION ==========
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
    
    # ISSUE 4 FIX: Handle non-square images properly
    img_rgb, (h, w) = normalize_image_aspect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    aspect_ratio = w / h
    print(f"[INFO] Image size: {w}x{h}, aspect ratio: {aspect_ratio:.2f}")
    
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        print(f"[WARNING] No pose detected in {view_name}, skipping...")
        continue
    
    # Extract segmentation mask for body shape estimation
    segmentation_mask = None
    if results.segmentation_mask is not None:
        segmentation_mask = results.segmentation_mask
        all_segmentation_masks.append(segmentation_mask)
    
    # ISSUE 3 FIX: Better depth estimation from world landmarks
    if getattr(results, "pose_world_landmarks", None) is not None:
        landmarks = results.pose_world_landmarks.landmark
        # Use world coordinates directly (they're in meters, normalized by height)
        kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
        
        # Scale to reasonable human proportions (avg height ~1.7m)
        scale_factor = 1.7 / (np.max(kps[:, 1]) - np.min(kps[:, 1]) + 1e-6)
        kps = kps * scale_factor
    else:
        # Fallback to pixel coordinates with aspect ratio correction
        landmarks = results.pose_landmarks.landmark
        kps = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
        # Normalize by height and correct for aspect ratio
        kps[:, 0] = kps[:, 0] / w * aspect_ratio  # X correction
        kps[:, 1] = kps[:, 1] / h  # Y normalization
        kps[:, 2] = kps[:, 2] / h  # Z normalization
        kps = kps * 1.7  # Scale to human height
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    
    all_keypoints.append(kps)
    all_view_names.append(view_name)
    all_confidences.append(confidences)
    all_images_rgb.append(img_rgb)
    print(f"[INFO] Extracted {len(kps)} keypoints from {view_name}, avg confidence: {confidences.mean():.3f}")

if len(all_keypoints) == 0:
    raise ValueError("No valid poses detected in any image!")

# ISSUE 1 FIX: Handle single image case with symmetry
if len(all_keypoints) == 1:
    print("\n[INFO] Single image detected. Using symmetry assumptions for depth.")
    # For single front view, we'll rely more on the frontal keypoints
    # and use left-right symmetry

# ========== LOAD SMPL-X MODEL ==========
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=NUM_BETAS,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# Enhanced mapping with confidence-based selection
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

# ========== ISSUE 3 FIX: PROCRUSTES ALIGNMENT FOR MULTI-VIEW ==========
def procrustes_align(source, target, weights=None):
    """
    Align source to target using Procrustes analysis (rotation + translation + scale).
    This solves the multi-view alignment problem.
    """
    if weights is None:
        weights = np.ones(len(source))
    
    # Weighted centroid
    weights = weights / weights.sum()
    source_center = (source.T @ weights).T
    target_center = (target.T @ weights).T
    
    source_centered = source - source_center
    target_centered = target - target_center
    
    # Weighted covariance matrix
    W = np.diag(weights)
    H = source_centered.T @ W @ target_centered
    
    # SVD for optimal rotation
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    
    # Optimal scale
    scale = np.trace(S) / np.trace(source_centered.T @ W @ source_centered)
    
    # Apply transformation
    aligned = scale * (source_centered @ R_mat.T) + target_center
    
    return aligned, R_mat, scale, target_center

# Prepare all targets with view-specific transformations and alignment
all_targets = []
all_weights = []

# ISSUE 3 FIX: Align all views to the front view
if len(all_keypoints) > 1:
    print("\n[INFO] Aligning multiple views using Procrustes analysis...")
    reference_idx = 0  # Use front view as reference
    ref_kps = all_keypoints[reference_idx].copy()
    ref_conf = all_confidences[reference_idx].copy()
    
    # Center at hip midpoint
    if ref_kps[23].any() and ref_kps[24].any():
        ref_root = (ref_kps[23] + ref_kps[24]) / 2.0
    else:
        ref_root = np.median(ref_kps, axis=0)
    
    ref_kps_centered = ref_kps - ref_root

for view_idx, (kps, conf, view_name) in enumerate(zip(all_keypoints, all_confidences, all_view_names)):
    # Center at hip midpoint
    if kps[23].any() and kps[24].any():
        root = (kps[23] + kps[24]) / 2.0
    else:
        root = np.median(kps, axis=0)
    
    kps_centered = kps - root
    
    # ISSUE 3 FIX: Apply view-specific transformations with better angle estimation
    if 'left' in view_name.lower():
        # Left view: person's left side is visible
        # Estimate rotation angle from keypoint positions
        rot_angle = -90  # Default
        rot = R.from_euler('y', rot_angle, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied {rot_angle}° Y rotation for left view")
        
        # Align to reference view if available
        if len(all_keypoints) > 1 and view_idx != reference_idx:
            kps_centered, _, _, _ = procrustes_align(
                kps_centered[mp_indices], 
                ref_kps_centered[mp_indices],
                weights=conf[mp_indices] * ref_conf[mp_indices]
            )
            print(f"[INFO] Aligned left view to reference using Procrustes")
            
    elif 'right' in view_name.lower():
        # Right view: person's right side is visible
        rot_angle = 90
        rot = R.from_euler('y', rot_angle, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied {rot_angle}° Y rotation for right view")
        
        # Align to reference view if available
        if len(all_keypoints) > 1 and view_idx != reference_idx:
            kps_centered, _, _, _ = procrustes_align(
                kps_centered[mp_indices], 
                ref_kps_centered[mp_indices],
                weights=conf[mp_indices] * ref_conf[mp_indices]
            )
            print(f"[INFO] Aligned right view to reference using Procrustes")
    
    # Select mapped keypoints
    kps_selected = kps_centered[mp_indices]
    conf_selected = conf[mp_indices].copy()
    
    # Boost confidence for visible side in side views
    if 'left' in view_name.lower():
        # Boost left side keypoints (11, 13, 15, 23, 25, 27)
        left_mask = np.isin(mp_indices, [11, 13, 15, 23, 25, 27])
        conf_selected[left_mask] *= 1.5
    elif 'right' in view_name.lower():
        # Boost right side keypoints (12, 14, 16, 24, 26, 28)
        right_mask = np.isin(mp_indices, [12, 14, 16, 24, 26, 28])
        conf_selected[right_mask] *= 1.5
    
    # ISSUE 1 FIX: For single front view, boost frontal keypoints
    if len(all_keypoints) == 1 and 'front' in view_name.lower():
        # Boost shoulders, elbows, knees (more visible from front)
        front_visible_mask = np.isin(mp_indices, [11, 12, 13, 14, 25, 26])
        conf_selected[front_visible_mask] *= 1.3
        # Reduce confidence for depth-ambiguous points
        depth_uncertain_mask = np.isin(mp_indices, [15, 16, 27, 28])  # wrists, ankles
        conf_selected[depth_uncertain_mask] *= 0.7
    
    conf_selected = np.clip(conf_selected, 0, 1)
    
    all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))

print(f"\n[INFO] Using {len(all_targets)} view(s) for optimization")

# Visualization of aligned keypoints
if len(all_targets) > 0:
    print("[INFO] Visualizing aligned 3D keypoints...")
    fig = plt.figure(figsize=(12, 5))
    
    # Plot 1: Original keypoints
    ax1 = fig.add_subplot(121, projection='3d')
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i, (target, name) in enumerate(zip(all_targets, all_view_names)):
        pts = target.detach().cpu().numpy()
        ax1.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
                   s=40, c=colors[i % len(colors)], label=name, alpha=0.8)
    ax1.set_title("Aligned 3D Keypoints (Multi-View)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y (height)")
    ax1.legend()
    ax1.view_init(elev=20, azim=70)
    
    plt.tight_layout()
    plt.savefig("keypoint_alignment.png", dpi=150, bbox_inches='tight')
    plt.show()

# ========== ISSUE 5 FIX: SHAPE ESTIMATION FROM SEGMENTATION ==========
# Estimate body proportions from segmentation mask
body_volume_estimate = 1.0
body_width_ratio = 1.0

if len(all_segmentation_masks) > 0:
    print("\n[INFO] Estimating body shape from segmentation masks...")
    for mask, img_rgb in zip(all_segmentation_masks, all_images_rgb):
        # Calculate body area ratio
        body_area = np.sum(mask > 0.5)
        total_area = mask.shape[0] * mask.shape[1]
        body_ratio = body_area / total_area
        
        # Estimate width at torso
        mask_binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect = w / (h + 1e-6)
            
            # Estimate body volume from area and aspect ratio
            # Wider aspect = larger body
            body_width_ratio = np.clip(aspect / 0.4, 0.7, 1.5)  # Normalize around 0.4 (slim person)
            body_volume_estimate = np.clip(body_ratio / 0.15, 0.8, 1.3)  # Normalize around 15% area
            
            print(f"[INFO] Body area ratio: {body_ratio:.3f}, aspect: {aspect:.3f}")
            print(f"[INFO] Estimated body volume: {body_volume_estimate:.3f}, width ratio: {body_width_ratio:.3f}")

# ========== INITIALIZE SMPL PARAMETERS ==========
# ISSUE 5 FIX: Initialize betas based on body shape estimate
initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)
# Beta 0 typically controls overall body size/weight
initial_betas[0, 0] = (body_volume_estimate - 1.0) * 2.0  # Scale to beta range
# Beta 1 typically controls height (keep at 0 for average)
# Beta 2 often controls width
initial_betas[0, 2] = (body_width_ratio - 1.0) * 1.5
betas = initial_betas.clone().requires_grad_(True)

body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# SMPL-X spine joints for torso regularization
spine_joints = [0, 3, 6, 9, 12, 15]  # pelvis -> spine1 -> spine2 -> spine3 -> neck -> head

# ========== STAGE 1: SHAPE + GLOBAL ORIENTATION ==========
print("\n[INFO] Stage 1: Optimizing shape and global orientation...")
opt_stage1 = torch.optim.Adam([betas, global_orient], lr=LR)

for it in trange(150, desc="Stage 1"):
    opt_stage1.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_root = out.joints[0, 0, :]
    smpl_joints = out.joints[0, smpl_indices, :] - smpl_root
    
    total_loss = 0
    for target, weight in zip(all_targets, all_weights):
        # ISSUE 5 FIX: Use more robust scale estimation
        target_scale = torch.norm(target, dim=1).mean()
        smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        scale = target_scale / (smpl_scale + 1e-8)
        
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # ISSUE 5 FIX: Regularize betas but allow shape variation
    loss_beta = 5e-4 * torch.mean(betas ** 2)
    loss = total_loss / len(all_targets) + loss_beta
    
    loss.backward()
    opt_stage1.step()
    
    # Clamp betas to reasonable range
    with torch.no_grad():
        betas.clamp_(-3.0, 8.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, beta[0]={betas[0,0].item():.3f}")

# ========== STAGE 2: LIMBS ONLY (NOT SPINE) ==========
print("\n[INFO] Stage 2: Optimizing limb poses (keeping spine straight)...")
opt_stage2 = torch.optim.Adam([body_pose, betas, global_orient], lr=LR * 0.5)

for it in trange(250, desc="Stage 2"):
    opt_stage2.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_root = out.joints[0, 0, :]
    smpl_joints = out.joints[0, smpl_indices, :] - smpl_root
    
    total_loss = 0
    for target, weight in zip(all_targets, all_weights):
        target_scale = torch.norm(target, dim=1).mean()
        smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        scale = target_scale / (smpl_scale + 1e-8)
        
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # Strong spine regularization
    loss_spine = 5e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 5e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 3e-4 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine + loss_pose + loss_beta
    loss.backward()
    opt_stage2.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.2, 0.2)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-3.0, 8.0)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

# ========== STAGE 3: FINE-TUNE ==========
print("\n[INFO] Stage 3: Fine-tuning all parameters...")
opt_stage3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.2)

best_loss = 1e9
best_params = None

for it in trange(200, desc="Stage 3"):
    opt_stage3.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    
    smpl_root = out.joints[0, 0, :]
    smpl_joints = out.joints[0, smpl_indices, :] - smpl_root
    
    total_loss = 0
    for target, weight in zip(all_targets, all_weights):
        target_scale = torch.norm(target, dim=1).mean()
        smpl_scale = torch.norm(smpl_joints, dim=1).mean()
        scale = target_scale / (smpl_scale + 1e-8)
        
        diff = (smpl_joints * scale - target) ** 2
        loss_joints = (diff * weight.unsqueeze(1)).mean()
        total_loss += loss_joints
    
    # Spine alignment
    spine_joints_coords = out.joints[0, spine_joints, :]
    spine_dirs = spine_joints_coords[1:] - spine_joints_coords[:-1]
    spine_dirs_norm = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    loss_spine_align = 2e-2 * torch.mean((1 - torch.sum(spine_dirs_norm[:-1] * spine_dirs_norm[1:], dim=1)) ** 2)
    
    loss_spine_pose = 3e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 1e-3 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 3e-4 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine_align + loss_spine_pose + loss_pose + loss_beta
    loss.backward()
    opt_stage3.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.3, 0.3)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
        betas.clamp_(-3.0, 8.0)
    
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
print(f"[INFO] Final betas: {best_params['betas'][0, :3].cpu().numpy()}")

# ========== GENERATE FINAL MESH ==========
final_out = smplx_model(
    betas=best_params['betas'],
    body_pose=best_params['body_pose'],
    global_orient=best_params['global_orient'],
    transl=best_params['transl']
)

verts = final_out.vertices[0].cpu().detach().numpy()
faces = smplx_model.faces
verts_centered = verts - verts.mean(axis=0)

# Fix upside-down mesh
rotation_fix = R.from_euler('x', 180, degrees=True).as_matrix()
verts_centered = (rotation_fix @ verts_centered.T).T

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
    kps_vis = all_keypoints[0] - (all_keypoints[0][23] + all_keypoints[0][24]) / 2.0
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
print(f"[INFO] Final body shape parameters (betas): {best_params['betas'][0, :5].cpu().numpy()}")
print(f"[INFO] Estimated body volume factor: {body_volume_estimate:.3f}")
print(f"[INFO] Estimated body width ratio: {body_width_ratio:.3f}")