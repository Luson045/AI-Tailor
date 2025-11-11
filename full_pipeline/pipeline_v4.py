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
# Use front + left + right views for best results
IMAGES = {
    'front': "dataset/image13.jpg",
    'left': "dataset/image13_right.jpg",   # Left side view (person's left visible)
    'right': "dataset/image13_left.jpg", # Right side view (person's right visible)
}
MODEL_PATH = "full_pipeline/models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 10
OPT_ITERS = 600
LR = 3e-2

# ========== EXTRACT KEYPOINTS FROM ALL VIEWS ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

all_keypoints = []
all_view_names = []
all_confidences = []
all_images_rgb = []

for view_name, img_path in IMAGES.items():
    print(f"\n[INFO] Processing {view_name} view: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not load {img_path}, skipping...")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    results = pose.process(img_rgb)
    
    if not results.pose_landmarks:
        print(f"[WARNING] No pose detected in {view_name}, skipping...")
        continue
    
    # Extract world landmarks
    if getattr(results, "pose_world_landmarks", None) is not None:
        landmarks = results.pose_world_landmarks.landmark
        kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    else:
        landmarks = results.pose_landmarks.landmark
        kps = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
        kps = kps / h * 1.7
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
    
    all_keypoints.append(kps)
    all_view_names.append(view_name)
    all_confidences.append(confidences)
    all_images_rgb.append(img_rgb)
    print(f"[INFO] Extracted {len(kps)} keypoints from {view_name}, avg confidence: {confidences.mean():.3f}")

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

# Prepare all targets with view-specific transformations
all_targets = []
all_weights = []

for view_idx, (kps, conf, view_name) in enumerate(zip(all_keypoints, all_confidences, all_view_names)):
    # Center at hip midpoint
    if kps[23].any() and kps[24].any():
        root = (kps[23] + kps[24]) / 2.0
    else:
        root = np.median(kps, axis=0)
    
    kps_centered = kps - root
    
    # Apply view-specific transformations
    if 'left' in view_name.lower():
        # Left view: rotate +90° around Y (person's left side visible)
        rot = R.from_euler('y', -90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied -90° Y rotation for left view")
    elif 'right' in view_name.lower():
        # Right view: rotate -90° around Y (person's right side visible)
        rot = R.from_euler('y', 90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied +90° Y rotation for right view")
    
    # Select mapped keypoints
    kps_selected = kps_centered[mp_indices]
    conf_selected = conf[mp_indices]
    
    # Boost confidence for visible side in side views
    if 'left' in view_name.lower():
        # Boost left side keypoints (11, 13, 15, 23, 25, 27)
        left_indices = [0, 2, 4, 6, 8, 10]  # indices in mp_indices for left joints
        conf_selected[left_indices] *= 1.5
    elif 'right' in view_name.lower():
        # Boost right side keypoints (12, 14, 16, 24, 26, 28)
        right_indices = [1, 3, 5, 7, 9, 11]  # indices in mp_indices for right joints
        conf_selected[right_indices] *= 1.5
    
    conf_selected = np.clip(conf_selected, 0, 1)
    
    all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))

print(f"\n[INFO] Using {len(all_targets)} view(s) for optimization")
print("[INFO] Visualizing pre-fitting 3D keypoints...")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y', 'm', 'c']
for i, (target, name) in enumerate(zip(all_targets, all_view_names)):
    pts = target.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],  # flip Y for upright view
               s=40, c=colors[i % len(colors)], label=name, alpha=0.8)

ax.set_title("3D Keypoints Point Cloud (Before SMPL-X Fitting)")
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y (height)")
ax.legend()
ax.view_init(elev=20, azim=70)
plt.tight_layout()
plt.show()
# ========== INITIALIZE SMPL PARAMETERS ==========
betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE, requires_grad=True)
body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# SMPL-X spine joints for torso regularization
spine_joints = [0, 3, 6, 9, 12, 15]  # pelvis -> spine1 -> spine2 -> spine3 -> neck -> head

# ========== STAGE 1: SHAPE + GLOBAL ORIENTATION ==========
print("\n[INFO] Stage 1: Optimizing shape and global orientation (no body pose)...")
opt_stage1 = torch.optim.Adam([betas, global_orient], lr=LR)

for it in trange(150, desc="Stage 1"):
    opt_stage1.zero_grad()
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
    
    loss_beta = 1e-3 * torch.mean(betas ** 2)
    loss = total_loss / len(all_targets) + loss_beta
    
    loss.backward()
    opt_stage1.step()
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

# ========== STAGE 2: LIMBS ONLY (NOT SPINE) ==========
print("\n[INFO] Stage 2: Optimizing limb poses (keeping spine straight)...")

# Create mask for body_pose: only optimize limbs, not spine
# body_pose has 21 joints × 3 = 63 params
# Joints: 0-2 (spine), 3-5 (spine), 6-8 (spine), 9-11 (neck), 12-14 (L shoulder), etc.
# We'll only optimize joints for: shoulders, elbows, wrists, hips, knees, ankles
limb_joint_indices = list(range(12, 63))  # Start from shoulders onwards
spine_joint_indices = list(range(0, 12))   # Spine joints (keep these at 0)

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
    
    # Strong spine regularization - keep it straight
    loss_spine = 5e-2 * torch.sum(body_pose[0, :12] ** 2)
    
    # Pose prior for limbs only
    loss_pose = 5e-4 * torch.sum(body_pose[0, 12:] ** 2)
    
    loss_beta = 5e-4 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine + loss_pose + loss_beta
    loss.backward()
    opt_stage2.step()
    
    # Hard constraint: clamp spine rotations to very small values
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.2, 0.2)  # Max ±11° for spine
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f} spine_loss={loss_spine.item():.6f}")

# ========== STAGE 3: FINE-TUNE EVERYTHING (WITH SPINE CONSTRAINT) ==========
print("\n[INFO] Stage 3: Fine-tuning all parameters with spine constraint...")
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
    
    # Spine straightness constraint
    spine_joints_coords = out.joints[0, spine_joints, :]
    spine_dirs = spine_joints_coords[1:] - spine_joints_coords[:-1]
    spine_dirs_norm = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    # Encourage consecutive spine segments to be aligned (dot product close to 1)
    loss_spine_align = 2e-2 * torch.mean((1 - torch.sum(spine_dirs_norm[:-1] * spine_dirs_norm[1:], dim=1)) ** 2)
    
    loss_spine_pose = 3e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose = 1e-3 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta = 5e-4 * torch.mean(betas ** 2)
    
    loss = total_loss / len(all_targets) + loss_spine_align + loss_spine_pose + loss_pose + loss_beta
    loss.backward()
    opt_stage3.step()
    
    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.3, 0.3)  # Max ±17° for spine
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = {
            'betas': betas.detach().clone(),
            'body_pose': body_pose.detach().clone(),
            'global_orient': global_orient.detach().clone(),
            'transl': transl.detach().clone(),
        }
    
    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f} spine_align={loss_spine_align.item():.6f}")

print(f"\n[INFO] Optimization finished. Best loss: {best_loss:.6f}")

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

# Fix upside-down mesh: rotate 180° around X-axis
rotation_fix = R.from_euler('x', 180, degrees=True).as_matrix()
verts_centered = (rotation_fix @ verts_centered.T).T

mesh_export = trimesh.Trimesh(verts_centered, faces)
mesh_export.export("fitted_smplx_mesh.obj")
print("[INFO] Exported fitted_smplx_mesh.obj")

# Also save with texture for better visualization
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

# Add skeleton lines for visualization
joint_positions = final_out.joints[0].cpu().detach().numpy()
joint_positions_centered = joint_positions - joint_positions.mean(axis=0)
# Apply same rotation fix to joints
joint_positions_centered = (rotation_fix @ joint_positions_centered.T).T

# Define skeleton connections
skeleton_pairs = [
    (0, 1), (0, 2),  # pelvis to hips
    (1, 4), (4, 7), (7, 10),  # left leg
    (2, 5), (5, 8), (8, 11),  # right leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # spine to head
    (9, 13), (9, 14),  # neck to shoulders
    (13, 16), (16, 18), (18, 20),  # left arm
    (14, 17), (17, 19), (19, 21),  # right arm
]

for j1, j2 in skeleton_pairs:
    p1, p2 = joint_positions_centered[j1], joint_positions_centered[j2]
    cylinder = trimesh.creation.cylinder(radius=0.01, height=np.linalg.norm(p2 - p1))
    
    # Align cylinder
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

# Add keypoints from first view (also rotate)
kps_vis = all_keypoints[0] - (all_keypoints[0][23] + all_keypoints[0][24]) / 2.0
kps_vis = (rotation_fix @ kps_vis.T).T
for idx in mp_indices:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.025)
    sphere.apply_translation(kps_vis[idx])
    sphere.visual.vertex_colors = [255, 60, 60, 255]
    scene.add(pyrender.Mesh.from_trimesh(sphere))

# Lighting - use valid transformation matrices
light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light1, pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], dtype=np.float32))
scene.add(light2, pose=np.array([[1,0,0,0],[0,0.707,-0.707,1],[0,0.707,0.707,1],[0,0,0,1]], dtype=np.float32))

# Camera - adjusted for upright mesh
cam_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2.5],[0,0,0,1]], dtype=np.float32)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
scene.add(camera, pose=cam_pose)

print("[INFO] Launching viewer. Use mouse to rotate. Close window to finish.")
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
print(f"[INFO] Final spine rotation range: {best_params['body_pose'][0, :12].abs().max().item():.3f} rad")