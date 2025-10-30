import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== CONFIG ==========
MODEL_PATH = 'full_pipeline/models/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== MEDIAPIPE SETUP ==========
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_keypoints_2d(image_path):
    """Extract 2D keypoints from a single image using MediaPipe."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Could not read image: {image_path}")
        return None, None
    
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, 
                      model_complexity=2, 
                      enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        results = pose.process(rgb)
    
    if not results.pose_landmarks:
        print(f"[Warning] No pose detected in {image_path}")
        return None, None
    
    # Extract 2D keypoints (normalized then converted to pixels)
    keypoints_2d = np.array([[lm.x * w, lm.y * h, lm.visibility] 
                             for lm in results.pose_landmarks.landmark])
    
    return keypoints_2d, (h, w)

def normalize_keypoints(keypoints_2d, img_size):
    """
    Normalize keypoints to a scale suitable for SMPL-X.
    Centers around the hip midpoint and scales to roughly match SMPL-X scale.
    """
    h, w = img_size
    
    # MediaPipe keypoint indices for hips
    left_hip_idx = 23
    right_hip_idx = 24
    
    # Calculate hip center
    hip_center = (keypoints_2d[left_hip_idx, :2] + keypoints_2d[right_hip_idx, :2]) / 2
    
    # Center keypoints around hip
    centered = keypoints_2d[:, :2] - hip_center
    
    # Scale to roughly match SMPL-X (typical human ~1.7m tall, image height is in pixels)
    # Assuming average person occupies ~80% of image height
    scale_factor = 1.7 / (h * 0.8)
    normalized = centered * scale_factor
    
    # Create 3D coordinates (z=0 for frontal image, will be optimized)
    keypoints_3d = np.zeros((keypoints_2d.shape[0], 3))
    keypoints_3d[:, :2] = normalized
    keypoints_3d[:, 2] = 0  # Assume frontal plane initially
    
    return keypoints_3d, keypoints_2d[:, 2]  # Return normalized coords and visibility scores

# ========== LOAD SMPL-X MODEL ==========
print("[INFO] Loading SMPL-X model...")
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='neutral',
    num_betas=10,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# ========== MAPPING (MediaPipe → SMPL-X) ==========
# More comprehensive mapping for better results
mp_to_smplx = {
    # Torso
    11: 16,  # left shoulder
    12: 17,  # right shoulder
    23: 1,   # left hip
    24: 2,   # right hip
    
    # Arms
    13: 18,  # left elbow
    14: 19,  # right elbow
    15: 20,  # left wrist
    16: 21,  # right wrist
    
    # Legs
    25: 4,   # left knee
    26: 5,   # right knee
    27: 7,   # left ankle
    28: 8,   # right ankle
}

# ========== LOAD AND PROCESS IMAGE ==========
front_img = 'dataset/image7.jpg'

print(f"[INFO] Processing image: {front_img}")
keypoints_2d, img_size = get_keypoints_2d(front_img)

if keypoints_2d is None:
    raise ValueError("No keypoints detected in the image.")

# Normalize keypoints
keypoints_3d, visibility = normalize_keypoints(keypoints_2d, img_size)

# Extract only the mapped keypoints
mp_indices = np.array(list(mp_to_smplx.keys()))
smplx_indices = np.array(list(mp_to_smplx.values()))

target_keypoints = keypoints_3d[mp_indices]
visibility_weights = visibility[mp_indices]

# Convert to torch tensors
target_keypoints_torch = torch.tensor(target_keypoints, dtype=torch.float32, device=DEVICE)
visibility_weights_torch = torch.tensor(visibility_weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

# ========== INITIALIZE SMPL-X PARAMETERS ==========
print("[INFO] Initializing SMPL-X parameters...")

# Shape parameters (body proportions)
betas = torch.zeros([1, 10], dtype=torch.float32, device=DEVICE, requires_grad=False)

# Body pose (joint rotations) - 21 body joints × 3 (axis-angle)
body_pose = torch.zeros([1, 63], dtype=torch.float32, device=DEVICE, requires_grad=True)

# Global orientation
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# Translation (to align with image)
transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# ========== OPTIMIZATION ==========
# Stage 1: Optimize global orientation and translation
print("\n[INFO] Stage 1: Optimizing global orientation and translation...")
optimizer_stage1 = torch.optim.Adam([global_orient, transl], lr=0.02)

for i in tqdm(range(100)):
    output = smplx_model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        return_verts=True
    )
    
    pred_joints = output.joints[0, smplx_indices, :]
    
    # Weighted loss based on visibility
    loss = torch.mean(visibility_weights_torch * (pred_joints - target_keypoints_torch) ** 2)
    
    optimizer_stage1.zero_grad()
    loss.backward()
    optimizer_stage1.step()

print(f"Stage 1 complete. Loss: {loss.item():.6f}")

# Stage 2: Optimize body pose
print("\n[INFO] Stage 2: Optimizing body pose...")
optimizer_stage2 = torch.optim.Adam([body_pose, global_orient, transl], lr=0.01)

for i in tqdm(range(300)):
    output = smplx_model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        return_verts=True
    )
    
    pred_joints = output.joints[0, smplx_indices, :]
    
    # Weighted MSE loss
    joint_loss = torch.mean(visibility_weights_torch * (pred_joints - target_keypoints_torch) ** 2)
    
    # Regularization to keep pose natural
    pose_reg = torch.mean(body_pose ** 2) * 0.001
    
    loss = joint_loss + pose_reg
    
    optimizer_stage2.zero_grad()
    loss.backward()
    optimizer_stage2.step()

print(f"Stage 2 complete. Final loss: {loss.item():.6f}")

# ========== GET FINAL OUTPUT ==========
with torch.no_grad():
    final_output = smplx_model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        return_verts=True
    )

# ========== VISUALIZE ==========
print("\n[INFO] Generating visualization...")

verts = final_output.vertices[0].cpu().numpy()
joints = final_output.joints[0].cpu().numpy()
faces = smplx_model.faces

# Create figure with subplots
fig = plt.figure(figsize=(15, 5))

# 3D mesh view 1 (front)
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                 triangles=faces, alpha=0.3, color='lightblue', edgecolor='none')
ax1.scatter(joints[smplx_indices, 0], joints[smplx_indices, 1], joints[smplx_indices, 2], 
            c='red', s=20, label='SMPL-X joints')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Front View')
ax1.view_init(elev=0, azim=-90)

# 3D mesh view 2 (side)
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                 triangles=faces, alpha=0.3, color='lightblue', edgecolor='none')
ax2.scatter(joints[smplx_indices, 0], joints[smplx_indices, 1], joints[smplx_indices, 2], 
            c='red', s=20)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Side View')
ax2.view_init(elev=0, azim=0)

# 3D mesh view 3 (isometric)
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                 triangles=faces, alpha=0.3, color='lightblue', edgecolor='none')
ax3.scatter(joints[smplx_indices, 0], joints[smplx_indices, 1], joints[smplx_indices, 2], 
            c='red', s=20)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Isometric View')
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('smplx_fitted_mesh.png', dpi=150, bbox_inches='tight')
print("[INFO] Saved visualization to 'smplx_fitted_mesh.png'")
plt.show()

print("\n[INFO] Pipeline complete!")
print(f"Final parameters:")
print(f"  - Body pose shape: {body_pose.shape}")
print(f"  - Global orientation: {global_orient.detach().cpu().numpy()}")
print(f"  - Translation: {transl.detach().cpu().numpy()}")