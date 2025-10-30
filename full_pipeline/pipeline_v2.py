import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== CONFIG ==========
MODEL_PATH = 'full_pipeline/models/smplx/SMPLX_NEUTRAL.npz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== MEDIAPIPE SETUP ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

def get_keypoints(image_path):
    """Extract 3D keypoints from a single image using MediaPipe."""
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        print(f"[Warning] No pose detected in {image_path}")
        return None

    keypoints = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in results.pose_landmarks.landmark])
    return keypoints

# ========== LOAD IMAGES ==========
front_img = 'dataset/image7.jpg'
left_img = 'dataset/image7_left.jpg'
right_img = 'dataset/image7_right.jpg'
back_img = 'dataset/image7_back.jpg'

keypoints_list = []
for img in [front_img, left_img, right_img, back_img]:
    kps = get_keypoints(img)
    if kps is not None:
        keypoints_list.append(kps)

# Average across views (simple multi-view fusion)
if len(keypoints_list) == 0:
    raise ValueError("No keypoints detected in any view.")
avg_keypoints = np.mean(np.stack(keypoints_list), axis=0)

# ========== LOAD SMPL-X MODEL ==========
smplx_model = smplx.create(
    model_path='full_pipeline/models/',
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=10,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# ========== INITIALIZE PARAMETERS ==========
betas = torch.zeros([1, 10], dtype=torch.float32, device=DEVICE)
body_pose = torch.zeros([1, smplx_model.NUM_BODY_JOINTS * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)

# ========== MAPPING (MediaPipe → SMPL-X) ==========
mp_to_smplx = {
    11: 1,  # left shoulder
    12: 2,  # right shoulder
    23: 4,  # left hip
    24: 5,  # right hip
    13: 16, # left elbow
    14: 17, # right elbow
    15: 18, # left wrist
    16: 19, # right wrist
    25: 7,  # left knee
    26: 8,  # right knee
    27: 9,  # left ankle
    28: 10, # right ankle
}

mp_indices = np.array(list(mp_to_smplx.keys()))
smplx_indices = np.array(list(mp_to_smplx.values()))
keypoints_used = torch.tensor(avg_keypoints[mp_indices, :], dtype=torch.float32, device=DEVICE)

# ========== OPTIMIZATION ==========
optimizer = torch.optim.Adam([body_pose, global_orient], lr=0.01)
n_iters = 300

print("[INFO] Fitting SMPL-X model to detected keypoints...")
for i in tqdm(range(n_iters)):
    output = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient)
    joints = output.joints[0, smplx_indices, :]  # select relevant SMPL-X joints
    loss = torch.mean((joints - keypoints_used) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Optimization complete. Final loss: {loss.item():.4f}")

# ========== VISUALIZE ==========
verts = output.vertices[0].cpu().detach().numpy()
x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=0.1, c='lightblue')
ax.set_title('3D SMPL-X Mesh fitted to pose')
plt.show()
