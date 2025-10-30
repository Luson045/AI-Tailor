import cv2
import mediapipe as mp
import numpy as np
import torch
import matplotlib.pyplot as plt
from smplx import SMPLX  # <-- changed from SMPL to SMPLX

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Path to SMPL-X model (point to the folder where model.npz or .pkl exists)
SMPLX_MODEL_PATH = "full_pipeline/models/smplx"  # adjust to your extracted folder

# Function to get keypoints from an image using MediaPipe
def get_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"No keypoints detected for {image_path}")
        return None

    keypoints = []
    h, w, _ = image.shape
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x * w, lm.y * h, lm.z * w])
    return np.array(keypoints)

# Load keypoints for all views
views = {
    "front": "dataset/image1.jpg",
    "left": "dataset/image1_left.jpg",
    "right": "dataset/image1_right.jpg",
    "back": "dataset/image1_back.jpg"
}

keypoints_3d = []
for name, path in views.items():
    kp = get_keypoints(path)
    if kp is not None:
        keypoints_3d.append(kp)
    else:
        keypoints_3d.append(np.zeros((33, 3)))

# Roughly average the 3D keypoints from all views
avg_keypoints = np.mean(keypoints_3d, axis=0)

# Initialize SMPL-X model
device = torch.device('cpu')
smplx_model = SMPLX(model_path=SMPLX_MODEL_PATH, gender='neutral', use_pca=False).to(device)

# Dummy pose and shape parameters (you can later optimize them using fitting)
betas = torch.zeros([1, 10]).to(device)
body_pose = torch.zeros([1, smplx_model.NUM_BODY_JOINTS * 3]).to(device)
global_orient = torch.zeros([1, 3]).to(device)
expression = torch.zeros([1, 10]).to(device)

# Generate SMPL-X mesh
output = smplx_model(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient,
    expression=expression
)

vertices = output.vertices[0].detach().cpu().numpy()
joints = output.joints[0].detach().cpu().numpy()

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)
ax.set_title("3D SMPL-X Body Mesh (Rough Fit)")
plt.show()

print("3D reconstruction complete (approximate).")
