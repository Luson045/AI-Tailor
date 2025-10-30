import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import torch
from smplx import SMPL
import trimesh
import pyrender
import os
#pip install torch torchvision smplx trimesh pyrender
# =========================================================
# 🧵 Virtual Tailor - Front View Body Measurement Estimation
# =========================================================

def euclidean(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def estimate_measurements(image_path, person_height_cm=170):
    # Initialize Mediapipe Pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)

    if not results.pose_landmarks:
        raise Exception("❌ No body landmarks detected! Please use a clear full-body front-view image.")

    # Extract 3D keypoints
    keypoints = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in results.pose_landmarks.landmark])

    # Annotate image with landmarks
    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
    )

    # Extract relevant landmarks
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    left_hip = keypoints[23]
    right_hip = keypoints[24]
    head_top = keypoints[0]
    left_ankle = keypoints[29]
    right_ankle = keypoints[30]
    avg_ankle = (left_ankle + right_ankle) / 2

    # Estimate real-world pixel-to-cm ratio
    pixel_height = euclidean(head_top, avg_ankle)
    pixel_to_cm = person_height_cm / pixel_height

    # Calculate measurements
    shoulder_width = euclidean(left_shoulder, right_shoulder) * pixel_to_cm
    hip_width = euclidean(left_hip, right_hip) * pixel_to_cm
    torso_length = euclidean(left_shoulder, left_hip) * pixel_to_cm
    leg_length = euclidean(left_hip, avg_ankle) * pixel_to_cm

    print("\n📏 Estimated Measurements:")
    print(f"• Height (approx)     : {person_height_cm:.2f} cm (user input)")
    print(f"• Shoulder Width      : {shoulder_width:.2f} cm")
    print(f"• Hip Width           : {hip_width:.2f} cm")
    print(f"• Torso Length        : {torso_length:.2f} cm")
    print(f"• Leg Length          : {leg_length:.2f} cm")

    # Display using matplotlib
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 8))
    plt.imshow(annotated_rgb)
    plt.axis('off')
    plt.title("🧍 Virtual Tailor - Detected Landmarks")
    plt.show()

    return {
        "height": person_height_cm,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,
        "leg_length": leg_length
    }


# =========================================================
# Optional: Render Neutral SMPL Model
# =========================================================
def render_smpl_model():
    """Render a neutral SMPL model for reference."""
    device = torch.device('cpu')
    model = SMPL(model_path='.', gender='NEUTRAL').to(device)
    betas = torch.zeros([1, 10]).to(device)
    pose = torch.zeros([1, 72]).to(device)

    output = model(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3])
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    pyrender.Viewer(scene, use_raymond_lighting=True)

# =========================================================
# Main execution
# =========================================================
if __name__ == "__main__":
    IMAGE_PATH = "luson.jpg"   # Change this to your image filename
    PERSON_HEIGHT_CM = 169      # Set your height manually for scaling

    if not os.path.exists(IMAGE_PATH):
        print(f"⚠️ Please place your front-view full-body image as '{IMAGE_PATH}' in this folder.")
    else:
        measurements = estimate_measurements(IMAGE_PATH, PERSON_HEIGHT_CM)
        print("\n✅ Measurement extraction complete!")

        # Uncomment if you want to see a basic 3D SMPL body
        # render_smpl_model()
