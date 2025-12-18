# ========== COMPLETE SYNTHETIC SMPL-X DATASET GENERATOR ==========
# Generates 50 models with multi-view images and anthropometric measurements

import os
import json
import numpy as np
import torch
import smplx
import trimesh
import pyrender
from PIL import Image

# ==================== CONFIGURATION ====================
MODEL_PATH = "full_pipeline/models/"
OUTPUT_BASE = "./synthetic_data"
NUM_SAMPLES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH, IMG_HEIGHT = 1024, 1024

# Create directory structure
MODELS_DIR = os.path.join(OUTPUT_BASE, "models")
IMAGES_DIR = os.path.join(OUTPUT_BASE, "images")
MEASURES_DIR = os.path.join(OUTPUT_BASE, "measures")

for dir_path in [MODELS_DIR, IMAGES_DIR, MEASURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"[INFO] Generating {NUM_SAMPLES} synthetic samples with images and measurements...")
print(f"[INFO] Output directory: {OUTPUT_BASE}")

# ==================== LANDMARK DEFINITIONS ====================
pose_landmarks = {
    "Acr_L": 3875, "Acr_R": 7215,
    "Rad_L": 4334, "Rad_R": 7078,
    "Styl_L": 4858, "Styl_R": 7594,
    "Troc_L": 3448, "Troc_R": 6208,
    "Iliocr_L": 5512, "Iliocr_R": 8237,
    "Tib_L": 3673, "Tib_R": 6437
}

shape_landmarks = {
    "Chest_L": 4490, "Chest_R": 7250,
    "Belly_L": 3263, "Belly_R": 6070,
    "Hip_L": 3447, "Hip_R": 6203,
    "BellySide_L": 5940, "BellySide_R": 5941,
    "LegSide_L": 7156, "LegSide_R": 6231,
    "FootKnee_L": 8626, "FootKnee_R": 6399,
    "KneeWaist_L": 6399, "KneeWaist_R": 7149,
    "WaistNeck_L": 7149, "WaistNeck_R": 6104,
    "NeckHead_L": 6104, "NeckHead_R": 8969
}

# ==================== HELPER FUNCTIONS ====================
def compute_distances(vertices, landmarks):
    """Compute anthropometric distances between paired landmarks"""
    distances = {}
    paired = sorted(set(k[:-2] for k in landmarks.keys()))
    for name in paired:
        left = vertices[landmarks[f"{name}_L"]]
        right = vertices[landmarks[f"{name}_R"]]
        dist = np.linalg.norm(left - right)
        distances[name] = round(dist * 1000, 2)  # mm
    distances["Mean"] = round(np.mean(list(distances.values())), 2)
    return distances

def render_view(mesh_trimesh, camera_pose, light_pose=None):
    """Render a single view of the mesh"""
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    
    # Add mesh
    mesh_trimesh.visual.vertex_colors = [200, 200, 230, 255]
    scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True))
    
    # Add lighting
    if light_pose is None:
        light_pose = np.eye(4)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=light_pose)
    
    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
    color, _ = r.render(scene)
    r.delete()
    
    return color

def get_camera_poses(mesh_center, distance=2.5):
    """Generate camera poses for front, left, and right views"""
    poses = {}
    
    # Front view
    poses['front'] = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, mesh_center[1]],
        [0, 0, 1, mesh_center[2] + distance],
        [0, 0, 0, 1]
    ])
    
    # Left view (rotate 90° around Y-axis)
    poses['left'] = np.array([
        [0, 0, 1, mesh_center[0] + distance],
        [0, 1, 0, mesh_center[1]],
        [-1, 0, 0, mesh_center[2]],
        [0, 0, 0, 1]
    ])
    
    # Right view (rotate -90° around Y-axis)
    poses['right'] = np.array([
        [0, 0, -1, mesh_center[0] - distance],
        [0, 1, 0, mesh_center[1]],
        [1, 0, 0, mesh_center[2]],
        [0, 0, 0, 1]
    ])
    
    return poses

# ==================== SMPL-X MODEL ====================
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=10,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# ==================== GENERATION LOOP ====================
num_standing = NUM_SAMPLES // 2
num_random = NUM_SAMPLES - num_standing

for i in range(NUM_SAMPLES):
    print(f"\n[{i+1}/{NUM_SAMPLES}] Processing sample {i+1}...")
    
    # ---------- Generate SMPL-X Parameters ----------
    betas = torch.randn([1, 10], dtype=torch.float32, device=DEVICE) * 2.0
    
    if i < num_standing:
        body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE)
        body_pose += torch.randn_like(body_pose) * 0.05
        pose_type = "standing"
    else:
        body_pose = torch.randn([1, 21 * 3], dtype=torch.float32, device=DEVICE) * 0.25
        pose_type = "random"
    
    global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE)
    transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE)
    
    # ---------- Generate Mesh ----------
    with torch.no_grad():
        output = smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        )
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces
    
    # ---------- 1) Save Model ----------
    model_filename = f"model_{i+1}.obj"
    model_path = os.path.join(MODELS_DIR, model_filename)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(model_path)
    print(f"  ✓ Saved model: {model_filename}")
    
    # ---------- 2) Render and Save Images ----------
    mesh_center = mesh.bounds.mean(axis=0)
    camera_poses = get_camera_poses(mesh_center)
    
    for view_name, camera_pose in camera_poses.items():
        img_data = render_view(mesh, camera_pose)
        
        if view_name == 'front':
            img_filename = f"image_{i+1}.png"
        else:
            img_filename = f"img_{view_name}_{i+1}.png"
        
        img_path = os.path.join(IMAGES_DIR, img_filename)
        Image.fromarray(img_data).save(img_path)
        print(f"  ✓ Saved {view_name} view: {img_filename}")
    
    # ---------- 3) Compute and Save Measurements ----------
    pose_distances = compute_distances(vertices, pose_landmarks)
    shape_distances = compute_distances(vertices, shape_landmarks)
    
    measurements = {
        "model_id": i + 1,
        "pose_type": pose_type,
        "pose_measurements_mm": pose_distances,
        "shape_measurements_mm": shape_distances,
        "parameters": {
            "betas": betas.cpu().numpy().tolist(),
            "body_pose": body_pose.cpu().numpy().tolist(),
            "global_orient": global_orient.cpu().numpy().tolist(),
            "transl": transl.cpu().numpy().tolist()
        }
    }
    
    measure_filename = f"measure_model_{i+1}.json"
    measure_path = os.path.join(MEASURES_DIR, measure_filename)
    with open(measure_path, "w") as f:
        json.dump(measurements, f, indent=4)
    print(f"  ✓ Saved measurements: {measure_filename}")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print(f"[COMPLETE] Successfully generated {NUM_SAMPLES} synthetic samples!")
print(f"  - {num_standing} standing poses")
print(f"  - {num_random} random poses")
print(f"\nOutput structure:")
print(f"  {OUTPUT_BASE}/")
print(f"    ├── models/     ({NUM_SAMPLES} .obj files)")
print(f"    ├── images/     ({NUM_SAMPLES * 3} .png files - front/left/right)")
print(f"    └── measures/   ({NUM_SAMPLES} .json files)")
print("="*70)

# ==================== DATASET STATISTICS ====================
print("\n[INFO] Computing dataset statistics...")

all_pose_means = []
all_shape_means = []

for i in range(1, NUM_SAMPLES + 1):
    measure_path = os.path.join(MEASURES_DIR, f"measure_model_{i}.json")
    with open(measure_path, 'r') as f:
        data = json.load(f)
        all_pose_means.append(data['pose_measurements_mm']['Mean'])
        all_shape_means.append(data['shape_measurements_mm']['Mean'])

print(f"\nDataset Statistics:")
print(f"  Pose Measurements (Mean):")
print(f"    - Average: {np.mean(all_pose_means):.2f} mm")
print(f"    - Std Dev: {np.std(all_pose_means):.2f} mm")
print(f"    - Range: [{np.min(all_pose_means):.2f}, {np.max(all_pose_means):.2f}] mm")
print(f"\n  Shape Measurements (Mean):")
print(f"    - Average: {np.mean(all_shape_means):.2f} mm")
print(f"    - Std Dev: {np.std(all_shape_means):.2f} mm")
print(f"    - Range: [{np.min(all_shape_means):.2f}, {np.max(all_shape_means):.2f}] mm")

# Save summary
summary = {
    "total_samples": NUM_SAMPLES,
    "standing_poses": num_standing,
    "random_poses": num_random,
    "statistics": {
        "pose_measurements": {
            "mean": float(np.mean(all_pose_means)),
            "std": float(np.std(all_pose_means)),
            "min": float(np.min(all_pose_means)),
            "max": float(np.max(all_pose_means))
        },
        "shape_measurements": {
            "mean": float(np.mean(all_shape_means)),
            "std": float(np.std(all_shape_means)),
            "min": float(np.min(all_shape_means)),
            "max": float(np.max(all_shape_means))
        }
    }
}

summary_path = os.path.join(OUTPUT_BASE, "dataset_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"\n[INFO] Dataset summary saved to: {summary_path}")
print("[INFO] Generation complete! ✓")