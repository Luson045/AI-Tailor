# ========== LANDMARK DISTANCE EVALUATION + VISUALIZATION ==========
import numpy as np
import trimesh
import pyrender
import json

print("\n[INFO] Computing anthropometric distances (Pose + Shape comparison)...")

mesh_path = "fitted_smplx_mesh.obj"
mesh = trimesh.load(mesh_path, process=True)
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# ✅ POSE LANDMARKS (as before)
pose_landmarks = {
    "Acr_L": 3875, "Acr_R": 7215,        # Acromiale (shoulder tip)
    "Rad_L": 4334, "Rad_R": 7078,        # Radiale (elbow)
    "Styl_L": 4858, "Styl_R": 7594,      # Stylion (wrist)
    "Troc_L": 3448, "Troc_R": 6208,      # Trochanterion (hip)
    "Iliocr_L": 5512, "Iliocr_R": 8237,  # Iliocristale (hip crest)
    "Tib_L": 3673, "Tib_R": 6437         # Tibiale laterale (knee)
}

# ✅ SHAPE LANDMARKS (you’ll set actual indices later)
shape_landmarks = {
    "Chest_L": 4490, "Chest_R": 7250,
    "Belly_L": 3263, "Belly_R": 6070,
    "Hip_L": 3447, "Hip_R": 6203,
    "BellySide_L": 5940, "BellySide_R": 5941,
    "LegSide_L": 7156, "LegSide_R": 6231,
    "FootKnee_L": 8626,
    "FootKnee_R": 6399,
    "KneeWaist_L": 6399,
    "KneeWaist_R": 7149,
    "WaistNeck_L": 7149,
    "WaistNeck_R": 6104,
    "NeckHead_L": 6104,
    "NeckHead_R": 8969
}

# ---------- FUNCTION TO COMPUTE DISTANCES ----------
def compute_distances(vertices, landmarks):
    distances = {}
    paired = sorted(set(k[:-2] for k in landmarks.keys()))
    for name in paired:
        left = vertices[landmarks[f"{name}_L"]]
        right = vertices[landmarks[f"{name}_R"]]
        dist = np.linalg.norm(left - right)
        distances[name] = round(dist * 1000, 2)  # convert to mm
    distances["Mean"] = round(np.mean(list(distances.values())), 2)
    return distances

# ---------- COMPUTE BOTH SECTIONS ----------
pose_distances = compute_distances(vertices, pose_landmarks)
shape_distances = compute_distances(vertices, shape_landmarks)

# ---------- PRINT RESULTS ----------
print("\n[RESULTS] Anthropometric Measurements (mm):")

print("\n--- Pose-based Distances ---")
print(f"{'Method':<20}{'Acr.':>8}{'Rad.':>8}{'Styl.':>8}{'Troc.':>8}{'Iliocr.':>10}{'Tib.':>8}{'Mean':>8}")
print("-" * 70)
print(f"{'Ours (Pose)':<20}"
      f"{pose_distances['Acr']:>8.2f}{pose_distances['Rad']:>8.2f}{pose_distances['Styl']:>8.2f}"
      f"{pose_distances['Troc']:>8.2f}{pose_distances['Iliocr']:>10.2f}{pose_distances['Tib']:>8.2f}{pose_distances['Mean']:>8.2f}")

print("\n--- Shape-based Distances ---")
names = list(shape_distances.keys())[:-1]  # exclude Mean
print(f"{'Method':<20}" + "".join([f"{n[:6]:>10}" for n in names]) + f"{'Mean':>10}")
print("-" * 80)
print(f"{'Ours (Shape)':<20}" + "".join([f"{shape_distances[n]:>10.2f}" for n in names]) + f"{shape_distances['Mean']:>10.2f}")

# ---------- SAVE RESULTS ----------
all_results = {"Pose": pose_distances, "Shape": shape_distances}
with open("anthropometric_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
print("\n[INFO] All results saved to anthropometric_results.json")

# ---------- VISUALIZATION ----------
print("\n[INFO] Launching 3D viewer for validation...")

scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
mesh_trimesh = trimesh.Trimesh(vertices, faces, process=False)
mesh_trimesh.visual.vertex_colors = [200, 200, 230, 255]
scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True))

# --- Pose landmarks (colored) ---
pose_colors = {
    "Acr": [255, 0, 0, 255],
    "Rad": [0, 255, 0, 255],
    "Styl": [0, 0, 255, 255],
    "Troc": [255, 255, 0, 255],
    "Iliocr": [255, 0, 255, 255],
    "Tib": [0, 255, 255, 255],
}

for name in ["Acr", "Rad", "Styl", "Troc", "Iliocr", "Tib"]:
    for side in ["L", "R"]:
        idx = pose_landmarks[f"{name}_{side}"]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
        sphere.apply_translation(vertices[idx])
        sphere.visual.vertex_colors = pose_colors[name]
        scene.add(pyrender.Mesh.from_trimesh(sphere))

# --- Shape landmarks (black) ---
for name in sorted(set(k[:-2] for k in shape_landmarks.keys())):
    for side in ["L", "R"]:
        idx = shape_landmarks[f"{name}_{side}"]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
        sphere.apply_translation(vertices[idx])
        sphere.visual.vertex_colors = [0, 0, 0, 255]
        scene.add(pyrender.Mesh.from_trimesh(sphere))

# --- Lighting & Camera ---
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=np.eye(4))
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1.0],
    [0, 0, 1, 2.5],
    [0, 0, 0, 1]
])
scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=camera_pose)

# --- Viewer / Fallback render ---
try:
    pyrender.Viewer(scene, use_raymond_lighting=True)
except Exception as e:
    print(f"[WARNING] Viewer error: {e}")
    r = pyrender.OffscreenRenderer(1200, 1200)
    color, _ = r.render(scene)
    trimesh.exchange.export.export_image(color, "anthropometric_validation.png")
    print("[INFO] Saved anthropometric_validation.png instead.")
