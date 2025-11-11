# ========== LANDMARK DISTANCE EVALUATION + VISUALIZATION ==========
import numpy as np
import trimesh
import pyrender
import json

print("\n[INFO] Computing landmark distances (Anthropometry comparison)...")

mesh_path = "fitted_smplx_mesh.obj"
mesh = trimesh.load(mesh_path, process=True)
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# ✅ Updated SMPL-X landmark indices (your verified set)
landmarks = {
    "Acr_L": 3875, "Acr_R": 7215,        # Acromiale (shoulder tip)
    "Rad_L": 4334, "Rad_R": 7078,        # Radiale (elbow)
    "Styl_L": 4858, "Styl_R": 7594,      # Stylion (wrist)
    "Troc_L": 3448, "Troc_R": 6208,      # Trochanterion (hip)
    "Iliocr_L": 5512, "Iliocr_R": 8237,  # Iliocristale (hip crest)
    "Tib_L": 3673, "Tib_R": 6437         # Tibiale laterale (knee)
}

# ---------- COMPUTE DISTANCES ----------
distances = {}
for name in ["Acr", "Rad", "Styl", "Troc", "Iliocr", "Tib"]:
    left = vertices[landmarks[f"{name}_L"]]
    right = vertices[landmarks[f"{name}_R"]]
    dist = np.linalg.norm(left - right)
    distances[name] = round(dist * 1000, 2)  # convert to mm for comparison

distances["Mean"] = round(np.mean(list(distances.values())), 2)

# ---------- PRINT RESULTS ----------
print("\n[RESULTS] Anthropometric Landmark Distances (mm):")
print(f"{'Method':<20}{'Acr.':>8}{'Rad.':>8}{'Styl.':>8}{'Troc.':>8}{'Iliocr.':>10}{'Tib.':>8}{'Mean':>8}")
print("-" * 70)
print(f"{'Ours (Generated)':<20}"
      f"{distances['Acr']:>8.2f}{distances['Rad']:>8.2f}{distances['Styl']:>8.2f}"
      f"{distances['Troc']:>8.2f}{distances['Iliocr']:>10.2f}{distances['Tib']:>8.2f}{distances['Mean']:>8.2f}")

# ---------- SAVE RESULTS ----------
with open("landmark_distances.json", "w") as f:
    json.dump(distances, f, indent=4)
print("\n[INFO] Landmark distances saved to landmark_distances.json")

# ---------- VISUALIZATION ----------
print("\n[INFO] Launching 3D viewer for landmark validation...")

scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
mesh_trimesh = trimesh.Trimesh(vertices, faces, process=False)
mesh_trimesh.visual.vertex_colors = [200, 200, 230, 255]
scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True))

# Assign distinct colors for each landmark pair
colors = {
    "Acr": [255, 0, 0, 255],        # Red
    "Rad": [0, 255, 0, 255],        # Green
    "Styl": [0, 0, 255, 255],       # Blue
    "Troc": [255, 255, 0, 255],     # Yellow
    "Iliocr": [255, 0, 255, 255],   # Magenta
    "Tib": [0, 255, 255, 255],      # Cyan
}

# Add spheres for each landmark
for name in ["Acr", "Rad", "Styl", "Troc", "Iliocr", "Tib"]:
    for side in ["L", "R"]:
        idx = landmarks[f"{name}_{side}"]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
        sphere.apply_translation(vertices[idx])
        sphere.visual.vertex_colors = colors[name]
        scene.add(pyrender.Mesh.from_trimesh(sphere))

# Add lighting & camera
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=np.eye(4))
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1.0],
    [0, 0, 1, 2.5],
    [0, 0, 0, 1]
])
scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=camera_pose)

# Show the viewer
try:
    pyrender.Viewer(scene, use_raymond_lighting=True)
except Exception as e:
    print(f"[WARNING] Viewer error: {e}")
    r = pyrender.OffscreenRenderer(1200, 1200)
    color, _ = r.render(scene)
    trimesh.util.attach_to_log()
    trimesh.exchange.export.export_image(color, "landmark_validation.png")
    print("[INFO] Saved landmark_validation.png instead.")
