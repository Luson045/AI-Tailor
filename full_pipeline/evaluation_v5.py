# ========== LANDMARK DISTANCE EVALUATION + VISUALIZATION (with Height Normalization) ==========
import numpy as np
import trimesh
import pyrender
import json
import argparse
import sys

# ========== CONFIGURATION ==========
# Set actual height in cm. If None, normalization will be skipped.
# You can also pass this via command line: python evaluation_v5.py --actual_height 165
ACTUAL_HEIGHT_CM = 165  # Set to None to disable normalization, or provide actual height in cm

print("\n[INFO] Computing anthropometric distances (Pose + Shape comparison)...")

mesh_path = "fitted_smplx_mesh.obj"
mesh = trimesh.load(mesh_path, process=True)
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# ========== COMPUTE PREDICTED HEIGHT ==========
def compute_predicted_height(vertices):
    """
    Compute predicted height from mesh vertices.
    Height is the Y-axis range (assuming Y is vertical).
    Returns height in cm.
    """
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    predicted_height_m = max_bounds[1] - min_bounds[1]  # Y-axis range in meters
    predicted_height_cm = predicted_height_m * 100  # Convert to cm
    return predicted_height_cm

predicted_height_cm = compute_predicted_height(vertices)
print(f"[INFO] Predicted height from mesh: {predicted_height_cm:.2f} cm")

# ========== HEIGHT NORMALIZATION ==========
# Parse command line arguments for actual height
parser = argparse.ArgumentParser(description='Evaluate body measurements with height normalization')
parser.add_argument('--actual_height', type=float, default=None,
                    help='Actual height in cm for normalization (overrides config)')
args = parser.parse_args()

# Use command line argument if provided, otherwise use config
actual_height_cm = args.actual_height if args.actual_height is not None else ACTUAL_HEIGHT_CM

scale_factor = 1.0  # Default: no scaling
if actual_height_cm is not None and actual_height_cm > 0:
    if predicted_height_cm > 0:
        scale_factor = actual_height_cm / predicted_height_cm
        print(f"[INFO] Actual height: {actual_height_cm:.2f} cm")
        print(f"[INFO] Scale factor: {scale_factor:.4f} (actual / predicted)")
        print(f"[INFO] Applying height normalization to all measurements...")
    else:
        print(f"[WARNING] Predicted height is invalid ({predicted_height_cm:.2f} cm). Skipping normalization.")
else:
    print(f"[INFO] Actual height not provided. Measurements will not be normalized.")
    print(f"[INFO] To enable normalization, set ACTUAL_HEIGHT_CM or use --actual_height flag")

# ✅ POSE LANDMARKS (as before)
pose_landmarks = {
    "Acr_L": 3875, "Acr_R": 7215,        # Acromiale (shoulder tip)
    "Rad_L": 4334, "Rad_R": 7078,        # Radiale (elbow)
    "Styl_L": 4858, "Styl_R": 7594,      # Stylion (wrist)
    "Troc_L": 3448, "Troc_R": 6208,      # Trochanterion (hip)
    "Iliocr_L": 5512, "Iliocr_R": 8237,  # Iliocristale (hip crest)
    "Tib_L": 3673, "Tib_R": 6437         # Tibiale laterale (knee)
}

# ✅ SHAPE LANDMARKS (you'll set actual indices later)
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

# ---------- APPLY HEIGHT NORMALIZATION TO ALL MEASUREMENTS ----------
def apply_scale_factor(measurements_dict, scale_factor):
    """
    Apply scale factor to all measurements in the dictionary.
    Excludes 'Mean' from scaling, recalculates it after scaling other values.
    """
    scaled_measurements = {}
    mean_values = []
    
    for key, value in measurements_dict.items():
        if key != "Mean":
            scaled_value = value * scale_factor
            scaled_measurements[key] = round(scaled_value, 2)
            mean_values.append(scaled_value)
        else:
            # Mean will be recalculated
            pass
    
    # Recalculate mean from scaled values
    if mean_values:
        scaled_measurements["Mean"] = round(np.mean(mean_values), 2)
    
    return scaled_measurements

if scale_factor != 1.0:
    pose_distances = apply_scale_factor(pose_distances, scale_factor)
    shape_distances = apply_scale_factor(shape_distances, scale_factor)
    print(f"[INFO] All measurements scaled by factor: {scale_factor:.4f}")

# ---------- PRINT RESULTS ----------
print("\n[RESULTS] Anthropometric Measurements (mm):")
if scale_factor != 1.0:
    print(f"[NOTE] Measurements have been normalized using height scale factor: {scale_factor:.4f}")

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
all_results = {
    "Pose": pose_distances,
    "Shape": shape_distances,
    "Height_Info": {
        "predicted_height_cm": round(predicted_height_cm, 2),
        "actual_height_cm": round(actual_height_cm, 2) if actual_height_cm is not None else None,
        "scale_factor": round(scale_factor, 4) if scale_factor != 1.0 else None
    }
}
with open("scaled_anthropometric_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
print("\n[INFO] All results saved to scaled_anthropometric_results.json")

# ---------- VISUALIZATION ----------
# print("\n[INFO] Launching 3D viewer for validation...")

# scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
# mesh_trimesh = trimesh.Trimesh(vertices, faces, process=False)
# mesh_trimesh.visual.vertex_colors = [200, 200, 230, 255]
# scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True))

# # --- Pose landmarks (colored) ---
# pose_colors = {
#     "Acr": [255, 0, 0, 255],
#     "Rad": [0, 255, 0, 255],
#     "Styl": [0, 0, 255, 255],
#     "Troc": [255, 255, 0, 255],
#     "Iliocr": [255, 0, 255, 255],
#     "Tib": [0, 255, 255, 255],
# }

# for name in ["Acr", "Rad", "Styl", "Troc", "Iliocr", "Tib"]:
#     for side in ["L", "R"]:
#         idx = pose_landmarks[f"{name}_{side}"]
#         sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
#         sphere.apply_translation(vertices[idx])
#         sphere.visual.vertex_colors = pose_colors[name]
#         scene.add(pyrender.Mesh.from_trimesh(sphere))

# # --- Shape landmarks (black) ---
# for name in sorted(set(k[:-2] for k in shape_landmarks.keys())):
#     for side in ["L", "R"]:
#         idx = shape_landmarks[f"{name}_{side}"]
#         sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
#         sphere.apply_translation(vertices[idx])
#         sphere.visual.vertex_colors = [0, 0, 0, 255]
#         scene.add(pyrender.Mesh.from_trimesh(sphere))

# # --- Lighting & Camera ---
# light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
# scene.add(light, pose=np.eye(4))
# camera_pose = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, -1.0],
#     [0, 0, 1, 2.5],
#     [0, 0, 0, 1]
# ])
# scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=camera_pose)

# --- Viewer / Fallback render ---
# try:
#     pyrender.Viewer(scene, use_raymond_lighting=True)
# except Exception as e:
#     print(f"[WARNING] Viewer error: {e}")
#     r = pyrender.OffscreenRenderer(1200, 1200)
#     color, _ = r.render(scene)
#     trimesh.exchange.export.export_image(color, "anthropometric_validation.png")
#     print("[INFO] Saved anthropometric_validation.png instead.")

