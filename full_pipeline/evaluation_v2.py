# ========== LANDMARK DISTANCE EVALUATION (Automatic) ==========
import numpy as np
import trimesh
import json

print("\n[INFO] Computing landmark distances from SMPL-X mesh...")

# Load mesh
mesh_path = "fitted_smplx_mesh.obj"
mesh = trimesh.load(mesh_path, process=True)
vertices = np.array(mesh.vertices)

# Basic normalization (if your model is rotated, you may need to reorient)
min_z = np.min(vertices[:, 2])
vertices[:, 2] -= min_z  # ground align

# Function to find a landmark in a given region based on spatial extrema
def find_landmark(region_verts, axis, side, mode):
    """region_verts: vertex subset, axis: int(0=x,1=y,2=z), side: 'L'/'R', mode: 'max'/'min'"""
    if len(region_verts) == 0:
        return None
    if mode == 'max':
        idx = np.argmax(region_verts[:, axis])
    else:
        idx = np.argmin(region_verts[:, axis])
    return region_verts[idx]

# Split by left/right using X-axis median
x_mid = np.median(vertices[:, 0])
left_side = vertices[vertices[:, 0] > x_mid]
right_side = vertices[vertices[:, 0] < x_mid]

# Heuristic landmark identification
landmark_points = {}

# Shoulders (Acr): highest Y near sides
landmark_points["Acr_L"] = find_landmark(left_side, axis=1, side='L', mode='max')
landmark_points["Acr_R"] = find_landmark(right_side, axis=1, side='R', mode='max')

# Elbows (Rad): mid arm height, outermost X
landmark_points["Rad_L"] = find_landmark(left_side[(left_side[:, 1] < 0.7 * np.max(left_side[:, 1])) &
                                                   (left_side[:, 1] > 0.4 * np.max(left_side[:, 1]))], axis=0, side='L', mode='max')
landmark_points["Rad_R"] = find_landmark(right_side[(right_side[:, 1] < 0.7 * np.max(right_side[:, 1])) &
                                                    (right_side[:, 1] > 0.4 * np.max(right_side[:, 1]))], axis=0, side='R', mode='min')

# Wrists (Styl): lower Y, outermost X
landmark_points["Styl_L"] = find_landmark(left_side[left_side[:, 1] < 0.5 * np.max(left_side[:, 1])], axis=0, side='L', mode='max')
landmark_points["Styl_R"] = find_landmark(right_side[right_side[:, 1] < 0.5 * np.max(right_side[:, 1])], axis=0, side='R', mode='min')

# Hips (Iliocr): wide and upper thigh region
landmark_points["Iliocr_L"] = find_landmark(left_side[left_side[:, 2] < 0.6 * np.max(left_side[:, 2])], axis=0, side='L', mode='max')
landmark_points["Iliocr_R"] = find_landmark(right_side[right_side[:, 2] < 0.6 * np.max(right_side[:, 2])], axis=0, side='R', mode='min')

# Trochanterion (Troc): upper leg start, similar to Iliocr but slightly lower
landmark_points["Troc_L"] = find_landmark(left_side[left_side[:, 2] < 0.5 * np.max(left_side[:, 2])], axis=0, side='L', mode='max')
landmark_points["Troc_R"] = find_landmark(right_side[right_side[:, 2] < 0.5 * np.max(right_side[:, 2])], axis=0, side='R', mode='min')

# Tibiale (Tib): near knee, mid-lower leg
landmark_points["Tib_L"] = find_landmark(left_side[left_side[:, 2] < 0.35 * np.max(left_side[:, 2])], axis=0, side='L', mode='max')
landmark_points["Tib_R"] = find_landmark(right_side[right_side[:, 2] < 0.35 * np.max(right_side[:, 2])], axis=0, side='R', mode='min')

# Compute Euclidean distances
distances = {}
for key in ["Acr", "Rad", "Styl", "Troc", "Iliocr", "Tib"]:
    if landmark_points[f"{key}_L"] is not None and landmark_points[f"{key}_R"] is not None:
        dist = np.linalg.norm(landmark_points[f"{key}_L"] - landmark_points[f"{key}_R"])
        distances[key] = round(dist * 1000, 2)  # convert to mm
    else:
        distances[key] = None

# Mean
valid = [v for v in distances.values() if v is not None]
distances["Mean"] = round(np.mean(valid), 2)

# Print formatted output
print("\n[RESULTS] Automatically estimated landmark distances (mm):")
print(f"{'Landmark':<12}{'Distance (mm)':>15}")
print("-" * 30)
for k, v in distances.items():
    print(f"{k:<12}{v:>15}")

# Save results
with open("auto_landmark_distances.json", "w") as f:
    json.dump(distances, f, indent=4)

print("\n[INFO] Saved to auto_landmark_distances.json")
