# ========== LANDMARK DISTANCE EVALUATION ==========
import numpy as np
import trimesh
import json

print("\n[INFO] Computing landmark distances (Anthropometry comparison)...")

mesh_path = "results/benchmarks/hunyuan_image9.obj"
mesh = trimesh.load(mesh_path, process=True)
vertices = np.array(mesh.vertices)

# SMPL-X landmark indices (approx., can be refined from SMPL-Anthropometry repo)
landmarks = {
    "Acr_L": 3721, "Acr_R": 1682,         # Acromiale
    "Rad_L": 5075, "Rad_R": 2815,         # Radiale
    "Styl_L": 5782, "Styl_R": 3501,       # Stylion
    "Troc_L": 1294, "Troc_R": 240,        # Trochanterion
    "Iliocr_L": 987, "Iliocr_R": 112,     # Iliocristale
    "Tib_L": 5125, "Tib_R": 2951          # Tibiale laterale
}

# Compute left-right Euclidean distances between symmetric landmarks
distances = {}
for name in ["Acr", "Rad", "Styl", "Troc", "Iliocr", "Tib"]:
    left = vertices[landmarks[f"{name}_L"]]
    right = vertices[landmarks[f"{name}_R"]]
    dist = np.linalg.norm(left - right)
    distances[name] = round(dist * 1000, 2)  # convert to mm for comparison

# Compute mean
distances["Mean"] = round(np.mean(list(distances.values())), 2)

# Display results in formatted style
print("\n[RESULTS] Anthropometric Landmark Distances (mm):")
print(f"{'Method':<20}{'Acr.':>8}{'Rad.':>8}{'Styl.':>8}{'Troc.':>8}{'Iliocr.':>10}{'Tib.':>8}{'Mean':>8}")
print("-" * 70)
print(f"{'Ours (Generated)':<20}"
      f"{distances['Acr']:>8.2f}{distances['Rad']:>8.2f}{distances['Styl']:>8.2f}"
      f"{distances['Troc']:>8.2f}{distances['Iliocr']:>10.2f}{distances['Tib']:>8.2f}{distances['Mean']:>8.2f}")

# Save results to JSON for later reference
with open("landmark_distances.json", "w") as f:
    json.dump(distances, f, indent=4)

print(f"\n[INFO] Landmark distances saved to landmark_distances.json")
