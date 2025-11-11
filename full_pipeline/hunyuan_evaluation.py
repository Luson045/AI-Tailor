# ========== ANTHROPOMETRIC MEASUREMENTS FOR HUNYUAN3D MESHES ==========
"""
CRITICAL UNDERSTANDING:
- Hunyuan3D generates ARBITRARY triangle meshes (500k-600k triangles)
- These are NOT SMPL/SMPL-X topology meshes
- You CANNOT use fixed vertex indices for landmarks
- You need to FIT an SMPL/SMPL-X model to your Hunyuan mesh first

This script shows 3 approaches:
1. Geometric measurements (bounding box based)
2. Landmark detection (requires fitting SMPL/SMPL-X)
3. Proper SMPL-Anthropometry workflow
"""

import numpy as np
import trimesh
import json
import sys

print("\n" + "="*80)
print("ANTHROPOMETRIC MEASUREMENTS FOR HUNYUAN3D MESHES")
print("="*80)

# ============================================================
# APPROACH 1: GEOMETRIC MEASUREMENTS (Simple but crude)
# ============================================================
def geometric_measurements(mesh):
    """
    Compute basic measurements from mesh geometry.
    This doesn't use anatomical landmarks but gives rough approximations.
    """
    vertices = mesh.vertices
    
    # Get bounding box
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    
    measurements = {}
    
    # Height (Y-axis range in standard orientation)
    measurements['height'] = (max_bounds[1] - min_bounds[1]) * 1000  # to mm
    
    # Width (X-axis range)
    measurements['width'] = (max_bounds[0] - min_bounds[0]) * 1000
    
    # Depth (Z-axis range)
    measurements['depth'] = (max_bounds[2] - min_bounds[2]) * 1000
    
    # Approximate shoulder width (upper 80-90% of height)
    upper_region = vertices[vertices[:, 1] > min_bounds[1] + 0.8 * (max_bounds[1] - min_bounds[1])]
    if len(upper_region) > 0:
        shoulder_width = (upper_region[:, 0].max() - upper_region[:, 0].min()) * 1000
        measurements['shoulder_width_approx'] = shoulder_width
    
    # Approximate hip width (around 45-55% of height)
    mid_region = vertices[
        (vertices[:, 1] > min_bounds[1] + 0.45 * (max_bounds[1] - min_bounds[1])) &
        (vertices[:, 1] < min_bounds[1] + 0.55 * (max_bounds[1] - min_bounds[1]))
    ]
    if len(mid_region) > 0:
        hip_width = (mid_region[:, 0].max() - mid_region[:, 0].min()) * 1000
        measurements['hip_width_approx'] = hip_width
    
    return measurements


# ============================================================
# APPROACH 2: LANDMARK DETECTION (Requires SMPL fitting)
# ============================================================
def detect_anatomical_landmarks(mesh):
    """
    Attempt to find anatomical landmarks on arbitrary mesh.
    This is very approximate and should be validated visually!
    
    Better approach: Fit SMPL/SMPL-X model to mesh first.
    """
    vertices = mesh.vertices
    
    # Find extremal points (crude landmark approximation)
    landmarks = {}
    
    # Top of head (highest Y point)
    landmarks['head_top'] = vertices[vertices[:, 1].argmax()]
    
    # Bottom of feet (lowest Y point)
    landmarks['feet_bottom'] = vertices[vertices[:, 1].argmin()]
    
    # Left and right shoulders (high + far left/right)
    height_threshold = vertices[:, 1].max() - 0.15 * (vertices[:, 1].max() - vertices[:, 1].min())
    upper_verts = vertices[vertices[:, 1] > height_threshold]
    
    if len(upper_verts) > 0:
        landmarks['shoulder_left'] = upper_verts[upper_verts[:, 0].argmin()]
        landmarks['shoulder_right'] = upper_verts[upper_verts[:, 0].argmax()]
    
    # Mid-height for hips
    mid_height = vertices[:, 1].min() + 0.5 * (vertices[:, 1].max() - vertices[:, 1].min())
    mid_verts = vertices[
        (vertices[:, 1] > mid_height - 0.05 * (vertices[:, 1].max() - vertices[:, 1].min())) &
        (vertices[:, 1] < mid_height + 0.05 * (vertices[:, 1].max() - vertices[:, 1].min()))
    ]
    
    if len(mid_verts) > 0:
        landmarks['hip_left'] = mid_verts[mid_verts[:, 0].argmin()]
        landmarks['hip_right'] = mid_verts[mid_verts[:, 0].argmax()]
    
    return landmarks


def compute_landmark_distances(landmarks):
    """Compute bilateral distances from detected landmarks."""
    measurements = {}
    
    if 'shoulder_left' in landmarks and 'shoulder_right' in landmarks:
        dist = np.linalg.norm(landmarks['shoulder_left'] - landmarks['shoulder_right'])
        measurements['shoulder_breadth'] = round(dist * 1000, 2)
    
    if 'hip_left' in landmarks and 'hip_right' in landmarks:
        dist = np.linalg.norm(landmarks['hip_left'] - landmarks['hip_right'])
        measurements['hip_breadth'] = round(dist * 1000, 2)
    
    if 'head_top' in landmarks and 'feet_bottom' in landmarks:
        dist = np.linalg.norm(landmarks['head_top'] - landmarks['feet_bottom'])
        measurements['total_height'] = round(dist * 1000, 2)
    
    return measurements


# ============================================================
# MAIN EXECUTION
# ============================================================
mesh_path = "results/benchmarks/hunyuan_image9.obj"

try:
    print(f"\n[INFO] Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=True)
    vertices = mesh.vertices
    faces = mesh.faces
    
    print(f"[INFO] Loaded mesh:")
    print(f"  - Vertices: {len(vertices):,}")
    print(f"  - Faces: {len(faces):,}")
    print(f"  - Bounds: {mesh.bounds}")
    
    # Check mesh type
    if len(vertices) in [6890, 10475]:
        print(f"\n[WARNING] Mesh has {len(vertices)} vertices - looks like SMPL/SMPL-X!")
        print("[INFO] You can use SMPL-Anthropometry library directly")
        is_smpl = True
    else:
        print(f"\n[INFO] Arbitrary mesh topology detected ({len(vertices):,} vertices)")
        print("[INFO] This is typical for Hunyuan3D generated meshes")
        is_smpl = False
    
except Exception as e:
    print(f"[ERROR] Failed to load mesh: {e}")
    sys.exit(1)


# ============================================================
# COMPUTE MEASUREMENTS
# ============================================================
print("\n" + "="*80)
print("APPROACH 1: GEOMETRIC MEASUREMENTS (Bounding Box Based)")
print("="*80)

geo_measurements = geometric_measurements(mesh)

print("\n[RESULTS] Geometric Measurements (mm):")
for name, value in geo_measurements.items():
    print(f"  {name:.<30} {value:>10.2f}")


print("\n" + "="*80)
print("APPROACH 2: LANDMARK DETECTION (Approximate)")
print("="*80)
print("[WARNING] These are CRUDE estimates based on extremal points!")
print("[WARNING] Results should be validated visually!\n")

landmarks = detect_anatomical_landmarks(mesh)
landmark_measurements = compute_landmark_distances(landmarks)

print("[RESULTS] Landmark-Based Measurements (mm):")
for name, value in landmark_measurements.items():
    print(f"  {name:.<30} {value:>10.2f}")

# Display landmarks for verification
print("\n[INFO] Detected landmark positions (for visual verification):")
for name, pos in landmarks.items():
    print(f"  {name:.<20} [{pos[0]:>8.4f}, {pos[1]:>8.4f}, {pos[2]:>8.4f}]")


# ============================================================
# SAVE RESULTS
# ============================================================
output = {
    "mesh_info": {
        "path": mesh_path,
        "vertices": len(vertices),
        "faces": len(faces),
        "is_smpl_topology": is_smpl
    },
    "geometric_measurements_mm": {k: round(v, 2) for k, v in geo_measurements.items()},
    "landmark_measurements_mm": {k: round(v, 2) for k, v in landmark_measurements.items()},
    "landmarks": {k: v.tolist() for k, v in landmarks.items()},
    "warnings": [
        "Measurements are APPROXIMATE for non-SMPL meshes",
        "Landmark detection uses crude extremal point heuristics",
        "For accurate measurements, fit SMPL/SMPL-X model first"
    ]
}

with open("hunyuan_measurements.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[INFO] Results saved to hunyuan_measurements.json")


# ============================================================
# RECOMMENDATIONS
# ============================================================
print("\n" + "="*80)
print("RECOMMENDED WORKFLOW FOR ACCURATE MEASUREMENTS")
print("="*80)
print("""
Since Hunyuan3D generates arbitrary triangle meshes, you need to:

METHOD 1: FIT SMPL/SMPL-X MODEL (Most Accurate)
─────────────────────────────────────────────────
1. Fit an SMPL/SMPL-X model to your Hunyuan mesh using:
   - PyMAF-X: https://github.com/HongwenZhang/PyMAF-X
   - PIXIE: https://github.com/yfeng95/PIXIE
   - PARE: https://github.com/mkocabas/PARE
   
2. Once you have SMPL/SMPL-X parameters, use SMPL-Anthropometry:
   
   from measure import MeasureBody
   measurer = MeasureBody('smplx')
   measurer.from_body_model(gender='neutral', shape=fitted_betas)
   measurer.measure(measurer.all_possible_measurements)
   measurements = measurer.measurements  # in cm


METHOD 2: VISUAL LANDMARK ANNOTATION (Manual but Accurate)
──────────────────────────────────────────────────────────
1. Open mesh in Blender/MeshLab
2. Manually mark anatomical landmarks
3. Export landmark vertex indices
4. Compute distances in Python


METHOD 3: USE GEOMETRIC MEASUREMENTS (Fast but Approximate)
───────────────────────────────────────────────────────────
Use the measurements above for rough estimates, but note:
- Not based on true anatomical landmarks
- Can't be compared to anthropometric standards
- Good for relative comparisons only


EXAMPLE: Fitting SMPL-X with PyMAF-X
─────────────────────────────────────
# Install PyMAF-X
git clone https://github.com/HongwenZhang/PyMAF-X
cd PyMAF-X
pip install -r requirements.txt

# Fit to your Hunyuan mesh
python demo.py --mesh_file /path/to/hunyuan_mesh.obj

# Use the output SMPL-X parameters with SMPL-Anthropometry
from measure import MeasureBody
measurer = MeasureBody('smplx')
measurer.from_body_model(gender='neutral', shape=fitted_betas)
measurer.measure(measurer.all_possible_measurements)

print("Measurements (cm):")
for name, value in measurer.measurements.items():
    print(f"{name}: {value:.2f}")
""")

print("\n" + "="*80)
print(f"Summary: {'SMPL topology detected' if is_smpl else 'Arbitrary mesh topology'}")
print("Action: " + ("Use SMPL-Anthropometry directly" if is_smpl else 
                     "Fit SMPL/SMPL-X model for accurate measurements"))
print("="*80 + "\n")