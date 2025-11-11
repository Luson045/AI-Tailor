# ========== LANDMARK DISTANCE EVALUATION (FIXED) ==========
import numpy as np
import trimesh
import json
import sys

print("\n[INFO] Computing anthropometric measurements...")
print("[WARNING] This script requires SMPL-X topology mesh!")

# ============================================================
# CRITICAL: These are GUESSED indices from the original code
# For accurate measurements, you MUST:
# 1. Use the SMPL-Anthropometry library (https://github.com/DavidBoja/SMPL-Anthropometry)
# 2. Or manually validate landmarks on your specific mesh
# ============================================================

mesh_path = "fitted_smplx_mesh.obj"  # Replace with your mesh path

try:
    mesh = trimesh.load(mesh_path, process=True)
    vertices = np.array(mesh.vertices)
    print(f"[INFO] Loaded mesh with {len(vertices)} vertices")
    
    # Check if this looks like SMPL-X topology
    expected_smplx_verts = 10475
    expected_smpl_verts = 6890
    
    if len(vertices) not in [expected_smpl_verts, expected_smplx_verts]:
        print(f"\n[ERROR] Vertex count ({len(vertices)}) doesn't match SMPL ({expected_smpl_verts}) or SMPL-X ({expected_smplx_verts})")
        print("[ERROR] Cannot reliably compute anthropometric measurements")
        print("[SOLUTION] Use meshes with proper SMPL/SMPL-X topology")
        sys.exit(1)
    
    model_type = "SMPL-X" if len(vertices) == expected_smplx_verts else "SMPL"
    print(f"[INFO] Detected {model_type} topology")
    
except Exception as e:
    print(f"[ERROR] Failed to load mesh: {e}")
    sys.exit(1)

# ============================================================
# APPROXIMATE landmark indices based on SMPL-Anthropometry repo
# These should be validated for YOUR specific mesh!
# 
# For SMPL-X (10475 vertices), common landmarks from literature:
# - Shoulder (acromion): vertices near shoulder joints
# - Radiale: vertices near elbow joints  
# - Stylion: vertices near wrist joints
# - Trochanterion: vertices near hip joints
# - Tibiale: vertices near knee joints
#
# WARNING: The indices below are ESTIMATES and may not be accurate!
# ============================================================

if model_type == "SMPL-X":
    # SMPL-X approximate landmarks (NEED VALIDATION!)
    # These are based on anatomical position estimates
    landmarks = {
        # Shoulders (acromion points)
        "Acr_L": 4920,   # Left shoulder - APPROXIMATE
        "Acr_R": 1682,   # Right shoulder - APPROXIMATE
        
        # Elbows (radiale points)  
        "Rad_L": 5075,   # Left elbow - APPROXIMATE
        "Rad_R": 2815,   # Right elbow - APPROXIMATE
        
        # Wrists (stylion points)
        "Styl_L": 5782,  # Left wrist - APPROXIMATE
        "Styl_R": 3501,  # Right wrist - APPROXIMATE
        
        # Hips (trochanterion points)
        "Troc_L": 6857,  # Left hip - APPROXIMATE  
        "Troc_R": 3387,  # Right hip - APPROXIMATE
        
        # Knees (tibiale points)
        "Tib_L": 5125,   # Left knee - APPROXIMATE
        "Tib_R": 2951    # Right knee - APPROXIMATE
    }
else:  # SMPL
    # SMPL approximate landmarks (NEED VALIDATION!)
    landmarks = {
        "Acr_L": 3721,   # Left shoulder - APPROXIMATE
        "Acr_R": 1682,   # Right shoulder - APPROXIMATE
        "Rad_L": 3216,   # Left elbow - APPROXIMATE
        "Rad_R": 881,    # Right elbow - APPROXIMATE
        "Styl_L": 3387,  # Left wrist - APPROXIMATE
        "Styl_R": 1052,  # Right wrist - APPROXIMATE
        "Troc_L": 4356,  # Left hip - APPROXIMATE
        "Troc_R": 959,   # Right hip - APPROXIMATE
        "Tib_L": 4453,   # Left knee - APPROXIMATE
        "Tib_R": 1096    # Right knee - APPROXIMATE
    }

print(f"\n[WARNING] Using APPROXIMATE landmark indices!")
print("[WARNING] For accurate measurements, use SMPL-Anthropometry library")
print("[INFO] Repository: https://github.com/DavidBoja/SMPL-Anthropometry\n")

# Validate that landmark indices are within bounds
invalid_landmarks = []
for name, idx in landmarks.items():
    if idx >= len(vertices):
        invalid_landmarks.append(f"{name} (index {idx})")

if invalid_landmarks:
    print(f"[ERROR] Invalid landmark indices: {', '.join(invalid_landmarks)}")
    print(f"[ERROR] Mesh only has {len(vertices)} vertices")
    sys.exit(1)

# Compute bilateral distances
measurements = {}
measurement_pairs = [
    ("Acromiale", "Acr"),
    ("Radiale", "Rad"),
    ("Stylion", "Styl"),
    ("Trochanterion", "Troc"),
    ("Tibiale", "Tib")
]

for full_name, short_name in measurement_pairs:
    left_idx = landmarks[f"{short_name}_L"]
    right_idx = landmarks[f"{short_name}_R"]
    
    left_pos = vertices[left_idx]
    right_pos = vertices[right_idx]
    
    # Euclidean distance in meters, convert to mm
    dist = np.linalg.norm(left_pos - right_pos) * 1000
    measurements[short_name] = round(dist, 2)

# Compute mean
measurements["Mean"] = round(np.mean(list(measurements.values())), 2)

# Display results
print("[RESULTS] Anthropometric Bilateral Distances (mm):")
print("=" * 80)
print(f"{'Measurement':<20}{'Distance (mm)':>12}")
print("-" * 80)

for full_name, short_name in measurement_pairs:
    print(f"{full_name:<20}{measurements[short_name]:>12.2f}")
    
print("-" * 80)
print(f"{'Mean':<20}{measurements['Mean']:>12.2f}")
print("=" * 80)

# Context for interpreting results
print("\n[INFO] Typical adult human anthropometric ranges (mm):")
print("  - Shoulder breadth (Acromiale):     300-450")
print("  - Elbow span (Radiale):             ~400-500")  
print("  - Wrist span (Stylion):             ~350-450")
print("  - Hip breadth (Trochanterion):      250-380")
print("  - Knee span (Tibiale):              ~350-450")

# Check if measurements are reasonable
unrealistic = []
if not (300 <= measurements["Acr"] <= 600):
    unrealistic.append("Shoulder")
if not (250 <= measurements["Troc"] <= 500):
    unrealistic.append("Hip")
    
if unrealistic:
    print(f"\n[WARNING] Measurements for {', '.join(unrealistic)} may be unrealistic!")
    print("[WARNING] This suggests landmark indices may be incorrect")
    print("[RECOMMENDATION] Use SMPL-Anthropometry library for accurate landmarks")

# Save results
output = {
    "model_type": model_type,
    "vertex_count": len(vertices),
    "measurements_mm": measurements,
    "warning": "These measurements use APPROXIMATE landmark indices and should be validated",
    "recommendation": "Use SMPL-Anthropometry library: https://github.com/DavidBoja/SMPL-Anthropometry"
}

with open("landmark_distances.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"\n[INFO] Results saved to landmark_distances.json")

# ============================================================
# PROPER SOLUTION (RECOMMENDED):
# ============================================================
print("\n" + "=" * 80)
print("RECOMMENDED APPROACH FOR ACCURATE MEASUREMENTS:")
print("=" * 80)
print("""
1. Install SMPL-Anthropometry:
   pip install git+https://github.com/DavidBoja/SMPL-Anthropometry.git

2. Use their validated landmarks:
   from measure import MeasureBody
   from measurement_definitions import STANDARD_LABELS
   
   measurer = MeasureBody('smplx')  # or 'smpl'
   measurer.from_verts(verts=your_vertices)
   measurer.measure(measurer.all_possible_measurements)
   
   # Get measurements in cm
   measurements = measurer.measurements

3. This provides validated measurements including:
   - Shoulder breadth, hip breadth
   - Arm/leg lengths
   - Circumferences (chest, waist, hip, etc.)
   - Height
   - And 15+ other standardized measurements

For more details: https://github.com/DavidBoja/SMPL-Anthropometry
""")