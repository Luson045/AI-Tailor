# ========== SYNTHETIC SMPL-X DATASET GENERATOR (Mixed Poses) ==========
import os
import json
import numpy as np
import torch
import smplx
import trimesh

# Configuration
MODEL_PATH = "full_pipeline/models/"   # Path to your SMPL-X models
OUTPUT_DIR = "synthetic_dataset"
NUM_SAMPLES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Generating {NUM_SAMPLES} synthetic SMPL-X samples (half standing, half random)...")

# Create SMPL-X model (neutral gender)
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender='NEUTRAL',
    num_betas=10,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

metadata = {}

# Half standing (neutral) + half random poses
num_standing = NUM_SAMPLES // 2
num_random = NUM_SAMPLES - num_standing

for i in range(NUM_SAMPLES):
    # ---------- Randomize SHAPE ----------
    betas = torch.randn([1, 10], dtype=torch.float32, device=DEVICE) * 2.0  # shape variation

    # ---------- Define POSE ----------
    if i < num_standing:
        # Standing (neutral) pose: small noise only
        body_pose = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE)
        body_pose += torch.randn_like(body_pose) * 0.05  # slight noise for realism
        pose_type = "standing"
    else:
        # Random pose: more varied body angles
        body_pose = torch.randn([1, 21 * 3], dtype=torch.float32, device=DEVICE) * 0.25
        pose_type = "random"

    global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE)
    transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE)
    
    # ---------- Generate mesh ----------
    with torch.no_grad():
        output = smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        )
        vertices = output.vertices[0].cpu().numpy()
        faces = smplx_model.faces

    # ---------- Export mesh ----------
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(os.path.join(OUTPUT_DIR, f"{pose_type}_{i+1:02d}.obj"))

    # ---------- Save parameters ----------
    metadata[f"{pose_type}_{i+1:02d}"] = {
        "pose_type": pose_type,
        "betas": betas.cpu().numpy().tolist(),
        "body_pose": body_pose.cpu().numpy().tolist(),
        "global_orient": global_orient.cpu().numpy().tolist(),
        "transl": transl.cpu().numpy().tolist()
    }

    print(f"[INFO] Saved {pose_type} sample {i+1:02d} to {OUTPUT_DIR}/{pose_type}_{i+1:02d}.obj")

# ---------- Save metadata ----------
with open(os.path.join(OUTPUT_DIR, "synthetic_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n[INFO] ✓ Generated {NUM_SAMPLES} SMPL-X samples total.")
print(f"[INFO] {num_standing} standing and {num_random} random poses saved in '{OUTPUT_DIR}/'.")
