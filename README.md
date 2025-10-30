```markdown
# 🧍‍♂️ Human Body Measurement Estimation using 3D Human Body Reconstruction from 2D Images

## 📘 Overview
This project focuses on estimating **human body measurements** through **3D human body reconstruction** from standard 2D images.

The system combines advanced pose estimation and mesh reconstruction models to achieve accurate, realistic 3D results.

---

## ⚙️ Pipeline v4 – Workflow

### 1. BlazePose
- Extracts **2D keypoints** from the input image.  
- Converts them into an approximate **3D point cloud** representing the human pose.

### 2. SMPL-X
- Uses the **3D keypoints** to generate a **parametric 3D human mesh**.  
- Provides detailed control over **body shape, pose, and expressions** for precise measurement estimation.

---

## 🧩 Model Setup

### 🔽 Download and Place SMPL-X Model

1. Visit the official **[SMPL-X website](https://smpl-x.is.tue.mpg.de/)**.  
2. **Sign in** (or create an account if you don’t have one).  
3. Navigate to the **Downloads** section.  
4. Download the **SMPL-X model package**.  
5. **Unzip** the downloaded archive.  
6. Place the extracted `smplx` folder inside the `full_pipeline/models/` directory.

Your folder structure should look like this:

```

full_pipeline/
├── models/
│   └── smplx/
├── SMPL_python_v.1.1.0/
│   └── smpl/
├── pipeline_v2.py
├── pipeline_v3.py
├── pipeline_v4.py
└── simple_pipeline.py

```

---

## 📂 Project Structure

```

AI-TAILOR/
├── benchmark models/
├── dataset/
├── full_pipeline/
│   ├── models/
│   │   └── smplx/
│   ├── SMPL_python_v.1.1.0/
│   │   └── smpl/
│   ├── pipeline_v2.py
│   ├── pipeline_v3.py
│   ├── pipeline_v4.py
│   └── simple_pipeline.py
├── images/
├── models/
├── papers/
├── results/
├── test codes/
└── README.md

```

---

## 📏 Output
- Reconstructed **3D human mesh**
- Estimated **body measurements** (height, shoulder width, waist, limb lengths, etc.)

---

## 🧠 Future Enhancements
- Support for **real-time video input**
- Improved **texture and detail reconstruction**
- Enhanced **measurement calibration** using demographic data

---

## 🧩 Credits
- [BlazePose](https://google.github.io/mediapipe/solutions/pose.html) for 2D/3D keypoint detection  
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) for 3D human mesh generation

---
```