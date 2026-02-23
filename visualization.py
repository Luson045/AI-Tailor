import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "AI-Tailor: End-to-End Visualization"

PIPELINE_SCRIPT = Path("full_pipeline") / "pipeline_v6.py"
EVAL_V4_SCRIPT = Path("full_pipeline") / "evaluation_v4.py"
EVAL_V5_SCRIPT = Path("full_pipeline") / "evaluation_v5.py"

ANTHRO_RESULTS = Path("anthropometric_results.json")
SCALED_RESULTS = Path("scaled_anthropometric_results.json")
MESH_OBJ = Path("fitted_smplx_mesh.obj")


def apply_app_theme() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
            :root {
                --bg: #0f1117;
                --panel: #151a23;
                --panel-2: #1b2230;
                --text: #e6e9ef;
                --muted: #9aa4b2;
                --accent: #46d2ff;
                --accent-2: #ffb86b;
                --border: #283044;
            }
            .stApp {
                background: radial-gradient(1200px 600px at 10% 0%, #1b2332 0%, var(--bg) 45%),
                            radial-gradient(1000px 500px at 90% 10%, #1a2c2a 0%, var(--bg) 50%);
                color: var(--text);
            }
            h1, h2, h3 {
                color: var(--text) !important;
                letter-spacing: 0.4px;
            }
            .card {
                background: linear-gradient(180deg, var(--panel), var(--panel-2));
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px 22px;
                box-shadow: 0 10px 30px rgba(5, 10, 20, 0.35);
            }
            .muted {
                color: var(--muted);
                font-size: 0.92rem;
            }
            .tag {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(70, 210, 255, 0.16);
                color: var(--accent);
                font-size: 0.8rem;
                margin-left: 8px;
            }
            .stTextArea textarea {
                background: #0b0f16 !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
            }
            .stButton>button {
                background: linear-gradient(90deg, var(--accent), #5eead4);
                border: none;
                color: #0b0f16;
                font-weight: 700;
                padding: 0.6rem 1.4rem;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_script_realtime(command: List[str], title: str) -> Tuple[int, str]:
    st.markdown(f"### {title} <span class='tag'>live</span>", unsafe_allow_html=True)
    log_box = st.empty()
    log_lines: List[str] = []

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    if process.stdout is None:
        return 1, ""

    for line in process.stdout:
        log_lines.append(line.rstrip())
        log_box.code(
            "\n".join(log_lines[-300:]),
            language="text",
        )

    process.wait()
    return process.returncode, "\n".join(log_lines)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_measurements(title: str, data: dict) -> None:
    st.markdown(f"### {title}")
    if not data:
        st.warning("No measurement data found.")
        return

    pose = data.get("Pose", {})
    shape = data.get("Shape", {})
    height_info = data.get("Height_Info", {})

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Pose-based distances (mm)**")
        st.dataframe(
            [{"Metric": k, "Value (mm)": v} for k, v in pose.items()],
            width="stretch",
        )
    with cols[1]:
        st.markdown("**Shape-based distances (mm)**")
        st.dataframe(
            [{"Metric": k, "Value (mm)": v} for k, v in shape.items()],
            width="stretch",
        )

    if height_info:
        st.markdown("**Height normalization**")
        st.json(height_info)


def load_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    idx = part.split("/")[0]
                    face.append(int(idx) - 1)
                if len(face) >= 3:
                    faces.append(face[:3])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def render_mesh(path: Path) -> None:
    st.markdown("### Final Fitted Mesh")
    if not path.exists():
        st.warning("Mesh file not found. Run the pipeline first.")
        return

    verts, faces = load_obj_mesh(path)
    if verts.size == 0 or faces.size == 0:
        st.warning("Mesh data is empty or invalid.")
        return

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#7dd3fc",
                opacity=0.92,
                flatshading=True,
                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.6, roughness=0.35),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
    )
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    apply_app_theme()
    st.title(APP_TITLE)
    st.markdown(
        "<div class='muted'>Run the full pipeline, inspect anthropometric measurements, and visualize the final SMPL-X mesh.</div>",
        unsafe_allow_html=True,
    )

    control_col, info_col = st.columns([1, 2])
    with control_col:
        st.markdown("### Controls")
        actual_height = st.number_input(
            "Actual height (cm) for evaluation_v5",
            min_value=0.0,
            max_value=250.0,
            value=165.0,
            step=0.5,
        )
        run_all = st.button("Run Pipeline + Evaluations")
    with info_col:
        st.markdown("### What will run")
        st.markdown(
            """
            - `pipeline_v6.py` to build the fitted mesh
            - `evaluation_v4.py` for anthropometric measurements
            - `evaluation_v5.py` for height-normalized measurements
            """,
        )

    if run_all:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        code, _ = run_script_realtime(
            [sys.executable, str(PIPELINE_SCRIPT)],
            "Stage 1: Pipeline (pipeline_v6.py)",
        )
        if code != 0:
            st.error("Pipeline failed. Check the logs above.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        code, _ = run_script_realtime(
            [sys.executable, str(EVAL_V4_SCRIPT)],
            "Stage 2: Anthropometric Evaluation (evaluation_v4.py)",
        )
        if code != 0:
            st.error("evaluation_v4 failed. Check the logs above.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        code, _ = run_script_realtime(
            [sys.executable, str(EVAL_V5_SCRIPT), "--actual_height", str(actual_height)],
            "Stage 3: Scaled Evaluation (evaluation_v5.py)",
        )
        if code != 0:
            st.error("evaluation_v5 failed. Check the logs above.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_measurements("Anthropometric Results", read_json(ANTHRO_RESULTS))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_measurements("Scaled Anthropometric Results", read_json(SCALED_RESULTS))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_mesh(MESH_OBJ)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
