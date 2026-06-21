"""
app.py
------
Streamlit dashboard for the IR-to-RGB Satellite Image Colorization & Enhancement system.
Swiss International Style — flat, boxy, functional.
"""

import os
import sys
import io
import time
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.model import Generator
from src.dataset import SyntheticIRDataset, denormalize
from src.inference import load_generator, preprocess_array, preprocess_png, run_inference, compute_metrics, preprocess_array as _preprocess


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DrishtiIR — IR Colorization System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Swiss Design System ───────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,200..800&display=swap');

:root {
    --bg:           #080D1F;
    --surface:      #0D1530;
    --surface-2:    #111C3A;
    --border:       #1C2B4A;
    --border-hi:    #2A4070;
    --accent:       #F5821F;
    --accent-dim:   #7a3e08;
    --text:         #E8EDF8;
    --text-mid:     #7A8FAD;
    --text-low:     #3A4E6E;
    --mono:         'Bricolage Grotesque', sans-serif;
    --sans:         'Bricolage Grotesque', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background-color: var(--bg);
}



/* Streamlit slider override - force orange */
.stSlider div { color: #F5821F !important; }
input[type=range] { accent-color: #F5821F !important; }

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: var(--sans) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-right: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text-mid) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.4rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    border-color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.5rem !important;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: #e06a10 !important;
}

/* ── File Uploader ── */
.stFileUploader {
    border: 1px dashed var(--border-hi) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border-hi); }

/* ── Swiss Component Classes ── */

.sw-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 1.5rem;
}
.sw-wordmark {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1;
}
.sw-wordmark span {
    color: var(--accent);
}
.sw-tagline {
    font-family: var(--sans);
    font-size: 0.82rem;
    color: var(--text-mid);
    margin-top: 0.4rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sw-tag {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    margin-right: 8px;
}

.sw-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.sw-card-title {
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-mid);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin-bottom: 0.9rem;
}
.sw-card-title.red { color: var(--accent); border-color: var(--accent); }

.sw-metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border: 1px solid var(--border);
    margin: 1rem 0;
}
.sw-metric {
    padding: 1.2rem;
    border-right: 1px solid var(--border);
    text-align: center;
}
.sw-metric:last-child { border-right: none; }
.sw-metric-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-low);
    margin-bottom: 0.4rem;
}
.sw-metric-value {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
}
.sw-metric-value.red { color: var(--accent); }
.sw-metric-unit {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-low);
    margin-top: 2px;
}

.sw-img-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-mid);
    padding: 4px 8px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-bottom: none;
    display: inline-block;
}

.sw-pipeline-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}
.sw-pipeline-item:last-child { border-bottom: none; }
.sw-step-num {
    background: var(--accent);
    color: #fff;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sw-step-label { font-size: 0.82rem; font-weight: 500; color: var(--text); }
.sw-step-sub   { font-size: 0.72rem; color: var(--text-mid); font-family: var(--mono); }

.sw-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    border: 1px dashed var(--border-hi);
    background: var(--surface);
    color: var(--text-low);
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-align: center;
    gap: 8px;
}
.sw-empty-icon {
    font-size: 2rem;
    opacity: 0.4;
    margin-bottom: 4px;
}

.sw-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 0.78rem;
}
.sw-table th {
    text-align: left;
    padding: 6px 12px;
    background: var(--surface-2);
    color: var(--text-mid);
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.65rem;
    border-bottom: 1px solid var(--border-hi);
}
.sw-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.sw-table td.red { color: var(--accent); }
.sw-table td.mid { color: var(--text-mid); }

.sw-code {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 0.8rem 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-mid);
    margin: 0.5rem 0;
    line-height: 1.9;
}
.sw-code .cmd { color: var(--text); }
.sw-code .cmt { color: var(--text-low); }

.stDownloadButton > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
.stDownloadButton > button:hover {
    background: var(--accent) !important;
    color: #fff !important;
}

.stSlider > div { font-family: var(--mono) !important; }
.stSelectSlider > div { font-family: var(--mono) !important; }
.stCaption { font-family: var(--mono) !important; font-size: 0.72rem !important; color: var(--text-low) !important; }
.stInfo { border-radius: 0 !important; }
.stSuccess { border-radius: 0 !important; }

</style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/generator_latest.pth"

@st.cache_resource(show_spinner=False)
def load_model() -> Generator:
    return load_generator(CHECKPOINT_PATH, DEVICE)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="sw-header">
    <div style="display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:12px;">
        <div>
            <div class="sw-wordmark">Drishti<span>IR</span></div>
            <div class="sw-tagline">Infrared Satellite Image Colorization &amp; Enhancement</div>
        </div>
        <div>
            <span class="sw-tag">ISRO</span>
            <span class="sw-tag">PS-10</span>
            <span class="sw-tag">Bharatiya Antariksh Hackathon 2026</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'Bricolage Grotesque',monospace; font-size:1rem; font-weight:600; color:#E8EDF8; letter-spacing:-0.01em;">
            DRISHTI<span style="color:#F5821F;">IR</span>
        </div>
        <div style="font-family:'Bricolage Grotesque',monospace; font-size:0.65rem; color:#3A4E6E; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">
            Deep Learning Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sw-card-title" style="font-family:\'Bricolage Grotesque\',monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:#3A4E6E;border-bottom:1px solid #1C2B4A;padding-bottom:6px;margin:12px 0 10px;">Configuration</div>', unsafe_allow_html=True)

    tile_size = st.select_slider("Tile Size (px)", options=[128, 256, 512], value=256)
    st.caption(f"Resolution: {tile_size} x {tile_size}")

    st.markdown('<div class="sw-card-title" style="font-family:\'Bricolage Grotesque\',monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:#3A4E6E;border-bottom:1px solid #1C2B4A;padding-bottom:6px;margin:16px 0 10px;">Device</div>', unsafe_allow_html=True)
    device_label = "GPU / CUDA" if DEVICE == "cuda" else "CPU"
    device_color = "#F5821F" if DEVICE == "cuda" else "#7A8FAD"
    st.markdown(f'<div style="font-family:\'Bricolage Grotesque\',monospace;font-size:0.82rem;color:{device_color};font-weight:600;">{device_label}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sw-card-title" style="font-family:\'Bricolage Grotesque\',monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:#3A4E6E;border-bottom:1px solid #1C2B4A;padding-bottom:6px;margin:16px 0 10px;">Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Input", "GeoTIFF / PNG"),
        ("2", "Normalize", "Scale to [-1, 1]"),
        ("3", "SR Upscale", "100m → 30m"),
        ("4", "U-Net Forward", "IR → RGB"),
        ("5", "Denormalize", "Output [0, 255]"),
    ]
    pipeline_html = ""
    for num, title, sub in steps:
        pipeline_html += f"""
        <div class="sw-pipeline-item">
            <div class="sw-step-num">{num}</div>
            <div>
                <div class="sw-step-label">{title}</div>
                <div class="sw-step-sub">{sub}</div>
            </div>
        </div>"""
    st.markdown(pipeline_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid #1C2B4A;">
        <div style="font-family:'Bricolage Grotesque',monospace;font-size:0.65rem;color:#3A4E6E;line-height:2.1;">
            <div>Model&nbsp;&nbsp;&nbsp;&nbsp;Pix2Pix GAN</div>
            <div>Generator&nbsp;U-Net (54M params)</div>
            <div>Disc.&nbsp;&nbsp;&nbsp;&nbsp;PatchGAN 70x70</div>
            <div>Loss&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L1 + Adv + SSIM</div>
            <div>Dataset&nbsp;&nbsp;Landsat 8/9</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_single, tab_demo, tab_batch, tab_about = st.tabs([
    "Single Image", "Demo Mode", "Batch Evaluate", "About"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Single Image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_single:
    col_upload, col_output = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.markdown('<div class="sw-card"><div class="sw-card-title">Input IR Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a monochrome IR / Thermal image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="single_upload",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sw-card">
            <div class="sw-card-title">Tips</div>
            <ul style="color:#7A8FAD;font-size:0.8rem;padding-left:1.2rem;margin:0;font-family:'Bricolage Grotesque',monospace;line-height:2.2;">
                <li>Landsat 8/9 Band 10 (Thermal) works best</li>
                <li>Any grayscale PNG accepted for demo</li>
                <li>Optimal tile size: 256 x 256 px</li>
                <li>Run mock_weights.py for demo checkpoint</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_output:
        if uploaded is not None:
            with st.spinner("Running IR to RGB inference..."):
                gen = load_model()
                img_bytes = uploaded.getvalue()
                try:
                    import rasterio
                    from rasterio.io import MemoryFile
                    with MemoryFile(img_bytes) as memfile:
                        with memfile.open() as src:
                            img_arr = src.read(1)
                            if img_arr.ndim == 2:
                                img_arr = img_arr[:, :, np.newaxis] # add channel dim for cvtColor logic below if needed, though it's 1 channel
                except:
                    # Fallback for PNG/JPG
                    img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    if img_arr is None:
                        img_pil = Image.open(uploaded)
                        img_arr = np.array(img_pil)
                
                if img_arr.ndim == 3 and img_arr.shape[2] > 1:
                    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_arr.squeeze()
                ir_preprocessed = preprocess_array(img_gray, tile_size=tile_size)
                ir_disp, rgb_out, elapsed_ms = run_inference(gen, ir_preprocessed, DEVICE)

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown('<div class="sw-img-label">Input IR (Thermal)</div>', unsafe_allow_html=True)
                st.image(ir_disp, use_container_width=True, clamp=True)
            with img_col2:
                st.markdown('<div class="sw-img-label">Colorized RGB Output</div>', unsafe_allow_html=True)
                st.image(rgb_out, use_container_width=True, clamp=True)

            st.markdown(f"""
            <div class="sw-metric-row">
                <div class="sw-metric">
                    <div class="sw-metric-label">Inference Time</div>
                    <div class="sw-metric-value red">{elapsed_ms:.0f}</div>
                    <div class="sw-metric-unit">milliseconds</div>
                </div>
                <div class="sw-metric">
                    <div class="sw-metric-label">Model</div>
                    <div class="sw-metric-value" style="font-size:1rem;padding-top:0.5rem;">U-Net</div>
                    <div class="sw-metric-unit">Pix2Pix GAN</div>
                </div>
                <div class="sw-metric">
                    <div class="sw-metric-label">Device</div>
                    <div class="sw-metric-value" style="font-size:1rem;padding-top:0.5rem;">{"GPU" if DEVICE=="cuda" else "CPU"}</div>
                    <div class="sw-metric-unit">{DEVICE.upper()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            rgb_pil = Image.fromarray(rgb_out)
            buf = io.BytesIO()
            rgb_pil.save(buf, format="PNG")
            st.download_button(
                label="Download Colorized Output",
                data=buf.getvalue(),
                file_name="colorized_output.png",
                mime="image/png",
            )
        else:
            st.markdown("""
            <div class="sw-empty">
                <div class="sw-empty-icon">[IR]</div>
                <div>Upload an IR image to begin</div>
                <div style="color:#1C2B4A;margin-top:4px;">Supports Landsat GeoTIFF and PNG/JPEG</div>
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Demo Mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_demo:
    st.markdown("""
    <div class="sw-card">
        <div class="sw-card-title">Synthetic IR Scene Generator</div>
        <div style="color:#7A8FAD;font-size:0.82rem;font-family:'Bricolage Grotesque',monospace;">
            Procedurally generates a synthetic thermal scene and runs the colorization model in real time.<br>
            Use this to test without real Landsat data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    demo_col1, demo_col2 = st.columns([1, 2], gap="large")

    with demo_col1:
        seed_val    = st.slider("Scene Seed",      min_value=0,   max_value=999, value=42)
        num_blobs   = st.slider("Thermal Blobs",   min_value=1,   max_value=8,   value=4)
        noise_level = st.slider("Noise Level",     min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        run_demo    = st.button("Generate and Colorize", use_container_width=True)

    with demo_col2:
        if run_demo:
            with st.spinner("Generating scene and running inference..."):
                rng = np.random.default_rng(seed_val)
                H = W = tile_size
                background = rng.uniform(0.05, 0.35, (H, W)).astype(np.float32)
                for _ in range(num_blobs):
                    cx = rng.integers(20, W - 20)
                    cy = rng.integers(20, H - 20)
                    radius = rng.integers(12, 45)
                    intensity = rng.uniform(0.55, 1.0)
                    y_c, x_c = np.ogrid[:H, :W]
                    mask = (x_c - cx)**2 + (y_c - cy)**2 <= radius**2
                    background[mask] = np.maximum(background[mask], intensity)
                background += rng.normal(0, noise_level, background.shape).astype(np.float32)
                background = np.clip(background, 0, 1)
                ir_arr = (background * 2.0 - 1.0)[np.newaxis, :, :]
                gen = load_model()
                ir_disp, rgb_out, elapsed_ms = run_inference(gen, ir_arr, DEVICE)

            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.markdown('<div class="sw-img-label">Synthetic Thermal IR</div>', unsafe_allow_html=True)
                st.image(ir_disp, use_container_width=True, clamp=True)
            with d_col2:
                st.markdown('<div class="sw-img-label">Colorized RGB</div>', unsafe_allow_html=True)
                st.image(rgb_out, use_container_width=True, clamp=True)
            with d_col3:
                st.markdown(f"""
                <div class="sw-card" style="margin-top:0;">
                    <div class="sw-card-title">Run Stats</div>
                    <table class="sw-table">
                        <tr><td class="mid">Inference</td><td class="red">{elapsed_ms:.0f} ms</td></tr>
                        <tr><td class="mid">Seed</td><td>{seed_val}</td></tr>
                        <tr><td class="mid">Blobs</td><td>{num_blobs}</td></tr>
                        <tr><td class="mid">Noise</td><td>{noise_level:.2f}</td></tr>
                        <tr><td class="mid">Tile</td><td>{tile_size}x{tile_size}</td></tr>
                        <tr><td class="mid">Device</td><td>{DEVICE.upper()}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sw-empty">
                <div class="sw-empty-icon">[GEN]</div>
                <div>Click Generate to create a synthetic thermal scene</div>
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Batch Evaluate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_batch:
    st.markdown("""
    <div class="sw-card">
        <div class="sw-card-title">Batch Evaluation</div>
        <div style="color:#7A8FAD;font-size:0.82rem;font-family:'Bricolage Grotesque',monospace;">
            Upload multiple IR images. The model runs inference on each and aggregates statistics across the batch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple IR images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed",
    )

    if batch_files:
        gen = load_model()
        results = []
        progress_bar = st.progress(0, text="Processing images...")
        for i, f in enumerate(batch_files):
            img_pil = Image.open(f).convert("L")
            img_arr = np.array(img_pil)
            ir_pre = preprocess_array(img_arr, tile_size)
            ir_disp, rgb_out, ms = run_inference(gen, ir_pre, DEVICE)
            results.append({
                "filename": f.name,
                "inference_ms": round(ms, 1),
                "output": rgb_out,
                "ir": ir_disp,
            })
            progress_bar.progress((i + 1) / len(batch_files), text=f"Processed {i+1}/{len(batch_files)}")
        progress_bar.empty()
        st.success(f"Processed {len(results)} images")

        avg_ms = np.mean([r["inference_ms"] for r in results])
        total_s = sum(r["inference_ms"] for r in results) / 1000

        st.markdown(f"""
        <div class="sw-metric-row">
            <div class="sw-metric">
                <div class="sw-metric-label">Images Processed</div>
                <div class="sw-metric-value red">{len(results)}</div>
                <div class="sw-metric-unit">files</div>
            </div>
            <div class="sw-metric">
                <div class="sw-metric-label">Avg Inference Time</div>
                <div class="sw-metric-value red">{avg_ms:.0f}</div>
                <div class="sw-metric-unit">ms per image</div>
            </div>
            <div class="sw-metric">
                <div class="sw-metric-label">Total Time</div>
                <div class="sw-metric-value red">{total_s:.1f}</div>
                <div class="sw-metric-unit">seconds</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols_per_row = 4
        for row_start in range(0, len(results), cols_per_row):
            row_results = results[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_results))
            for col, r in zip(cols, row_results):
                with col:
                    st.markdown(f'<div class="sw-img-label">{r["filename"]}</div>', unsafe_allow_html=True)
                    combined = np.concatenate([
                        cv2.cvtColor(r["ir"], cv2.COLOR_GRAY2RGB),
                        r["output"]
                    ], axis=1)
                    st.image(combined, use_container_width=True)
                    st.caption(f"{r['inference_ms']} ms")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — About
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_about:
    ab1, ab2 = st.columns(2, gap="large")

    with ab1:
        st.markdown("""
        <div class="sw-card">
            <div class="sw-card-title red">Architecture</div>
            <table class="sw-table">
                <tr><th>Component</th><th>Detail</th></tr>
                <tr><td class="red">Generator</td><td>U-Net, 8-level enc-dec, skip connections, 54M params</td></tr>
                <tr><td class="red">Discriminator</td><td>PatchGAN, 70x70 patches, real/fake texture classification</td></tr>
                <tr><td class="red">SR Module</td><td>ESPCN PixelShuffle, 2x upscale, 100m to 30m</td></tr>
                <tr><td class="red">Input</td><td>1-channel IR, 256x256, normalized [-1, 1]</td></tr>
                <tr><td class="red">Output</td><td>3-channel RGB, 256x256, Tanh activation</td></tr>
            </table>
        </div>

        <div class="sw-card">
            <div class="sw-card-title red">Loss Functions</div>
            <div class="sw-code">
                <span class="cmt"># Generator</span><br>
                <span class="cmd">L_G = L_adv + 100 * L_L1 + 20 * (1 - SSIM)</span><br><br>
                <span class="cmt"># Discriminator</span><br>
                <span class="cmd">L_D = 0.5 * (L_real + L_fake)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ab2:
        st.markdown("""
        <div class="sw-card">
            <div class="sw-card-title red">Evaluation Metrics</div>
            <table class="sw-table">
                <tr><th>Metric</th><th>Target</th><th>Description</th></tr>
                <tr><td class="red">PSNR</td><td>&gt; 28 dB</td><td>Pixel-level reconstruction quality</td></tr>
                <tr><td class="red">SSIM</td><td>&gt; 0.85</td><td>Structural similarity preservation</td></tr>
                <tr><td class="red">FID</td><td>&lt; 50</td><td>Realism of generated images</td></tr>
                <tr><td class="red">Inference</td><td>&lt; 100 ms</td><td>Per-tile speed (CPU, 256x256)</td></tr>
            </table>
        </div>

        <div class="sw-card">
            <div class="sw-card-title red">Dataset — Landsat 8/9 (USGS)</div>
            <table class="sw-table">
                <tr><th>Band</th><th>Type</th><th>Resolution</th><th>Role</th></tr>
                <tr><td class="red">B10 TIRS-1</td><td>10.6-11.2 um Thermal</td><td>100m</td><td>IR Input</td></tr>
                <tr><td class="red">B4</td><td>Red 0.64-0.67 um</td><td>30m</td><td>RGB Target</td></tr>
                <tr><td class="red">B3</td><td>Green 0.53-0.59 um</td><td>30m</td><td>RGB Target</td></tr>
                <tr><td class="red">B2</td><td>Blue 0.45-0.51 um</td><td>30m</td><td>RGB Target</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sw-card">
        <div class="sw-card-title red">Quick Start Commands</div>
        <div class="sw-code">
            <span class="cmt"># Install dependencies</span><br>
            <span class="cmd">pip install -r requirements.txt</span><br><br>
            <span class="cmt"># Generate mock checkpoint for demo</span><br>
            <span class="cmd">python mock_weights.py</span><br><br>
            <span class="cmt"># Launch dashboard</span><br>
            <span class="cmd">python -m streamlit run app.py</span><br><br>
            <span class="cmt"># Train on synthetic data (no download needed)</span><br>
            <span class="cmd">python src/train.py --num-epochs 100 --batch-size 8</span><br><br>
            <span class="cmt"># Train on real Landsat data</span><br>
            <span class="cmd">python src/train.py --ir-dir data/train/ir --rgb-dir data/train/rgb --num-epochs 100</span><br><br>
            <span class="cmt"># CLI inference on a single image</span><br>
            <span class="cmd">python src/inference.py --input my_ir.png --output result.png</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<hr style="border-color:#1C2B4A; margin-top:2rem;">
<div style="display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0 1rem;">
    <div style="font-family:'Bricolage Grotesque',monospace;font-size:0.65rem;color:#2A4070;letter-spacing:0.08em;">
        DRISHTIIR &mdash; BHARATIYA ANTARIKSH HACKATHON 2026 &mdash; PROBLEM STATEMENT 10
    </div>
    <div style="font-family:'Bricolage Grotesque',monospace;font-size:0.65rem;color:#2A4070;letter-spacing:0.08em;">
        PYTORCH &middot; PIX2PIX GAN &middot; LANDSAT 8/9
    </div>
</div>
""", unsafe_allow_html=True)
