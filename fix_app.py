"""
fix_app.py
Inserts GAN vs Diffusion + Object Detection tab bodies into app.py
"""

GAN_VS_DIFF_TAB = '''

# ============================================================
# TAB 5 -- GAN vs Diffusion Model Comparison
# ============================================================

with tab_compare:
    st.markdown("""
    <div class="sw-card">
        <div class="sw-card-title">GAN vs Diffusion -- Side-by-Side Model Comparison</div>
        <div style="color:#7A8FAD;font-size:0.82rem;font-family:'Bricolage Grotesque',monospace;">
            Run both the Pix2Pix GAN and the Conditional Diffusion Model on the same IR image.
            Compare colorization quality, inference time, and view a hallucination deviation heatmap.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cmp_upload = st.file_uploader(
        "Upload IR image for model comparison",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="cmp_upload",
        label_visibility="collapsed",
    )
    ddim_steps_cmp = st.slider("DDIM Steps (Diffusion quality vs speed)", 10, 100, 50, 5)

    if cmp_upload:
        with st.spinner("Running both models..."):
            img_pil = Image.open(cmp_upload).convert("L")
            ir_arr  = np.array(img_pil)
            ir_pre  = preprocess_array(ir_arr, tile_size)
            land_mask = classify_landcover(ir_pre)
            gan_model = load_model()
            ir_disp_gan, gan_rgb, gan_ms = run_inference(gan_model, ir_pre, DEVICE)
            gan_rgb = apply_post_processing(gan_rgb)
            if semantic_strength > 0:
                gan_rgb = apply_semantic_correction(gan_rgb, land_mask, strength=semantic_strength)
            diff_available = os.path.exists(DIFFUSION_CKPT)
            if diff_available:
                ir_disp_d, diff_rgb, diff_ms = run_diffusion_inference_app(ir_pre, ddim_steps=ddim_steps_cmp)
                if diff_rgb is not None and semantic_strength > 0:
                    diff_rgb = apply_semantic_correction(diff_rgb, land_mask, strength=semantic_strength)
            else:
                diff_rgb, diff_ms = None, 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(\'<div class="sw-img-label">Input Thermal IR</div>\', unsafe_allow_html=True)
            st.image(ir_disp_gan, use_container_width=True, clamp=True)
            st.caption("Landsat B10 Thermal (100m)")
        with c2:
            st.markdown(\'<div class="sw-img-label">Pix2Pix GAN</div>\', unsafe_allow_html=True)
            st.image(gan_rgb, use_container_width=True, clamp=True)
            st.caption(f"U-Net Generator -- {gan_ms:.0f} ms")
        with c3:
            if diff_rgb is not None:
                st.markdown(\'<div class="sw-img-label">Diffusion EMA (DDIM+CFG)</div>\', unsafe_allow_html=True)
                st.image(diff_rgb, use_container_width=True, clamp=True)
                st.caption(f"DDPM {ddim_steps_cmp}-step DDIM -- {diff_ms:.0f} ms")
            else:
                st.markdown(\'<div class="sw-empty"><div class="sw-empty-icon">[DIFF]</div><div>Train on Kaggle then download diffusion_ema_latest.pth</div></div>\', unsafe_allow_html=True)

        sem_color = get_semantic_color_map(land_mask)
        total = land_mask.size
        w_pct = round(100 * (land_mask == 1).sum() / total, 1)
        v_pct = round(100 * (land_mask == 2).sum() / total, 1)
        u_pct = round(100 * (land_mask == 3).sum() / total, 1)
        sm1, sm2 = st.columns([1, 3])
        with sm1:
            st.markdown(\'<div class="sw-img-label" style="margin-top:1rem;">Semantic Land Cover</div>\', unsafe_allow_html=True)
            st.image(sem_color, use_container_width=True)
        with sm2:
            st.markdown(f"""
            <div class="sw-card" style="margin-top:1rem;">
                <div class="sw-card-title">Spectral Breakdown (B10 Thermal / B6 SWIR / B5 NIR)</div>
                <div style="font-family:\'Bricolage Grotesque\',monospace;font-size:0.85rem;line-height:2.5;">
                    Water: <b style="color:#2196F3">{w_pct}%</b> &nbsp;|&nbsp;
                    Vegetation: <b style="color:#4CAF50">{v_pct}%</b> &nbsp;|&nbsp;
                    Urban/Bare: <b style="color:#9E9E9E">{u_pct}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if diff_rgb is not None:
            st.markdown(\'<div class="sw-card-title" style="margin-top:1.5rem;font-family:None;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:#7A8FAD;">Hallucination Analysis -- Inter-Model Disagreement Heatmap</div>\', unsafe_allow_html=True)
            deviation = np.abs(gan_rgb.astype(np.float32) - diff_rgb.astype(np.float32)).mean(axis=2)
            dev_norm  = (deviation / max(deviation.max(), 1) * 255).astype(np.uint8)
            heatmap   = cv2.applyColorMap(dev_norm, cv2.COLORMAP_INFERNO)
            heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            h1, h2, h3 = st.columns(3)
            with h1:
                st.image(gan_rgb,  use_container_width=True, caption="GAN Output")
            with h2:
                st.image(diff_rgb, use_container_width=True, caption="Diffusion Output")
            with h3:
                st.image(heatmap,  use_container_width=True, caption="Disagreement Heatmap (bright = hallucination risk)")
            st.caption(f"Mean deviation: {deviation.mean():.1f}/255. Low = both models agree = low hallucination risk.")
    else:
        st.markdown(\'<div class="sw-empty"><div class="sw-empty-icon">[CMP]</div><div>Upload an IR image to compare GAN vs Diffusion</div></div>\', unsafe_allow_html=True)


# ============================================================
# TAB 6 -- Object Detection Benchmark (Downstream Task)
# ============================================================

with tab_detect:
    st.markdown("""
    <div class="sw-card">
        <div class="sw-card-title">Downstream Object Detection Benchmark</div>
        <div style="color:#7A8FAD;font-size:0.82rem;font-family:\'Bricolage Grotesque\',monospace;">
            Proves PS-10: colorization boosts downstream CV tasks.
            Faster-RCNN (ResNet-50, COCO-pretrained) detects objects in raw IR vs colorized RGB.
        </div>
    </div>
    """, unsafe_allow_html=True)

    det_upload = st.file_uploader(
        "Upload IR image for detection benchmark",
        type=["png", "jpg", "jpeg"],
        key="det_upload",
        label_visibility="collapsed",
    )
    det_threshold = st.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

    if det_upload:
        with st.spinner("Colorizing image..."):
            img_pil   = Image.open(det_upload).convert("L")
            ir_arr    = np.array(img_pil)
            ir_pre    = preprocess_array(ir_arr, tile_size)
            gan_model = load_model()
            ir_disp, rgb_out, _ = run_inference(gan_model, ir_pre, DEVICE)
            rgb_out   = apply_post_processing(rgb_out)
            land_mask = classify_landcover(ir_pre)
            if semantic_strength > 0:
                rgb_out = apply_semantic_correction(rgb_out, land_mask, strength=semantic_strength)

        if DETECTION_AVAILABLE:
            with st.spinner("Running Faster-RCNN (downloads approx 170MB on first run)..."):
                from src.detection import compare_detection as _cdet
                det_results = _cdet(ir_disp, rgb_out, DEVICE, det_threshold)

            ir_det  = det_results["ir"]
            rgb_det = det_results["rgb"]
            delta   = det_results["delta_count"]
            delta_color = "#4CAF50" if delta > 0 else "#F5821F" if delta == 0 else "#ef5350"
            delta_sign  = "+" if delta >= 0 else ""

            st.markdown(f"""
            <div class="sw-card" style="margin-top:1rem;">
                <div class="sw-card-title red">Faster-RCNN Results -- IR vs Colorized RGB</div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;border:1px solid #1C2B4A;margin-top:0.8rem;">
                    <div style="padding:1.2rem;text-align:center;border-right:1px solid #1C2B4A;">
                        <div class="sw-metric-label">IR Detections</div>
                        <div class="sw-metric-value" style="font-size:2rem;">{ir_det["count"]}</div>
                        <div class="sw-metric-unit">objects</div>
                    </div>
                    <div style="padding:1.2rem;text-align:center;border-right:1px solid #1C2B4A;">
                        <div class="sw-metric-label">RGB Detections</div>
                        <div class="sw-metric-value" style="font-size:2rem;">{rgb_det["count"]}</div>
                        <div class="sw-metric-unit">objects</div>
                    </div>
                    <div style="padding:1.2rem;text-align:center;border-right:1px solid #1C2B4A;">
                        <div class="sw-metric-label">Delta</div>
                        <div class="sw-metric-value" style="color:{delta_color};font-size:2rem;">{delta_sign}{delta}</div>
                        <div class="sw-metric-unit">{"Colorization helped!" if delta > 0 else "No change"}</div>
                    </div>
                    <div style="padding:1.2rem;text-align:center;">
                        <div class="sw-metric-label">Avg Confidence</div>
                        <div class="sw-metric-value" style="font-size:1.5rem;">{rgb_det["mean_conf"]:.2f}</div>
                        <div class="sw-metric-unit">RGB vs {ir_det["mean_conf"]:.2f} IR</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(\'<div class="sw-img-label">Raw IR -- Detections</div>\', unsafe_allow_html=True)
                st.image(ir_det["annotated"], use_container_width=True, clamp=True)
                st.caption(f"{ir_det[\'count\']} objects | {\', \'.join(set(ir_det[\'labels\'][:4])) or \'none\'}")
            with d2:
                st.markdown(\'<div class="sw-img-label">Colorized RGB -- Detections</div>\', unsafe_allow_html=True)
                st.image(rgb_det["annotated"], use_container_width=True, clamp=True)
                st.caption(f"{rgb_det[\'count\']} objects | {\', \'.join(set(rgb_det[\'labels\'][:4])) or \'none\'}")
        else:
            st.error("Install torchvision>=0.16.0 to enable detection benchmark.")
    else:
        st.markdown(\'<div class="sw-empty"><div class="sw-empty-icon">[DET]</div><div>Upload IR image for downstream detection benchmark</div></div>\', unsafe_allow_html=True)

'''

EVAL_HEADER = '''

# ============================================================
# TAB 7 -- Evaluation (ISRO PS-10 Metrics)
# ============================================================

with tab_eval:'''

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the old TAB 5 marker
import re
# Match the unicode bar line before "TAB 5"
old_section = (
    "\n\n# \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
    "# TAB 5 \u2014 Evaluation (ISRO PS-10 Metrics)\n"
    "# \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
    "\nwith tab_eval:"
)

if old_section in content:
    content = content.replace(old_section, GAN_VS_DIFF_TAB + EVAL_HEADER, 1)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Inserted GAN vs Diffusion and Detection tabs")
else:
    print("ERROR: Could not find old marker")
    idx = content.find("with tab_eval:")
    print(f"with tab_eval: found at index {idx}")
    print(repr(content[max(0,idx-200):idx+20]))
