# =========================================
# UNISOLE AI LABS — Unified ML Studio (Pro UX)
# Single-file app with sidebar navigation
# =========================================

import os, io, uuid, json, time
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

# ---------- Tabular AutoML ----------
from automl_tabular import (
    quick_eda, detect_task, run_automl,
    plot_classification_curves, plot_regression_plots
)

# ---------- EDA ----------
from eda_tools import (
    overview, plot_missingness, target_summary, plot_target_distribution,
    correlation_matrix, plot_corr_heatmap, top_corr_with_target,
    plot_numeric_distributions, plot_categorical_bars,
    outlier_report, plot_boxplots, class_imbalance, leakage_scan
)

# ---------- Explainability ----------
from explainability import (
    global_feature_importance, build_shap_explainer,
    shap_summary_fig, shap_single_waterfall_fig, pick_interesting_row
)

# ---------- Feature Engineering ----------
from feature_engineering import (
    FEConfig, default_config, fit_feature_pipeline,
    transform_with_pipeline, export_processed_dataframe
)

# ---------- LLM Mentor ----------
from llm_helper import explain_dataset, explain_model_choice, explain_metrics, ask_ai_mentor

# ---------- CV (Image Classification) ----------
from cv_trainer import (
    CVTrainConfig, best_device, build_loaders, build_model, train_loop,
    plot_history, save_model, load_model_for_infer, predict_image, make_transforms
)

from cv_data_prep import (
    infer_labels_auto, apply_csv_labels, build_org_structure,
    add_manual_labels, zip_directory, small_dataset_hint,
    _extract_zip_to_tmp, _scan_images
)

from cv_notebook_builder import build_cv_notebook_bytes

# ---------- Notebook Export (Tabular) ----------
from notebook_builder import build_notebook_bytes

# ---------- YOLO (Object Detection) ----------
from yolo_trainer import (
    extract_zip_to_dir, find_data_yaml, train_yolo,
    infer_image_bytes, infer_video_file
)


# =========================
# Page config + Directories
# =========================
st.set_page_config(page_title="UNISOLE AI LABS", layout="wide", page_icon="🧪")

for folder in ["uploads", "models", "reports", "pipelines", "runs"]:
    os.makedirs(folder, exist_ok=True)

# =========================
# Minimal dark styling
# =========================
STYLES = """
<style>
/* clean pro dark tweaks */
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; }
.sidebar .sidebar-content { background: #0e1117 !important; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: .6rem 0 1.2rem; }
.unisole-banner {
  padding: 10px 16px; border-radius: 12px;
  background: linear-gradient(135deg, #0f172a 0%, #0b1220 100%);
  border: 1px solid rgba(255,255,255,.08); margin-bottom: 10px;
}
.unisole-title {
  font-size: 22px; font-weight: 700; letter-spacing: .5px;
}
.unisole-sub {
  font-size: 12px; opacity: .75; margin-top: 2px;
}
.small-cap { font-size: 12px; opacity: .7; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

def banner():
    st.markdown(
        """
        <div class="unisole-banner">
          <div class="unisole-title">UNISOLE AI LABS</div>
          <div class="unisole-sub">Unified platform for Data Lab • AutoML • Vision • Explainability • Notebooks • Mentor</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Session State
# =========================
SS = st.session_state
defaults = {
    "df": None,                  # raw dataframe
    "processed_df": None,        # FE-processed dataframe (with target)
    "feature_pipe": None,
    "feature_summary": None,
    "last_result": None,         # AutoML result dict
    "last_task": None,
    "target": None,
    "chat": [],
    # CV
    "cv_raw_dir": None,
    "cv_img_paths": [],
    "cv_mapping": {},
    "cv_unknowns": [],
    "cv_org_dir": None,
    "cv_classes": [],
    "cv_small_hint": {},
    "cv_label_names": None,
    "cv_is_labeled": False,
    "cv_labeled_root": None,
    "cv_last_model": None,
    "cv_last_labels": None,
    "cv_model_name": "resnet18",
    # YOLO
    "yolo_data_dir": None,
    "yolo_data_yaml": None,
    "yolo_best": None,
    "yolo_names": None,
}
for k,v in defaults.items():
    if k not in SS: SS[k] = v

# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("**📊 Data Lab**")
    page_data = st.radio(
        "Data Lab",
        ["📥 Upload Data", "📊 EDA Dashboard", "🔧 Feature Engineering"],
        label_visibility="collapsed",
        key="nav_data"
    )
    st.markdown("**⚙️ Machine Learning**")
    page_ml = st.radio(
        "Machine Learning",
        ["🤖 AutoML (Tabular)", "⚡ Explainability", "📓 Notebook Export"],
        label_visibility="collapsed",
        key="nav_ml"
    )
    st.markdown("**🖼️ Vision AI**")
    page_cv = st.radio(
        "Vision AI",
        ["🖼️ Image Classification", "🧿 Object Detection (YOLO)"],
        label_visibility="collapsed",
        key="nav_cv"
    )
    st.markdown("**🧠 Mentor**")
    page_mentor = st.radio(
        "Mentor",
        ["🧑‍🏫 AI Learning Assistant"],
        label_visibility="collapsed",
        key="nav_mentor"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Global Settings")
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("CV Folds", 2, 10, 5, 1)
    max_time = st.slider("Max Train Time (minutes)", 1, 60, 10, 1)
    edu_mode = st.toggle("🎓 Education Mode", value=True)

# ============== Helper UI ==============
def page_header(title: str):
    banner()
    st.markdown(f"### {title}")
    st.markdown("<hr/>", unsafe_allow_html=True)

# =========================================================
# Pages — DATA LAB
# =========================================================
def page_upload():
    page_header("📥 Upload Data")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        file_id = str(uuid.uuid4())[:8]
        file_path = os.path.join("uploads", f"data_{file_id}.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Saved as `{file_path}`")
        df = pd.read_csv(file_path)
        SS.df = df
        SS.processed_df = None
        SS.feature_pipe = None
        SS.feature_summary = None
        SS.last_result = None
        st.write("#### Preview")
        st.dataframe(df.head(), use_container_width=True)
        if edu_mode and st.button("🧠 Explain this dataset"):
            st.info(explain_dataset(quick_eda(df), sample_schema={c: str(df[c].dtype) for c in df.columns}))

def page_eda():
    page_header("📊 EDA Dashboard")
    if SS.df is None:
        st.warning("Upload a CSV first in Data Lab → Upload Data.")
        return
    df = SS.df
    ov = overview(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", ov["rows"]); c2.metric("Columns", ov["cols"]); c3.metric("Duplicates", ov["duplicates"])

    st.write("#### Missing Values")
    st.pyplot(plot_missingness(df), clear_figure=True)
    if edu_mode and st.button("🧠 Explain Missing Data"):
        st.info(ask_ai_mentor("Explain missing data handling methods and trade-offs.", {"missing": ov["missing_perc"]}))

    st.write("---")
    target_sel = st.selectbox("🎯 Target column (optional)", options=df.columns)
    if target_sel:
        st.pyplot(plot_target_distribution(df, target_sel), clear_figure=True)

    st.write("---")
    st.write("#### Correlation Heatmap (numeric)")
    corr = correlation_matrix(df)
    st.pyplot(plot_corr_heatmap(corr), clear_figure=True)

    st.write("---")
    st.write("#### Numeric Distributions")
    for fig in plot_numeric_distributions(df, max_cols=6): st.pyplot(fig, clear_figure=True)

    cat_cols = ov["categorical_cols"]
    if len(cat_cols) > 0:
        st.write("---")
        st.write("#### Categorical Insights")
        for fig in plot_categorical_bars(df, max_cols=6, top_n=12): st.pyplot(fig, clear_figure=True)

def page_feature_eng():
    page_header("🔧 Feature Engineering")
    if SS.df is None:
        st.warning("Upload a CSV first.")
        return
    df = SS.df
    target = st.selectbox("Target column", options=df.columns, key="fe_target")
    if not target: return
    SS.target = target

    c1, c2 = st.columns(2)
    with c1:
        num_impute = st.selectbox("Numeric imputation", ["median","mean","most_frequent"], index=0)
    with c2:
        cat_impute = st.selectbox("Categorical imputation", ["most_frequent"], index=0)

    c3, c4 = st.columns(2)
    with c3:
        enable_rare = st.toggle("Group rare categories", value=True)
    with c4:
        rare_frac = st.slider("Rare min fraction", 0.0, 0.1, 0.01, 0.005)

    c5, c6 = st.columns(2)
    with c5:
        fix_skew = st.toggle("Fix skew (log1p)", value=True)
    with c6:
        skew_thr = st.slider("Skew threshold (|skew|≥)", 0.25, 2.0, 0.75, 0.05)

    scale_opt = st.selectbox("Scaling", ["standard", "minmax", "none"], index=0)
    encode_opt = st.selectbox("Encoding", ["onehot", "none"], index=0)

    c7, c8 = st.columns(2)
    with c7:
        low_var = st.toggle("Remove low-variance", value=True)
    with c8:
        var_thr = st.number_input("Variance threshold", min_value=0.0, value=0.0, step=0.01)

    if st.button("🚀 Build Pipeline & Transform"):
        cfg = FEConfig(
            num_impute=num_impute, cat_impute=cat_impute, encode=encode_opt,
            enable_rare_group=enable_rare, rare_min_frac=rare_frac,
            scale_numeric=scale_opt, fix_skew=fix_skew, skew_threshold=skew_thr,
            enable_low_variance=low_var, variance_threshold=var_thr
        )
        with st.spinner("Fitting feature pipeline…"):
            pipe, X_out, names, summary = fit_feature_pipeline(df.copy(), target=target, config=cfg)
            proc_df = export_processed_dataframe(pipe, df.copy(), target=target)
        SS.feature_pipe = pipe; SS.feature_summary = summary; SS.processed_df = proc_df
        st.success("Pipeline built & dataset transformed!")
        st.json(summary); st.dataframe(proc_df.head(), use_container_width=True)

    if SS.processed_df is not None:
        colA, colB = st.columns(2)
        with colA:
            if st.button("💾 Save Pipeline (.pkl)"):
                pid = str(uuid.uuid4())[:8]
                path = os.path.join("pipelines", f"feature_pipeline_{pid}.pkl")
                joblib.dump(SS.feature_pipe, path)
                st.success(f"Saved pipeline to `{path}`")
        with colB:
            buf = io.StringIO(); SS.processed_df.to_csv(buf, index=False)
            st.download_button("⬇️ Download Cleaned CSV", data=buf.getvalue(), file_name="cleaned_dataset.csv", mime="text/csv")

# =========================================================
# Pages — MACHINE LEARNING
# =========================================================
def page_automl():
    page_header("🤖 AutoML (Tabular)")
    if SS.df is None:
        st.warning("Upload a CSV first.")
        return
    df = SS.processed_df if SS.processed_df is not None else SS.df
    target = st.selectbox("Target (what to predict)", options=df.columns, key="auto_target")
    if not target: return
    SS.target = target
    task = detect_task(df, target); SS.last_task = task
    st.info(f"Detected task: **{task.upper()}**")

    if st.button("🚀 Run AutoML"):
        with st.spinner("Training models…"):
            start = time.time()
            result = run_automl(
                df=df, target=target, task=task, seed=seed,
                test_size=test_size, cv_folds=cv_folds, max_minutes=max_time
            )
            elapsed = time.time() - start
        SS.last_result = result
        st.success(f"Training complete in {elapsed/60:.1f} min")
        st.subheader("🏆 Best Model Summary")
        st.json(result["summary"])
        st.markdown("#### 📊 Visualizations")
        if task == "classification":
            for fig in plot_classification_curves(result): st.pyplot(fig, clear_figure=True)
        else:
            for fig in plot_regression_plots(result): st.pyplot(fig, clear_figure=True)
        uid = str(uuid.uuid4())[:8]
        model_id = f"{result['summary']['best_model_name']}_{uid}"
        model_path = os.path.join("models", f"{model_id}.pkl")
        card_path = os.path.join("reports", f"{model_id}.json")
        joblib.dump(result["artifact"], model_path)
        with open(card_path, "w") as f: json.dump(result["summary"], f, indent=2)
        st.download_button("Download model (.pkl)", data=open(model_path,"rb").read(), file_name=os.path.basename(model_path))
        st.download_button("Download model card (.json)", data=open(card_path,"rb").read(), file_name=os.path.basename(card_path))
        if edu_mode:
            colA, colB = st.columns(2)
            with colA:
                if st.button("🤔 Why was this model chosen?"):
                    st.info(explain_model_choice(result["summary"]))
            with colB:
                if st.button("📐 Explain these metrics"):
                    st.info(explain_metrics(result["summary"]))

def page_explain():
    page_header("⚡ Explainability")
    if SS.last_result is None or SS.df is None or SS.target is None:
        st.warning("Train a model first in Machine Learning → AutoML.")
        return
    df = SS.df; target = SS.target; artifact = SS.last_result["artifact"]
    X = df.drop(columns=[target])

    st.markdown("#### 1) Global Feature Importance")
    try:
        fig_imp, imp_df = global_feature_importance(artifact, X, top_n=20)
        st.pyplot(fig_imp, clear_figure=True); st.dataframe(imp_df, use_container_width=True)
    except Exception as e:
        st.error(f"Global importance failed: {e}")

    st.markdown("---")
    st.markdown("#### 2) SHAP Summary (Global)")
    try:
        explainer, Xt_sample, feat_names = build_shap_explainer(artifact, X, sample_size=500)
        st.pyplot(shap_summary_fig(explainer, Xt_sample, feat_names), clear_figure=True)
    except Exception as e:
        st.error(f"SHAP summary failed: {e}")

    st.markdown("---")
    st.markdown("#### 3) Single Prediction")
    c1, c2 = st.columns(2)
    with c1:
        row_idx = st.number_input("Row index", 0, len(X)-1, 0, 1)
        btn = st.button("🔍 Explain row")
    with c2:
        strat = st.selectbox("Auto pick", ["max", "min"], index=0)
        btn_auto = st.button("🎯 Auto-pick & explain")

    if btn:
        try:
            explainer, _, feat_names = build_shap_explainer(artifact, X, sample_size=500)
            fig_one, contribs = shap_single_waterfall_fig(explainer, artifact, X, int(row_idx))
            st.pyplot(fig_one, clear_figure=True); st.write(contribs)
        except Exception as e:
            st.error(f"Single explanation failed: {e}")
    if btn_auto:
        try:
            idx = pick_interesting_row(artifact, X, strategy=strat)
            explainer, _, feat_names = build_shap_explainer(artifact, X, sample_size=500)
            fig_one, contribs = shap_single_waterfall_fig(explainer, artifact, X, idx)
            st.info(f"Auto-picked row: {idx}")
            st.pyplot(fig_one, clear_figure=True); st.write(contribs)
        except Exception as e:
            st.error(f"Auto-pick failed: {e}")

def page_notebook():
    page_header("📓 Notebook Export (Tabular, Education Mode)")
    if SS.df is None or SS.target is None:
        st.warning("Upload data and choose a target first.")
        return
    include_mini_eda = st.toggle("Include Mini-EDA", value=True)
    include_ai_intro = st.toggle("Include AI Mentor intro", value=True)

    model_path = None; model_card = None
    if SS.last_result:
        model_card = SS.last_result.get("summary")

    pipe_path = None
    if SS.feature_pipe is not None and st.button("💾 Save current pipeline for notebook"):
        pid = str(uuid.uuid4())[:8]
        path = os.path.join("pipelines", f"feature_pipeline_{pid}.pkl")
        joblib.dump(SS.feature_pipe, path)
        st.success(f"Saved pipeline to `{path}`"); pipe_path = path

    model_path = st.text_input("Saved model .pkl (optional)", value=model_path or "")
    pipe_path  = st.text_input("Saved feature pipeline .pkl (optional)", value=pipe_path or "")

    ai_intro_text = None
    if include_ai_intro:
        try:
            ai_intro_text = ask_ai_mentor("Write a short friendly notebook intro (6–10 lines).", {"task": SS.last_task, "target": SS.target})
        except Exception:
            ai_intro_text = None

    if st.button("📥 Generate Notebook"):
        csv_guess = ""
        try:
            files = sorted([os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".csv")], key=os.path.getmtime, reverse=True)
            if files: csv_guess = files[0]
        except Exception: pass

        nb_bytes = build_notebook_bytes(
            csv_path=csv_guess or "YOUR_DATA.csv",
            target=SS.target,
            task=SS.last_task or "classification",
            model_card=model_card,
            saved_model_path=(model_path or "").strip() or None,
            saved_pipeline_path=(pipe_path or "").strip() or None,
            include_mini_eda=include_mini_eda,
            include_llm_intro=ai_intro_text,
            kernel_name="auto_ml"
        )
        st.download_button("⬇️ Download Notebook (.ipynb)", data=nb_bytes, file_name="UNISOLE_AutoAI_Notebook.ipynb", mime="application/x-ipynb+json")

# =========================================================
# Pages — VISION AI: Image Classification (flexible)
# =========================================================
def page_cv():
    page_header("🖼️ Image Classification — Flexible Import")
    # Upload any zip
    upzip = st.file_uploader("Upload dataset (.zip, any structure)", type=["zip"])
    if upzip and st.button("📦 Extract ZIP"):
        raw_dir = _extract_zip_to_tmp(upzip.read(), base_dir="uploads")
        imgs = _scan_images(raw_dir)
        if not imgs: st.error("No images found."); return
        SS.cv_raw_dir = raw_dir; SS.cv_img_paths = imgs

        # try detecting labeled root by subfolders (handled in previous convo code)
        from torchvision import datasets
        # find a directory that ImageFolder can read with >=2 classes
        labeled_root = None
        for r, dirs, _ in os.walk(raw_dir):
            if len(dirs) >= 2:
                try:
                    tmp = datasets.ImageFolder(r)
                    if len(tmp.classes) >= 2:
                        labeled_root = r; break
                except Exception: pass
        if labeled_root:
            SS.cv_is_labeled = True
            SS.cv_labeled_root = labeled_root
            SS.cv_org_dir = labeled_root
            SS.cv_classes = datasets.ImageFolder(labeled_root).classes
            st.success(f"Labeled dataset detected. Preserving structure at: {labeled_root}")
            st.write("Classes (preserved):", SS.cv_classes)
        else:
            SS.cv_is_labeled = False
            st.success(f"Extracted {len(imgs)} images. Not labeled yet — choose labeling below.")

    # Unlabeled → labeling flow
    if SS.cv_raw_dir and not SS.cv_is_labeled:
        st.markdown("#### Labeling Options")
        mode = st.radio("Mode", ["Auto (filenames)", "Use labels.csv", "Manual fix unknowns"], index=0)
        mapping = {p: None for p in SS.cv_img_paths}
        if mode == "Auto (filenames)":
            mapping = infer_labels_auto(SS.cv_img_paths); st.info("Auto-labeled from filenames.")
        elif mode == "Use labels.csv":
            csv_up = st.file_uploader("Upload labels.csv (filename,label)", type=["csv"], key="labels_csv")
            if csv_up:
                mapping = apply_csv_labels(SS.cv_img_paths, csv_up.getvalue().decode("utf-8", "ignore"))
                st.success("Applied CSV labels where matched.")
        if st.button("🗂️ Organize into class folders"):
            org_dir, report, unknowns = build_org_structure(mapping, out_base="uploads")
            SS.cv_org_dir = org_dir; SS.cv_mapping = mapping; SS.cv_unknowns = unknowns
            SS.cv_classes = sorted(list(report.class_counts.keys()))
            st.success(f"Organized {report.labeled}/{report.total_images}; unknown: {report.unknown}")
            SS.cv_small_hint = small_dataset_hint(report.class_counts, min_per_class=50)
            if SS.cv_small_hint: st.warning(f"Small dataset detected. Suggested extras: {SS.cv_small_hint}")

        if SS.cv_unknowns:
            st.markdown("#### Manual Label Unknowns")
            edits = {}
            cols = st.columns(4); max_show = min(24, len(SS.cv_unknowns))
            for i, p in enumerate(SS.cv_unknowns[:max_show]):
                with cols[i % 4]:
                    try: st.image(p, caption=os.path.basename(p), use_container_width=True)
                    except Exception: st.write(os.path.basename(p))
                    lab = st.text_input(f"Label for {os.path.basename(p)}", key=f"cv_fix_{i}")
                    edits[p] = lab.strip()
            if st.button("✅ Apply Manual Labels"):
                new_map = add_manual_labels(SS.cv_mapping, edits)
                org_dir2, report2, unknowns2 = build_org_structure(new_map, out_base="uploads")
                SS.cv_org_dir = org_dir2; SS.cv_mapping = new_map; SS.cv_unknowns = unknowns2
                SS.cv_classes = sorted(list(report2.class_counts.keys()))
                st.success(f"Updated: {report2.labeled} labeled, {report2.unknown} unknown left.")

    # If we have organized or preserved data → rename + train
    if SS.cv_org_dir:
        st.markdown("#### (Optional) Rename Display Names")
        from torchvision import datasets
        classes_for_training = datasets.ImageFolder(SS.cv_org_dir).classes
        renamed = []
        cols = st.columns(2)
        for i, cname in enumerate(classes_for_training):
            with cols[i % 2]:
                new = st.text_input(f"Rename '{cname}' →", value=(SS.cv_label_names[i] if (SS.cv_label_names and i < len(SS.cv_label_names)) else cname), key=f"cv_rename_{i}")
                renamed.append(new if new.strip() else cname)
        if st.button("💾 Save display names"):
            SS.cv_label_names = renamed
            st.success("Saved display names for labels.json")

        st.markdown("#### Download Organized Dataset")
        zbytes = zip_directory(SS.cv_org_dir)
        st.download_button("⬇️ Download organized dataset (.zip)", data=zbytes, file_name="organized_dataset.zip")

        st.markdown("---")
        st.subheader("Train Classifier")
        model_name = st.selectbox("Backbone", ["resnet18", "efficientnet_b0", "mobilenet_v3_small"], index=0)
        img_size   = st.selectbox("Image Size", [224, 256, 299], index=0)
        batch_size = st.number_input("Batch Size", 1, 128, 16, 1)
        epochs     = st.number_input("Epochs", 1, 100, 6, 1)
        lr         = st.number_input("Learning rate", 1e-5, 1.0, 1e-3, format="%.5f")
        val_split  = st.slider("Validation split", 0.05, 0.5, 0.2, 0.05)
        auto_boost = st.toggle("Auto-Boost Augmentation (small datasets)", value=bool(SS.cv_small_hint))
        augment    = st.toggle("Enable augmentation", value=True)
        device_choice = st.selectbox("Device", ["auto","cpu","mps","cuda"], index=0)
        seed_local    = st.number_input("Seed", 0, 99999, seed, 1)

        if st.button("🚀 Train Model"):
            try:
                cfg = CVTrainConfig(
                    model_name=model_name, img_size=int(img_size), batch_size=int(batch_size),
                    lr=float(lr), epochs=int(epochs), val_split=float(val_split),
                    augment=bool(augment), augment_strength=("heavy" if auto_boost else "auto"),
                    seed=int(seed_local), device=device_choice
                )
                device = best_device(cfg.device); st.info(f"Using device: **{device}**")
                train_loader, val_loader, classes = build_loaders(SS.cv_org_dir, cfg)
                display_names = SS.cv_label_names if (SS.cv_label_names and len(SS.cv_label_names)==len(classes)) else classes
                model = build_model(cfg.model_name, len(classes), device)
                model, history = train_loop(model, train_loader, val_loader, cfg, device)
                for f in plot_history(history): st.pyplot(f, clear_figure=True)
                mpath, lpath = save_model(model, display_names, out_dir="models")
                SS.cv_last_model = mpath; SS.cv_last_labels = lpath; SS.cv_model_name = model_name
                st.download_button("Download Model (.pt)", data=open(mpath,"rb").read(), file_name=os.path.basename(mpath))
                st.download_button("Download Labels (.json)", data=open(lpath,"rb").read(), file_name=os.path.basename(lpath))
                st.success("Training complete and artifacts saved.")
            except Exception as e:
                st.error(f"Training failed: {e}")

        st.markdown("#### Inference")
        c1, c2 = st.columns(2)
        with c1: pred_img = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="cv_pred_img2")
        with c2: cam_img  = st.camera_input("Or capture from camera")
        target_img = None
        if pred_img: target_img = Image.open(pred_img).convert("RGB")
        elif cam_img: target_img = Image.open(io.BytesIO(cam_img.getvalue())).convert("RGB")
        if target_img:
            st.image(target_img, caption="Input", use_container_width=True)
            if not (SS.cv_last_model and SS.cv_last_labels):
                st.warning("Train or load a model first.")
            else:
                try:
                    device = best_device(device_choice)
                    model, classes = load_model_for_infer(SS.cv_last_model, SS.cv_model_name, SS.cv_last_labels, device)
                    label, prob, _, _ = predict_image(model, target_img, img_size=int(img_size), device=device, classes=classes)
                    st.success(f"Prediction: **{label}** ({prob*100:.1f}%)")
                except Exception as e:
                    st.error(f"Inference failed: {e}")

        st.markdown("---")
        st.subheader("📓 Export CV Training Notebook")
        if st.button("🧠 Generate CV Notebook (.ipynb)"):
            nb_bytes = build_cv_notebook_bytes(
                data_dir=SS.cv_org_dir, model_name=model_name, img_size=int(img_size),
                epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
                kernel_name="auto_ml"
            )
            st.download_button("⬇️ Download Notebook", data=nb_bytes, file_name="UNISOLE_CV_Training_Notebook.ipynb", mime="application/x-ipynb+json")

# ========= 🧿 Object Detection (YOLO) — Smart Basic Metrics =========
with tab_yolo:
    st.subheader("🧿 YOLO Object Detection — Train & Inference")
    st.caption("Upload a YOLO-format dataset (Roboflow export works). We’ll find `data.yaml`, train, and summarize metrics.")

    if "yolo_data_dir" not in st.session_state: st.session_state.yolo_data_dir = None
    if "yolo_data_yaml" not in st.session_state: st.session_state.yolo_data_yaml = None
    if "yolo_best" not in st.session_state: st.session_state.yolo_best = None
    if "yolo_names" not in st.session_state: st.session_state.yolo_names = None
    if "yolo_last_run" not in st.session_state: st.session_state.yolo_last_run = None  # save_dir

    # --- helpers for robust metrics parsing ---
    def _read_results_csv(save_dir: str):
        import pandas as _pd
        csv_path = os.path.join(save_dir, "results.csv")
        if not os.path.exists(csv_path):
            return None, None
        try:
            df = _pd.read_csv(csv_path)
            return df, csv_path
        except Exception:
            return None, None

    def _get_last_metrics(df):
        """
        Return a dict with precision, recall, map50, map5095 from the last row.
        Columns vary by version; select by substring.
        """
        if df is None or len(df) == 0:
            return {}
        row = df.iloc[-1]
        cols = {c.lower(): c for c in df.columns}

        def pick(*keys):
            # find first column whose lowercase name contains any of the substrings
            for k in keys:
                for c in df.columns:
                    if k in c.lower():
                        return float(row[c])
            return None

        return {
            "precision": pick("metrics/precision", "precision"),
            "recall": pick("metrics/recall", "recall"),
            "map50": pick("metrics/map50", "map50"),
            "map5095": pick("metrics/map50-95", "map50-95", "map_0.5:0.95"),
            "epochs": (int(row.get("epoch", len(df)-1)) + 1) if "epoch" in df.columns else len(df)
        }

    def _show_training_summary(save_dir: str, names_dict: dict):
        # 1) Metrics (results.csv)
        df, csv_path = _read_results_csv(save_dir)
        metrics = _get_last_metrics(df) if df is not None else {}
        # 2) Curves (results.png)
        png_path = os.path.join(save_dir, "results.png")

        # Clean minimal summary
        classes_str = ", ".join([names_dict[k] for k in sorted(names_dict.keys())]) if names_dict else "—"
        m50   = metrics.get("map50")
        m5095 = metrics.get("map5095")
        prec  = metrics.get("precision")
        rec   = metrics.get("recall")
        ep    = metrics.get("epochs")

        summary_lines = ["✅ Training Complete"]
        if m50 is not None:   summary_lines.append(f"📊 mAP50: **{m50:.3f}**")
        if m5095 is not None: summary_lines.append(f"📈 mAP50-95: **{m5095:.3f}**")
        if prec is not None:  summary_lines.append(f"🎯 Precision: **{prec:.3f}**")
        if rec is not None:   summary_lines.append(f"👌 Recall: **{rec:.3f}**")
        if ep is not None:    summary_lines.append(f"⏱️ Epochs: **{ep}**")
        summary_lines.append(f"🍎 Classes: **{classes_str}**")

        st.success("\n".join(summary_lines))

        # Show training curves image if available
        if os.path.exists(png_path):
            st.image(png_path, caption="Training curves (loss, P/R, mAP)", use_container_width=True)
        # Show link to CSV if available
        if csv_path and os.path.exists(csv_path):
            st.download_button("⬇️ Download results.csv", data=open(csv_path, "rb").read(),
                               file_name="results.csv", mime="text/csv")

    # --- Upload YOLO dataset as .zip ---
    dz = st.file_uploader("Upload YOLO dataset (.zip with images/labels + data.yaml)", type=["zip"])
    if dz and st.button("📦 Extract YOLO ZIP"):
        ydir = extract_zip_to_dir(dz.read(), base_dir="uploads")
        dyaml = find_data_yaml(ydir)
        if not dyaml:
            st.error("❌ Could not find data.yaml. Please upload a YOLO-format dataset (Roboflow export).")
        else:
            st.session_state.yolo_data_dir = ydir
            st.session_state.yolo_data_yaml = dyaml
            st.success(f"✅ Found data.yaml at: {dyaml}")

    # --- Train controls ---
    if st.session_state.yolo_data_yaml:
        st.markdown("### 🏋️ Train")
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        with c2:
            imgsz = st.selectbox("Image size", [416, 512, 640], index=2)
        with c3:
            device = st.selectbox("Device", ["", "cpu", "mps", "cuda"], index=0)

        c4, c5, c6 = st.columns(3)
        with c4:
            epochs = st.number_input("Epochs", min_value=1, value=50, step=1)
        with c5:
            batch = st.number_input("Batch size", min_value=4, value=16, step=2)
        with c6:
            seed_val = st.number_input("Seed", min_value=0, value=42, step=1)

        run_name = st.text_input("Run name", value="apple_detector_v1")

        if st.button("🚀 Train YOLO"):
            with st.spinner("Training YOLO… this can take a while depending on hardware and dataset size."):
                out = train_yolo(
                    model_name=model_name,
                    data_yaml=st.session_state.yolo_data_yaml,
                    imgsz=int(imgsz),
                    epochs=int(epochs),
                    batch=int(batch),
                    device=device,
                    name=run_name,
                    seed=int(seed_val),
                )
            st.success("✅ Training complete!")
            st.write("Save dir:", out["save_dir"])
            st.session_state.yolo_best = out["best"]
            st.session_state.yolo_names = out["names"]
            st.session_state.yolo_last_run = out["save_dir"]

            # Downloads
            st.download_button("⬇️ Download best.pt",
                               data=open(out["best"], "rb").read(),
                               file_name="best.pt")

            # 🔹 NEW: Minimal & Clean summary + curves + results.csv
            _show_training_summary(out["save_dir"], out["names"])

    # --- Inference ---
    st.markdown("---")
    st.subheader("🔮 Inference & Counting")
    if not st.session_state.yolo_best:
        st.info("Train a model first, or provide a path to a YOLO weights .pt")
        custom_best = st.text_input("Or provide a path to a YOLO weights .pt", "")
        if custom_best and os.path.exists(custom_best):
            st.session_state.yolo_best = custom_best

    if st.session_state.yolo_best:
        c1, c2, c3 = st.columns(3)
        with c1:
            inf_imgsz = st.selectbox("Inference size", [416, 512, 640], index=2)
        with c2:
            conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
        with c3:
            iou = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05)
        dev_inf = st.selectbox("Device", ["", "cpu", "mps", "cuda"], index=0, key="yolo_dev_inf")

        # Image inference
        st.markdown("**Image:**")
        img_up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="yolo_img")
        cam_img = st.camera_input("Or capture from camera", key="yolo_cam")

        pil_target = None
        if img_up:
            pil_target = Image.open(img_up).convert("RGB")
        elif cam_img:
            pil_target = Image.open(io.BytesIO(cam_img.getvalue())).convert("RGB")

        if pil_target:
            st.image(pil_target, caption="Input", use_container_width=True)
            with st.spinner("Running detection…"):
                annotated, counts = infer_image_bytes(
                    st.session_state.yolo_best,
                    pil_target,
                    imgsz=int(inf_imgsz),
                    conf=float(conf),
                    iou=float(iou),
                    device=dev_inf or ""
                )
            st.image(annotated, caption=f"Detections — counts: {counts}", use_container_width=True)

    # --- Optional: quick access to last run metrics again ---
    if st.session_state.get("yolo_last_run"):
        with st.expander("📈 Show last training summary again"):
            _show_training_summary(st.session_state["yolo_last_run"], st.session_state.get("yolo_names", {}))

# =========================================================
# Pages — MENTOR
# =========================================================
def page_mentor_ui():
    page_header("🧑‍🏫 AI Learning Assistant")
    for role, content in SS.chat:
        with st.chat_message(role): st.markdown(content)
    msg = st.chat_input("Ask about algorithms, metrics, tuning, errors, or next steps…")
    if msg:
        SS.chat.append(("user", msg))
        with st.chat_message("user"): st.markdown(msg)
        context = {
            "task": SS.last_task,
            "eda": quick_eda(SS.df) if SS.df is not None else None,
            "last_summary": (SS.last_result or {}).get("summary") if SS.last_result else None,
            "feature_summary": SS.feature_summary,
            "goal": "Teach, guide, and propose practical next steps."
        }
        with st.chat_message("assistant"):
            with st.spinner("Mentor is thinking…"):
                reply = ask_ai_mentor(msg, context)
                st.markdown(reply)
        SS.chat.append(("assistant", reply))

# =========================
# Router: render selected
# =========================
# Show just one page at a time, based on sidebar
current = None
if page_mentor == "🧑‍🏫 AI Learning Assistant":
    current = "mentor"
elif page_cv == "🖼️ Image Classification":
    current = "cv"
elif page_cv == "🧿 Object Detection (YOLO)":
    current = "yolo"
elif page_ml == "🤖 AutoML (Tabular)":
    current = "automl"
elif page_ml == "⚡ Explainability":
    current = "explain"
elif page_ml == "📓 Notebook Export":
    current = "nb"
elif page_data == "📥 Upload Data":
    current = "upload"
elif page_data == "📊 EDA Dashboard":
    current = "eda"
elif page_data == "🔧 Feature Engineering":
    current = "fe"

if current == "upload": page_upload()
elif current == "eda": page_eda()
elif current == "fe": page_feature_eng()
elif current == "automl": page_automl()
elif current == "explain": page_explain()
elif current == "nb": page_notebook()
elif current == "cv": page_cv()
elif current == "yolo": page_yolo()
elif current == "mentor": page_mentor_ui()
else:
    page_upload()

st.caption("UNISOLE AI LABS • Data Lab • AutoML • Vision • Explainability • Notebooks • Mentor")
