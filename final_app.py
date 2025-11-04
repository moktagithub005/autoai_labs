import os
import io
import uuid
import json
import time
import joblib
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# HF-FRIENDLY TEMP STORAGE CONFIGURATION (insert near top of app.py)
# ============================================================================
import tempfile

# Use environment variable override for dev, or default to system temp
BASE_TMP = os.getenv("UNISOLE_TMPDIR", tempfile.gettempdir())

# Define all working directories under temp base
UPLOAD_DIR   = os.path.join(BASE_TMP, "uploads")
MODEL_DIR    = os.path.join(BASE_TMP, "models")
REPORT_DIR   = os.path.join(BASE_TMP, "reports")
PIPELINE_DIR = os.path.join(BASE_TMP, "pipelines")
PLOT_DIR     = os.path.join(BASE_TMP, "plots")

# Ensure all directories exist
for _d in (UPLOAD_DIR, MODEL_DIR, REPORT_DIR, PIPELINE_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

# Helper to create unique file paths in temp directories
def _tmp_filepath(dirpath, basename=None, ext=None):
    """Generate a unique file path in the given directory."""
    name = basename or str(uuid.uuid4())[:8]
    if ext and not name.endswith(ext):
        name = f"{name}{ext}"
    return os.path.join(dirpath, name)

# Compatibility shim: modules expecting "uploads", "models" etc. in current dir
# will still work because app.py now passes full temp paths
# End temp-storage block
# ============================================================================

from cv_trainer import (
    CVTrainConfig,
    best_device,
    prepare_dataset_from_zip,
    build_loaders,
    build_model,
    train_loop,
    plot_history,
    save_model,
    load_model_for_infer,
    predict_image
)

from PIL import Image
import io

from cv_data_prep import (
    infer_labels_auto,
    apply_csv_labels,
    build_org_structure,
    add_manual_labels,
    zip_directory,
    small_dataset_hint,
    _extract_zip_to_tmp,
    _scan_images
)

from yolo_trainer import (
    extract_zip_to_dir, find_data_yaml, train_yolo,
    infer_image_bytes, infer_video_file
)

from cv_notebook_builder import build_cv_notebook_bytes

# ========== AutoML ==========
from automl_tabular import (
    quick_eda, detect_task, run_automl,
    plot_classification_curves, plot_regression_plots
)

# ========== EDA ==========
from eda_tools import (
    overview, plot_missingness, target_summary, plot_target_distribution,
    correlation_matrix, plot_corr_heatmap, top_corr_with_target,
    plot_numeric_distributions, plot_categorical_bars,
    outlier_report, plot_boxplots, class_imbalance, leakage_scan
)

# ========== Explainability ==========
from explainability import (
    global_feature_importance, build_shap_explainer,
    shap_summary_fig, shap_single_waterfall_fig, pick_interesting_row
)

# ========== Feature Engineering ==========
from feature_engineering import (
    FEConfig, default_config, fit_feature_pipeline,
    transform_with_pipeline, export_processed_dataframe
)

from notebook_builder import build_notebook_bytes

# ========== LLM Mentor ==========
from llm_helper import explain_dataset, explain_model_choice, explain_metrics, ask_ai_mentor

# ======= Page config =======
st.set_page_config(page_title="UNISOLE Auto-AI — ML Lab", layout="wide")
st.title("🤖 UNISOLE Auto-AI LAB")
st.caption("Machine Learning Studio • Pro EDA • Feature Engineering • AutoML • Explainability • LLM Mentor")

# ======= Show temp storage info in sidebar =======
st.sidebar.info(f"✅ Using temporary storage at: `{BASE_TMP}`\n\n(Set `UNISOLE_TMPDIR` env var to override)")

# ======= Session State =======
if "df" not in st.session_state:
    st.session_state.df = None              # original dataset
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None    # FE-processed DataFrame (with target)
if "feature_pipe" not in st.session_state:
    st.session_state.feature_pipe = None    # fitted FE pipeline
if "feature_summary" not in st.session_state:
    st.session_state.feature_summary = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None     # AutoML result dict
if "last_task" not in st.session_state:
    st.session_state.last_task = None
if "target" not in st.session_state:
    st.session_state.target = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# ========= Sidebar =========
with st.sidebar:
    st.header("⚙️ Training Settings")
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("CV Folds", 2, 10, 5, 1)
    max_time = st.slider("Max Train Time (minutes)", 1, 60, 10, 1)
    st.markdown("---")
    edu_mode = st.toggle("🎓 Education Mode (AI explanations)", value=True)

tab_upload, tab_eda, tab_fe, tab_auto, tab_cv, tab_yolo, tab_explain, tab_nb, tab_forecast, tab_nlp, tab_mentor = st.tabs([
    "📥 Upload Data",
    "📊 EDA Dashboard",
    "🔧 Feature Engineering",
    "🤖 AutoML (Tabular)",
    "🖼️ Image Classification (CV)",
    "🧿 Object Detection (YOLO)",
    "⚡ Explainability",
    "📓 Notebook Export",
    "🔮 AI Forecasting",
    "🧠 NLP Tasks",
    "🧑‍🏫 AI Mentor"
])

# ===== NLP session slots (persist model + AI responses) =====
for _k, _v in {
    "nlp_df": None,
    "nlp_pipe": None,
    "nlp_label_encoder": None,
    "nlp_classes": None,
    "nlp_metrics": None,
    "nlp_model_name": None,
    "nlp_ai_model_explain": None,
    "nlp_ai_pred_explain": None
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ========= 🧠 NLP Tasks — Guided Wizard (TF-IDF + Logistic/NB) =========
with tab_nlp:
    st.subheader("🧠 NLP Tasks — Text Classification (TF-IDF + Logistic/NB)")
    st.caption("Upload a CSV with a text column and a label column. Clean text, train a fast baseline model, evaluate, and predict. LLM can explain the model or a specific prediction.")

    import io, os, joblib, re, string
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay, classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder

    SS = st.session_state

    # AI flag to trigger model explanation without refresh
    if "nlp_explain_model_flag" not in SS:
        SS["nlp_explain_model_flag"] = False

    # ---------- helpers ----------
    def simple_clean(text: str,
                     lower=True, rm_urls=True, rm_emails=True, rm_punct=True, rm_digits=False, squeeze_ws=True):
        if text is None: return ""
        s = str(text)
        if lower: s = s.lower()
        if rm_urls: s = re.sub(r"https?://\S+|www\.\S+", " ", s)
        if rm_emails: s = re.sub(r"\b[\w\.-]+?@[\w\.-]+?\.\w+\b", " ", s)
        if rm_punct: s = s.translate(str.maketrans("", "", string.punctuation))
        if rm_digits: s = re.sub(r"\d+", " ", s)
        if squeeze_ws: s = re.sub(r"\s+", " ", s).strip()
        return s

    def build_pipeline(model_type: str, max_features: int, ngram: str):
        ngram_range = (1,1) if ngram == "Unigram" else (1,2) if ngram == "Uni+Bi" else (1,3)
        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        if model_type == "Logistic Regression":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
        else:
            clf = MultinomialNB()
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        return pipe, {"ngram_range": ngram_range, "max_features": max_features}

    def compute_metrics(y_true, y_pred, labels):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        rep = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
        return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "report": rep}

    def plot_confusion(y_true, y_pred, labels):
        fig, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", xticks_rotation=45, ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        return fig

    def top_token_contributions(pipe: Pipeline, text: str, top_k: int = 10):
        vec = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        X = vec.transform([text])

        # Try to access vocab / feature names
        vocab = None
        if hasattr(vec, "get_feature_names_out"):
            feat_names = vec.get_feature_names_out()
        elif hasattr(vec, "vocabulary_"):
            inv_vocab = {v:k for k,v in vec.vocabulary_.items()}
            feat_names = [inv_vocab.get(i, f"id_{i}") for i in range(len(inv_vocab))]
        else:
            feat_names = None

        # LogisticRegression: contributions via coef * tfidf
        if isinstance(clf, LogisticRegression):
            probs = clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            coefs = clf.coef_[pred_idx]
            vals = X.toarray()[0] * coefs
            top_ids = np.argsort(-np.abs(vals))[:top_k]
            items = []
            for fid in top_ids:
                token = feat_names[fid] if feat_names is not None and fid < len(feat_names) else f"id_{fid}"
                items.append({"token": token, "contribution": float(vals[fid])})
            return pred_idx, probs, items

        # MultinomialNB
        if isinstance(clf, MultinomialNB):
            probs = clf.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            log_prob = clf.feature_log_prob_[pred_idx]
            vals = X.toarray()[0] * log_prob
            top_ids = np.argsort(-np.abs(vals))[:top_k]
            items = []
            if hasattr(vec, "get_feature_names_out"):
                for fid in top_ids:
                    token = feat_names[fid]
                    items.append({"token": token, "contribution": float(vals[fid])})
            else:
                for fid in top_ids:
                    items.append({"token": f"id_{fid}", "contribution": float(vals[fid])})
            return pred_idx, probs, items

        # Fallback
        probs = getattr(clf, "predict_proba", lambda X: np.array([[1.0]]))(X)[0]
        return int(np.argmax(probs)), probs, []

    # ---------- 1) Load Data ----------
    st.markdown("### 1) Load Data")
    source = st.radio(
        "Source",
        ["Use uploaded CSV (Data Lab)", "Upload CSV here"],
        horizontal=True,
        key="nlp_source"
    )

    df = None
    if source == "Use uploaded CSV (Data Lab)":
        if SS.get("df") is not None:
            df = SS["df"].copy()
            st.success("Using dataset from Data Lab")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.warning("No dataset present in memory. Upload below.")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"], key="nlp_csv")
        if up:
            try:
                df = pd.read_csv(up)
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # ---------- 2) Configure Task ----------
    if df is not None and len(df.columns) >= 2:
        st.markdown("### 2) Configure Task")
        text_col = st.selectbox("Text column", options=df.columns, key="nlp_text_col")
        label_col = st.selectbox("Label column", options=[c for c in df.columns if c != text_col], key="nlp_label_col")

        c1, c2, c3 = st.columns(3)
        with c1:
            model_type = st.selectbox("Model", ["Logistic Regression", "Naive Bayes"], index=0, key="nlp_model_type")
        with c2:
            ngram = st.selectbox("TF-IDF n-grams", ["Unigram", "Uni+Bi", "Uni+Bi+Tri"], index=1, key="nlp_ngram")
        with c3:
            max_feat = st.number_input("Max features", min_value=1000, max_value=200000, value=20000, step=1000, key="nlp_maxfeat")

        st.markdown("**Text cleaning:**")
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1: opt_lower = st.toggle("lowercase", True, key="nlp_clean_lower")
        with d2: opt_urls  = st.toggle("remove URLs", True, key="nlp_clean_urls")
        with d3: opt_mail  = st.toggle("remove emails", True, key="nlp_clean_emails")
        with d4: opt_punc  = st.toggle("remove punctuation", True, key="nlp_clean_punct")
        with d5: opt_digs  = st.toggle("remove digits", False, key="nlp_clean_digits")

        # ---------- 3) Train & Evaluate ----------
        st.markdown("### 3) Train & Evaluate")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="nlp_testsize")
        random_state = st.number_input("Random seed", 0, 9999, 42, key="nlp_seed")

        if st.button("🚀 Train NLP Model", key="nlp_btn_train"):
            try:
                df_work = df.dropna(subset=[text_col, label_col]).copy()
                df_work[text_col] = df_work[text_col].apply(lambda x: simple_clean(
                    x, lower=opt_lower, rm_urls=opt_urls, rm_emails=opt_mail, rm_punct=opt_punc, rm_digits=opt_digs
                ))

                # Encode labels
                le = LabelEncoder()
                y = le.fit_transform(df_work[label_col].astype(str).values)
                X = df_work[text_col].astype(str).values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

                pipe, vec_params = build_pipeline(model_type, max_features=int(max_feat), ngram=ngram)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                labels = list(le.classes_)
                mets = compute_metrics(y_test, y_pred, labels)
                fig_cm = plot_confusion(y_test, y_pred, labels)

                # Persist all essentials to session
                SS["nlp_pipe"] = pipe
                SS["nlp_label_encoder"] = le
                SS["nlp_classes"] = labels
                SS["nlp_metrics"] = mets
                SS["nlp_model_name"] = f"{model_type} + TF-IDF({vec_params['ngram_range']}, {vec_params['max_features']})"
                SS["nlp_ai_model_explain"] = None
                SS["nlp_ai_pred_explain"]  = None

                st.success("✅ Training complete!")
                a,b,c,d = st.columns(4)
                a.metric("Accuracy", f"{mets['accuracy']:.3f}")
                b.metric("Precision (macro)", f"{mets['precision_macro']:.3f}")
                c.metric("Recall (macro)", f"{mets['recall_macro']:.3f}")
                d.metric("F1 (macro)", f"{mets['f1_macro']:.3f}")
                st.pyplot(fig_cm, clear_figure=True)

                # Download model
                buf = io.BytesIO()
                joblib.dump(
                    {"pipeline": pipe, "label_encoder": le, "classes": labels, "model_name": SS["nlp_model_name"]},
                    buf
                )
                st.download_button("⬇️ Download model (.pkl)", data=buf.getvalue(), file_name="nlp_pipeline.pkl")

                # Run AI explanation safely after training
                if SS.get("nlp_explain_model_flag") and SS.get("nlp_pipe") is not None:
                    SS["nlp_explain_model_flag"] = False
                try:
                    with st.spinner("⏳ AI is explaining your model…"):
                        prompt = f"""
                        Explain this text classification pipeline clearly for business users.
                        Model used: {SS['nlp_model_name']}
                        Classes: {', '.join(SS['nlp_classes'])}
                        Accuracy: {SS['nlp_metrics']['accuracy']:.3f}
                        F1 Score: {SS['nlp_metrics']['f1_macro']:.3f}
                        Explain how it works in simple language. No jargon. Keep it short and useful.
                        """
                        SS["nlp_ai_model_explain"] = ask_ai_mentor(prompt, {"module": "nlp"})
                except Exception as e:
                    st.error(f"AI explanation failed: {e}")

                # Persist explanation
                if SS.get("nlp_ai_model_explain"):
                    st.info(SS["nlp_ai_model_explain"])

                # Set flag when button is clicked
                if st.button("💬 Ask AI to explain this model", key="nlp_btn_ai_model"):
                    SS["nlp_explain_model_flag"] = True

                # Persisted AI text shown directly under metrics
                if SS.get("nlp_ai_model_explain"):
                    st.info(SS["nlp_ai_model_explain"])

            except Exception as e:
                st.error(f"Training failed: {e}")

        # ---------- 4) Try a Prediction ----------
        st.markdown("### 4) Try a Prediction")
        text_input = st.text_area("Enter text to classify", height=140, placeholder="Paste a review, message, or sentence here…", key="nlp_text_input")

        if st.button("🔮 Predict", key="nlp_btn_predict"):
            if SS["nlp_pipe"] is None:
                st.warning("Train a model first.")
            else:
                try:
                    # Validate input text
                    if not text_input or text_input.strip() == "":
                        st.warning("Please enter valid text to classify.")
                    else:
                        clean_in = simple_clean(
                            text_input,
                            lower=SS.get("nlp_clean_lower", True),
                            rm_urls=SS.get("nlp_clean_urls", True),
                            rm_emails=SS.get("nlp_clean_emails", True),
                            rm_punct=SS.get("nlp_clean_punct", True),
                            rm_digits=SS.get("nlp_clean_digits", False)
                        )

                        pipe = SS["nlp_pipe"]
                        le = SS["nlp_label_encoder"]
                        labels = SS["nlp_classes"]

                        # Safe prediction
                        try:
                            pred = pipe.predict([clean_in])[0]
                        except Exception:
                            st.error("⚠️ Unable to classify this text. Try a longer or clearer sentence.")
                            st.stop()

                        pred_label = le.inverse_transform([pred])[0]

                        # Simple display
                        st.success(f"✅ **Prediction:** {pred_label}")

                        # Local contribution explainer
                        pred_idx, probs, items = top_token_contributions(pipe, clean_in, top_k=8)
                        with st.expander("🔎 Why this prediction? (token contributions)"):
                            st.write(pd.DataFrame(items))

                        # LLM: Explain this prediction
                        if st.button("💬 Ask AI to explain this prediction", key="nlp_btn_ai_pred"):
                            try:
                                with st.spinner("⏳ AI is analyzing this prediction…"):
                                    top_words = ", ".join([it["token"] for it in items[:6]])
                                    prompt = f"""
                                    Explain this prediction for a business audience.
                                    Predicted class: {pred_label}
                                    Top indicative words: {top_words}
                                    If probabilities are available, comment on confidence and any ambiguity.
                                    Provide 3 short recommendations on how to improve text for a different target class if needed.
                                    """
                                    SS["nlp_ai_pred_explain"] = ask_ai_mentor(prompt, {"module":"nlp","pred":pred_label,"tokens":items[:6]})
                            except Exception as e:
                                st.error(f"AI explanation failed: {e}")

                        # show persisted AI prediction explanation
                        if SS.get("nlp_ai_pred_explain"):
                            st.info(SS["nlp_ai_pred_explain"])

                except Exception as e:
                    st.error(f"Inference failed: {e}")

    else:
        st.info("Upload a CSV (or use Data Lab), then select text and label columns.")

# ========= Upload Tab =========
with tab_upload:
    st.subheader("📥 Upload a CSV Dataset")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        # Save to temp UPLOAD_DIR with unique ID
        file_id = str(uuid.uuid4())[:8]
        file_path = _tmp_filepath(UPLOAD_DIR, f"data_{file_id}", ".csv")

        with open(file_path, "wb") as f:
            f.write(uploaded.read())

        st.success(f"✅ File uploaded successfully (saved to temp storage)")

        # Read and store in session
        try:
            df = pd.read_csv(file_path)
            st.session_state.df = df
            st.session_state.processed_df = None
            st.session_state.feature_pipe = None
            st.session_state.feature_summary = None
            st.session_state.last_result = None

            st.write("### 🔎 Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Optional AI explanation
            if edu_mode and st.button("🧠 Explain this dataset"):
                explanation = explain_dataset(
                    quick_eda(df),
                    sample_schema={c: str(df[c].dtype) for c in df.columns}
                )
                st.info(explanation)

        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")

# ========= EDA Tab =========
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis")
    if st.session_state.df is None:
        st.warning("⚠ Please upload a CSV file in the Upload tab.")
    else:
        df = st.session_state.df
        ov = overview(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", ov["rows"])
        c2.metric("Columns", ov["cols"])
        c3.metric("Duplicates", ov["duplicates"])

        st.write("### 🔧 Missing Values")
        st.pyplot(plot_missingness(df), clear_figure=True)
        if edu_mode and st.button("🧠 Explain Missing Data"):
            st.info(ask_ai_mentor("Explain missing data handling methods and tradeoffs.", {"missing": ov["missing_perc"]}))

        st.write("---")
        st.write("### 🎯 Target Distribution")
        target_sel = st.selectbox("Select target column to analyze", options=df.columns, key="eda_target")
        if target_sel:
            st.pyplot(plot_target_distribution(df, target_sel), clear_figure=True)
            if edu_mode and st.button("🧠 Explain Target"):
                st.info(ask_ai_mentor("Explain why understanding the target distribution matters for modeling.", {"target": target_sel}))

        st.write("---")
        st.write("### 🔗 Correlation Heatmap (numeric features)")
        corr = correlation_matrix(df)
        st.pyplot(plot_corr_heatmap(corr), clear_figure=True)
        if edu_mode and st.button("🧠 Explain Correlations"):
            st.info(ask_ai_mentor("Explain how to read a correlation heatmap and multicollinearity risks.", {}))

        st.write("---")
        st.write("### 📈 Feature Distributions")
        for fig in plot_numeric_distributions(df, max_cols=6):
            st.pyplot(fig, clear_figure=True)
        if edu_mode and st.button("🧠 Explain Numeric Distributions"):
            st.info(ask_ai_mentor("Explain what numeric distributions tell us and when to scale/transform.", {}))

        cat_cols = ov["categorical_cols"]
        if len(cat_cols) > 0:
            st.write("### 🏷️ Categorical Insights")
            for fig in plot_categorical_bars(df, max_cols=6, top_n=12):
                st.pyplot(fig, clear_figure=True)
            if edu_mode and st.button("🧠 Explain Categorical Encoding"):
                st.info(ask_ai_mentor("Explain one-hot vs target encoding and when to use which.", {"categorical_cols": cat_cols}))

        st.write("---")
        st.write("### ⚠ Outliers (IQR)")
        out = outlier_report(df, k=1.5)
        st.json(out)
        for fig in plot_boxplots(df, max_cols=6):
            st.pyplot(fig, clear_figure=True)
        if edu_mode and st.button("🧠 Explain Outliers"):
            st.info(ask_ai_mentor("Explain outlier handling (cap, transform, or robust models).", {"outliers": out}))

        st.write("---")
        st.write("### 🧪 Data Quality Checks")
        if target_sel:
            imb = class_imbalance(df, target_sel)
            if imb:
                st.write("**Class Imbalance (fractions):**", imb)
            leak = leakage_scan(df, target_sel, threshold=0.98)
            st.write("**Potential Leakage / Duplicates:**", leak)
            if edu_mode and st.button("🧠 Explain Data Quality Warnings"):
                st.info(ask_ai_mentor("Explain leakage risks and class imbalance mitigation.", {"imbalance": imb, "leakage": leak}))

# ========= Feature Engineering Tab =========
with tab_fe:
    st.subheader("🔧 Feature Engineering (Guided • Pro Pipeline)")
    if st.session_state.df is None:
        st.warning("⚠ Upload a CSV first.")
    else:
        df = st.session_state.df
        st.write("Select your **target** column first, then follow the steps.")
        target = st.selectbox("Target column", options=df.columns, key="fe_target")
        if target:
            st.session_state.target = target

            st.markdown("#### Step 1 — Configure Missing Values")
            c1, c2 = st.columns(2)
            with c1:
                num_impute = st.selectbox("Numeric imputation", ["median", "mean", "most_frequent"], index=0)
            with c2:
                cat_impute = st.selectbox("Categorical imputation", ["most_frequent"], index=0)

            st.markdown("#### Step 2 — Rare Category Grouping")
            c3, c4 = st.columns(2)
            with c3:
                enable_rare = st.toggle("Enable rare category grouping", value=True)
            with c4:
                rare_frac = st.slider("Rare min fraction", 0.0, 0.1, 0.01, 0.005)

            st.markdown("#### Step 3 — Skew Fix (numeric)")
            c5, c6 = st.columns(2)
            with c5:
                fix_skew = st.toggle("Apply log1p to skewed columns", value=True)
            with c6:
                skew_thr = st.slider("Skew threshold (|skew| ≥)", 0.25, 2.0, 0.75, 0.05)

            st.markdown("#### Step 4 — Scaling")
            scale_opt = st.selectbox("Scale numeric features with", ["standard", "minmax", "none"], index=0)

            st.markdown("#### Step 5 — Encoding")
            encode_opt = st.selectbox("Categorical encoding", ["onehot", "none"], index=0)

            st.markdown("#### Step 6 — Low-Variance Removal")
            c7, c8 = st.columns(2)
            with c7:
                low_var = st.toggle("Remove low-variance features", value=True)
            with c8:
                var_thr = st.number_input("Variance threshold", min_value=0.0, value=0.0, step=0.01)

            st.markdown("#### Step 7 — Build & Apply Pipeline")
            if st.button("🚀 Build Pipeline and Transform Data"):
                cfg = FEConfig(
                    num_impute=num_impute,
                    cat_impute=cat_impute,
                    encode=encode_opt,
                    enable_rare_group=enable_rare,
                    rare_min_frac=rare_frac,
                    scale_numeric=scale_opt,
                    fix_skew=fix_skew,
                    skew_threshold=skew_thr,
                    enable_low_variance=low_var,
                    variance_threshold=var_thr
                )
                with st.spinner("Fitting feature pipeline…"):
                    pipe, X_out, names, summary = fit_feature_pipeline(df.copy(), target=target, config=cfg)
                    proc_df = export_processed_dataframe(pipe, df.copy(), target=target)

                st.session_state.feature_pipe = pipe
                st.session_state.feature_summary = summary
                st.session_state.processed_df = proc_df

                st.success("✅ Feature pipeline built & dataset transformed!")
                st.write("**Pipeline summary:**")
                st.json(summary)
                st.write("**Processed preview:**")
                st.dataframe(proc_df.head(), use_container_width=True)

            if st.session_state.processed_df is not None:
                st.markdown("#### Save / Download")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("💾 Save Pipeline (.pkl)"):
                        pid = str(uuid.uuid4())[:8]
                        # Save to temp PIPELINE_DIR
                        path = _tmp_filepath(PIPELINE_DIR, f"feature_pipeline_{pid}", ".pkl")
                        joblib.dump(st.session_state.feature_pipe, path)
                        st.success(f"Saved pipeline to temp storage")

                with colB:
                    # Download processed CSV
                    buf = io.StringIO()
                    st.session_state.processed_df.to_csv(buf, index=False)
                    st.download_button(
                        "⬇️ Download Cleaned CSV",
                        data=buf.getvalue(),
                        file_name="cleaned_dataset.csv",
                        mime="text/csv"
                    )

                if edu_mode and st.button("🧠 Suggest feature engineering improvements"):
                    st.info(ask_ai_mentor(
                        "Given this feature pipeline summary, suggest 5 improvements and why.",
                        {"feature_summary": st.session_state.feature_summary}
                    ))

# ========= AutoML Tab =========
with tab_auto:
    st.subheader("🤖 AutoML Trainer")
    if st.session_state.df is None:
        st.warning("⚠ Upload a CSV first.")
    else:
        df = st.session_state.df
        target = st.selectbox("Target (what to predict)", options=df.columns, key="auto_target")
        use_clean = False
        if st.session_state.processed_df is not None and target:
            st.info("A cleaned dataset from Feature Engineering is available.")
            use_clean = st.toggle("Use cleaned dataset for training", value=True, key="use_clean")

        train_df = st.session_state.processed_df if (use_clean and st.session_state.processed_df is not None) else df

        if target:
            st.session_state.target = target
            task = detect_task(train_df, target)
            st.session_state.last_task = task
            st.info(f"Detected task: **{task.upper()}**")

            if st.button("🚀 Run AutoML"):
                with st.spinner("Training models…"):
                    start = time.time()
                    result = run_automl(
                        df=train_df,
                        target=target,
                        task=task,
                        seed=seed,
                        test_size=test_size,
                        cv_folds=cv_folds,
                        max_minutes=max_time
                    )
                    elapsed = time.time() - start

                st.session_state.last_result = result
                st.success(f"✅ Training complete in {elapsed/60:.1f} min")
                st.subheader("🏆 Best Model Summary")
                st.json(result["summary"])

                st.markdown("### 📊 Visualizations")
                if task == "classification":
                    for fig in plot_classification_curves(result):
                        st.pyplot(fig, clear_figure=True)
                else:
                    for fig in plot_regression_plots(result):
                        st.pyplot(fig, clear_figure=True)

                # Save artifacts to temp directories
                uid = str(uuid.uuid4())[:8]
                model_id = f"{result['summary']['best_model_name']}_{uid}"
                model_path = _tmp_filepath(MODEL_DIR, model_id, ".pkl")
                card_path = _tmp_filepath(REPORT_DIR, model_id, ".json")
                
                joblib.dump(result["artifact"], model_path)
                with open(card_path, "w") as f:
                    json.dump(result["summary"], f, indent=2)

                st.markdown("### ⬇️ Downloads")
                st.download_button("Download model (.pkl)",
                                   data=open(model_path, "rb").read(),
                                   file_name=os.path.basename(model_path))
                st.download_button("Download model card (.json)",
                                   data=open(card_path, "rb").read(),
                                   file_name=os.path.basename(card_path))

                if edu_mode:
                    st.markdown("### 🎓 Education & Explanations")
                    colA, colB = st.columns(2)
                    with colA:
                        if st.button("🤔 Why was this model chosen?"):
                            with st.spinner("Mentor is preparing explanation..."):
                                st.info(explain_model_choice(result["summary"]))
                    with colB:
                        if st.button("📐 Explain these metrics"):
                            with st.spinner("Mentor is preparing metrics explanation..."):
                                st.info(explain_metrics(result["summary"]))

# ========= 🖼️ CV Training Tab =========
with tab_cv:
    st.subheader("🖼️ Image Classification — Flexible Import (Preserve Labeled Datasets)")
    st.caption("Upload ANY .zip. If it's already in class folders, we keep them EXACTLY. If not, we'll help you label/organize. You can also rename classes before training.")

    # Session state
    ss = st.session_state
    for k, v in {
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
    }.items():
        if k not in ss:
            ss[k] = v

    # Helpers for labeled detection
    def _is_image_file(p: str) -> bool:
        return os.path.splitext(p.lower())[1] in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    def _scan_images(root: str):
        paths = []
        for r, _, files in os.walk(root):
            for fn in files:
                fp = os.path.join(r, fn)
                if _is_image_file(fp):
                    paths.append(fp)
        return sorted(paths)

    def _immediate_subdirs(p: str):
        return [os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

    def _dir_has_images(p: str) -> bool:
        try:
            for fn in os.listdir(p):
                if _is_image_file(os.path.join(p, fn)):
                    return True
        except Exception:
            pass
        return False

    def _find_labeled_root(root: str) -> str | None:
        candidates = []
        for r, dirs, _ in os.walk(root):
            if not dirs:
                continue
            subdirs = [os.path.join(r, d) for d in dirs]
            class_dirs = [d for d in subdirs if _dir_has_images(d)]
            if len(class_dirs) >= 2:
                candidates.append((r, len(class_dirs)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # Upload ZIP
    from torchvision import datasets

    upzip = st.file_uploader("Upload dataset (.zip)", type=["zip"])

    if upzip and st.button("📦 Extract ZIP"):
        # Extract to temp UPLOAD_DIR
        raw_dir = _extract_zip_to_tmp(upzip.read(), base_dir=UPLOAD_DIR)
        imgs = _scan_images(raw_dir)

        if not imgs:
            st.error("❌ No images found in ZIP.")
        else:
            ss.cv_raw_dir = raw_dir
            ss.cv_img_paths = imgs

            # Detect labeled structure
            labeled_root = _find_labeled_root(raw_dir)
            if labeled_root:
                ss.cv_is_labeled = True
                ss.cv_labeled_root = labeled_root
                ss.cv_org_dir = labeled_root
                try:
                    tmp = datasets.ImageFolder(labeled_root)
                    ss.cv_classes = list(tmp.classes)
                except Exception:
                    ss.cv_classes = []
                st.success(f"✅ Labeled dataset detected and preserved")
                st.write("Classes (preserved):", ss.cv_classes)
            else:
                ss.cv_is_labeled = False
                ss.cv_labeled_root = None
                st.success(f"✅ Extracted {len(imgs)} images. Not labeled yet — choose labeling below.")

    # If NOT labeled: labeling flow
    if ss.cv_raw_dir and not ss.cv_is_labeled:
        st.markdown("### 2️⃣ Labeling Options (for unlabeled data)")
        mode = st.radio("Choose labeling mode:", ["Auto (from filenames)", "Use labels.csv", "Manual fix unknowns"], index=0)
        mapping = {p: None for p in ss.cv_img_paths}

        if mode == "Auto (from filenames)":
            mapping = infer_labels_auto(ss.cv_img_paths)
            st.info("🧠 Auto-labeled from filenames. You can fix unknowns in the next step.")
        elif mode == "Use labels.csv":
            csv_up = st.file_uploader("Upload labels.csv (filename,label)", type=["csv"], key="labels_csv")
            if csv_up:
                csv_text = csv_up.getvalue().decode("utf-8", errors="ignore")
                mapping = apply_csv_labels(ss.cv_img_paths, csv_text)
                st.success("✅ Applied CSV labels where matched.")

        if st.button("🗂️ Organize into class folders"):
            # Pass temp UPLOAD_DIR as base
            org_dir, report, unknowns = build_org_structure(mapping, out_base=UPLOAD_DIR)
            ss.cv_org_dir = org_dir
            ss.cv_mapping = mapping
            ss.cv_unknowns = unknowns
            ss.cv_classes = sorted(list(report.class_counts.keys()))
            st.success(f"Organized {report.labeled}/{report.total_images} images into {len(report.class_counts)} classes.")
            st.write("📊 Class counts:", report.class_counts)
            ss.cv_small_hint = small_dataset_hint(report.class_counts, min_per_class=50)
            if ss.cv_small_hint:
                st.warning(f"⚠️ Small dataset detected. Suggested extra samples per class: {ss.cv_small_hint}")

        if ss.cv_unknowns:
            st.markdown("### 🏷️ Manual Label Unknowns")
            st.caption("Enter labels for some unknown images (you can label only a subset).")
            edits = {}
            cols = st.columns(4)
            max_show = min(24, len(ss.cv_unknowns))
            for i, p in enumerate(ss.cv_unknowns[:max_show]):
                with cols[i % 4]:
                    try:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
                    except Exception:
                        st.write(os.path.basename(p))
                    lab = st.text_input(f"Label for {os.path.basename(p)}", key=f"cv_fix_{i}")
                    edits[p] = lab.strip()
            if st.button("✅ Apply Manual Labels"):
                new_map = add_manual_labels(ss.cv_mapping, edits)
                # Pass temp UPLOAD_DIR
                org_dir2, report2, unknowns2 = build_org_structure(new_map, out_base=UPLOAD_DIR)
                ss.cv_org_dir = org_dir2
                ss.cv_mapping = new_map
                ss.cv_unknowns = unknowns2
                ss.cv_classes = sorted(list(report2.class_counts.keys()))
                st.success(f"Updated: {report2.labeled} labeled, {report2.unknown} unknown left.")
                st.write("📊 Class counts:", report2.class_counts)

    # If labeled OR organized: allow renaming + training
    if ss.cv_org_dir:
        st.markdown("### 3️⃣ (Optional) Rename Classes For Display")
        st.caption("We will NOT rename folders — only the names **saved in labels.json** and used for inference outputs.")
        try:
            from torchvision import datasets
            imgf = datasets.ImageFolder(ss.cv_org_dir)
            classes_for_training = list(imgf.classes)
        except Exception:
            classes_for_training = ss.cv_classes or []
        if not classes_for_training:
            st.error("Could not read classes from dataset. Ensure there are class subfolders with images.")
        else:
            renamed = []
            cols = st.columns(2)
            for i, cname in enumerate(classes_for_training):
                with cols[i % 2]:
                    new = st.text_input(f"Rename '{cname}' →", value=(ss.cv_label_names[i] if (ss.cv_label_names and i < len(ss.cv_label_names)) else cname), key=f"cv_rename_{i}")
                    renamed.append(new if new.strip() else cname)
            if st.button("💾 Save display names (labels.json will use these)"):
                ss.cv_label_names = renamed
                st.success("Saved display names. Training will still use folder order; inference will report these names.")

        st.markdown("### 💾 Download Organized Dataset")
        zbytes = zip_directory(ss.cv_org_dir)
        st.download_button("⬇️ Download organized dataset (.zip)", data=zbytes, file_name="organized_dataset.zip")

        st.markdown("---")
        st.subheader("🎯 Train Image Classifier (PyTorch)")

        # Controls
        model_name = st.selectbox("Backbone", ["resnet18", "efficientnet_b0", "mobilenet_v3_small"], index=0)
        img_size = st.selectbox("Image Size", [224, 256, 299], index=0)
        batch_size = st.number_input("Batch Size", 1, 128, 16, 1)
        epochs = st.number_input("Epochs", 1, 100, 6, 1)
        lr = st.number_input("Learning rate", 1e-5, 1.0, 1e-3, format="%.5f")
        val_split = st.slider("Validation split", 0.05, 0.5, 0.2, 0.05)
        auto_boost = st.toggle("Auto-Boost Augmentation (good for small datasets)", value=bool(ss.cv_small_hint))
        augment = st.toggle("Enable augmentation", value=True)
        device_choice = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
        seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

        if st.button("🚀 Train Model"):
            try:
                cfg = CVTrainConfig(
                    model_name=model_name,
                    img_size=int(img_size),
                    batch_size=int(batch_size),
                    lr=float(lr),
                    epochs=int(epochs),
                    val_split=float(val_split),
                    augment=bool(augment),
                    augment_strength=("heavy" if auto_boost else "auto"),
                    seed=int(seed),
                    device=device_choice,
                )
                device = best_device(cfg.device)
                st.info(f"Using device: **{device}**")

                train_loader, val_loader, classes_from_loader = build_loaders(ss.cv_org_dir, cfg)
                display_names = ss.cv_label_names if (ss.cv_label_names and len(ss.cv_label_names) == len(classes_from_loader)) else classes_from_loader

                model = build_model(cfg.model_name, len(classes_from_loader), device)
                model, history = train_loop(model, train_loader, val_loader, cfg, device)
                figs = plot_history(history)
                st.success("✅ Training Complete!")
                for f in figs:
                    st.pyplot(f, clear_figure=True)

                # Save model + labels to temp MODEL_DIR
                mpath, lpath = save_model(model, display_names, out_dir=MODEL_DIR)
                ss.cv_last_model = mpath
                ss.cv_last_labels = lpath
                ss.cv_model_name = model_name

                st.download_button("Download Model (.pt)", data=open(mpath, "rb").read(), file_name=os.path.basename(mpath))
                st.download_button("Download Labels (.json)", data=open(lpath, "rb").read(), file_name=os.path.basename(lpath))
                st.caption("labels.json preserves the EXACT display names you set above; inference will report them.")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

        # Inference
        st.markdown("### 🔮 Try Inference")
        c1, c2 = st.columns(2)
        with c1:
            pred_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="cv_pred_img2")
        with c2:
            cam_img = st.camera_input("Or Capture from Camera")
        target_img = None
        if pred_img:
            target_img = Image.open(pred_img).convert("RGB")
        elif cam_img:
            target_img = Image.open(io.BytesIO(cam_img.getvalue())).convert("RGB")

        if target_img:
            st.image(target_img, caption="Prediction Input", use_container_width=True)
            if not (ss.get("cv_last_model") and ss.get("cv_last_labels")):
                st.warning("Train or load a model first.")
            else:
                try:
                    device = best_device(device_choice)
                    model, classes_for_infer = load_model_for_infer(
                        ss.cv_last_model, ss.cv_model_name, ss.cv_last_labels, device
                    )
                    label, prob, _, _ = predict_image(model, target_img, img_size=int(img_size), device=device, classes=classes_for_infer)
                    st.success(f"Prediction: **{label}** ({prob*100:.1f}%)")
                    st.caption(f"Classes: {classes_for_infer}")
                except Exception as e:
                    st.error(f"❌ Inference failed: {e}")

        # Export CV Notebook
        st.markdown("---")
        st.subheader("📓 Export CV Training Notebook")
        if st.button("🧠 Generate CV Notebook (.ipynb)"):
            nb_bytes = build_cv_notebook_bytes(
                data_dir=ss.cv_org_dir,
                model_name=model_name,
                img_size=int(img_size),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                kernel_name="auto_ml",
            )
            st.download_button(
                "⬇️ Download Notebook",
                data=nb_bytes,
                file_name="UNISOLE_CV_Training_Notebook.ipynb",
                mime="application/x-ipynb+json",
            )