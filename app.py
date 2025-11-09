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
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check


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

# ======= Ensure folders =======
for folder in ["uploads", "models", "reports", "pipelines"]:
    os.makedirs(folder, exist_ok=True)

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
    "🧠 NLP Tasks",          # ✅ New NLP Tab
    "🧑‍🏫 AI Mentor"
])

# ===== NLP session slots (persist model + AI responses) =====
for _k, _v in {
    "nlp_df": None,                  # last loaded NLP dataframe (optional)
    "nlp_pipe": None,                # trained Pipeline
    "nlp_label_encoder": None,       # LabelEncoder
    "nlp_classes": None,             # list of class names
    "nlp_metrics": None,             # dict of metrics from last train
    "nlp_model_name": None,          # string summary of pipeline
    "nlp_ai_model_explain": None,    # last AI model explanation text
    "nlp_ai_pred_explain": None      # last AI prediction explanation text
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
            coefs = clf.coef_[pred_idx]  # shape [n_features]
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
        key="nlp_source"  # ✅ unique key
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
                SS["nlp_ai_model_explain"] = None  # reset prior AI text when retraining
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

                # ✅ Run AI explanation safely after training
                if SS.get("nlp_explain_model_flag") and SS.get("nlp_pipe") is not None:
                    SS["nlp_explain_model_flag"] = False  # reset click
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

                # ✅ Persist explanation
                if SS.get("nlp_ai_model_explain"):
                    st.info(SS["nlp_ai_model_explain"])



                # ---- LLM: Explain the trained model (Business clarity) ----
                # ✅ Set the flag when button is clicked (no AI call here)
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
                    # ✅ Validate input text
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

                        # ✅ Safe prediction (prevents "index out of bounds")
                        try:
                            pred = pipe.predict([clean_in])[0]
                        except Exception:
                            st.error("⚠️ Unable to classify this text. Try a longer or clearer sentence.")
                            st.stop()

                        pred_label = le.inverse_transform([pred])[0]
 
                        # ✅ Simple display (label only)
                        st.success(f"✅ **Prediction:** {pred_label}")

                except Exception as e:
                    st.error(f"⚠️ Inference failed: {e}")


                    # Local contribution explainer
                    pred_idx, probs, items = top_token_contributions(pipe, clean_in, top_k=8)
                    with st.expander("🔎 Why this prediction? (token contributions)"):
                        st.write(pd.DataFrame(items))

                    # LLM: Explain this prediction (persist response)
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

                    # show persisted AI prediction explanation (if any)
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
        file_id = str(uuid.uuid4())[:8]
        file_path = os.path.join("uploads", f"data_{file_id}.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"✅ File uploaded and saved as `{file_path}`")

        df = pd.read_csv(file_path)
        st.session_state.df = df
        st.session_state.processed_df = None
        st.session_state.feature_pipe = None
        st.session_state.feature_summary = None
        st.session_state.last_result = None

        st.write("### 🔎 Preview")
        st.dataframe(df.head(), use_container_width=True)

        if edu_mode and st.button("🧠 Explain this dataset"):
            explanation = explain_dataset(quick_eda(df), sample_schema={c: str(df[c].dtype) for c in df.columns})
            st.info(explanation)


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
                    # Create processed dataframe (keep target at the end)
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
                        path = os.path.join("pipelines", f"feature_pipeline_{pid}.pkl")
                        joblib.dump(st.session_state.feature_pipe, path)
                        st.success(f"Saved pipeline to `{path}`")

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

        # Note: If using cleaned dataset, target column must exist in processed_df
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

                # Save artifacts
                uid = str(uuid.uuid4())[:8]
                model_id = f"{result['summary']['best_model_name']}_{uid}"
                model_path = os.path.join("models", f"{model_id}.pkl")
                card_path = os.path.join("reports", f"{model_id}.json")
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



# ========= 🖼️ CV Training Tab (Preserve labeled datasets + Auto/Manual labeling + Rename) =========
with tab_cv:
    st.subheader("🖼️ Image Classification — Flexible Import (Preserve Labeled Datasets)")
    st.caption("Upload ANY .zip. If it’s already in class folders, we keep them EXACTLY. If not, we’ll help you label/organize. You can also rename classes before training.")

    # --- Session state ---
    ss = st.session_state
    for k, v in {
        "cv_raw_dir": None,
        "cv_img_paths": [],
        "cv_mapping": {},
        "cv_unknowns": [],
        "cv_org_dir": None,              # final training directory (either preserved labeled root or organized output)
        "cv_classes": [],
        "cv_small_hint": {},
        "cv_label_names": None,          # optional renamed labels (aligned to training class indices)
        "cv_is_labeled": False,          # True if upload already had class subfolders
        "cv_labeled_root": None,         # root that contains class subfolders (when cv_is_labeled=True)
    }.items():
        if k not in ss:
            ss[k] = v

    # --- tiny helpers for labeled detection ---
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
        """
        Heuristic: pick a directory whose immediate subfolders each contain images (class folders).
        Prefer the one with most such subfolders.
        """
        candidates = []
        for r, dirs, _ in os.walk(root):
            if not dirs:
                continue
            subdirs = [os.path.join(r, d) for d in dirs]
            class_dirs = [d for d in subdirs if _dir_has_images(d)]
            if len(class_dirs) >= 2:  # at least 2 classes
                candidates.append((r, len(class_dirs)))
        if not candidates:
            return None
        # pick the one with most class subdirs (more likely the dataset root)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # --- 1) Upload ZIP ---
    upzip = st.file_uploader("Upload dataset (.zip)", type=["zip"])
    if upzip and st.button("📦 Extract ZIP"):
        raw_dir = _extract_zip_to_tmp(upzip.read(), base_dir="uploads")
        imgs = _scan_images(raw_dir)
        if not imgs:
            st.error("❌ No images found in ZIP.")
        else:
            ss.cv_raw_dir = raw_dir
            ss.cv_img_paths = imgs
            # detect labeled structure
            labeled_root = _find_labeled_root(raw_dir)
            if labeled_root:
                ss.cv_is_labeled = True
                ss.cv_labeled_root = labeled_root
                ss.cv_org_dir = labeled_root   # train directly here (preserve!)
                try:
                    from torchvision import datasets
                    tmp = datasets.ImageFolder(labeled_root)
                    ss.cv_classes = list(tmp.classes)  # this is the exact order used by ImageFolder (alpha)
                except Exception:
                    ss.cv_classes = []
                st.success(f"✅ Labeled dataset detected. Preserving classes and structure at: {labeled_root}")
                st.write("Classes (preserved):", ss.cv_classes)
            else:
                ss.cv_is_labeled = False
                ss.cv_labeled_root = None
                st.success(f"✅ Extracted {len(imgs)} images. Not labeled yet — choose labeling below.")
                st.caption(f"Extracted to: {raw_dir}")

    # --- 2) If NOT labeled: labeling flow (Auto/CSV/Manual) ---
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
            org_dir, report, unknowns = build_org_structure(mapping, out_base="uploads")
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
                org_dir2, report2, unknowns2 = build_org_structure(new_map, out_base="uploads")
                ss.cv_org_dir = org_dir2
                ss.cv_mapping = new_map
                ss.cv_unknowns = unknowns2
                ss.cv_classes = sorted(list(report2.class_counts.keys()))
                st.success(f"Updated: {report2.labeled} labeled, {report2.unknown} unknown left.")
                st.write("📊 Class counts:", report2.class_counts)

    # --- 3) If labeled (preserved) OR we produced an organized dataset: allow renaming + training ---
    if ss.cv_org_dir:
        st.markdown("### 3️⃣ (Optional) Rename Classes For Display")
        st.caption("We will NOT rename folders — only the names **saved in labels.json** and used for inference outputs.")
        # Get current training classes from ImageFolder so indices match exactly
        try:
            from torchvision import datasets
            imgf = datasets.ImageFolder(ss.cv_org_dir)
            classes_for_training = list(imgf.classes)  # alpha order (ImageFolder)
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
        # small dataset hint (for organized flows). For preserved labeled datasets, offer it too:
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

                # Build loaders straight from ss.cv_org_dir (preserved OR organized)
                train_loader, val_loader, classes_from_loader = build_loaders(ss.cv_org_dir, cfg)

                # Ensure our display names align 1:1 in the exact order used by ImageFolder
                display_names = ss.cv_label_names if (ss.cv_label_names and len(ss.cv_label_names) == len(classes_from_loader)) else classes_from_loader

                model = build_model(cfg.model_name, len(classes_from_loader), device)
                model, history = train_loop(model, train_loader, val_loader, cfg, device)
                figs = plot_history(history)
                st.success("✅ Training Complete!")
                for f in figs:
                    st.pyplot(f, clear_figure=True)

                # Save model + labels using the display_names (do not rename folders)
                mpath, lpath = save_model(model, display_names, out_dir="models")
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

        # -----------------------
    # 🏋️ Train (with Advanced Settings)
    # -----------------------
    st.markdown("### 🏋️ Train")

    # Basic controls (keep your existing model/imgsz/device etc. if you already defined them above)
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0, key="y_model_name")
    with c2:
        imgsz = st.selectbox("Image size", [416, 512, 640], index=2, key="y_imgsz")
    with c3:
        device = st.selectbox("Device", ["", "cpu", "mps", "cuda"], index=0, key="y_device")

    c4, c5, c6 = st.columns(3)
    with c4:
        epochs = st.number_input("Epochs", min_value=1, value=50, step=1, key="y_epochs")
    with c5:
        batch = st.number_input("Batch size", min_value=4, value=16, step=2, key="y_batch")
    with c6:
        seed_val = st.number_input("Seed", min_value=0, value=42, step=1, key="y_seed")

    run_name = st.text_input("Run name", value="apple_detector_v1", key="y_run_name")

    # ========= Advanced Settings (collapsible) =========
    with st.expander("⚙️ Advanced Training Settings"):
        g1c1, g1c2, g1c3 = st.columns(3)
        with g1c1:
            optimizer = st.selectbox("Optimizer", ["AdamW", "SGD"], index=0, key="y_opt")
        with g1c2:
            lr0 = st.number_input("Learning rate (lr0)", min_value=1e-6, max_value=1.0, value=0.001, step=1e-5, format="%.5f", key="y_lr0")
        with g1c3:
            cos_lr = st.toggle("Cosine LR scheduler", value=True, key="y_coslr")

        g2c1, g2c2, g2c3 = st.columns(3)
        with g2c1:
            warmup_epochs = st.number_input("Warmup epochs", min_value=0, value=3, step=1, key="y_warmup")
        with g2c2:
            early_stop = st.toggle("Early stopping", value=True, key="y_es")
        with g2c3:
            patience = st.number_input("Patience (epochs w/o improvement)", min_value=5, value=50, step=5, key="y_patience")

        g3c1, g3c2, g3c3 = st.columns(3)
        with g3c1:
            amp = st.toggle("AMP (mixed precision)", value=True, key="y_amp")
        with g3c2:
            freeze_n = st.number_input("Freeze first N layers (0 = none)", min_value=0, value=0, step=1, key="y_freeze")
        with g3c3:
            save_every = st.toggle("Save model every N epochs", value=False, key="y_saveevery")

        g4c1, g4c2 = st.columns(2)
        with g4c1:
            save_period = st.number_input("Save period (epochs)", min_value=1, value=10, step=1, key="y_saveperiod", disabled=not save_every)
        with g4c2:
            verbose = st.toggle("Verbose training logs", value=False, key="y_verbose")

        st.markdown("**Resume Training**")
        rc1, rc2 = st.columns([0.6, 0.4])
        with rc1:
            resume = st.toggle("Resume from a weights file (last.pt)", value=False, key="y_resume")
        default_resume_path = ""
        # suggest last run's 'last.pt' if we have it
        try:
            if st.session_state.get("yolo_last_run"):
                maybe_last = os.path.join(st.session_state["yolo_last_run"], "weights", "last.pt")
                if os.path.exists(maybe_last):
                    default_resume_path = maybe_last
        except Exception:
            pass
        with rc2:
            resume_path = st.text_input("Resume path (.pt)", value=default_resume_path, key="y_resumepath", disabled=not resume)

        st.caption("Tip: For small datasets, try freezing some layers and using AdamW with a slightly lower LR (e.g., 0.0005).")

    # ---- Train button: run Ultralytics with advanced args ----
    if st.button("🚀 Train YOLO", key="y_train_btn"):
        from ultralytics import YOLO

        # Build model: resume from path if requested & exists, else from selected base model
        use_resume = bool(resume and resume_path and os.path.exists(resume_path))
        model = YOLO(resume_path if use_resume else model_name)

        # If resuming, Ultralytics supports resume=True; if not, leave resume=False
        resume_flag = True if use_resume else False

        with st.spinner("Training YOLO…"):
            results = model.train(
                data=st.session_state.yolo_data_yaml,
                imgsz=int(imgsz),
                epochs=int(epochs),
                batch=int(batch),
                device=(device or None),
                project="runs/detect",
                name=run_name,
                seed=int(seed_val),

                # Pro controls
                optimizer=optimizer.lower(),     # "adamw" or "sgd"
                lr0=float(lr0),
                cos_lr=bool(cos_lr),
                warmup_epochs=int(warmup_epochs),
                patience=int(patience) if early_stop else 1000000,  # effectively disable if ES off
                amp=bool(amp),
                resume=resume_flag,
                freeze=int(freeze_n) if int(freeze_n) > 0 else None,
                save_period=int(save_period) if save_every else -1,
                verbose=bool(verbose),
            )

        save_dir = results.save_dir  # e.g., runs/detect/<run_name>
        best_path = os.path.join(save_dir, "weights", "best.pt")
        last_path = os.path.join(save_dir, "weights", "last.pt")

        # Load names from best
        try:
            model_best = YOLO(best_path)
            names = model_best.names  # {id:name}
        except Exception:
            names = {}

        # persist in session
        st.session_state.yolo_best = best_path
        st.session_state.yolo_names = names
        st.session_state.yolo_last_run = save_dir

        st.success("✅ Training complete!")
        st.write("Save dir:", save_dir)
        st.download_button("⬇️ Download best.pt", data=open(best_path, "rb").read(), file_name="best.pt")

        # ------- Minimal & Clean training summary (reuse helper if present; define if not) -------
        try:
            _show_training_summary  # type: ignore # noqa
        except NameError:
            # define helpers if your earlier block didn't define them
            import pandas as _pd
            def _read_results_csv(_save_dir: str):
                _csv = os.path.join(_save_dir, "results.csv")
                if not os.path.exists(_csv):
                    return None, None
                try:
                    return _pd.read_csv(_csv), _csv
                except Exception:
                    return None, None

            def _get_last_metrics(df):
                if df is None or len(df) == 0: return {}
                row = df.iloc[-1]
                def pick(*keys):
                    for k in keys:
                        for c in df.columns:
                            if k in c.lower(): return float(row[c])
                    return None
                return {
                    "precision": pick("metrics/precision", "precision"),
                    "recall": pick("metrics/recall", "recall"),
                    "map50": pick("metrics/map50", "map50"),
                    "map5095": pick("metrics/map50-95", "map50-95", "map_0.5:0.95"),
                    "epochs": int(row.get("epoch", len(df)-1)) + 1 if "epoch" in df.columns else len(df)
                }

            def _show_training_summary(_save_dir: str, _names: dict):
                df, csv_path = _read_results_csv(_save_dir)
                metrics = _get_last_metrics(df) if df is not None else {}
                png_path = os.path.join(_save_dir, "results.png")
                classes_str = ", ".join([_names[k] for k in sorted(_names.keys())]) if _names else "—"
                lines = ["✅ Training Complete"]
                if metrics.get("map50")  is not None:  lines.append(f"📊 mAP50: **{metrics['map50']:.3f}**")
                if metrics.get("map5095") is not None: lines.append(f"📈 mAP50-95: **{metrics['map5095']:.3f}**")
                if metrics.get("precision") is not None: lines.append(f"🎯 Precision: **{metrics['precision']:.3f}**")
                if metrics.get("recall") is not None:    lines.append(f"👌 Recall: **{metrics['recall']:.3f}**")
                if metrics.get("epochs") is not None:    lines.append(f"⏱️ Epochs: **{metrics['epochs']}**")
                lines.append(f"🍎 Classes: **{classes_str}**")
                st.success("\n".join(lines))
                if os.path.exists(png_path):
                    st.image(png_path, caption="Training curves (loss, P/R, mAP)", use_container_width=True)
                if csv_path and os.path.exists(csv_path):
                    st.download_button("⬇️ Download results.csv", data=open(csv_path, "rb").read(),
                                       file_name="results.csv", mime="text/csv")

        # Show summary now
        _show_training_summary(save_dir, names)

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
        # =========================
    # 📊 Model Evaluation (Validation set)
    # =========================
    st.markdown("---")
    with st.expander("📊 Model Evaluation (Validation set)", expanded=False):

        if not (st.session_state.get("yolo_best") and st.session_state.get("yolo_data_yaml")):
            st.info("Train a model first (or set best.pt) and make sure `data.yaml` is detected.")
        else:
            import yaml, glob
            from ultralytics import YOLO

            # Small helper: run validation and return metrics + where plots were saved
            def _run_yolo_eval(weights_path: str, data_yaml: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45, device: str = ""):
                model = YOLO(weights_path)
                results = model.val(
                    data=data_yaml,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=(device or None),
                    split="val",
                    verbose=False,
                )
                # results.metrics has mp, mr, map50, map, maps (per-class)
                m = results.results_dict if hasattr(results, "results_dict") else None
                # Fallback to metrics object
                metrics = getattr(results, "metrics", None)
                box = getattr(metrics, "box", None) if metrics else None

                # Compose summary dict robustly
                summary = {}
                if m:
                    # Ultralytics often provides these keys
                    summary["precision"] = float(m.get("metrics/precision(B)", m.get("precision", 0.0)))
                    summary["recall"]    = float(m.get("metrics/recall(B)",    m.get("recall", 0.0)))
                    summary["map50"]     = float(m.get("metrics/mAP50(B)",     m.get("map50", 0.0)))
                    summary["map5095"]   = float(m.get("metrics/mAP50-95(B)",  m.get("map", 0.0)))
                elif box:
                    summary["precision"] = float(getattr(metrics, "precision", getattr(box, "mp", 0.0)))
                    summary["recall"]    = float(getattr(metrics, "recall",    getattr(box, "mr", 0.0)))
                    summary["map50"]     = float(getattr(box, "map50", 0.0))
                    summary["map5095"]   = float(getattr(box, "map", 0.0))
                else:
                    summary = {"precision": None, "recall": None, "map50": None, "map5095": None}

                # Per-class mAP (0.5:0.95)
                per_class_map = []
                names = model.names  # {id: name}
                try:
                    maps = list(box.maps) if box and hasattr(box, "maps") else None
                    if maps:
                        for cid, ap in enumerate(maps):
                            per_class_map.append({"class_id": cid, "class_name": names.get(cid, str(cid)), "mAP50-95": float(ap)})
                except Exception:
                    pass

                return {
                    "summary": summary,
                    "per_class_map": per_class_map,
                    "save_dir": getattr(results, "save_dir", None),
                    "names": names
                }

            # Helper: pull some images from val folder declared in data.yaml
            def _sample_val_images_from_yaml(yaml_path: str, limit: int = 6):
                try:
                    with open(yaml_path, "r") as f:
                        y = yaml.safe_load(f)
                    val_path = y.get("val")
                    if not val_path:
                        return []
                    patterns = []
                    if os.path.isdir(val_path):
                        # common image globs
                        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                            patterns += glob.glob(os.path.join(val_path, "**", ext), recursive=True)
                    else:
                        # if val is a txt with list of image paths
                        if os.path.isfile(val_path) and val_path.endswith(".txt"):
                            with open(val_path, "r") as vf:
                                patterns = [ln.strip() for ln in vf if ln.strip()]
                    return sorted(patterns)[:limit]
                except Exception:
                    return []

            # UI controls for evaluation
            c1, c2, c3 = st.columns(3)
            with c1:
                eval_imgsz = st.selectbox("Eval size", [416, 512, 640], index=2, key="y_eval_imgsz")
            with c2:
                eval_conf = st.slider("Eval conf", 0.1, 0.9, 0.25, 0.05, key="y_eval_conf")
            with c3:
                eval_iou  = st.slider("Eval IoU", 0.1, 0.9, 0.45, 0.05, key="y_eval_iou")
            dev_eval = st.selectbox("Device", ["", "cpu", "mps", "cuda"], index=0, key="y_eval_dev")

            if st.button("📏 Run Evaluation on Val Set"):
                with st.spinner("Evaluating on validation set…"):
                    ev = _run_yolo_eval(
                        weights_path=st.session_state.yolo_best,
                        data_yaml=st.session_state.yolo_data_yaml,
                        imgsz=int(eval_imgsz),
                        conf=float(eval_conf),
                        iou=float(eval_iou),
                        device=dev_eval or "",
                    )

                # ✅ Minimal & Clean summary
                s = ev["summary"]
                class_list = ", ".join([ev["names"][k] for k in sorted(ev["names"].keys())]) if ev["names"] else "—"
                lines = ["✅ Evaluation Complete"]
                if s.get("map50")  is not None:  lines.append(f"📊 mAP50: **{s['map50']:.3f}**")
                if s.get("map5095") is not None: lines.append(f"📈 mAP50-95: **{s['map5095']:.3f}**")
                if s.get("precision") is not None: lines.append(f"🎯 Precision: **{s['precision']:.3f}**")
                if s.get("recall") is not None:    lines.append(f"👌 Recall: **{s['recall']:.3f}**")
                lines.append(f"🍎 Classes: **{class_list}**")
                st.success("\n".join(lines))

                # Per-class mAP table
                if ev["per_class_map"]:
                    import pandas as _pd
                    df_ap = _pd.DataFrame(ev["per_class_map"]).sort_values("mAP50-95", ascending=False)
                    st.markdown("#### Per-class mAP (0.5:0.95)")
                    st.dataframe(df_ap.reset_index(drop=True), use_container_width=True)

                # Confusion matrix image (Ultralytics saves it under val run dir)
                if ev["save_dir"]:
                    cm_png = os.path.join(ev["save_dir"], "confusion_matrix.png")
                    if os.path.exists(cm_png):
                        st.markdown("#### Confusion Matrix")
                        st.image(cm_png, use_container_width=True)

                # Validation previews (sample a few val images and run inference)
                st.markdown("#### Validation Previews")
                val_samples = _sample_val_images_from_yaml(st.session_state.yolo_data_yaml, limit=6)
                if not val_samples:
                    st.caption("No val images found (check your `data.yaml`).")
                else:
                    cols = st.columns(3)
                    for i, p in enumerate(val_samples):
                        try:
                            img = Image.open(p).convert("RGB")
                            annotated, counts = infer_image_bytes(
                                st.session_state.yolo_best,
                                img,
                                imgsz=int(eval_imgsz),
                                conf=float(eval_conf),
                                iou=float(eval_iou),
                                device=dev_eval or ""
                            )
                            with cols[i % 3]:
                                st.image(annotated, caption=f"{os.path.basename(p)} • {counts}", use_container_width=True)
                        except Exception as e:
                            pass


    # --- Optional: quick access to last run metrics again ---
    if st.session_state.get("yolo_last_run"):
        with st.expander("📈 Show last training summary again"):
            _show_training_summary(st.session_state["yolo_last_run"], st.session_state.get("yolo_names", {}))




# ========= Explainability Tab =========
with tab_explain:
    st.subheader("⚡ Explainability (Teaching Mode)")
    if st.session_state.last_result is None or st.session_state.df is None or st.session_state.target is None:
        st.warning("Train a model in the **AutoML** tab first.")
    else:
        df = st.session_state.df
        target = st.session_state.target
        artifact = st.session_state.last_result["artifact"]

        # Use original features here (artifact contains its own preprocessing)
        X = df.drop(columns=[target])

        # 1) Global Feature Importance
        st.markdown("#### 1) Global Feature Importance")
        try:
            fig_imp, imp_df = global_feature_importance(artifact, X, top_n=20)
            st.pyplot(fig_imp, clear_figure=True)
            st.dataframe(imp_df, use_container_width=True)
            if st.toggle("🧠 Explain this importance chart", value=False):
                ctx = {"top_features": imp_df.to_dict(orient="records")}
                st.info(ask_ai_mentor("Explain global feature importance and how to use it.", ctx))
        except Exception as e:
            st.error(f"Could not compute global importance: {e}")

        st.markdown("---")

        # 2) SHAP Global Summary
        st.markdown("#### 2) SHAP Summary (Global Behavior)")
        try:
            with st.spinner("Building SHAP explainer (sampling for speed)…"):
                explainer, Xt_sample, feat_names = build_shap_explainer(artifact, X, sample_size=500)
            fig_shap = shap_summary_fig(explainer, Xt_sample, feat_names)
            st.pyplot(fig_shap, clear_figure=True)
            if st.toggle("🧠 Explain this SHAP summary", value=False):
                st.info(ask_ai_mentor("Explain the SHAP beeswarm plot in simple terms.", {}))
        except Exception as e:
            st.error(f"Could not produce SHAP summary: {e}")

        st.markdown("---")

        # 3) SHAP Single Prediction
        st.markdown("#### 3) Explain a Single Prediction")
        c1, c2 = st.columns(2)
        with c1:
            row_idx = st.number_input("Row index to explain", min_value=0, max_value=len(X)-1, value=0, step=1)
            btn_explain = st.button("🔍 Explain this row")
        with c2:
            auto_pick = st.selectbox("Auto-pick strategy", ["max (highest prediction)", "min (lowest prediction)"], index=0)
            btn_auto = st.button("🎯 Auto-pick interesting row and explain")

        if btn_explain:
            try:
                with st.spinner("Computing SHAP for the selected row…"):
                    explainer, _, feat_names = build_shap_explainer(artifact, X, sample_size=500)
                    fig_one, contribs = shap_single_waterfall_fig(explainer, artifact, X, int(row_idx))
                st.pyplot(fig_one, clear_figure=True)
                st.caption("Top contributions (feature, SHAP value):")
                st.write(contribs)
                if st.toggle("🧠 Explain this prediction", value=False):
                    st.info(ask_ai_mentor("Explain why the model produced this prediction for the row.", {"top_contrib": contribs}))
            except Exception as e:
                st.error(f"Could not explain this row: {e}")

        if btn_auto:
            try:
                strategy = "max" if "max" in auto_pick else "min"
                idx = pick_interesting_row(artifact, X, strategy=strategy)
                with st.spinner(f"Explaining auto-picked row #{idx}…"):
                    explainer, _, feat_names = build_shap_explainer(artifact, X, sample_size=500)
                    fig_one, contribs = shap_single_waterfall_fig(explainer, artifact, X, idx)
                st.info(f"Auto-picked row index: **{idx}**")
                st.pyplot(fig_one, clear_figure=True)
                st.caption("Top contributions (feature, SHAP value):")
                st.write(contribs)
                if st.toggle("🧠 Explain this auto-picked prediction", value=False):
                    st.info(ask_ai_mentor("Explain this prediction in plain English.", {"top_contrib": contribs}))
            except Exception as e:
                st.error(f"Could not auto-explain a row: {e}")
# ========= Notebook Export Tab =========
with tab_nb:
    st.subheader("📓 Notebook Export (Education Mode)")
    if st.session_state.df is None or st.session_state.target is None:
        st.warning("Upload data and pick a target first (Feature Engineering/AutoML tabs).")
    else:
        df = st.session_state.df
        target = st.session_state.target

        st.write("This will generate a **reproducible Jupyter Notebook** with:")
        st.markdown("- Data loading\n- Mini-EDA\n- Optional Feature Engineering\n- AutoML training\n- Metrics & model card\n- Inference example")

        include_mini_eda = st.toggle("Include Mini-EDA", value=True)
        include_ai_intro = st.toggle("Include AI Mentor intro section", value=True)

        # Figure out saved artifacts (if any)
        model_path = None
        model_card = None
        if st.session_state.last_result:
            # save model card dict
            model_card = st.session_state.last_result.get("summary")
            # find the latest saved model path in /models (user already downloaded earlier, but we embed path)
            # It's okay if None; notebook will fallback to baseline
            # (You could store exact model path during AutoML save if desired)
        pipe_path = None
        if st.session_state.feature_pipe is not None:
            # Let the user save pipeline first if they haven't
            st.info("A feature pipeline is available. Save it to embed its path in the notebook (optional).")
            if st.button("💾 Save current pipeline for notebook"):
                pid = str(uuid.uuid4())[:8]
                path = os.path.join("pipelines", f"feature_pipeline_{pid}.pkl")
                joblib.dump(st.session_state.feature_pipe, path)
                st.success(f"Saved pipeline to `{path}`")
                pipe_path = path

        # Allow user to manually enter artifact paths (optional)
        st.markdown("**(Optional)** If you have already saved artifacts, you can specify them here:")
        model_path = st.text_input("Saved model .pkl path (optional; leave blank to train baseline in notebook)", value=model_path or "")
        pipe_path = st.text_input("Saved feature pipeline .pkl path (optional)", value=pipe_path or "")

        # AI intro text (optional)
        ai_intro_text = None
        if include_ai_intro:
            try:
                ai_intro_text = ask_ai_mentor(
                    "Write a short, friendly introduction (6–10 lines) for a student opening this notebook. Encourage exploration.",
                    {"task": st.session_state.last_task, "target": target}
                )
            except Exception:
                ai_intro_text = None

        # Build and download
        if st.button("📥 Generate Notebook (.ipynb)"):
            csv_guess = ""
            # Find uploaded CSV path (best-effort: latest file in uploads/)
            try:
                files = sorted([os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".csv")], key=os.path.getmtime, reverse=True)
                if files:
                    csv_guess = files[0]
            except Exception:
                pass

            with st.spinner("Building notebook..."):
                nb_bytes = build_notebook_bytes(
                    csv_path=csv_guess or "YOUR_DATA.csv",
                    target=target,
                    task=st.session_state.last_task or "classification",
                    model_card=model_card,
                    saved_model_path=model_path if model_path.strip() else None,
                    saved_pipeline_path=pipe_path if pipe_path.strip() else None,
                    include_mini_eda=include_mini_eda,
                    include_llm_intro=ai_intro_text,
                    kernel_name="auto_ml",  # your conda env name
                )

            st.success("✅ Notebook generated!")
            st.download_button(
                "Download Notebook (.ipynb)",
                data=nb_bytes,
                file_name="UNISOLE_AutoAI_Notebook.ipynb",
                mime="application/x-ipynb+json"
            )
            st.caption("Open it in Jupyter, VS Code, or Google Colab.")


# ========= 🔮 AI Forecasting (Flexible + LLM Business Insight, Safe) =========
with tab_forecast:
    st.subheader("🔮 AI Forecasting — Flexible Time Series + Business Insights")
    st.caption("Works with messy datasets: auto-detects date column, coerces target to numeric, and handles resampling safely.")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Prophet availability
    try:
        from prophet import Prophet
        _HAS_PROPHET = True
    except Exception:
        _HAS_PROPHET = False

    # ---------- Session slots ----------
    if "ts_forecast" not in st.session_state: st.session_state.ts_forecast = None
    if "ts_metrics"  not in st.session_state: st.session_state.ts_metrics  = None
    if "ts_target"   not in st.session_state: st.session_state.ts_target   = None
    if "ts_history"  not in st.session_state: st.session_state.ts_history  = None  # cleaned historical data (ds,y)

    # ---------- Helpers (robust) ----------
    def _auto_date_col(df: pd.DataFrame) -> str | None:
        # look for obvious names first
        candidates = [c for c in df.columns if c.lower() in ("date","ds","datetime","timestamp","time")]
        for c in candidates:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
        # try any column that can parse
        for c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
        return None

    def _clean_and_prepare(df: pd.DataFrame, date_col: str, target_col: str, freq: str | None) -> tuple[pd.DataFrame, list[str]]:
        """
        Returns (dfc, warnings) where dfc has columns ['ds','y'].
        - Coerces date to datetime, target to numeric
        - Drops invalid rows
        - Optional resample on numeric target only
        """
        warns = []

        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataset.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        work = df[[date_col, target_col]].copy()

        # Parse date
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        bad_dates = work[date_col].isna().sum()
        if bad_dates > 0:
            warns.append(f"Dropped {bad_dates} rows with invalid dates.")
        work = work.dropna(subset=[date_col])

        # Coerce numeric target
        pre_n = len(work)
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        bad_t = work[target_col].isna().sum()
        if bad_t > 0:
            warns.append(f"Coerced target to numeric: dropped {bad_t} non-numeric rows.")
        work = work.dropna(subset=[target_col])

        # Sort
        work = work.sort_values(date_col)

        # Optional resample (mean over target)
        if freq:
            if work.empty:
                raise ValueError("No valid rows after cleaning; cannot resample.")
            try:
                work = (work
                        .set_index(date_col)
                        .resample(freq)[target_col]
                        .mean()
                        .to_frame()
                        .reset_index())
            except Exception as e:
                warns.append(f"Resample failed ({e}); continuing without resample.")
        # Rename for Prophet
        work = work.rename(columns={date_col: "ds", target_col: "y"})

        # Final sanity
        if work.empty or work["y"].nunique() <= 1:
            raise ValueError("Insufficient variation in target after cleaning. Need more than one distinct numeric value.")

        # Drop duplicates / sort
        work = (work.drop_duplicates(subset=["ds"]).sort_values("ds")).reset_index(drop=True)
        return work, warns

    def _tail_split(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(df)
        ts = int(max(1, min(test_size, max(1, n // 5))))  # cap to 20% if large
        if n <= ts:
            ts = max(1, n - 1)
        return df.iloc[:-ts].copy(), df.iloc[-ts:].copy()

    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        # avoid divide by zero
        denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
        mape = float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0
        return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

    # ---------- UI ----------
    st.markdown("### 📥 Step 1: Load data")
    source = st.radio("Source", ["Use uploaded CSV (Data Lab)", "Upload CSV here"], horizontal=True)
    df_ts = None
    if source == "Use uploaded CSV (Data Lab)":
        if st.session_state.get("df") is not None:
            df_ts = st.session_state["df"]
            st.success("Using Data Lab dataset")
            st.dataframe(df_ts.head(), use_container_width=True)
        else:
            st.warning("No dataset present in Data Lab. Upload below.")
    else:
        up = st.file_uploader("Upload time series CSV", type=["csv"])
        if up:
            try:
                df_ts = pd.read_csv(up)
                st.dataframe(df_ts.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if df_ts is not None:
        st.markdown("### ⚙️ Step 2: Configure")
        guess_date = _auto_date_col(df_ts)
        date_col = st.selectbox("Date/Time column", options=df_ts.columns,
                                index=(list(df_ts.columns).index(guess_date) if guess_date in df_ts.columns else 0))
        target_col = st.selectbox("Target column (numeric to forecast)", options=[c for c in df_ts.columns if c != date_col])

        c1, c2, c3 = st.columns(3)
        with c1:
            freq = st.selectbox("Resampling", ["None", "D - Daily", "W - Weekly", "M - Monthly"], index=0)
        with c2:
            horizon = st.number_input("Forecast horizon (periods)", min_value=5, value=30, step=1)
        with c3:
            test_tail = st.number_input("Validation tail size", min_value=3, value= min(30, max(3, len(df_ts)//10 or 3)), step=1)

        freq_map = {"None": None, "D - Daily": "D", "W - Weekly": "W", "M - Monthly": "M"}
        use_freq = freq_map[freq]

        if st.button("🚀 Run Forecast"):
            if not _HAS_PROPHET:
                st.error("`prophet` not installed. Install: `pip install prophet`")
            else:
                try:
                    # Clean & prepare safely
                    dfc, warns = _clean_and_prepare(df_ts, date_col, target_col, use_freq)
                    if warns:
                        for w in warns:
                            st.caption(f"ℹ️ {w}")
                    if len(dfc) < 10:
                        st.warning("Dataset is small after cleaning; forecasts may be unstable.")

                    # Train/validation split on the tail
                    train, val = _tail_split(dfc, int(test_tail))

                    # Fit on train, validate on val
                    m = Prophet(seasonality_mode="additive")
                    m.fit(train)
                    val_fore = m.predict(val[["ds"]])
                    mets = _metrics(val["y"].values, val_fore["yhat"].values)

                    # Fit on full cleaned history for final forecast
                    m_full = Prophet(seasonality_mode="additive")
                    m_full.fit(dfc)
                    future = m_full.make_future_dataframe(periods=int(horizon), include_history=True)
                    forecast = m_full.predict(future)

                    # Persist to session for LLM
                    st.session_state.ts_forecast = forecast
                    st.session_state.ts_metrics  = mets
                    st.session_state.ts_target   = target_col
                    st.session_state.ts_history  = dfc

                    st.success("✅ Forecast complete!")

                    # Plot
                    fig = plt.figure(figsize=(10, 4))
                    plt.plot(dfc["ds"], dfc["y"], label="Actual")
                    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
                    try:
                        plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.25, label="Uncertainty")
                    except Exception:
                        pass
                    plt.title(f"Forecast for {target_col}")
                    plt.legend()
                    st.pyplot(fig, clear_figure=True)

                    # Metrics
                    a, b, c = st.columns(3)
                    a.metric("MAE", f"{mets['MAE']:.3f}")
                    b.metric("RMSE", f"{mets['RMSE']:.3f}")
                    c.metric("MAPE %", f"{mets['MAPE%']:.2f}")

                    # Download
                    out = forecast[["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"date"})
                    st.download_button("⬇️ Download forecast.csv", data=out.to_csv(index=False).encode("utf-8"),
                                       file_name="forecast.csv", mime="text/csv")

                    # Components (optional)
                    with st.expander("📈 Trend & Seasonality (Prophet components)"):
                        comp = m_full.plot_components(forecast)
                        st.pyplot(comp, clear_figure=True)

                except ValueError as ve:
                    st.error(f"Data validation error: {ve}")
                except Exception as e:
                    st.error(f"Forecast failed: {e}")

    # ---------- 💬 Business AI Insight (works after forecast; uses session) ----------
    if st.session_state.ts_forecast is not None and st.session_state.ts_metrics is not None:
        st.markdown("---")
        st.markdown("### 💬 AI Business Insight")
        if st.button("💬 Explain this forecast"):
            try:
                forecast = st.session_state.ts_forecast
                mets     = st.session_state.ts_metrics
                target   = st.session_state.ts_target
                history  = st.session_state.ts_history

                # Basic business cues
                trend_dir = "upward 📈" if forecast["yhat"].iloc[-1] > forecast["yhat"].iloc[0] else "downward 📉"
                uncertainty = float((forecast.get("yhat_upper", forecast["yhat"]) - forecast.get("yhat_lower", forecast["yhat"])).abs().mean())
                volatility  = float(forecast["yhat"].std())

                # Build prompt
                prompt = f"""
                You are a senior business analyst. Explain this forecast and give actionable guidance.
                Target: {target}
                Trend: {trend_dir}
                Metrics: MAE={mets['MAE']:.3f}, RMSE={mets['RMSE']:.3f}, MAPE={mets['MAPE%']:.2f}%
                Uncertainty (avg band width): {uncertainty:.3f}
                Volatility (pred std): {volatility:.3f}

                Provide a concise, executive-friendly analysis with:
                1) Business interpretation of the trend and any seasonal patterns,
                2) Risks and confidence considerations,
                3) 3–5 specific, actionable recommendations (inventory, pricing, marketing, ops),
                4) One short KPI to watch next.
                Keep it practical and outcome-oriented.
                """

                # Use your existing LLM helper
                reply = ask_ai_mentor(prompt, {
                    "module": "forecast",
                    "target": target,
                    "metrics": mets,
                    "trend": trend_dir,
                    "uncertainty": uncertainty,
                    "volatility": volatility
                })
                st.success(reply)
            except Exception as e:
                st.error(f"AI explanation failed: {e}")

    # ==============================
    # 🔧 Multi-Model Forecast (Prophet + AutoARIMA + XGBoost)
    # ==============================
    st.markdown("---")
    st.subheader("🔧 Multi-Model Forecast (Prophet • AutoARIMA • XGBoost)")

    # Only proceed if we have a cleaned time series in memory
    if st.session_state.ts_history is None:
        st.info("Run a forecast above first to prepare the cleaned time series (ds,y).")
    else:
        dfc = st.session_state.ts_history.copy()  # cleaned df with columns ds,y
        horizon_mm = st.number_input("Horizon (periods)", min_value=5, value=30, step=1, key="mm_horizon")
        tail_mm    = st.number_input("Validation tail size", min_value=5, value=20, step=1, key="mm_tail")

        # Choose which models to run
        models_to_run = st.multiselect(
            "Choose models",
            ["Prophet", "AutoARIMA", "XGBoost"],
            default=["Prophet", "AutoARIMA", "XGBoost"],
            key="mm_models"
        )

        # Try imports for optional models
        _HAS_ARIMA = True
        _HAS_XGB   = True
        try:
            import pmdarima as pm
        except Exception:
            _HAS_ARIMA = False
        try:
            from xgboost import XGBRegressor
        except Exception:
            _HAS_XGB = False

        # Helpers
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt

        def _tail_split(df: _pd.DataFrame, test_size: int):
            n = len(df)
            ts = int(max(1, min(test_size, max(1, n // 5))))
            if n <= ts:
                ts = max(1, n - 1)
            return df.iloc[:-ts].copy(), df.iloc[-ts:].copy()

        def _metrics(y_true, y_pred):
            y_true = _np.asarray(y_true, dtype=float)
            y_pred = _np.asarray(y_pred, dtype=float)
            mae  = float(_np.mean(_np.abs(y_true - y_pred)))
            rmse = float(_np.sqrt(_np.mean((y_true - y_pred)**2)))
            denom = _np.where(_np.abs(y_true) < 1e-9, 1e-9, _np.abs(y_true))
            mape = float(_np.mean(_np.abs((y_true - y_pred) / denom))) * 100.0
            return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

        # Fit/predict wrappers
        def _fit_predict_prophet(train_df, full_df, horizon):
            from prophet import Prophet
            m = Prophet(seasonality_mode="additive")
            m.fit(train_df)
            # Validation prediction (on val ds)
            val_hat = m.predict(val_df[["ds"]])["yhat"].values
            # Full + future
            m_full = Prophet(seasonality_mode="additive")
            m_full.fit(full_df)
            future = m_full.make_future_dataframe(periods=horizon, include_history=True)
            fore  = m_full.predict(future)
            return val_hat, fore.rename(columns={"ds":"date"})

        def _fit_predict_autoarima(train_df, full_df, horizon):
            # train on y only (regular spacing assumed post-clean; if irregular, ARIMA roughly still ok)
            ar = pm.auto_arima(
                train_df["y"].values,
                error_action="ignore", suppress_warnings=True, seasonal=False,
                information_criterion="aic"
            )
            # validation prediction, same length as val
            val_hat = ar.predict(n_periods=len(val_df))
            # refit on full and predict future
            ar_full = pm.auto_arima(full_df["y"].values, error_action="ignore", suppress_warnings=True, seasonal=False)
            fut_hat = ar_full.predict(n_periods=horizon)
            # build forecast frame aligned with end
            last_ds = full_df["ds"].iloc[-1]
            # infer frequency from cleaned index deltas if possible
            freq = _pd.infer_freq(full_df["ds"].sort_values())
            if freq is None:
                # default daily if unknown
                freq = "D"
            future_dates = _pd.date_range(start=last_ds, periods=horizon+1, freq=freq)[1:]
            fore = _pd.DataFrame({
                "date": _pd.concat([full_df["ds"], _pd.Series(future_dates)]).reset_index(drop=True)
            })
            # stitch yhat: history + future
            hist = full_df["y"].reset_index(drop=True)
            yhat_full = _pd.concat([hist, _pd.Series(fut_hat)], ignore_index=True)
            fore["yhat"] = yhat_full.values
            fore["yhat_lower"] = _np.nan
            fore["yhat_upper"] = _np.nan
            return val_hat, fore

        def _fit_predict_xgb(train_df, full_df, horizon, lags=14):
            """
            Build a lag-based supervised problem for XGBoost.
            """
            # Build supervised matrix from full_df for training on train_df region
            y = full_df["y"].values.astype(float)
            # create lag features
            X_all = []
            for i in range(lags, len(y)):
                X_all.append(y[i-lags:i])
            X_all = _np.array(X_all)
            y_all = y[lags:]

            # align ds
            ds_all = full_df["ds"].iloc[lags:].reset_index(drop=True)

            # determine split index that separates train/val inside this lagged space
            split_ds = train_df["ds"].iloc[-1]
            split_idx = int((ds_all <= split_ds).sum())

            X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
            X_va, y_va = X_all[split_idx:], y_all[split_idx:]

            if len(X_tr) < 10 or len(X_va) < 1:
                raise ValueError("Not enough samples for XGBoost after lag transformation. Try reducing 'Validation tail size' or 'lags'.")

            # Train XGB
            xgb = XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                n_jobs=4,
                random_state=42
            )
            xgb.fit(X_tr, y_tr)

            # Validation preds (one-shot on aligned matrix)
            val_hat = xgb.predict(X_va)

            # Recursive future forecast: roll-forward using last lags from full y + predicted
            history = list(y[-lags:])  # last lags
            future_vals = []
            for _ in range(int(horizon)):
                x = _np.array(history[-lags:]).reshape(1, -1)
                yhat = float(xgb.predict(x))
                future_vals.append(yhat)
                history.append(yhat)

            # Build forecast frame
            freq = _pd.infer_freq(full_df["ds"].sort_values())
            if freq is None:
                freq = "D"
            future_dates = _pd.date_range(start=full_df["ds"].iloc[-1], periods=horizon+1, freq=freq)[1:]
            fore = _pd.DataFrame({
                "date": _pd.concat([full_df["ds"], _pd.Series(future_dates)]).reset_index(drop=True)
            })
            yhat_full = _np.concatenate([y, _np.array(future_vals)], axis=0)
            fore["yhat"] = yhat_full
            fore["yhat_lower"] = _np.nan
            fore["yhat_upper"] = _np.nan
            return val_hat, fore

        # Build train/val split once
        train_df, val_df = _tail_split(dfc, int(tail_mm))

        # Run selected models
        rows = []
        forecasts = {}  # name -> forecast df with columns [date, yhat, yhat_lower, yhat_upper] (lower/upper may be NaN)
        val_truth = val_df["y"].values

        if "Prophet" in models_to_run:
            try:
                vhat, fore = _fit_predict_prophet(train_df, dfc, horizon_mm)
                m = _metrics(val_truth, vhat)
                rows.append({"Model":"Prophet", **m})
                forecasts["Prophet"] = fore[["date","yhat","yhat_lower","yhat_upper"]]
            except Exception as e:
                st.error(f"Prophet failed: {e}")

        if "AutoARIMA" in models_to_run:
            if not _HAS_ARIMA:
                st.warning("AutoARIMA unavailable — install with `pip install pmdarima`")
            else:
                try:
                    vhat, fore = _fit_predict_autoarima(train_df, dfc, horizon_mm)
                    m = _metrics(val_truth, vhat)
                    rows.append({"Model":"AutoARIMA", **m})
                    forecasts["AutoARIMA"] = fore[["date","yhat","yhat_lower","yhat_upper"]]
                except Exception as e:
                    st.error(f"AutoARIMA failed: {e}")

        if "XGBoost" in models_to_run:
            if not _HAS_XGB:
                st.warning("XGBoost unavailable — install with `pip install xgboost`")
            else:
                try:
                    vhat, fore = _fit_predict_xgb(train_df, dfc, horizon_mm, lags=14)
                    m = _metrics(val_truth, vhat)
                    rows.append({"Model":"XGBoost (lags=14)", **m})
                    forecasts["XGBoost (lags=14)"] = fore[["date","yhat","yhat_lower","yhat_upper"]]
                except Exception as e:
                    st.error(f"XGBoost failed: {e}")

        if not rows:
            st.info("No models ran successfully yet.")
        else:
            # Leaderboard
            lb = _pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
            st.markdown("### 🏆 Leaderboard (lower RMSE is better)")
            st.dataframe(lb, use_container_width=True)

            # Combined plot
            st.markdown("### 📈 Combined Forecast")
            fig = _plt.figure(figsize=(10,4))
            _plt.plot(dfc["ds"], dfc["y"], label="Actual", linewidth=1.5)
            # Plot each model's forecast line
            for name, fdf in forecasts.items():
                _plt.plot(fdf["date"], fdf["yhat"], label=name)
            _plt.legend()
            st.pyplot(fig, clear_figure=True)

            # Download best forecast
            best_name = lb.iloc[0]["Model"]
            best_fore = forecasts[best_name].copy()
            best_fore_csv = best_fore.to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️ Download best forecast CSV ({best_name})", data=best_fore_csv, file_name="best_forecast.csv", mime="text/csv")

            # Small tip for users
            st.caption("Tip: Different models excel on different patterns. Try adjusting validation tail size and horizon to match your real decision window.")






# ========= Mentor Tab =========
with tab_mentor:
    st.subheader("🧑‍🏫 LLM Mentor — Ask anything about your data or models")
    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("Ask about algorithms, metrics, tuning, errors, or next steps…")
    if user_msg:
        st.session_state.chat.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)
        context = {
            "task": st.session_state.last_task,
            "eda": quick_eda(st.session_state.df) if st.session_state.df is not None else None,
            "last_summary": (st.session_state.last_result or {}).get("summary") if st.session_state.last_result else None,
            "feature_summary": st.session_state.feature_summary,
            "goal": "Teach, guide, and propose practical next steps."
        }
        with st.chat_message("assistant"):
            with st.spinner("Mentor is thinking…"):
                reply = ask_ai_mentor(user_msg, context)
                st.markdown(reply)
        st.session_state.chat.append(("assistant", reply))

st.caption("UNISOLE Auto-AI — Pro EDA • Feature Engineering • AutoML • Explainability • LLM Mentor")