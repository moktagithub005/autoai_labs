# explainability.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Tuple, List, Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ------------- Helpers to work with our sklearn Pipeline( pre -> model ) -------------

def split_pipeline(artifact: Pipeline):
    """Return (preprocessor, estimator) from our fitted Pipeline."""
    if not isinstance(artifact, Pipeline):
        raise ValueError("Artifact must be an sklearn Pipeline(pre -> model).")
    pre = artifact.named_steps.get("pre")
    est = artifact.named_steps.get("model")
    if pre is None or est is None:
        # fallback for unnamed steps
        try:
            pre, est = artifact.steps[0][1], artifact.steps[-1][1]
        except Exception:
            raise ValueError("Could not find pre/model steps in pipeline.")
    return pre, est

def transform_X_and_names(artifact: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transform X with the pipeline preprocessor and return feature names after encoding."""
    pre, _ = split_pipeline(artifact)
    Xt = pre.transform(X)
    # sklearn >=1.0 has get_feature_names_out
    try:
        names = pre.get_feature_names_out().tolist()
    except Exception:
        # fallback: use original columns count
        if hasattr(Xt, "shape"):
            names = [f"f{i}" for i in range(Xt.shape[1])]
        else:
            names = list(X.columns)
    # Convert sparse to dense for SHAP safety if needed
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return Xt, names

# ------------- Global Feature Importance -------------

def global_feature_importance(artifact: Pipeline, X: pd.DataFrame, top_n: int = 20):
    """Compute global importance from model (feature_importances_/coef_) or permutation fallback."""
    pre, est = split_pipeline(artifact)
    Xt, names = transform_X_and_names(artifact, X)

    importances = None
    title = "Global Feature Importance"
    # Tree models
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_)
        importances = pd.Series(vals, index=names)
    # Linear models
    elif hasattr(est, "coef_"):
        coefs = est.coef_
        if coefs.ndim > 1:  # multi-class
            coefs = np.mean(np.abs(coefs), axis=0)
        importances = pd.Series(np.abs(coefs), index=names)
        title = "Global Importance (|coef|)"
    # Fallback: permutation importance on a small sample
    if importances is None:
        sample_idx = np.random.choice(len(X), size=min(300, len(X)), replace=False)
        Xs = X.iloc[sample_idx]
        # Need y for permutation importance; derive from pipeline? Not available here.
        # So as minimal fallback, rank by variance after preprocessing.
        Xt_s, _ = transform_X_and_names(artifact, Xs)
        importances = pd.Series(np.var(Xt_s, axis=0), index=names)
        title = "Proxy Importance (feature variance)"

    imp = importances.sort_values(ascending=False).head(top_n)
    fig = plt.figure(figsize=(8, 5))
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    return fig, imp.reset_index().rename(columns={"index": "feature", 0: "importance"})

# ------------- SHAP: Global & Single Prediction -------------

def build_shap_explainer(artifact: Pipeline, X: pd.DataFrame, sample_size: int = 500):
    """Return (explainer, X_sample_transformed, feature_names). Uses auto Explainer with masker."""
    Xt, names = transform_X_and_names(artifact, X)
    # sample to keep it CPU-friendly
    n = Xt.shape[0]
    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    Xt_sample = Xt[idx]
    # Create a model function that takes already-transformed input
    _, est = split_pipeline(artifact)

    def model_fn(Z):
        # Z will already be in transformed space; ensure dense
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        return est.predict_proba(Z) if hasattr(est, "predict_proba") else est.predict(Z)

    # let shap auto-pick the right explainer; provide masker for performance
    masker = shap.maskers.Independent(Xt_sample)
    explainer = shap.Explainer(model_fn, masker)
    return explainer, Xt_sample, names

def shap_summary_fig(explainer, Xt_sample, feature_names):
    """Return a matplotlib figure for SHAP summary (beeswarm)."""
    shap_values = explainer(Xt_sample)
    # Make names safe
    try:
        shap_values.feature_names = feature_names
    except Exception:
        pass
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, features=Xt_sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig

def shap_single_waterfall_fig(explainer, artifact: Pipeline, X: pd.DataFrame, row_index: int):
    """Explain a single row (after preprocessing) with a waterfall plot. Returns (fig, ordered_contrib_list)."""
    Xt, names = transform_X_and_names(artifact, X)
    if row_index < 0 or row_index >= Xt.shape[0]:
        raise IndexError("Row index out of range for X.")
    sv = explainer(Xt[row_index:row_index+1])
    # Waterfall plot (matplotlib)
    try:
        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(sv[0], show=False)
        fig = plt.gcf()
        plt.tight_layout()
    except Exception:
        # Fallback: bar chart of top contributions
        vals = sv.values[0] if hasattr(sv, "values") else np.ravel(sv[0].values)
        order = np.argsort(np.abs(vals))[::-1][:15]
        fig = plt.figure(figsize=(8, 5))
        plt.barh(np.array(names)[order][::-1], np.array(vals)[order][::-1])
        plt.title("Top feature contributions (absolute)")
        plt.tight_layout()
    # Return ordered contributions for text explanation
    try:
        vals = sv.values[0]
    except Exception:
        vals = np.ravel(sv[0].values)
    order = np.argsort(np.abs(vals))[::-1][:10]
    contrib_list = [(names[i], float(vals[i])) for i in order]
    return fig, contrib_list

# ------------- Utility -------------

def pick_interesting_row(artifact: Pipeline, X: pd.DataFrame, strategy: str = "max") -> int:
    """Pick an 'interesting' row index based on prediction extremes."""
    pre, est = split_pipeline(artifact)
    Xt, _ = transform_X_and_names(artifact, X)
    try:
        preds = est.predict(Xt)
        # if classification with probabilities, use max class prob
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(Xt)
            preds = probs.max(axis=1)
    except Exception:
        preds = np.sum(Xt, axis=1)
    if strategy == "min":
        return int(np.argmin(preds))
    return int(np.argmax(preds))
