# feature_engineering.py
# Pro Feature Engineering Pipeline (Guided UI ready)
# - Missing values (numeric/categorical)
# - Rare category grouping
# - Skew fix (log1p on skewed numeric)
# - Scaling (Standard/MinMax)
# - OneHot encoding
# - Low variance removal
# Exposes helpers to build, fit, apply, and export the pipeline.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


# --------------------------- Config ---------------------------

@dataclass
class FEConfig:
    # Missing values
    num_impute: str = "median"          # "mean" | "median" | "most_frequent"
    cat_impute: str = "most_frequent"   # "most_frequent" only for stability

    # Categorical encoding
    encode: str = "onehot"              # "onehot" | "none"

    # Rare category grouping
    enable_rare_group: bool = True
    rare_min_frac: float = 0.01         # categories below this fraction -> 'RARE'

    # Scaling
    scale_numeric: str = "standard"     # "standard" | "minmax" | "none"

    # Skew fix (applied after imputation, before scaling)
    fix_skew: bool = True
    skew_threshold: float = 0.75        # abs(skew) >= threshold -> log1p

    # Low variance removal (after encoding)
    enable_low_variance: bool = True
    variance_threshold: float = 0.0     # 0.0 removes constant features

    # For reporting
    keep_feature_names: bool = True


# --------------------------- Transformers ---------------------------

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group infrequent categories per column into 'RARE' by min_frac."""
    def __init__(self, min_frac: float = 0.01):
        self.min_frac = float(min_frac)
        self._kept_: List[Dict[Any, bool]] | None = None  # list per column -> dict of kept categories

    def fit(self, X, y=None):
        X = self._to_2d(X)
        n_samples = X.shape[0]
        kept_list = []
        for j in range(X.shape[1]):
            col = pd.Series(X[:, j], dtype="object")
            frac = (col.value_counts(dropna=False) / n_samples)
            kept = {k: bool(v >= self.min_frac) for k, v in frac.items()}
            kept_list.append(kept)
        self._kept_ = kept_list
        return self

    def transform(self, X):
        X = self._to_2d(X)
        if self._kept_ is None:
            return X
        X_out = X.astype("object").copy()
        for j in range(X.shape[1]):
            kept = self._kept_[j]
            col = X_out[:, j]
            X_out[:, j] = np.where(
                [kept.get(v, False) for v in col],
                col,
                "RARE"
            )
        return X_out

    @staticmethod
    def _to_2d(X):
        if isinstance(X, pd.DataFrame):
            return X.values
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


class SkewFixer(BaseEstimator, TransformerMixin):
    """Apply log1p to numeric columns with abs(skew) >= threshold."""
    def __init__(self, skew_threshold: float = 0.75):
        self.skew_threshold = float(skew_threshold)
        self.cols_: Optional[List[int]] = None  # indices to transform

    def fit(self, X, y=None):
        X = self._to_2d_float(X)
        self.cols_ = []
        for j in range(X.shape[1]):
            col = pd.Series(X[:, j])
            # guard against negative values; skew calc requires finite
            s = col.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                continue
            skew_val = s.skew()
            if np.isfinite(skew_val) and abs(skew_val) >= self.skew_threshold:
                self.cols_.append(j)
        return self

    def transform(self, X):
        X = self._to_2d_float(X.copy())
        if not self.cols_:
            return X
        # log1p requires non-negative; shift if needed
        for j in self.cols_:
            col = X[:, j]
            minv = np.nanmin(col)
            if np.isfinite(minv) and minv < 0:
                col = col - minv  # shift to >= 0
            X[:, j] = np.log1p(col)
        return X

    @staticmethod
    def _to_2d_float(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


# --------------------------- Detection ---------------------------

def detect_column_types(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, List[str]]:
    cols = df.columns.tolist()
    if target and target in cols:
        cols.remove(target)
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    return {"numeric": num_cols, "categorical": cat_cols}


# --------------------------- Build Pipeline ---------------------------

def build_preprocessor(df: pd.DataFrame, config: FEConfig, target: Optional[str] = None) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    types = detect_column_types(df, target)
    num_cols, cat_cols = types["numeric"], types["categorical"]

    report = {
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "steps": []
    }

    # Numeric pipeline
    num_steps: List[Tuple[str, Any]] = []
    num_steps.append(("imputer", SimpleImputer(strategy=config.num_impute)))
    report["steps"].append(f"Numeric imputation: {config.num_impute}")

    if config.fix_skew:
        num_steps.append(("skewfix", SkewFixer(skew_threshold=config.skew_threshold)))
        report["steps"].append(f"Skew fix: log1p for |skew| >= {config.skew_threshold}")

    if config.scale_numeric == "standard":
        num_steps.append(("scaler", StandardScaler()))
        report["steps"].append("Scaling: StandardScaler")
    elif config.scale_numeric == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
        report["steps"].append("Scaling: MinMaxScaler")
    else:
        report["steps"].append("Scaling: none")

    num_pipe = Pipeline(steps=num_steps) if num_cols else "drop"

    # Categorical pipeline
    cat_steps: List[Tuple[str, Any]] = []
    if config.enable_rare_group:
        cat_steps.append(("rare", RareCategoryGrouper(min_frac=config.rare_min_frac)))
        report["steps"].append(f"Rare category grouping: min_frac={config.rare_min_frac}")
    cat_steps.append(("imputer", SimpleImputer(strategy=config.cat_impute)))
    report["steps"].append(f"Categorical imputation: {config.cat_impute}")

    if config.encode == "onehot":
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        report["steps"].append("Encoding: OneHotEncoder")
    else:
        report["steps"].append("Encoding: none")

    cat_pipe = Pipeline(steps=cat_steps) if cat_cols else "drop"

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return pre, report


def build_feature_pipeline(df: pd.DataFrame, config: FEConfig, target: Optional[str] = None):
    """Create the full pipeline: ColumnTransformer -> VarianceThreshold."""
    pre, report = build_preprocessor(df, config, target=target)

    steps: List[Tuple[str, Any]] = [("pre", pre)]
    if config.enable_low_variance:
        steps.append(("var", VarianceThreshold(threshold=config.variance_threshold)))
        report["steps"].append(f"Low-variance removal: threshold={config.variance_threshold}")

    pipe = Pipeline(steps=steps)
    return pipe, report


# --------------------------- Names & Export ---------------------------

def get_feature_names_after_pre(pre: ColumnTransformer) -> List[str]:
    """Get feature names right after the ColumnTransformer step."""
    try:
        names = pre.get_feature_names_out().tolist()
    except Exception:
        # fallback generic names
        if hasattr(pre, "transformers_"):
            # count total columns
            total = 0
            for name, trans, cols in pre.transformers_:
                if trans == "drop":
                    continue
                if isinstance(cols, list):
                    total += len(cols)
            names = [f"f{i}" for i in range(total)]
        else:
            names = []
    return names


def get_final_feature_names(pipe: Pipeline, X: pd.DataFrame) -> List[str]:
    """Get final output feature names after variance threshold (if present)."""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    names = get_feature_names_after_pre(pre)
    if "var" in pipe.named_steps:
        var: VarianceThreshold = pipe.named_steps["var"]
        # Force a pass to initialize support_
        _ = pipe.named_steps["pre"].fit_transform(X)
        _ = var.fit(pipe.named_steps["pre"].transform(X))
        mask = var.get_support()
        if len(names) == len(mask):
            names = [n for n, keep in zip(names, mask) if keep]
        else:
            # length mismatch fallback
            names = [f"f{i}" for i, keep in enumerate(mask) if keep]
    return names


# --------------------------- Public API ---------------------------

def fit_feature_pipeline(df: pd.DataFrame, target: Optional[str], config: FEConfig):
    """
    Fit the feature pipeline on df (excluding target), return:
      - pipe: fitted Pipeline
      - X_out: transformed numpy array
      - names: final feature names
      - summary: dict describing steps and columns
    """
    X = df.drop(columns=[target]) if target and target in df.columns else df.copy()

    pipe, report = build_feature_pipeline(df, config, target=target)
    X_out = pipe.fit_transform(X)
    names = get_final_feature_names(pipe, X)

    summary = {
        "applied_steps": report["steps"],
        "numeric_cols": report.get("numeric_cols", []),
        "categorical_cols": report.get("categorical_cols", []),
        "output_shape": (int(X_out.shape[0]), int(X_out.shape[1])),
    }
    return pipe, X_out, names, summary


def transform_with_pipeline(pipe: Pipeline, df: pd.DataFrame, target: Optional[str]) -> Tuple[np.ndarray, Optional[pd.Series]]:
    X = df.drop(columns=[target]) if target and target in df.columns else df.copy()
    y = df[target] if target and target in df.columns else None
    X_out = pipe.transform(X)
    return X_out, y


def export_processed_dataframe(pipe: Pipeline, df: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    """Return a DataFrame of processed features (for download)."""
    X = df.drop(columns=[target]) if target and target in df.columns else df.copy()
    X_out = pipe.transform(X)
    names = get_final_feature_names(pipe, X)
    try:
        X_df = pd.DataFrame(X_out, columns=names, index=df.index)
    except Exception:
        X_df = pd.DataFrame(X_out, index=df.index)
    if target and target in df.columns:
        X_df[target] = df[target].values
    return X_df


# --------------------------- Defaults ---------------------------

def default_config() -> FEConfig:
    return FEConfig()
