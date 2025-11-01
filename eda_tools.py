# eda_tools.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import skew, kurtosis

sns.set(style="whitegrid")

# ---------- OVERVIEW & HEALTH ----------

def overview(df: pd.DataFrame) -> Dict[str, Any]:
    dup_rows = int(df.duplicated().sum())
    missing_perc = df.isna().mean().round(3).to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "duplicates": dup_rows,
        "missing_perc": missing_perc,
        "dtypes": dtypes,
        "numeric_cols": numeric,
        "categorical_cols": categorical,
    }

def plot_missingness(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=miss.values, y=miss.index)
    ax.set_title("Missingness by Column")
    ax.set_xlabel("Fraction Missing")
    ax.set_ylabel("Column")
    plt.tight_layout()
    return fig

# ---------- TARGET ANALYSIS ----------

def target_summary(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    s = df[target].dropna()
    info = {
        "dtype": str(df[target].dtype),
        "n_unique": int(s.nunique()),
        "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) else None,
        "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) else None,
        "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else None,
        "median": float(s.median()) if pd.api.types.is_numeric_dtype(s) else None,
    }
    if pd.api.types.is_numeric_dtype(s):
        info["skew"] = float(skew(s, nan_policy="omit"))
        info["kurtosis"] = float(kurtosis(s, nan_policy="omit"))
    return info

def plot_target_distribution(df: pd.DataFrame, target: str):
    fig = plt.figure(figsize=(7, 4))
    if pd.api.types.is_numeric_dtype(df[target]):
        ax = sns.histplot(df[target].dropna(), kde=True)
        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel(target)
    else:
        counts = df[target].value_counts().sort_values(ascending=False)
        ax = sns.barplot(x=counts.index, y=counts.values)
        ax.set_title(f"Target Class Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        plt.xticks(rotation=20)
    plt.tight_layout()
    return fig

# ---------- CORRELATIONS ----------

def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method)

def plot_corr_heatmap(corr: pd.DataFrame, vmax: float = 1.0):
    if corr.empty:
        fig = plt.figure()
        plt.title("No numeric columns to compute correlation.")
        return fig
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return fig

def top_corr_with_target(df: pd.DataFrame, target: str, k: int = 10, method: str = "pearson") -> List[Tuple[str, float]]:
    if not pd.api.types.is_numeric_dtype(df[target]):
        return []
    num = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    if num.empty:
        return []
    vals = num.corrwith(df[target], method=method).dropna().abs().sort_values(ascending=False)
    return list(vals.head(k).items())

# ---------- FEATURE DISTRIBUTIONS ----------

def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 8):
    figs = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:max_cols]:
        fig = plt.figure(figsize=(6, 3.5))
        ax = sns.histplot(df[col].dropna(), kde=True)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        plt.tight_layout()
        figs.append(fig)
    return figs

def plot_categorical_bars(df: pd.DataFrame, max_cols: int = 6, top_n: int = 12):
    figs = []
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols[:max_cols]:
        counts = df[col].value_counts().head(top_n)
        fig = plt.figure(figsize=(6, 3.5))
        ax = sns.barplot(x=counts.index, y=counts.values)
        ax.set_title(f"Top-{top_n} Categories: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=25)
        plt.tight_layout()
        figs.append(fig)
    return figs

# ---------- OUTLIERS (IQR) ----------

def outlier_report(df: pd.DataFrame, k: float = 1.5) -> Dict[str, Any]:
    report = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        count = int(((s < lower) | (s > upper)).sum())
        report[col] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower": float(lower), "upper": float(upper),
            "outliers": count
        }
    return {"rule": f"IQR with k={k}", "columns": report}

def plot_boxplots(df: pd.DataFrame, max_cols: int = 8):
    figs = []
    for col in df.select_dtypes(include=[np.number]).columns[:max_cols]:
        fig = plt.figure(figsize=(5, 3.5))
        ax = sns.boxplot(x=df[col])
        ax.set_title(f"Boxplot: {col}")
        plt.tight_layout()
        figs.append(fig)
    return figs

# ---------- QUALITY CHECKS ----------

def class_imbalance(df: pd.DataFrame, target: str) -> Optional[Dict[str, float]]:
    if pd.api.types.is_numeric_dtype(df[target]):
        return None
    counts = df[target].value_counts(normalize=True)
    return counts.round(3).to_dict()

def leakage_scan(df: pd.DataFrame, target: str, threshold: float = 0.98) -> Dict[str, Any]:
    """Heuristic: very high correlation with target or exact duplicates."""
    issues = []
    if pd.api.types.is_numeric_dtype(df[target]):
        corr = df.select_dtypes(include=[np.number]).corr()[target].drop(target).abs()
        high = corr[corr >= threshold]
        for c, v in high.items():
            issues.append({"column": c, "corr_with_target": float(v)})
    # duplicate columns
    seen = {}
    for c in df.columns:
        s = df[c]
        key = (str(s.dtype), tuple(pd.util.hash_pandas_object(s.fillna("NaN")).values))
        if key in seen:
            issues.append({"duplicate_of": seen[key], "column": c})
        else:
            seen[key] = c
    return {"potential_leakage": issues}
