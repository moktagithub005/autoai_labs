import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import joblib

# Optional learners
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def quick_eda(df: pd.DataFrame) -> Dict[str, Any]:
    info = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_perc": df.isna().mean().round(3).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude=[np.number]).columns.tolist(),
        "sample_values": {c: df[c].dropna().unique()[:5].tolist() for c in df.columns[:8]}
    }
    return info


def detect_task(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    # Heuristic: few unique values (<=20 and ratio < 0.05) → classification
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(y):
        if nunique <= 20 and nunique / len(y) < 0.05:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


def build_preprocessor(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])
    return pre, numeric, categorical


def model_candidates(task: str):
    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300),
            "SVC": SVC(probability=True),
            "KNNClassifier": KNeighborsClassifier(n_neighbors=7),
        }
        if HAS_XGB:
            models["XGBClassifier"] = XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", tree_method="hist"
            )
        if HAS_LGBM:
            models["LGBMClassifier"] = LGBMClassifier(n_estimators=600, learning_rate=0.05)
        return models
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300),
            "SVR": SVR(),
            "KNNRegressor": KNeighborsRegressor(n_neighbors=7),
        }
        if HAS_XGB:
            models["XGBRegressor"] = XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                tree_method="hist"
            )
        if HAS_LGBM:
            models["LGBMRegressor"] = LGBMRegressor(n_estimators=800, learning_rate=0.05)
        return models


def run_automl(df: pd.DataFrame, target: str, task: str, seed=42, test_size=0.2, cv_folds=5, max_minutes=10):
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    pre, num_cols, cat_cols = build_preprocessor(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                        stratify=y if task=="classification" else None)

    # CV splitter
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed) if task=="classification" \
         else KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    candidates = model_candidates(task)
    scores = []
    best_name, best_score, best_pipe = None, -np.inf, None

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        scoring = "roc_auc_ovr" if task=="classification" else "r2"
        try:
            cv_score = np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None))
        except Exception:
            # fallback for models without probas
            if task=="classification":
                cv_score = np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None))
            else:
                cv_score = np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=None))

        scores.append({"model": name, "cv_score": float(cv_score)})
        if cv_score > best_score:
            best_score, best_name, best_pipe = cv_score, name, pipe

    # Fit best on full train
    best_pipe.fit(X_train, y_train)

    # Evaluate on test
    summary = {"task": task, "best_model_name": best_name, "cv_scores": scores}

    if task == "classification":
        proba_ok = hasattr(best_pipe.named_steps["model"], "predict_proba")
        y_pred = best_pipe.predict(X_test)
        metrics = {}
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        metrics["accuracy"] = float(acc)
        metrics["f1_weighted"] = float(f1)
        if proba_ok and len(np.unique(y_test)) == 2:
            y_proba = best_pipe.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        metrics["confusion_matrix"] = cm
        summary["test_metrics"] = metrics

        # Persist a small model card
        summary["features"] = {"numeric": num_cols, "categorical": cat_cols}
        summary["class_names"] = [str(x) for x in sorted(y.unique())]

    else:
        y_pred = best_pipe.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred)**0.5)
        }
        summary["test_metrics"] = metrics
        summary["features"] = {"numeric": num_cols, "categorical": cat_cols}

    return {
        "summary": summary,
        "artifact": best_pipe,
    }


# ---------- PLOTTING HELPERS (matplotlib; one plot per figure) ----------

def plot_classification_curves(result):
    figs = []
    # Confusion matrix heatmap (simple)
    cm = np.array(result["summary"]["test_metrics"]["confusion_matrix"])
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    figs.append(fig)

    # We can’t reliably plot ROC/PR without raw probs & binarity for every model; skip if not available.

    return figs


def plot_regression_plots(result):
    # Placeholders for MVP; could be enhanced after storing predictions
    figs = []
    fig = plt.figure()
    plt.title("Residuals (placeholder)")
    plt.xlabel("Samples")
    plt.ylabel("Residual")
    # In MVP we don’t store residuals; leave empty axes to avoid seaborn and keep simple.
    figs.append(fig)
    return figs
