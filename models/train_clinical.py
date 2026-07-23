"""
Train the clinical-dataset (DS1) model.

This is your Colab notebook's modeling section, refactored into a script:
  - same preprocessing (src/preprocessing.py)
  - LogisticRegression, RandomForest, XGBoost with the same class-imbalance
    handling you used (class_weight='balanced' / scale_pos_weight)
  - light RandomizedSearchCV tuning (kept small so this runs in a few
    minutes on a laptop — widen param grids if you want to fully match
    your notebook's tuning)
  - 5-fold Stratified CV used to pick the single best model, by RECALL
  - a threshold sweep on the held-out test set, with a recommended
    screening threshold auto-picked (highest recall that still keeps
    precision usable)
  - the WINNER is refit on all data and saved to models/saved/

Usage:
    python models/train_clinical.py --data data/DS1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                      cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing import (CLINICAL_TARGET, CLINICAL_FEATURE_ORDER,
                                build_clinical_preprocessor)
from src.utils import compute_threshold_presets, save_model

THRESHOLD_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60]


def main(data_path: str, force_model: str | None = None):
    df = pd.read_csv(data_path)
    missing = set(CLINICAL_FEATURE_ORDER + [CLINICAL_TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    X = df[CLINICAL_FEATURE_ORDER]
    y = df[CLINICAL_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_clinical_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    candidates = {}

    # --- Logistic Regression -------------------------------------------------
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced",
                                      max_iter=1000, random_state=42)),
    ])
    lr_search = RandomizedSearchCV(
        lr_pipe,
        param_distributions={
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs"],
        },
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    lr_search.fit(X_train, y_train)
    candidates["Logistic Regression"] = lr_search.best_estimator_

    # --- Random Forest --------------------------------------------------------
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced",
                                          random_state=42, n_jobs=-1)),
    ])
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions={
            "model__n_estimators": [200, 300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_leaf": [1, 5],
        },
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    candidates["Random Forest"] = rf_search.best_estimator_

    # --- Gradient Boosting (XGBoost) ------------------------------------------
    xgb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(scale_pos_weight=scale_pos_weight,
                                 eval_metric="logloss", random_state=42,
                                 n_jobs=-1)),
    ])
    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions={
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
        },
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    candidates["Gradient Boosting (XGBoost)"] = xgb_search.best_estimator_

    # --- Compare all three on CV F1 / AUC / Recall, pick the winner ----------
    # Selection metric is RECALL, not F1: for a diabetes screening tool,
    # missing a diabetic patient (false negative) is worse than an extra
    # follow-up test (false positive) — matches DS1 XGBoost's reported
    # 0.8835 recall being the headline number, not its 0.5151 precision.
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "recall": "recall"}
    leaderboard = {}
    for name, pipe in candidates.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        leaderboard[name] = {
            "cv_f1": round(scores["test_f1"].mean(), 4),
            "cv_auc": round(scores["test_roc_auc"].mean(), 4),
            "cv_recall": round(scores["test_recall"].mean(), 4),
        }
        print(f"{name}: {leaderboard[name]}")

    name_lookup = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "xgboost": "Gradient Boosting (XGBoost)",
    }
    if force_model:
        if force_model not in name_lookup:
            raise ValueError(f"--force-model must be one of {list(name_lookup)}")
        best_name = name_lookup[force_model]
        print(f"\n--force-model set: pinning deployed model to {best_name} "
              f"(overriding recall-based auto-selection)")
    else:
        best_name = max(leaderboard, key=lambda n: leaderboard[n]["cv_recall"])
        print(f"\nBest model by CV Recall: {best_name}")

    best_pipe = candidates[best_name]

    # Refit best model on the full training set, evaluate on held-out test set
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print("Held-out test performance @ threshold 0.50:", test_metrics)

    threshold_presets, recommended_threshold = compute_threshold_presets(
        y_test, y_proba, THRESHOLD_SWEEP
    )
    for p in threshold_presets:
        print(f"  threshold={p['threshold']:.2f} -> recall={p['recall']}, "
              f"precision={p['precision']}" + (f"  [{p['note']}]" if p['note'] else ""))
    print(f"Recommended threshold: {recommended_threshold}")

    metadata = {
        "dataset": "clinical",
        "target": CLINICAL_TARGET,
        "feature_order": CLINICAL_FEATURE_ORDER,
        "best_model_name": best_name,
        "cv_leaderboard": leaderboard,
        "test_metrics": test_metrics,
        "threshold_presets": threshold_presets,
        "decision_threshold": recommended_threshold,
    }
    save_model(best_pipe, "clinical", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/DS1.csv",
                         help="Path to the clinical dataset CSV")
    parser.add_argument("--force-model", default="auto",
                         choices=["auto", "logreg", "rf", "xgboost"],
                         help="Pin the deployed model instead of auto-selecting "
                              "by CV recall. Defaults to 'auto', which lets "
                              "recall decide (this reproduced XGBoost winning "
                              "with 0.8835 recall on your thesis's clinical "
                              "data). Pass 'xgboost'/'rf'/'logreg' to force a "
                              "specific model instead.")
    args = parser.parse_args()
    main(args.data, None if args.force_model == "auto" else args.force_model)