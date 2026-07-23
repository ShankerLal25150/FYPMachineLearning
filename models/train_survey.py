"""
Train the survey-dataset (DS2 / BRFSS) model — binary target (pre-diabetes
merged into no-diabetes, same as your notebook).

Candidates compared (matches your full results table, including the LR/RF +
ADASYN variants that were missing from earlier versions of this script):
  - Logistic Regression (class_weight, baseline)
  - Random Forest (class_weight, baseline)
  - Gradient Boosting / XGBoost (scale_pos_weight, tuned, recall-scored)
  - XGBoost + ADASYN (tuned, recall-scored)
  - Logistic Regression + ADASYN (tuned, recall-scored)
  - Random Forest + ADASYN (tuned, recall-scored)
  - Soft-voting ensemble (LR + XGBoost)

Selection: whichever candidate has the best cross-validated RECALL wins and
gets deployed — not F1, not accuracy. This is a conscious choice for a
screening tool: missing a diabetic (false negative) is worse than an extra
follow-up test (false positive), and F1 penalizes exactly the low-precision/
high-recall trade-off that screening wants.

Threshold: after the winning model is picked, the script sweeps decision
thresholds on the held-out test set and auto-picks a "recommended" one — the
lowest threshold (highest recall) that still keeps precision at or above a
floor (default 0.30), so the deployed default isn't unusably noisy. The full
sweep (including a "maximum recall" extreme) is saved so the app can offer
it as one-click presets, same idea as your own recall/precision table.

Usage:
    python models/train_survey.py --data data/DS2.csv
    python models/train_survey.py --data data/DS2.csv --force-model xgb_adasyn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                      cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing import (SURVEY_BINARY_TARGET, SURVEY_FEATURE_ORDER,
                                SURVEY_RAW_TARGET, build_survey_preprocessor,
                                merge_survey_target)
from src.utils import compute_threshold_presets, save_model

THRESHOLD_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60]


def main(data_path: str, force_model: str | None):
    df = pd.read_csv(data_path)
    missing = set(SURVEY_FEATURE_ORDER + [SURVEY_RAW_TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df = merge_survey_target(df)
    X = df[SURVEY_FEATURE_ORDER]
    y = df[SURVEY_BINARY_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_survey_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    candidates = {}

    # --- Logistic Regression (baseline, class_weight) ------------------------
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced",
                                      max_iter=1000, random_state=42)),
    ])
    lr_search = RandomizedSearchCV(
        lr_pipe,
        param_distributions={"model__C": [0.01, 0.1, 1, 10]},
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    lr_search.fit(X_train, y_train)
    candidates["Logistic Regression"] = lr_search.best_estimator_

    # --- Random Forest (baseline, class_weight) -------------------------------
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
        },
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    candidates["Random Forest"] = rf_search.best_estimator_

    # --- Gradient Boosting (XGBoost), scale_pos_weight, tuned -----------------
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

    # --- ADASYN variants -------------------------------------------------------
    # ADASYN oversamples the minority (diabetic) class synthetically. It lives
    # *inside* the pipeline (imblearn Pipeline, not sklearn's) so it only ever
    # resamples the training folds, never the validation/test data — otherwise
    # CV scores would be optimistic and leak information. Applied to all three
    # base algorithms (this was missing for LR/RF in earlier versions).
    xgb_adasyn_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("adasyn", ADASYN(random_state=42)),
        ("model", XGBClassifier(eval_metric="logloss", random_state=42,
                                 n_jobs=-1)),
    ])
    xgb_adasyn_search = RandomizedSearchCV(
        xgb_adasyn_pipe,
        param_distributions={
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
        },
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    xgb_adasyn_search.fit(X_train, y_train)
    candidates["XGBoost + ADASYN"] = xgb_adasyn_search.best_estimator_

    lr_adasyn_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("adasyn", ADASYN(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_adasyn_search = RandomizedSearchCV(
        lr_adasyn_pipe,
        param_distributions={"model__C": [0.01, 0.1, 1, 10]},
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    lr_adasyn_search.fit(X_train, y_train)
    candidates["Logistic Regression + ADASYN"] = lr_adasyn_search.best_estimator_

    rf_adasyn_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("adasyn", ADASYN(random_state=42)),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    rf_adasyn_search = RandomizedSearchCV(
        rf_adasyn_pipe,
        param_distributions={
            "model__n_estimators": [200, 300],
            "model__max_depth": [10, 20, None],
        },
        n_iter=4, scoring="recall", cv=cv, random_state=42, n_jobs=-1,
    )
    rf_adasyn_search.fit(X_train, y_train)
    candidates["Random Forest + ADASYN"] = rf_adasyn_search.best_estimator_

    # --- Soft-voting ensemble (LR + XGBoost), matches your notebook ----------
    ensemble = VotingClassifier(
        estimators=[
            ("lr", candidates["Logistic Regression"]),
            ("xgb", candidates["Gradient Boosting (XGBoost)"]),
        ],
        voting="soft",
    )
    candidates["Voting Ensemble (LR + XGBoost)"] = ensemble

    # --- Compare everything on CV, select by RECALL ---------------------------
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
        "xgb_adasyn": "XGBoost + ADASYN",
        "logreg_adasyn": "Logistic Regression + ADASYN",
        "rf_adasyn": "Random Forest + ADASYN",
        "voting": "Voting Ensemble (LR + XGBoost)",
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
    best_pipe.fit(X_train, y_train)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    # Default-threshold (0.5) metrics, for the headline numbers
    y_pred_05 = (y_proba >= 0.5).astype(int)
    test_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred_05), 4),
        "f1": round(f1_score(y_test, y_pred_05), 4),
        "recall": round(recall_score(y_test, y_pred_05), 4),
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
        "dataset": "survey",
        "target": SURVEY_BINARY_TARGET,
        "feature_order": SURVEY_FEATURE_ORDER,
        "best_model_name": best_name,
        "cv_leaderboard": leaderboard,
        "test_metrics": test_metrics,
        "threshold_presets": threshold_presets,
        "decision_threshold": recommended_threshold,
    }
    save_model(best_pipe, "survey", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/DS2.csv",
                         help="Path to the survey dataset CSV")
    parser.add_argument("--force-model", default="auto",
                         choices=["auto", "logreg", "rf", "xgboost", "xgb_adasyn",
                                  "logreg_adasyn", "rf_adasyn", "voting"],
                         help="Pin the deployed model instead of auto-selecting "
                              "by CV recall. Defaults to 'auto', which lets "
                              "recall decide across all 7 candidates (matches "
                              "your full results table).")
    args = parser.parse_args()
    main(args.data, None if args.force_model == "auto" else args.force_model)