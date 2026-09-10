"""
RailSync AI - train the criticality model, run the CP-SAT scheduler, and
export everything the FastAPI backend needs to serve a local dashboard.

Run this once (and again any time you change the data or model):
    python train_and_export.py
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import shap
from ortools.sat.python import cp_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DROP_COLUMNS = [
    "task_id",
    "criticality_score",
    "priority_class",
    "recommended_window",
    "station_code",
]
CATEGORICAL_COLUMNS = ["department", "corridor_id", "zone_id", "asset_type"]


def get_priority(score):
    if score >= 80:
        return "CRITICAL"
    elif score >= 65:
        return "HIGH"
    elif score >= 35:
        return "MEDIUM"
    else:
        return "LOW"


def main():
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "railsync_maintenance_tasks_synthetic.csv"))

    X = df.drop(columns=DROP_COLUMNS).copy()
    y = df["criticality_score"].copy()
    priority_labels = df["priority_class"].copy()

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, dtype=int)
    FEATURE_COLUMNS = list(X.columns)

    X_train, X_test, y_train, y_test, pc_train, pc_test = train_test_split(
        X, y, priority_labels, test_size=0.20, random_state=RANDOM_STATE, stratify=priority_labels
    )

    print("Fitting baselines...")
    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = np.clip(lin.predict(X_test), 0, 100)

    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    rf_pred = np.clip(rf.predict(X_test), 0, 100)

    baseline_results = [
        {"model": "Linear Regression", "MAE": float(mean_absolute_error(y_test, lin_pred)), "R2": float(r2_score(y_test, lin_pred))},
        {"model": "Random Forest", "MAE": float(mean_absolute_error(y_test, rf_pred)), "R2": float(r2_score(y_test, rf_pred))},
    ]

    print("Tuning XGBoost (small randomized search)...")
    param_dist = {
        "n_estimators": [200, 300, 450, 600, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.03, 0.045, 0.07, 0.1],
        "subsample": [0.7, 0.8, 0.85, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.85, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 8],
    }
    search = RandomizedSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=2),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("  best CV MAE:", round(-search.best_score_, 4), " params:", search.best_params_)

    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE)
    best_params = dict(search.best_params_)
    best_params.update(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=2,
        n_estimators=max(best_params.get("n_estimators", 450), 800),
        eval_metric="mae",
        early_stopping_rounds=30,
    )
    model = XGBRegressor(**best_params)
    model.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)
    print("  early stopping best_iteration:", model.best_iteration)

    pred = np.clip(model.predict(X_test), 0, 100)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    train_pred = np.clip(model.predict(X_train), 0, 100)
    train_mae = float(mean_absolute_error(y_train, train_pred))
    train_r2 = float(r2_score(y_train, train_pred))

    per_class = pd.DataFrame({"actual": y_test.values, "pred": pred, "priority_class": pc_test.values})
    per_class_mae = (
        per_class.groupby("priority_class")
        .apply(lambda g: pd.Series({"n": int(len(g)), "MAE": float(mean_absolute_error(g["actual"], g["pred"]))}))
        .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    )

    metrics = {
        "test": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "train": {"MAE": train_mae, "R2": train_r2},
        "per_class_test": per_class_mae.reset_index().rename(columns={"index": "priority_class"}).to_dict("records"),
        "baselines": baseline_results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("Computing SHAP...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_sample = X_test.sample(min(1500, len(X_test)), random_state=RANDOM_STATE)
    shap_values = explainer(shap_sample)
    shap_importance = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("Scoring full dataset + building test-set prediction table...")
    all_pred = np.clip(model.predict(X), 0, 100)
    df["predicted_criticality"] = np.round(all_pred, 2)
    df["predicted_priority"] = df["predicted_criticality"].apply(get_priority)

    test_task_ids = df.loc[X_test.index, "task_id"]
    test_table = df.loc[X_test.index, [
        "task_id", "department", "corridor_id", "asset_type", "station_code",
        "criticality_score", "priority_class", "predicted_criticality", "predicted_priority",
    ]].copy()
    test_table["abs_error"] = (test_table["criticality_score"] - test_table["predicted_criticality"]).abs().round(2)
    test_table = test_table.sort_values("predicted_criticality", ascending=False)

    print("Running OR-Tools CP-SAT scheduler on the demo subset...")
    demo_df = pd.read_csv(os.path.join(DATA_DIR, "railsync_demo_250_rows.csv")).reset_index(drop=True)
    demo_X = demo_df.drop(columns=[c for c in DROP_COLUMNS if c in demo_df.columns], errors="ignore")
    demo_X = pd.get_dummies(demo_X, columns=CATEGORICAL_COLUMNS, dtype=int)
    demo_X = demo_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    demo_df["predicted_criticality"] = np.round(np.clip(model.predict(demo_X), 0, 100), 2)
    demo_df["predicted_priority"] = demo_df["predicted_criticality"].apply(get_priority)

    HORIZON_DAYS = 14
    CORRIDOR_DAILY_BUDGET_MIN = 240
    CREW_DAILY_CAPACITY = 90
    LATE_DAY_PENALTY = 2

    cp = cp_model.CpModel()
    n = len(demo_df)
    presence, day, intervals, day_x_presence = [], [], [], []
    for i, row in demo_df.iterrows():
        p = cp.NewBoolVar(f"presence_{i}")
        earliest = min(HORIZON_DAYS - 1, math.ceil(row["nearest_available_block_hours"] / 24))
        d = cp.NewIntVar(earliest, HORIZON_DAYS - 1, f"day_{i}")
        iv = cp.NewOptionalIntervalVar(d, 1, d + 1, p, f"iv_{i}")
        dp = cp.NewIntVar(0, HORIZON_DAYS - 1, f"dxp_{i}")
        cp.AddMultiplicationEquality(dp, [d, p])
        presence.append(p); day.append(d); intervals.append(iv); day_x_presence.append(dp)
        if row["permit_ready"] == 0:
            cp.Add(p == 0)

    for corridor, grp in demo_df.groupby("corridor_id"):
        idx = grp.index.tolist()
        cp.AddCumulative(
            [intervals[i] for i in idx],
            [int(demo_df.loc[i, "requested_block_duration_min"]) for i in idx],
            CORRIDOR_DAILY_BUDGET_MIN,
        )
    cp.AddCumulative(intervals, [int(c) for c in demo_df["crew_count_required"]], CREW_DAILY_CAPACITY)

    weight_terms = []
    for i, row in demo_df.iterrows():
        safety_bonus = 1.4 if (row["safety_critical"] == 1 and row["redundancy_available"] == 0) else 1.0
        w = int(round(row["predicted_criticality"] * safety_bonus * 10))
        weight_terms.append(w * presence[i])
    cp.Maximize(sum(weight_terms) - LATE_DAY_PENALTY * sum(day_x_presence))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8
    status = solver.Solve(cp)

    demo_df["scheduled"] = [bool(solver.Value(presence[i])) for i in range(n)]
    demo_df["scheduled_day"] = [int(solver.Value(day[i])) if solver.Value(presence[i]) else None for i in range(n)]

    schedule_summary = {
        "status": solver.StatusName(status),
        "objective_value": float(solver.ObjectiveValue()),
        "horizon_days": HORIZON_DAYS,
        "corridor_daily_budget_min": CORRIDOR_DAILY_BUDGET_MIN,
        "crew_daily_capacity": CREW_DAILY_CAPACITY,
        "n_tasks": n,
        "n_scheduled": int(demo_df["scheduled"].sum()),
        "scheduling_rate_by_priority": (
            demo_df.groupby("predicted_priority")["scheduled"].mean().round(3).to_dict()
        ),
    }

    print("Saving artifacts...")
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))
    joblib.dump(X, os.path.join(ARTIFACT_DIR, "X_all_encoded.joblib"))
    joblib.dump(df[["task_id"] + [c for c in df.columns if c != "task_id"]], os.path.join(ARTIFACT_DIR, "df_scored.joblib"))

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(ARTIFACT_DIR, "shap_importance.json"), "w") as f:
        json.dump(shap_importance.to_dict("records"), f, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "test_predictions.json"), "w") as f:
        json.dump(test_table.to_dict("records"), f, indent=2, default=str)
    # pandas upcasts the mixed int/None `scheduled_day` column to float, turning
    # unscheduled tasks' day into NaN instead of None/null — sanitize before export,
    # since NaN is not valid JSON and Starlette's JSONResponse rejects it on serve.
    schedule_tasks = demo_df.to_dict("records")
    for t in schedule_tasks:
        if isinstance(t.get("scheduled_day"), float) and math.isnan(t["scheduled_day"]):
            t["scheduled_day"] = None
        else:
            t["scheduled_day"] = int(t["scheduled_day"]) if t["scheduled_day"] is not None else None

    with open(os.path.join(ARTIFACT_DIR, "schedule.json"), "w") as f:
        json.dump(
            {"summary": schedule_summary, "tasks": schedule_tasks},
            f, indent=2, default=str,
        )

    print("Done. Artifacts written to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
