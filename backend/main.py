"""
RailSync AI - local demo backend.

Serves precomputed model results (predictions, metrics, SHAP importance,
CP-SAT schedule) plus one live endpoint (per-task SHAP explanation) to the
static frontend in ../frontend.

Run:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend"))

app = FastAPI(title="RailSync AI - Local Demo API")


def _load_json(name):
    with open(os.path.join(ARTIFACT_DIR, name)) as f:
        return json.load(f)


print("Loading model + artifacts...")
MODEL = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
FEATURE_COLUMNS = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))
X_ALL = joblib.load(os.path.join(ARTIFACT_DIR, "X_all_encoded.joblib"))
DF_SCORED = joblib.load(os.path.join(ARTIFACT_DIR, "df_scored.joblib")).set_index("task_id", drop=False)
EXPLAINER = shap.TreeExplainer(MODEL, feature_perturbation="tree_path_dependent")

METRICS = _load_json("metrics.json")
SHAP_IMPORTANCE = _load_json("shap_importance.json")
TEST_PREDICTIONS = _load_json("test_predictions.json")
SCHEDULE = _load_json("schedule.json")
print(f"Loaded. {len(TEST_PREDICTIONS)} test-set rows, {len(SCHEDULE['tasks'])} scheduling-demo rows.")


@app.get("/api/health")
def health():
    return {"status": "ok", "test_rows": len(TEST_PREDICTIONS), "schedule_rows": len(SCHEDULE["tasks"])}


@app.get("/api/metrics")
def get_metrics():
    """Overall + per-priority-class model performance, and the baseline comparison."""
    return METRICS


@app.get("/api/predictions")
def get_predictions(limit: int = 200, priority: str | None = None):
    """Test-set predictions: actual vs predicted criticality/priority, sorted by predicted criticality desc."""
    rows = TEST_PREDICTIONS
    if priority:
        rows = [r for r in rows if r["predicted_priority"] == priority.upper()]
    return {"total": len(rows), "rows": rows[:limit]}


@app.get("/api/feature-importance")
def get_feature_importance(limit: int = 20):
    return SHAP_IMPORTANCE[:limit]


@app.get("/api/explain/{task_id}")
def explain_task(task_id: str):
    """Live SHAP explanation for one task, by task_id (e.g. MT-000123)."""
    if task_id not in DF_SCORED.index:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' not found")

    # X_ALL is positionally indexed like the original df (0..n-1); map task_id -> row position
    row_pos = DF_SCORED.index.get_loc(task_id)
    task_row = X_ALL.iloc[[row_pos]]

    prediction = float(np.clip(MODEL.predict(task_row)[0], 0, 100))
    exp = EXPLAINER(task_row)

    contributors = (
        pd.DataFrame({
            "feature": task_row.columns,
            "feature_value": task_row.iloc[0].values,
            "shap_value": exp.values[0],
        })
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(12)
        .drop(columns="abs_shap")
    )

    meta = DF_SCORED.loc[task_id][[
        "department", "corridor_id", "asset_type", "station_code",
        "criticality_score", "priority_class", "predicted_criticality", "predicted_priority",
    ]].to_dict()

    return {
        "task_id": task_id,
        "prediction": round(prediction, 2),
        "meta": meta,
        "base_value": float(exp.base_values[0]),
        "contributors": contributors.to_dict("records"),
    }


@app.get("/api/schedule")
def get_schedule():
    """CP-SAT scheduling result on the 250-row demo subset."""
    return SCHEDULE


# Serve the static frontend last, so /api/* routes above take priority.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
