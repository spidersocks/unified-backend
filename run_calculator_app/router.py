# run_calculator_app/router.py

import pandas as pd
import pickle
import math
import os
from typing import List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(
    prefix="/running",
    tags=["800m Calculator"],
)

# === Helper Functions (Unchanged) ===
def convert_to_seconds(time_str: str) -> float:
    try:
        if not any(c.isdigit() for c in time_str): raise ValueError("Invalid input: Only numbers, colons, and periods are allowed.")
        if ":" not in time_str and "." not in time_str: return float(time_str)
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), float(parts[1] + "0" if len(parts[1]) < 2 else parts[1])
                return minutes * 60 + seconds
            elif len(parts) > 2: raise ValueError("Too many colons.")
        if "." in time_str:
            parts = time_str.split(".")
            if len(parts) == 2:
                if len(parts[0]) == 1: minutes = int(parts[0])
                else: return float(time_str)
                seconds = int(parts[1] + "0" if len(parts[1]) < 2 else parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                minutes, seconds, fractional_seconds = int(parts[0]), int(parts[1]), float("0." + parts[2])
                return minutes * 60 + seconds + fractional_seconds
            else: raise ValueError("Too many dots.")
        raise ValueError("Could not parse input.")
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")

def seconds_to_minutes(seconds: float) -> str:
    total_seconds = round(seconds, 2)
    minutes = int(total_seconds // 60)
    remaining_seconds = total_seconds - minutes * 60
    if remaining_seconds >= 59.995:
        minutes += 1
        remaining_seconds = 0.0
    return f"{minutes}:{remaining_seconds:05.2f}" if minutes > 0 else f"{remaining_seconds:05.2f}"

def predict_800m(model, feature_cols, input_values):
    processed = [sum(convert_to_seconds(x) for x in val) / len(val) if isinstance(val, list) else convert_to_seconds(val) for val in input_values]
    X = pd.DataFrame([processed], columns=feature_cols)
    prediction = model.predict(X)[0]
    if not 96 <= prediction <= 240:
        raise ValueError("Predicted time is unrealistic (outside 1:36-4:00). Check inputs.")
    return {"predicted_seconds": float(prediction), "predicted_formatted": seconds_to_minutes(prediction)}

def reverse_predict(df, target_col, goal_time, interval_cols, rounding=None):
    val = convert_to_seconds(goal_time)
    upper, lower = math.ceil(val), math.floor(val)
    frac = val - lower
    upper_row, lower_row = df[df[target_col] == upper], df[df[target_col] == lower]
    if upper_row.empty or lower_row.empty: raise ValueError("Goal time is out of range.")
    rounding = [0.5] * len(interval_cols) if rounding is None else ([rounding] * len(interval_cols) if isinstance(rounding, (float, int)) else rounding)
    splits = []
    for idx, col in enumerate(interval_cols):
        interp = upper_row[col].values[0] * frac + lower_row[col].values[0] * (1 - frac)
        rounded = round(interp / rounding[idx]) * rounding[idx]
        splits.append({"interval": col, "seconds": float(rounded)})
    return splits

# === Paths (Unchanged) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TABLES_DIR = os.path.join(BASE_DIR, "tables")

# === OPTIMIZED: Training Configurations now point to .parquet files ===
TRAINING_CONFIG = {
    "600m_x3": {"model_file": os.path.join(MODELS_DIR, "model_600.pkl"), "feature_cols": ["First 600m", "Second 600m", "Third 600m"], "table_file": os.path.join(TABLES_DIR, "600.parquet"), "interval_cols": ["First 600m", "Second 600m", "Third 600m"]},
    "600m_400m_x3": {"model_file": os.path.join(MODELS_DIR, "model_600_400.pkl"), "feature_cols": ["600m", "3x400m average"], "table_file": os.path.join(TABLES_DIR, "600_400.parquet"), "interval_cols": ["600m", "3x400m average"]},
    "600m_300m_x4": {"model_file": os.path.join(MODELS_DIR, "model_600_300.pkl"), "feature_cols": ["600m", "4x300m average"], "table_file": os.path.join(TABLES_DIR, "600_300.parquet"), "interval_cols": ["600m", "4x300m average"]},
    "500m_x3": {"model_file": os.path.join(MODELS_DIR, "model_500.pkl"), "feature_cols": ["First 500m", "Second 500m", "Third 500m"], "table_file": os.path.join(TABLES_DIR, "500.parquet"), "interval_cols": ["First 500m", "Second 500m", "Third 500m"]},
    "300m_x3x2": {"model_file": os.path.join(MODELS_DIR, "model_300.pkl"), "feature_cols": ["Set 1 3x300m average", "Set 2 3x300m average"], "table_file": os.path.join(TABLES_DIR, "300.parquet"), "interval_cols": ["Set 1 3x300m average", "Set 2 3x300m average"]},
    "ladder": {"model_file": os.path.join(MODELS_DIR, "model_ladder.pkl"), "feature_cols": ["First 300m", "First 400m", "500m", "Second 400m", "Second 300m", "200m"], "table_file": os.path.join(TABLES_DIR, "ladder.parquet"), "interval_cols": ["First 300m", "First 400m", "500m", "Second 400m", "Second 300m", "200m"]},
    "200m_x8": {"model_file": os.path.join(MODELS_DIR, "model_200.pkl"), "feature_cols": ["First 200m", "Second 200m", "Third 200m", "Fourth 200m", "Fifth 200m", "Sixth 200m", "Seventh 200m", "Eighth 200m"], "table_file": os.path.join(TABLES_DIR, "200.parquet"), "interval_cols": ["First 200m", "Second 200m", "Third 200m", "Fourth 200m", "Fifth 200m", "Sixth 200m", "Seventh 200m", "Eighth 200m"]}
}

# === OPTIMIZED: Load Models and Tables at Startup ===
MODELS, TABLES = {}, {}
for key, cfg in TRAINING_CONFIG.items():
    with open(cfg["model_file"], "rb") as f: MODELS[key] = pickle.load(f)
    # OPTIMIZED: Use pd.read_parquet for much lower memory usage
    TABLES[key] = pd.read_parquet(cfg["table_file"])

# === Request Models (Unchanged) ===
class PredictRequest(BaseModel):
    training_type: str
    input_values: List[Union[str, List[str]]]

class ReversePredictRequest(BaseModel):
    training_type: str
    goal_time: str

# === API Endpoints (Unchanged) ===
@router.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "800m Calculator submodule is up."}

@router.get("/get-training-types")
def get_training_types():
    return [{"key": k, "features": v["feature_cols"], "intervals": v["interval_cols"]} for k, v in TRAINING_CONFIG.items()]

@router.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        config = TRAINING_CONFIG[req.training_type]
        model = MODELS[req.training_type]
        return predict_800m(model, config["feature_cols"], req.input_values)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reverse-predict")
def reverse_predict_endpoint(req: ReversePredictRequest):
    try:
        config = TRAINING_CONFIG[req.training_type]
        df = TABLES[req.training_type]
        return reverse_predict(df, "TARGET", req.goal_time, config["interval_cols"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))