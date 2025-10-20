# pokemon_app/router.py

import os
import sys
import requests
import pandas as pd
import numpy as np
from typing import List, Tuple

# CHANGED: From FastAPI to APIRouter
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from joblib import load

# CHANGED: Create an APIRouter instance with a prefix
# All routes in this file will now start with /pokemon
router = APIRouter(
    prefix="/pokemon",
    tags=["Pokémon VGC Teammate Predictor"],
)

# ───────────────────────────────────────────────
# Paths (These will still work due to the file's location)
# ───────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DATAFRAMES_DIR  = os.path.join(BASE_DIR, "dataframes")

MODEL_FILE      = os.path.join(MODELS_DIR, "vgc_regi_restrictedcore_model.joblib")
X_DF_FILE       = os.path.join(DATAFRAMES_DIR, "X_df.csv")

# ───────────────────────────────────────────────
# Globals (populated in `load_pokemon_assets`)
# ───────────────────────────────────────────────
model           = None
label_columns   = None
X_df            = None

# ───────────────────────────────────────────────
# Fallbacks and sprite helper (No changes needed here)
# ───────────────────────────────────────────────
fallbacks: dict[str, str] = {
    "ogerpon-cornerstone": "ogerpon-cornerstone-mask",
    "ogerpon-hearthflame": "ogerpon-hearthflame-mask",
    "ogerpon-wellspring":  "ogerpon-wellspring-mask",
    "ogerpon":             "ogerpon-teal-mask",
    "landorus":            "landorus-incarnate",
    "tornadus":            "tornadus-incarnate",
    "thundurus":           "thundurus-incarnate",
    "enamorus":            "enamorus-incarnate",
    "urshifu":             "urshifu-single-strike",
    "indeedee-f":          "indeedee-female",
    "giratina":            "giratina-altered",
}

def get_sprite_url(poke_name: str) -> str:
    base = poke_name.lower().replace(" ", "-").replace("’", "").replace("'", "")
    attempts: List[str] = []
    if base in fallbacks:
        attempts.append(fallbacks[base])
    attempts.extend([base, base.split("-")[0]])

    for attempt in attempts:
        try:
            res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{attempt}", timeout=4)
            res.raise_for_status()
            data = res.json()
            sprite = (
                data["sprites"]["other"]["official-artwork"]["front_default"]
                or data["sprites"]["front_default"]
            )
            if sprite:
                return sprite
        except Exception:
            continue
    return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png"

# ───────────────────────────────────────────────
# Prediction helper (No changes needed here)
# ───────────────────────────────────────────────
def predict_teammates(core: Tuple[str, str], top_n: int = 20) -> pd.DataFrame:
    if model is None or X_df is None or label_columns is None:
        raise RuntimeError("Pokémon model and data not loaded. Check startup event.")
    input_row = pd.DataFrame(0, index=[0], columns=X_df.columns, dtype=np.int8)
    for mon in core:
        col = f"core_{mon}"
        if col in input_row.columns:
            input_row.at[0, col] = 1
        else:
            print(f"[WARN] Unknown core feature: {col}", flush=True)
    probs: List[float] = []
    for est, prob_arr in zip(model.estimators_, model.predict_proba(input_row)):
        if prob_arr.shape[1] == 2:
            probs.append(prob_arr[0, 1])
        else:
            label_idx = est.classes_[1] if 1 in est.classes_ else est.classes_[0]
            probs.append(1.0 if label_idx == 1 else 0.0)
    teammate_names = [c.replace("teammate_", "") for c in label_columns]
    ranked = sorted(zip(teammate_names, probs), key=lambda x: x[1], reverse=True)
    ranked = [row for row in ranked if row[1] > 0][:top_n]
    return pd.DataFrame(ranked, columns=["Teammate", "Predicted Probability"])

# ───────────────────────────────────────────────
# Request / response models (No changes needed here)
# ───────────────────────────────────────────────
class TeammateRequest(BaseModel):
    core1: str
    core2: str

# ───────────────────────────────────────────────
# API routes
# ───────────────────────────────────────────────
# CHANGED: Decorator now uses `router` instead of `app`
@router.get("/")
def read_pokemon_root():
    return {"message": "Pokémon VGC Teammate Predictor submodule is running!"}

@router.post("/predict-teammates")
async def predict_teammates_endpoint(req: TeammateRequest):
    try:
        results = predict_teammates((req.core1, req.core2))
        results["sprite_url"] = results["Teammate"].apply(get_sprite_url)
        return results.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ───────────────────────────────────────────────
# CHANGED: Startup logic moved to a standard function
# This will be called by the main app's startup event.
# ───────────────────────────────────────────────
def load_pokemon_assets() -> None:
    global model, label_columns, X_df
    print("[INFO] Loading Pokémon assets...", flush=True)
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Pokémon model file not found: {MODEL_FILE}", flush=True)
        sys.exit(1)
    bundle = load(MODEL_FILE, )
    model = bundle["model"]
    label_columns = bundle["label_columns"]
    if not os.path.exists(X_DF_FILE):
        print(f"[ERROR] Pokémon X_df.csv not found: {X_DF_FILE}", flush=True)
        sys.exit(1)
    X_df = pd.read_csv(X_DF_FILE, nrows=0)
    print("[INFO] Pokémon assets loaded successfully.", flush=True)