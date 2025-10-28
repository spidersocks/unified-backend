# C:\Users\sfont\unified_backend\main.py

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the routers from your app subdirectories
from pokemon_app.router import router as pokemon_router
from news_app.router import router as news_router
from run_calculator_app.router import router as run_calculator_router
from llm.router import router as llm_router

from pokemon_app.router import load_pokemon_assets

app = FastAPI(
    title="Unified Portfolio Backend",
    description="A single FastAPI service combining multiple portfolio projects.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STARTED = False

@app.on_event("startup")
def startup_event():
    global _STARTED
    print("[INFO] Main app startup: Loading assets for all submodules...", flush=True)
    load_pokemon_assets()
    _STARTED = True
    print("[INFO] All assets loaded.", flush=True)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Unified Portfolio Backend is running!"}

@app.get("/healthz")
def healthz():
    # Minimal check: process is alive and app constructed
    return {"ok": True, "started": _STARTED}

app.include_router(pokemon_router)
app.include_router(news_router)
app.include_router(run_calculator_router)
app.include_router(llm_router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)