# C:\Users\sfont\unified_backend\main.py

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the routers from your app subdirectories
from pokemon_app.router import router as pokemon_router
from news_app.router import router as news_router
from run_calculator_app.router import router as run_calculator_router
# --- START: NEW IMPORT ---
from dialogflow_app.router import router as dialogflow_router 
# --- END: NEW IMPORT ---

# Import the asset loading function from the Pokémon app
from pokemon_app.router import load_pokemon_assets

app = FastAPI(
    title="Unified Portfolio Backend",
    description="A single FastAPI service combining multiple portfolio projects.",
    version="1.0.0",
)

# Add CORS middleware once for the entire application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a startup event for the main app
@app.on_event("startup")
def startup_event():
    """
    Load all necessary models and data on application startup.
    The other apps (news, running) load their data when the module is imported,
    so we only need to explicitly call the one for the Pokémon app.
    """
    print("[INFO] Main app startup: Loading assets for all submodules...", flush=True)
    load_pokemon_assets()
    print("[INFO] All assets loaded.", flush=True)


# Add a root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Unified Portfolio Backend is running!"}


# Include the routers from each of the sub-apps
# The prefixes we defined in each router.py ensure the URLs don't clash
app.include_router(pokemon_router)
app.include_router(news_router)
app.include_router(run_calculator_router)
# --- START: NEW ROUTER INCLUSION ---
app.include_router(dialogflow_router) 
# --- END: NEW ROUTER INCLUSION ---

# Add the entrypoint for running with uvicorn
if __name__ == "__main__":
    # Render uses the PORT environment variable; fall back to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)