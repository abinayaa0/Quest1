"""
FastAPI Application Entry Point — Video Dialogue Localization Server
"""

import sys
from pathlib import Path

# Add src directory to sys.path so imports resolve cleanly
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="Video Dialogue Localization API",
    description="FastAPI Wrapper for Video Dialogue Localization System (V2 Pipeline)",
    version="2.0",
)

# CORS middleware for web frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists before mounting static files
output_dir = Path("output").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

# Include routes at root
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
