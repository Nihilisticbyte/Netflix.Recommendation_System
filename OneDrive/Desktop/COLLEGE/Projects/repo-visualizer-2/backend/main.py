"""
Repository Structure Analysis & Visualisation System
Backend entry point — FastAPI app with CORS and router registration.
"""
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import graph, ai

app = FastAPI(
    title="Repo Visualizer API",
    description="Parses local Git repositories into interactive dependency graphs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router, prefix="/api/graph", tags=["Graph"])
app.include_router(ai.router,    prefix="/api/ai",    tags=["AI"])


@app.get("/health")
def health():
    return {"status": "ok"}
