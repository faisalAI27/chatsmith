import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.router import api_router

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def get_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


def get_application() -> FastAPI:
    app = FastAPI(
        title="ChatSMITH Backend",
        description="FastAPI backend for the ChatSMITH scraping and chat pipeline",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),  # use ["*"] for dev if desired
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")
    return app


app = get_application()


@app.get("/health")
def health():
    return {"status": "ok"}
