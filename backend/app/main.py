from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.router import api_router


def get_application() -> FastAPI:
    app = FastAPI(
        title="ChatSMITH Backend",
        description="FastAPI backend for ChatSMITH pipeline and auth orchestration",
        version="0.1.0",
    )

    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # use ["*"] for dev if desired
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
