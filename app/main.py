from __future__ import annotations

from fastapi import FastAPI

from app.routers.glossaries import router as glossaries_router
from app.routers.rag import router as rag_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Glossary RAG Service",
        version="0.1.0",
        description="FastAPI service for managing glossary datasets and generating RAG answers.",
    )

    app.include_router(glossaries_router)
    app.include_router(rag_router)

    @app.get("/healthz", tags=["system"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
