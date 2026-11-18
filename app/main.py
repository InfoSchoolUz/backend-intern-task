from fastapi import FastAPI

from app import llm  # noqa: F401  (you will need it)
from app.chunking import chunk_text  # noqa: F401  (you will need it)


def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # TODO: POST /ingest
    # TODO: POST /ask
    # TODO: GET /documents
    # TODO: DELETE /documents/{doc_id}

    return app


app = create_app()
