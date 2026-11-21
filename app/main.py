from threading import Lock

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from app import llm
from app.chunking import chunk_text


def _not_blank(value: str) -> str:
    """Reject strings that are empty or contain only whitespace."""
    if not value.strip():
        raise ValueError("must not be empty or whitespace only")
    return value


class DocumentIn(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    _check_id = field_validator("id")(_not_blank)
    _check_text = field_validator("text")(_not_blank)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    _check_question = field_validator("question")(_not_blank)


def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()

    # doc_id -> list of chunks. Every chunk is a dict:
    # {"chunk_id": str, "text": str, "vector": list[float]}
    # The store lives inside create_app(), so each app has its own copy.
    store: dict[str, list[dict]] = {}
    lock = Lock()  # FastAPI runs these functions in threads; the lock keeps the store safe

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(body: IngestRequest):
        total_chunks = 0
        with lock:
            for doc in body.documents:
                chunks = []
                for index, chunk in enumerate(chunk_text(doc.text)):
                    chunks.append(
                        {
                            "chunk_id": f"{doc.id}:{index}",
                            "text": chunk,
                            "vector": llm.embed(chunk),
                        }
                    )
                store[doc.id] = chunks  # replaces old chunks if the id already exists
                total_chunks += len(chunks)
        return {"ingested": len(body.documents), "chunks": total_chunks}

    @app.post("/ask")
    def ask(body: AskRequest):
        question_vector = llm.embed(body.question)
        with lock:
            if not store:
                raise HTTPException(status_code=404, detail="No documents ingested")

            scored = []
            for doc_id, chunks in store.items():
                for chunk in chunks:
                    # vectors are unit length, so the dot product is the cosine similarity
                    score = sum(a * b for a, b in zip(question_vector, chunk["vector"]))
                    scored.append((score, doc_id, chunk))

        # sort() is stable: equal scores keep their original (insertion) order
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[: body.top_k]

        answer = llm.generate(body.question, [chunk["text"] for _, _, chunk in best])
        sources = [
            {
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
            }
            for score, doc_id, chunk in best
        ]
        return {"answer": answer, "sources": sources}

    @app.get("/documents")
    def list_documents():
        with lock:
            documents = [
                {"id": doc_id, "chunks": len(chunks)}
                for doc_id, chunks in sorted(store.items())
            ]
        return {"documents": documents}

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        with lock:
            if doc_id not in store:
                raise HTTPException(status_code=404, detail="Document not found")
            del store[doc_id]
        return Response(status_code=204)

    return app


app = create_app()
            
