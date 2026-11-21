# Notes

- Design: the in-memory store is a dict (`doc_id -> list of chunks`) created inside `create_app()`, so every app instance starts empty. Request validation uses Pydantic models, and a `Lock` protects the store because FastAPI runs sync endpoints in threads.
- `/ask` scores every chunk with a dot product (vectors are unit length), uses a stable sort, and returns the top `top_k` chunks; re-ingesting an existing id replaces its chunks.
- With more time I would add more tests (rest of the edge cases), a vector index for faster search, and a persistent store.
- AI tools used: Claude (Anthropic) helped me write and test the code; I reviewed and understood each part.


• chunk_text: har bir bo'lak size so'z, keyingisi size - overlap so'z oldinga suriladi, oxirgi so'zga yetganda to'xtaydi.
• store: create_app() ichidagi lug'at (hujjat id → bo'laklar). Shuning uchun har bir ilovada u bo'sh boshlanadi.
• /ask: savolni vektorga aylantiradi, har bir bo'lak bilan skalyar ko'paytma (o'xshashlik) hisoblaydi, eng yuqori top_k tasini qaytaradi.
• Lock: bir vaqtda kelgan so'rovlar ma'lumotni buzmasligi uchun.
