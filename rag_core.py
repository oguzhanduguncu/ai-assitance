import os
from pathlib import Path
import numpy as np
import faiss
import hashlib
import requests
import json

from sentence_transformers import SentenceTransformer
import pdfplumber

# =========================
# CONFIG
# =========================

DATA_DIR = "data"

INDEX_FILE = "faiss.index"
META_FILE = "metadata.npy"
CHUNKS_FILE = "chunks.npy"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

# =========================
# PROMPT
# =========================

BASE_PROMPT = """
You are a C++ language expert.
Answer clearly and concisely in English.

Use the provided documents as supporting context.
If the documents do not explicitly define something,
you may answer based on general C++ language rules.
If behavior is undefined, say so explicitly.

DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:
"""

# =========================
# HELPERS
# =========================

def make_chunk_id(text, source, page):
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(str(source).encode())
    h.update(str(page).encode())
    return h.hexdigest()

# =========================
# INGEST
# =========================

def load_documents():
    docs = []
    for f in Path(DATA_DIR).rglob("*"):
        if f.suffix.lower() == ".pdf":
            with pdfplumber.open(f) as pdf:
                for i, p in enumerate(pdf.pages):
                    t = p.extract_text()
                    if t:
                        docs.append((t, str(f), i + 1))
        elif f.suffix.lower() == ".txt":
            docs.append((f.read_text(encoding="utf-8"), str(f), None))
    return docs

def chunk_documents(docs):
    chunks, meta = [], []
    for text, src, page in docs:
        start = 0
        while start < len(text):
            part = text[start:start + CHUNK_SIZE]
            cid = make_chunk_id(part, src, page)
            chunks.append(part)
            meta.append({"id": cid, "source": src, "page": page})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks, meta

# =========================
# INDEX
# =========================

def build_index(chunks, meta):
    model = SentenceTransformer(EMBEDDING_MODEL)

    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        old_meta = np.load(META_FILE, allow_pickle=True).tolist()
        known = {m["id"] for m in old_meta}

        new_chunks, new_meta = [], []
        for c, m in zip(chunks, meta):
            if m["id"] not in known:
                new_chunks.append(c)
                new_meta.append(m)

        if new_chunks:
            emb = model.encode(
                ["query: " + c for c in new_chunks],
                normalize_embeddings=True,
                show_progress_bar=True
            )
            index.add(emb)
            old_meta.extend(new_meta)
            faiss.write_index(index, INDEX_FILE)
            np.save(META_FILE, old_meta)

        return index, old_meta, model

    emb = model.encode(
        ["query: " + c for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True
    )
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, INDEX_FILE)
    np.save(META_FILE, meta)
    return index, meta, model

# =========================
# QUERY
# =========================

def ask(question, index, meta, model, chunks, k=3):
    q_emb = model.encode(["query: " + question], normalize_embeddings=True)
    _, ids = index.search(q_emb, k)

    context = "\n\n".join(chunks[i] for i in ids[0])

    prompt = BASE_PROMPT.format(
        context=context,
        question=question
    )

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        timeout=120
    )
    r.raise_for_status()

    # ---- STREAMED JSON HANDLING ----
    answer_parts = []

    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        msg = obj.get("message", {}).get("content")
        if msg:
            answer_parts.append(msg)

    if not answer_parts:
        return "[LLM returned no usable content.]"

    return "".join(answer_parts).strip()


# =========================
# MAIN
# =========================

def main():
    if os.path.exists(CHUNKS_FILE) and os.path.exists(META_FILE):
        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
        meta = np.load(META_FILE, allow_pickle=True).tolist()
    else:
        docs = load_documents()
        chunks, meta = chunk_documents(docs)
        np.save(CHUNKS_FILE, chunks)
        np.save(META_FILE, meta)

    index, meta, model = build_index(chunks, meta)

    print("C++ RAG ready. Type 'exit' to quit.\n")
    while True:
        q = input(">> ").strip()
        if q.lower() == "exit":
            break
        print("\n" + ask(q, index, meta, model, chunks))
        print("-" * 60)

if __name__ == "__main__":
    main()
