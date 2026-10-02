"""
docstore.py — General unstructured document ingestion and retrieval.

Supports PDF, DOCX, plain text, and URLs. Chunks content and stores
in a per-user ChromaDB collection so the hybrid endpoint can retrieve
relevant context alongside SQL results.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from typing import Literal

import chromadb
import httpx
from rank_bm25 import BM25Okapi


CHUNK_WORDS   = 300
OVERLAP_WORDS = 50


def _chunk(text: str) -> list[str]:
    words = text.split()
    step, chunks = CHUNK_WORDS - OVERLAP_WORDS, []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext in (".docx", ".doc"):
        return _extract_docx(data)
    # txt, md, csv, json, html — treat as plain text
    return data.decode("utf-8", errors="replace")


def fetch_url(url: str) -> str:
    resp = httpx.get(url, follow_redirects=True, timeout=15)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "pdf" in content_type:
        return _extract_pdf(resp.content)
    # Strip HTML tags for web pages
    import re
    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = re.sub(r"\s+", " ", text)
    return text


class DocStore:
    """Per-user unstructured document store backed by ChromaDB + BM25 hybrid search."""

    def __init__(self, user_id: str, persist_dir: str = ".chroma_data"):
        path = os.path.join(persist_dir, user_id)
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name="unstructured",
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 index is rebuilt from ChromaDB on load — no separate persistence needed
        self._bm25:   BM25Okapi | None = None
        self._corpus: list[str]        = []   # raw chunk texts, parallel to BM25 index
        self._ids:    list[str]        = []   # ChromaDB ids, parallel to corpus
        self._metas:  list[dict]       = []   # metadata, parallel to corpus
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Load all chunks from ChromaDB and rebuild the BM25 index in memory."""
        if self._col.count() == 0:
            self._bm25, self._corpus, self._ids, self._metas = None, [], [], []
            return
        result = self._col.get(include=["documents", "metadatas"])
        self._ids     = result["ids"]
        self._corpus  = result["documents"]
        self._metas   = result["metadatas"]
        tokenised     = [doc.lower().split() for doc in self._corpus]
        self._bm25    = BM25Okapi(tokenised)

    @staticmethod
    def _uid(source: str, chunk_idx: int, chunk_text: str) -> str:
        return hashlib.sha256(
            f"{source}:{chunk_idx}:{chunk_text[:40]}".encode()
        ).hexdigest()[:16]

    def _upsert(self, ids: list, docs: list, metas: list) -> None:
        self._col.upsert(ids=ids, documents=docs, metadatas=metas)
        self._rebuild_bm25()

    def add_file(self, filename: str, data: bytes) -> int:
        text   = extract_text(filename, data)
        chunks = _chunk(text)
        if not chunks:
            return 0
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(self._uid(filename, i, chunk))
            docs.append(chunk)
            metas.append({"source": filename, "chunk": i, "kind": "file"})
        self._upsert(ids, docs, metas)
        return len(chunks)

    def add_url(self, url: str) -> int:
        text   = fetch_url(url)
        chunks = _chunk(text)
        if not chunks:
            return 0
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(self._uid(url, i, chunk))
            docs.append(chunk)
            metas.append({"source": url, "chunk": i, "kind": "url"})
        self._upsert(ids, docs, metas)
        return len(chunks)

    def search(self, question: str, n_results: int = 4) -> list[dict]:
        total = self._col.count()
        if total == 0:
            return []

        n = min(n_results, total)

        # ── Vector search (semantic) ──────────────────────────
        vec_res   = self._col.query(query_texts=[question], n_results=total)
        vec_ids   = vec_res["ids"][0]
        vec_dists = vec_res["distances"][0]   # cosine distance: lower = more similar
        # convert distance → similarity score (0–1)
        vec_scores = {vid: 1.0 - d for vid, d in zip(vec_ids, vec_dists)}

        # ── BM25 search (keyword) ─────────────────────────────
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None:
            raw      = self._bm25.get_scores(question.lower().split())
            bm25_max = max(raw) if max(raw) > 0 else 1.0
            bm25_scores = {cid: float(raw[i]) / bm25_max
                           for i, cid in enumerate(self._ids)}

        # ── Hybrid score: 60% vector + 40% BM25 ──────────────
        all_ids = set(vec_scores) | set(bm25_scores)
        hybrid  = {
            cid: 0.6 * vec_scores.get(cid, 0.0) + 0.4 * bm25_scores.get(cid, 0.0)
            for cid in all_ids
        }
        top_ids = sorted(hybrid, key=lambda x: hybrid[x], reverse=True)[:n]

        # ── Build result from in-memory corpus ────────────────
        id_to_idx = {cid: i for i, cid in enumerate(self._ids)}
        return [
            {
                "text":   self._corpus[id_to_idx[cid]],
                "source": self._metas[id_to_idx[cid]].get("source", ""),
                "chunk":  self._metas[id_to_idx[cid]].get("chunk", 0),
                "score":  round(hybrid[cid], 4),
            }
            for cid in top_ids if cid in id_to_idx
        ]

    def list_sources(self) -> list[str]:
        return sorted({m.get("source", "") for m in self._metas})

    def count(self) -> int:
        return self._col.count()
