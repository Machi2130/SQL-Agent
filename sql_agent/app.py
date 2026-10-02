
from __future__ import annotations

import asyncio
import datetime
import decimal
import json
import os
import re
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from . import cache as redis_cache
from . import storage
from .auth import get_current_user
from .docstore import DocStore
from .knowledge import KnowledgeEngine
from .query_planner import build_query_plan, plan_to_prompt, reset_concept_cache

load_dotenv()

try:
    from pymongo import MongoClient
    from bson import ObjectId
except ImportError:
    MongoClient = None
    ObjectId    = None

try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
except (ImportError, Exception):
    Cluster = None

import faiss


# ═══════════════════════════════════════════════════════════════
#  CONCURRENCY POOLS
#
#  ThreadPoolExecutor  — I/O-bound: DB queries, Redis calls, Groq HTTP
#    Threads release the GIL while waiting on sockets/network, so
#    multiple threads run concurrently even in CPython.
#
#  ProcessPoolExecutor — CPU-bound: SentenceTransformer.encode(), FAISS
#    These hold the GIL; threads cannot parallelize them.
#    Spawning separate interpreter processes bypasses the GIL entirely.
# ═══════════════════════════════════════════════════════════════

_THREAD_POOL  = ThreadPoolExecutor(max_workers=int(os.getenv("THREAD_WORKERS", 10)))
_PROCESS_POOL = ProcessPoolExecutor(max_workers=int(os.getenv("PROCESS_WORKERS", 2)))

DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT_SECONDS", 30))


# ─────────────────────────────────────────────────────────────
#  LIFESPAN — startup / shutdown
#  Replaces Flask's @app.before_first_request.
#  Guarantees clean pool shutdown so in-flight tasks finish.
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_PROCESS_POOL, _warm_embed_model)
    yield
    _THREAD_POOL.shutdown(wait=False)
    _PROCESS_POOL.shutdown(wait=False)


app = FastAPI(title="AI Query Agent", version="2.0.0", lifespan=lifespan)
_cors_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if os.path.isdir("static"):  # web UI assets; absent when only the MCP server is deployed
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC REQUEST MODELS
#  FastAPI validates these at the HTTP boundary automatically and
#  generates OpenAPI docs at /docs — no extra code needed.
# ═══════════════════════════════════════════════════════════════

class ConnectRequest(BaseModel):
    db_type:  str
    groq_key: str = ""   # empty = use server-side GROQ_API_KEY env var
    host:     str = "localhost"
    port:     str = ""
    username: str = ""
    password: str = ""
    database: str = ""
    keyspace: str = ""
    db_path:  str = ""


class QueryRequest(BaseModel):
    question:   str
    session_id: str = "default"
    page:       int = Field(default=1, ge=1, description="1-based page number")
    page_size:  int = Field(default=50, ge=1, le=500)


class StreamQueryRequest(BaseModel):
    question:   str
    session_id: str = "default"


class ExplainRequest(BaseModel):
    sql: str


class FeedbackRequest(BaseModel):
    question:   str
    sql:        str
    is_correct: bool = True
    correction: Optional[str] = None


class ChartRequest(BaseModel):
    columns: list[str]
    rows:    list[list[Any]]


class HybridRequest(BaseModel):
    question:   str
    session_id: str = "default"
    url:        str = ""   # optional: ingest a URL on-the-fly before answering


# ═══════════════════════════════════════════════════════════════
#  ASYNC HELPERS
# ═══════════════════════════════════════════════════════════════

async def run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_THREAD_POOL, fn, *args)


async def run_in_process(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_PROCESS_POOL, fn, *args)


async def exec_sql_async(engine, query: str) -> tuple[list, list]:
    """Execute SQL in a thread with a hard timeout."""
    def _exec():
        with engine.connect() as conn:
            result = conn.execute(text(query))
            cols = list(result.keys())
            return cols, safe_rows(cols, result.fetchall())
    return await asyncio.wait_for(run_in_thread(_exec), timeout=DB_QUERY_TIMEOUT)


# ═══════════════════════════════════════════════════════════════
#  CONNECTION POOL — production SQLAlchemy config
# ═══════════════════════════════════════════════════════════════

def _make_engine(url: str):
    """
    pool_size=5       — persistent connections per worker process
    max_overflow=10   — extra connections during traffic spikes
    pool_timeout=30   — wait before QueuePool raises timeout error
    pool_recycle=1800 — recycle after 30 min; prevents MySQL 'gone away'
    pool_pre_ping=True — sends SELECT 1 before checkout; detects stale connections
    """
    return create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PROCESS-LEVEL CACHES
# ═══════════════════════════════════════════════════════════════

_engines:    dict[str, Any] = {}
_mongos:     dict[str, Any] = {}
_cassandras: dict[str, Any] = {}
_retrievers: dict[str, Any] = {}
_knowledge:  dict[str, Any] = {}
_docstores:  dict[str, Any] = {}

class _OnnxEmbedder:
    """all-MiniLM-L6-v2 through chromadb's bundled ONNX runtime: the same
    384-dim L2-normalized vectors sentence-transformers produced, without the
    ~750 MB torch/transformers stack. Model file (~80 MB) downloads once."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        return np.asarray(self._ef(list(texts)), dtype=np.float32)


_EMBED_MODEL: _OnnxEmbedder | None = None


def _warm_embed_model() -> None:
    global _EMBED_MODEL
    _EMBED_MODEL = _OnnxEmbedder()


def _get_embed_model() -> _OnnxEmbedder:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = _OnnxEmbedder()
    return _EMBED_MODEL


# ═══════════════════════════════════════════════════════════════
#  SERIALIZATION
# ═══════════════════════════════════════════════════════════════

def serialize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if ObjectId and isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, set):
        return list(v)
    return v


def safe_rows(columns: list, raw_rows: list) -> list:
    return [[serialize_value(cell) for cell in row] for row in raw_rows]


# ═══════════════════════════════════════════════════════════════
#  EMBEDDING MODEL  (lazy-loaded once per process)
# ═══════════════════════════════════════════════════════════════

def _encode_texts(texts: list[str]) -> np.ndarray:
    """CPU-bound — called via run_in_process() to bypass the GIL."""
    return _get_embed_model().encode(texts, normalize_embeddings=True).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  SCHEMA RETRIEVER  (50% semantic + 30% BM25 + 20% column signal)
# ═══════════════════════════════════════════════════════════════

class SchemaRetriever:
    COLUMN_SIGNALS = {
        "sales":     ["total_amount", "sale_amount", "revenue", "net_amount"],
        "revenue":   ["total_amount", "price",       "amount",  "net_sales"],
        "amount":    ["total_amount", "price",       "amt",     "amount"],
        "stock":     ["qty_on_hand",  "stock_qty",   "quantity","available_qty"],
        "inventory": ["qty_on_hand",  "stock_qty",   "quantity","reorder_level"],
        "customer":  ["customer_id",  "cust_id",     "email",   "phone"],
        "product":   ["sku",          "product_id",  "barcode", "category"],
        "order":     ["order_id",     "order_date",  "total",   "status"],
        "date":      ["order_date",   "created_at",  "updated_at","invoice_date"],
        "tax":       ["cgst",         "sgst",        "igst",    "tax_amount"],
        "gst":       ["cgst",         "sgst",        "igst",    "hsn_code"],
        "purchase":  ["po_id",        "vendor_id",   "purchase_date","grn_id"],
        "employee":  ["emp_id",       "designation", "department","salary"],
        "store":     ["store_id",     "branch_id",   "store_name","location"],
    }

    def __init__(self, schema_dict: dict):
        self.schema_dict  = schema_dict
        self.table_names  = list(schema_dict.keys())
        self._build_docs()
        self._build_faiss_index()
        self._build_bm25()

    def _build_docs(self):
        self.docs = []
        for tbl, cols in self.schema_dict.items():
            clean_cols = [c.split(" ")[0] for c in cols]
            readable   = tbl.replace("_", " ")
            self.docs.append(f"{readable} {readable} {tbl} " + " ".join(clean_cols))

    def _build_faiss_index(self):
        model = _get_embed_model()
        embs  = model.encode(self.docs, normalize_embeddings=True).astype(np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def _build_bm25(self):
        self.bm25 = BM25Okapi([doc.lower().split() for doc in self.docs])

    def _column_score(self, question: str, table_name: str) -> float:
        ql   = question.lower()
        cols = [c.split(" ")[0].lower() for c in self.schema_dict[table_name]]
        score = sum(
            1.0 for kw, sigs in self.COLUMN_SIGNALS.items()
            if kw in ql and any(s in cols for s in sigs)
        )
        return min(score, 3.0) / 3.0

    def retrieve(self, question: str, top_k: int = 5, db_type: str = "sql") -> dict:
        model  = _get_embed_model()
        q_emb  = model.encode([question], normalize_embeddings=True).astype(np.float32)
        dists, idxs = self.index.search(q_emb, len(self.table_names))
        sem_scores  = {self.table_names[i]: float(dists[0][j]) for j, i in enumerate(idxs[0])}

        bm25_raw = self.bm25.get_scores(question.lower().split())
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_scores = {self.table_names[i]: float(bm25_raw[i]) / bm25_max
                       for i in range(len(self.table_names))}

        hybrid = {
            tbl: (0.50 * sem_scores.get(tbl, 0.0)
                + 0.30 * bm25_scores.get(tbl, 0.0)
                + 0.20 * self._column_score(question, tbl))
            for tbl in self.table_names
        }
        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)

        if db_type == "mongo":
            best = ranked[0][0]
            return {best: self.schema_dict[best]}

        selected = [t for t, s in ranked[:top_k] if s > 0.05] or [t for t, _ in ranked[:5]]
        return {t: self.schema_dict[t] for t in selected}


# ═══════════════════════════════════════════════════════════════
#  COLUMN COMPRESSION
# ═══════════════════════════════════════════════════════════════

_NOISE_COLS = {
    "created_by","updated_by","deleted_by","created_at","updated_at",
    "deleted_at","is_deleted","is_active","row_version","last_modified_by","modified_at",
}


def compress_schema(matched: dict, question: str, max_cols: int = 12) -> dict:
    model  = _get_embed_model()
    q_emb  = model.encode([question.lower()], normalize_embeddings=True).astype(np.float32)
    result = {}
    for tbl, cols in matched.items():
        clean = [c for c in cols if c.split(" ")[0].lower() not in _NOISE_COLS]
        if len(clean) <= max_cols:
            result[tbl] = clean
            continue
        col_names = [c.split(" ")[0].replace("_", " ") for c in clean]
        col_embs  = model.encode(col_names, normalize_embeddings=True).astype(np.float32)
        sims      = (col_embs @ q_emb.T).flatten()
        result[tbl] = [c for _, c in sorted(zip(sims, clean), reverse=True)][:max_cols]
    return result


def build_schema_str(matched: dict, db_type: str) -> str:
    label = "Collection" if db_type == "mongo" else "Table"
    field = "Fields"     if db_type == "mongo" else "Columns"
    return "\n".join(
        f"{label}: {name}\n  {field}: {', '.join(cols)}"
        for name, cols in matched.items()
    )


# ═══════════════════════════════════════════════════════════════
#  AGENT 1 — Schema Intelligence Agent (SIA)
# ═══════════════════════════════════════════════════════════════

SIA_PROMPT = (
    "You are a database schema analyst. Understand the schema and\n"
    "produce a structured business context JSON.\n\n"
    "Identify:\n"
    "1. Business domain (retail, banking, hotel, healthcare, logistics, etc.)\n"
    "2. For each table: its role, primary date column, primary value/amount column,\n"
    "   primary key column, and table type\n"
    "3. Core business metrics (how to compute them from columns)\n"
    "4. Common join relationships between tables\n\n"
    "Return ONLY this JSON (no markdown, no explanation):\n"
    "{\n"
    "  \"business_type\": \"<domain>\",\n"
    "  \"tables\": {\n"
    "    \"<table_name>\": {\n"
    "      \"role\": \"<what this table represents>\",\n"
    "      \"date_column\": \"<most relevant date/time column or null>\",\n"
    "      \"value_column\": \"<most relevant amount/revenue column or null>\",\n"
    "      \"id_column\": \"<primary key column>\",\n"
    "      \"type\": \"<transaction|master|lookup|log|mapping>\"\n"
    "    }\n"
    "  },\n"
    "  \"metrics\": {\n"
    "    \"revenue\":    \"<SQL expression>\",\n"
    "    \"quantity\":   \"<SQL expression or null>\",\n"
    "    \"count\":      \"<SQL expression>\",\n"
    "    \"date_trunc\": \"<SQL expression for monthly grouping>\"\n"
    "  },\n"
    "  \"joins\": [\n"
    "    { \"from\": \"<table.column>\", \"to\": \"<table.column>\", \"type\": \"INNER|LEFT\" }\n"
    "  ],\n"
    "  \"primary_transaction_table\": \"<most important table for queries>\"\n"
    "}"
)


def run_schema_agent(schema_dict: dict, db_flavor: str, client: Groq) -> dict:
    lines = [f"{t}: {', '.join(c.split(' ')[0] for c in cols)}"
             for t, cols in schema_dict.items()]
    user_msg = f"Database: {db_flavor}\n\nSchema:\n" + "\n".join(lines)
    try:
        res = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": SIA_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            # 1024 truncated the JSON on schemas of ~70 tables -> unparseable reply.
            max_completion_tokens=4096,
        )
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "business_type": "unknown", "tables": {}, "metrics": {},
            "joins": [], "primary_transaction_table": next(iter(schema_dict), ""),
            "_error": str(e),
        }


def infer_joins(schema_dict: dict) -> list:
    joins, table_names = [], list(schema_dict.keys())
    for tbl, cols in schema_dict.items():
        for c in cols:
            col = c.split(" ")[0].lower()
            if col.endswith("_id") and col != "id":
                prefix = col[:-3]
                for other in table_names:
                    if other == tbl:
                        continue
                    other_l = other.lower().replace("_", "")
                    if prefix in other_l or other_l in prefix:
                        other_cols = [x.split(" ")[0].lower() for x in schema_dict[other]]
                        target = "id" if "id" in other_cols else (col if col in other_cols else None)
                        if target:
                            entry = {"from": f"{tbl}.{col}", "to": f"{other}.{target}", "type": "LEFT"}
                            if entry not in joins:
                                joins.append(entry)
    return joins[:15]


def enrich_context_joins(context: dict, schema_dict: dict) -> dict:
    if not context.get("joins"):
        context["joins"] = infer_joins(schema_dict)
    return context


# ═══════════════════════════════════════════════════════════════
#  AGENT 2 — Query Generation Agent (QGA)
# ═══════════════════════════════════════════════════════════════

def make_sql_prompt(schema_str: str, flavor: str, context: dict, memory: dict) -> str:
    notes = {
        "MySQL":      "Use MySQL syntax. Backtick identifiers. LIMIT N.",
        "MariaDB":    "Use MariaDB/MySQL syntax. Backtick identifiers. LIMIT N.",
        "PostgreSQL": "Use PostgreSQL syntax. Double-quote identifiers if needed. LIMIT N.",
        "SQL Server": "Use T-SQL. SELECT TOP N instead of LIMIT. Square-bracket identifiers.",
        "SQLite":     "Use SQLite syntax. LIMIT N.",
    }.get(flavor, "Use standard SQL. LIMIT N.")

    ctx_parts = []
    biz = context.get("business_type", "")
    if biz and biz != "unknown":
        ctx_parts.append(f"Business Domain: {biz}")
    metrics = context.get("metrics", {})
    if metrics:
        ctx_parts.append("Business Metrics:\n" + "\n".join(
            f"  - {k} = {v}" for k, v in metrics.items() if v
        ))
    tbl_ctx = context.get("tables", {})
    if tbl_ctx:
        ctx_parts.append("Table Roles:\n" + "\n".join(
            f"  - {t}: date={m.get('date_column')}, value={m.get('value_column')}, role={m.get('role')}"
            for t, m in tbl_ctx.items()
        ))
    joins = context.get("joins", [])
    if joins:
        ctx_parts.append("Known Joins:\n" + "\n".join(
            f"  - {j['from']} -> {j['to']} ({j['type']} JOIN)" for j in joins
        ))
    mem_parts = []
    if memory:
        mem_parts.append("Learned Mappings (use exactly):\n" + "\n".join(
            f"  - '{k}' means -> {v}" for k, v in memory.items()
        ))

    return (
        f"You are an expert {flavor} SQL query generator.\n{notes}\n\n"
        f"Business Context:\n" + ("\n".join(ctx_parts) or "No context.") + "\n\n"
        + ("\n".join(mem_parts) + "\n\n" if mem_parts else "")
        + f"Schema:\n{schema_str}\n\n"
        "Rules:\n"
        "- Return ONLY the raw SQL SELECT — no markdown, no explanation.\n"
        "- Never use DROP, DELETE, TRUNCATE, ALTER, INSERT, UPDATE.\n"
        "- Always include LIMIT (default 50).\n"
        "- Use table aliases on JOINs.\n"
        "- Use date_column for time-based questions, value_column for amounts."
    )


def make_mongo_prompt(schema_str: str, valid_collections: list, context: dict) -> str:
    biz   = context.get("business_type", "")
    hints = "\n".join(
        f"  - {c}.{k.split('_')[0]} = '{context['tables'][c][k]}'"
        for c in valid_collections
        if c in context.get("tables", {})
        for k in ("date_column", "value_column")
        if context["tables"][c].get(k)
    )
    return (
        f"You are a MongoDB query expert.\n"
        + (f"Business Domain: {biz}\n" if biz else "")
        + (f"Field Hints:\n{hints}\n" if hints else "")
        + f"VALID COLLECTIONS (use only these exact names): {', '.join(valid_collections)}\n\n"
        + f"Schema:\n{schema_str}\n\n"
        'Return ONLY this JSON:\n'
        '{"collection":"<name>","filter":{},"projection":null,"sort":null,"limit":50}\n\n'
        'Rules:\n'
        '- filter: valid MongoDB filter dict. {} for no filter.\n'
        '- Text search: {"field":{"$regex":"term","$options":"i"}}\n'
        '- sort: null or {"field":1/-1}. limit: integer default 50.'
    )


def make_cassandra_prompt(schema_str: str, keyspace: str, context: dict) -> str:
    biz   = context.get("business_type", "")
    hints = "\n".join(
        f"  - {t}.{k.split('_')[0]} = '{m[k]}'"
        for t, m in context.get("tables", {}).items()
        for k in ("date_column", "value_column") if m.get(k)
    )
    return (
        f"You are a CQL expert. Keyspace: {keyspace}\n"
        + (f"Business Domain: {biz}\n" if biz else "")
        + (f"Field Hints:\n{hints}\n" if hints else "")
        + f"Schema:\n{schema_str}\n\n"
        "Rules:\n"
        "- Return ONLY raw CQL SELECT — no markdown, no explanation.\n"
        "- Use ALLOW FILTERING for non-partition-key filters.\n"
        "- Always add LIMIT (default 50).\n"
        "- No JOINs in Cassandra."
    )


def build_llm_messages(sys_prompt: str, question: str, history: list) -> list:
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return messages


# ═══════════════════════════════════════════════════════════════
#  AGENT 3 — Validator
# ═══════════════════════════════════════════════════════════════

_SENSITIVE_COLS = {"password","pwd","secret","token","api_key","private_key","hash","salt"}
_BLOCKED_KW = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|MERGE|EXEC|EXECUTE|"
    r"xp_\w+|sp_\w+|OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|DBCC|BACKUP|RESTORE|"
    r"SHUTDOWN|RECONFIGURE|WAITFOR|GRANT|REVOKE|DENY)\b", re.IGNORECASE
)
# SELECT ... INTO creates a table even though the statement starts with SELECT.
_SELECT_INTO = re.compile(r"\bINTO\s+[\[\"`#\w]", re.IGNORECASE)


class QueryValidationError(Exception):
    pass


def validate_sql(query: str, schema_dict: dict, flavor: str, question: str) -> str:
    q = query.strip()
    m = _BLOCKED_KW.search(q)
    if m:
        # The user's question wording never licenses a write (asking to "update me on sales"
        # must not let an UPDATE statement through).
        raise QueryValidationError(f"Blocked keyword '{m.group(0).upper()}' detected.")
    if not re.match(r"^\s*(SELECT|WITH)\b", q, re.IGNORECASE):
        raise QueryValidationError("Query must start with SELECT or WITH.")
    if _SELECT_INTO.search(q):
        raise QueryValidationError("SELECT ... INTO is not allowed (read-only).")
    schema_lower = {t.lower() for t in schema_dict}
    # CTE names (WITH a AS (...), b AS (...)) are legitimate FROM/JOIN targets too.
    schema_lower |= {m.lower() for m in re.findall(
        r"(?:\bWITH|,)\s+[\[`\"]?(\w+)[\]`\"]?\s*(?:\([^)]*\))?\s+AS\s*\(", q, re.IGNORECASE)}
    for match in re.finditer(r"\bFROM\s+`?(\w+)`?|\bJOIN\s+`?(\w+)`?", q, re.IGNORECASE):
        tbl = (match.group(1) or match.group(2) or "").lower()
        if tbl and tbl not in schema_lower:
            raise QueryValidationError(f"Table '{tbl}' not in schema.")
    q_lower = q.lower()
    for col in _SENSITIVE_COLS:
        if re.search(rf"\b{col}\b", q_lower):
            raise QueryValidationError(f"Access to sensitive column '{col}' is blocked.")
    if not re.search(r"\bLIMIT\b|\bTOP\b", q, re.IGNORECASE):
        if flavor == "SQL Server":
            q = re.sub(r"\bSELECT\b", "SELECT TOP 100", q, count=1, flags=re.IGNORECASE)
        else:
            q = q.rstrip(";") + "\nLIMIT 100"
    return q.rstrip(";").strip()


def validate_mongo(query_obj: dict, schema_dict: dict) -> dict:
    if not isinstance(query_obj, dict):
        raise QueryValidationError("MongoDB query must be a JSON object.")
    if "collection" not in query_obj:
        raise QueryValidationError("Missing 'collection' field.")
    if query_obj["collection"] not in schema_dict:
        raise QueryValidationError(f"Collection '{query_obj['collection']}' not in schema.")
    if not query_obj.get("limit"):
        query_obj["limit"] = 50
    if "$where" in (query_obj.get("filter") or {}):
        raise QueryValidationError("'$where' operator is blocked.")
    return query_obj


# ═══════════════════════════════════════════════════════════════
#  AUTO-FIX  (one LLM retry on execution error)
# ═══════════════════════════════════════════════════════════════

def attempt_fix(bad_query: str, error: str, schema_str: str, flavor: str, client: Groq) -> str:
    res = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        reasoning_effort="low",
        messages=[{"role": "user", "content": (
            f"You are an expert {flavor} SQL debugger.\n\n"
            f"Schema:\n{schema_str}\n\nBroken Query:\n{bad_query}\n\nError:\n{error}\n\n"
            "Fix the query. Return ONLY the corrected raw SQL."
        )}],
        temperature=0.0,
        max_completion_tokens=512,
    )
    fixed = res.choices[0].message.content.strip()
    fixed = re.sub(r"^```[a-z]*\n?", "", fixed)
    fixed = re.sub(r"\n?```$", "", fixed).strip()
    return fixed


# ═══════════════════════════════════════════════════════════════
#  MEMORY HELPERS
# ═══════════════════════════════════════════════════════════════

def update_memory(question: str, context: dict, memory: dict) -> dict:
    alias_map = {
        "revenue": "value_column", "sales":    "value_column",
        "income":  "value_column", "amount":   "value_column",
        "date":    "date_column",  "time":      "date_column",
        "month":   "date_column",  "quantity":  "qty_column",
        "qty":     "qty_column",   "stock":     "qty_column",
    }
    q_lower = question.lower()
    for word, meta_key in alias_map.items():
        if word in q_lower:
            for tbl, meta in context.get("tables", {}).items():
                col = meta.get(meta_key)
                if col:
                    memory[f"{word}_column"] = col
    return memory


# ═══════════════════════════════════════════════════════════════
#  SCHEMA FETCHING
# ═══════════════════════════════════════════════════════════════

def fetch_sql_schema(conn, flavor: str) -> dict:
    schema: dict[str, list] = {}
    if flavor == "SQLite":
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()
        for (tbl,) in tables:
            cols = conn.execute(text(f'PRAGMA table_info("{tbl}")')).fetchall()
            schema[tbl] = [f"{c[1]} ({c[2]})" for c in cols]
    elif flavor in ("MySQL", "MariaDB"):
        rows = conn.execute(text(
            "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA=DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )).fetchall()
        for tbl, col, dtype in rows:
            schema.setdefault(tbl, []).append(f"{col} ({dtype})")
    elif flavor == "PostgreSQL":
        rows = conn.execute(text(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' ORDER BY table_name, ordinal_position"
        )).fetchall()
        for tbl, col, dtype in rows:
            schema.setdefault(tbl, []).append(f"{col} ({dtype})")
    elif flavor == "SQL Server":
        rows = conn.execute(text(
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )).fetchall()
        for tbl, col, dtype in rows:
            schema.setdefault(tbl, []).append(f"{col} ({dtype})")
    return schema


def _quote_table(tbl: str, flavor: str) -> str:
    if flavor == "SQL Server":
        return f"[{tbl}]"
    if flavor in ("MySQL", "MariaDB"):
        return f"`{tbl}`"
    return f'"{tbl}"'  # PostgreSQL, SQLite, others


def filter_empty_tables(conn, schema_dict: dict, flavor: str) -> tuple[dict, list]:
    active = {}
    empty  = []
    for tbl in schema_dict:
        try:
            if flavor == "SQL Server":
                row = conn.execute(text(
                    "SELECT SUM(p.rows) FROM sys.partitions p "
                    "JOIN sys.tables t ON p.object_id = t.object_id "
                    "WHERE t.name = :tbl AND p.index_id IN (0,1)"
                ), {"tbl": tbl}).fetchone()
            else:
                quoted = _quote_table(tbl, flavor)
                row = conn.execute(text(f"SELECT COUNT(*) FROM {quoted}")).fetchone()
            count = int(row[0] or 0) if row else 0
            if count > 0:
                active[tbl] = schema_dict[tbl]
            else:
                empty.append(tbl)
        except Exception:
            active[tbl] = schema_dict[tbl]
    return active, empty


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_retriever(user_id: str) -> SchemaRetriever | None:
    if user_id in _retrievers:
        return _retrievers[user_id]
    schema = redis_cache.get_schema(user_id)
    if schema:
        r = SchemaRetriever(schema)
        _retrievers[user_id] = r
        return r
    return None


def _get_groq_client(session: dict) -> Groq:
    return Groq(api_key=session["groq_key"])


def _evt(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────────────────
#  CONNECT — SSE streaming with async generator
#
#  FastAPI SSE: StreamingResponse wraps an async generator.
#  Each blocking step (DB connect, LLM call) is awaited via
#  run_in_thread() so the event loop stays free for other requests.
#  Flask equivalent used stream_with_context() with sync generators.
# ─────────────────────────────────────────────────────────────

@app.post("/api/connect")
async def connect(req: ConnectRequest, user_id: str = Depends(get_current_user)):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield _evt({"phase": "init", "msg": "Initializing AI engine..."})
            resolved_key = req.groq_key or os.getenv("GROQ_API_KEY", "")
            if not resolved_key:
                yield _evt({"error": "No Groq API key — set GROQ_API_KEY env var or provide one."})
                return
            client = Groq(api_key=resolved_key)
            schema_dict: dict = {}
            db_flavor   = req.db_type

            yield _evt({"phase": "db", "msg": f"Connecting to {req.db_type}..."})

            # ── DB connection (I/O-bound → thread pool) ───────
            def _connect():
                nonlocal schema_dict, db_flavor
                if req.db_type == "SQLite":
                    engine = _make_engine(f"sqlite:///{req.db_path}")
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                        schema_dict = fetch_sql_schema(conn, "SQLite")
                    _engines[user_id] = engine

                elif req.db_type == "MongoDB":
                    if MongoClient is None:
                        raise RuntimeError("pymongo not installed.")
                    uri = (
                        f"mongodb://{req.username}:{urllib.parse.quote_plus(req.password)}"
                        f"@{req.host}:{req.port}/{req.database}?authSource=admin"
                        if req.username else f"mongodb://{req.host}:{req.port}/"
                    )
                    mc = MongoClient(uri, serverSelectionTimeoutMS=5000)
                    mc.server_info()
                    mongo_db = mc[req.database]
                    for coll in sorted(mongo_db.list_collection_names()):
                        sample = mongo_db[coll].find_one()
                        if sample:
                            sample.pop("_id", None)
                        schema_dict[coll] = list(sample.keys()) if sample else ["(empty)"]
                    _mongos[user_id] = mongo_db

                elif req.db_type == "Cassandra":
                    if Cluster is None:
                        raise RuntimeError("cassandra-driver not installed.")
                    auth = PlainTextAuthProvider(req.username, req.password) if req.username else None
                    cluster = Cluster([req.host], port=int(req.port or 9042), auth_provider=auth)
                    cass_session = cluster.connect(req.keyspace)
                    rows = cass_session.execute(
                        "SELECT table_name,column_name,type FROM system_schema.columns "
                        f"WHERE keyspace_name='{req.keyspace}'"
                    )
                    for row in rows:
                        schema_dict.setdefault(row.table_name, []).append(f"{row.column_name} ({row.type})")
                    _cassandras[user_id] = cass_session

                else:
                    default_ports = {"MySQL": 3306, "MariaDB": 3306, "PostgreSQL": 5432, "SQL Server": 1433}
                    p   = req.port or default_ports.get(req.db_type, 3306)
                    epw = urllib.parse.quote_plus(req.password)
                    edb = urllib.parse.quote_plus(req.database)
                    odbc_conn = urllib.parse.quote_plus(
                        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                        f"SERVER={req.host},{p};DATABASE={req.database};"
                        f"UID={req.username};PWD={req.password};"
                        f"Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
                    )
                    urls = {
                        "MySQL":      f"mysql+pymysql://{req.username}:{epw}@{req.host}:{p}/{edb}",
                        "MariaDB":    f"mysql+pymysql://{req.username}:{epw}@{req.host}:{p}/{edb}",
                        "PostgreSQL": f"postgresql+psycopg2://{req.username}:{epw}@{req.host}:{p}/{edb}",
                        "SQL Server": f"mssql+pyodbc:///?odbc_connect={odbc_conn}",
                    }
                    if req.db_type not in urls:
                        raise ValueError(f"Unknown db_type: {req.db_type}")
                    engine = _make_engine(urls[req.db_type])
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                        schema_dict = fetch_sql_schema(conn, req.db_type)
                    _engines[user_id] = engine

            await run_in_thread(_connect)

            total_count = len(schema_dict)
            yield _evt({"phase": "db_done", "msg": f"Connected — {total_count} tables found. Checking for data..."})

            # ── Filter empty tables (I/O-bound → thread pool) ─
            if req.db_type not in ("MongoDB", "Cassandra"):
                try:
                    def _filter():
                        with _engines[user_id].connect() as conn:
                            return filter_empty_tables(conn, schema_dict, db_flavor)
                    schema_dict, empty_tables = await run_in_thread(_filter)
                    yield _evt({"phase": "filter", "msg": f"{len(schema_dict)} tables with data, {len(empty_tables)} empty skipped"})
                except Exception:
                    empty_tables = []
            else:
                empty_tables = []

            reset_concept_cache()

            # ── Schema indexing (run in thread — CPU inside, but non-blocking) ──
            yield _evt({"phase": "index", "msg": f"Indexing {len(schema_dict)} tables..."})
            retriever = await run_in_thread(SchemaRetriever, schema_dict)
            _retrievers[user_id] = retriever

            # ── Knowledge Graph (I/O-bound → thread pool) ─────
            yield _evt({"phase": "graph", "msg": "Loading Knowledge Graph & RAG..."})
            ke = KnowledgeEngine(user_id=user_id)
            ke.set_groq_client(client)
            kg_stats = await run_in_thread(ke.learn_schema, schema_dict, db_flavor)
            _knowledge[user_id] = ke
            cached_tag = " (from cache)" if kg_stats.get("from_cache") else ""
            yield _evt({"phase": "graph_done", "msg": f"Graph ready — {kg_stats.get('graph_nodes', 0)} nodes{cached_tag}"})

            # ── AI Analysis (LLM call is I/O-bound → thread pool) ─
            db_name = req.database or req.db_path or req.keyspace or "default"
            cached_context = ke.load_sia_context(db_name)
            if cached_context:
                yield _evt({"phase": "ai", "msg": "Loading cached AI analysis (0 tokens)..."})
                context = enrich_context_joins(cached_context, schema_dict)
                biz = context.get("business_type", "unknown")
                yield _evt({"phase": "ai_done", "msg": f"Loaded: {biz} (from cache)"})
            else:
                yield _evt({"phase": "ai", "msg": "AI analysing database structure..."})
                top_for_sia = retriever.retrieve("business schema transactions orders products", top_k=20, db_type=req.db_type)
                context = await run_in_thread(run_schema_agent, top_for_sia, req.db_type, client)
                context = enrich_context_joins(context, schema_dict)
                ke.save_sia_context(context, db_name)
                biz = context.get("business_type", "unknown")
                yield _evt({"phase": "ai_done", "msg": f"Detected: {biz} database"})

            ke.learn_context(context)

            yield _evt({"phase": "session", "msg": "Setting up session..."})
            memory = storage.load_memory(user_id)
            session_data = {
                "db_type":   "sql" if req.db_type not in ("MongoDB", "Cassandra") else req.db_type.lower(),
                "db_flavor": db_flavor,
                "keyspace":  req.keyspace,
                "context":   context,
                "memory":    memory,
                "conn": {
                    "db_type": req.db_type, "host": req.host, "port": req.port,
                    "username": req.username,
                    "database": req.database, "db_path": req.db_path, "keyspace": req.keyspace,
                },
                "groq_key": resolved_key,
            }
            redis_cache.set_session(user_id, session_data)
            redis_cache.set_schema(user_id, schema_dict)
            redis_cache.clear_user_query_cache(user_id)

            yield _evt({
                "phase": "done", "success": True,
                "db_flavor": db_flavor,
                "tables": list(schema_dict.keys()),
                "schema": schema_dict,
                "business_type": biz,
                "knowledge_graph": ke.get_graph_stats(),
            })

        except Exception as e:
            yield _evt({"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
#  REFRESH SCHEMA
# ─────────────────────────────────────────────────────────────

@app.post("/api/refresh")
async def refresh_schema(user_id: str = Depends(get_current_user)):
    session = redis_cache.get_session(user_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]

    try:
        if db_type == "sql":
            engine = _engines.get(user_id)
            if not engine:
                raise HTTPException(status_code=400, detail="Engine not in memory — please reconnect.")
            def _fetch():
                with engine.connect() as conn:
                    return fetch_sql_schema(conn, db_flavor)
            schema_dict = await run_in_thread(_fetch)

        elif db_type == "mongo":
            mongo_db = _mongos.get(user_id)
            if not mongo_db:
                raise HTTPException(status_code=400, detail="Mongo client not in memory — please reconnect.")
            def _fetch_mongo():
                result = {}
                for coll in sorted(mongo_db.list_collection_names()):
                    sample = mongo_db[coll].find_one()
                    if sample:
                        sample.pop("_id", None)
                    result[coll] = list(sample.keys()) if sample else ["(empty)"]
                return result
            schema_dict = await run_in_thread(_fetch_mongo)
        else:
            raise HTTPException(status_code=400, detail="Schema refresh not supported for Cassandra.")

        retriever = await run_in_thread(SchemaRetriever, schema_dict)
        _retrievers[user_id] = retriever

        client  = _get_groq_client(session)
        top     = retriever.retrieve("business schema transactions orders", top_k=20, db_type=db_type)
        context = await run_in_thread(run_schema_agent, top, db_flavor, client)
        context = enrich_context_joins(context, schema_dict)

        session["context"] = context
        redis_cache.set_session(user_id, session)
        redis_cache.set_schema(user_id, schema_dict)
        cleared = redis_cache.clear_user_query_cache(user_id)

        return {
            "success":       True,
            "tables":        list(schema_dict.keys()),
            "business_type": context.get("business_type", "unknown"),
            "cache_cleared": cleared,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  QUERY — with pagination
#
#  Pagination: the DB runs the full query, but we slice rows before
#  returning them. Avoids sending millions of rows over HTTP.
#  The full result is stored in Redis cache so any page is fast.
#
#  BackgroundTasks: Supabase logging runs AFTER the response is sent.
#  The client gets the answer immediately without waiting for logging.
# ─────────────────────────────────────────────────────────────

@app.post("/api/query")
async def query(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")

    allowed, count, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({count} req). Retry in {ttl}s.",
            headers={"Retry-After": str(ttl)},
        )

    session = redis_cache.get_session(user_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    redis_cache.refresh_session(user_id)

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})
    client    = _get_groq_client(session)
    schema    = redis_cache.get_schema(user_id) or {}

    # ── Cache hit — apply pagination and return immediately ───
    cached = redis_cache.get_cached_query(user_id, db_flavor, req.question)
    if cached:
        all_rows  = cached.get("rows", [])
        total     = len(all_rows)
        start     = (req.page - 1) * req.page_size
        cached.update({
            "from_cache":  True,
            "latency_ms":  0,
            "tokens":      {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "rows":        all_rows[start:start + req.page_size],
            "page":        req.page,
            "page_size":   req.page_size,
            "total_rows":  total,
            "total_pages": -(-total // req.page_size),
        })
        background_tasks.add_task(
            storage.log_query,
            user_id=user_id, question=req.question,
            generated_query=cached.get("generated_sql", ""),
            db_flavor=db_flavor, tables_used=cached.get("tables_used", []),
            row_count=total, latency_ms=0, success=True, from_cache=True,
        )
        return cached

    history = redis_cache.get_turns(user_id, req.session_id, last_n=6)
    if not history:
        history = [
            {"role": h["role"], "content": h["content"]}
            for h in storage.get_history(user_id, req.session_id, limit=12)
        ]

    ke = _knowledge.get(user_id)
    kg_joins, similar_examples = [], ""
    if ke:
        try:
            kg_ctx = ke.get_context_for_question(req.question, db_type)
            kg_joins = kg_ctx.get("join_paths", [])
            sq = kg_ctx.get("similar_queries", [])
            if sq:
                similar_examples = "\n\nSimilar past queries:\n" + "\n".join(
                    f"  Q: {q.get('question','')} -> {q.get('sql','')}" for q in sq[:3]
                )
        except Exception:
            pass

    plan = build_query_plan(req.question, schema, kg_joins, flavor=db_flavor)

    if plan["tables"] and db_type == "sql":
        sys_prompt = plan_to_prompt(plan, db_flavor)
        if similar_examples:
            sys_prompt += similar_examples
        matched    = {t: schema[t] for t in plan["tables"] if t in schema}
        schema_str = build_schema_str(matched, db_type)
    else:
        retriever  = _get_retriever(user_id)
        matched    = retriever.retrieve(req.question, top_k=4, db_type=db_type) if retriever else {}
        matched    = compress_schema(matched, req.question, max_cols=10)
        schema_str = build_schema_str(matched, db_type)
        if db_type == "mongo":
            sys_prompt = make_mongo_prompt(schema_str, list(matched.keys()), context)
        elif db_type == "cassandra":
            sys_prompt = make_cassandra_prompt(schema_str, session.get("keyspace", ""), context)
        else:
            sys_prompt = make_sql_prompt(schema_str, db_flavor, context, memory)
            if similar_examples:
                sys_prompt += similar_examples

    messages = build_llm_messages(sys_prompt, req.question, history)

    t_start     = time.monotonic()
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # LLM call is I/O-bound — run in thread pool so event loop stays free
    def _llm_call():
        return client.chat.completions.create(
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
            messages=messages,
            temperature=0.1,
            max_completion_tokens=512,
        )

    try:
        res = await run_in_thread(_llm_call)
        if res.usage:
            token_usage = {
                "prompt_tokens":     res.usage.prompt_tokens     or 0,
                "completion_tokens": res.usage.completion_tokens or 0,
                "total_tokens":      res.usage.total_tokens      or 0,
            }
        generated_query = res.choices[0].message.content.strip()
        generated_query = re.sub(r"^```[a-z]*\n?", "", generated_query)
        generated_query = re.sub(r"\n?```$", "", generated_query).strip()
        generated_query = re.sub(r"^.*?(?=SELECT\b|WITH\b)", "", generated_query, flags=re.IGNORECASE | re.DOTALL)
        generated_query = re.sub(r"(?<=;).*$", "", generated_query, flags=re.DOTALL).strip()
        generated_query = generated_query.rstrip(";").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    try:
        if db_type == "sql":
            generated_query = validate_sql(generated_query, schema, db_flavor, req.question)
        elif db_type == "mongo":
            parsed          = json.loads(generated_query)
            parsed          = validate_mongo(parsed, schema)
            generated_query = json.dumps(parsed, indent=2)
    except QueryValidationError as ve:
        if db_type == "sql" and "must start with SELECT" in str(ve):
            try:
                def _retry_llm():
                    return client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        reasoning_effort="low",
                        messages=[{"role": "user", "content":
                            f"Generate ONLY a raw SQL SELECT query for: {req.question}\n"
                            f"Tables: {', '.join(matched.keys())}\n"
                            "Return NOTHING except the SQL."}],
                        temperature=0.0,
                        max_completion_tokens=512,
                    )
                retry_res = await run_in_thread(_retry_llm)
                generated_query = retry_res.choices[0].message.content.strip()
                generated_query = re.sub(r"^```[a-z]*\n?", "", generated_query)
                generated_query = re.sub(r"\n?```$", "", generated_query).strip()
                generated_query = re.sub(r"^.*?(?=SELECT\b|WITH\b)", "", generated_query, flags=re.IGNORECASE | re.DOTALL)
                generated_query = generated_query.rstrip(";").strip()
                generated_query = validate_sql(generated_query, schema, db_flavor, req.question)
            except Exception:
                raise HTTPException(status_code=400, detail=str(ve))
        else:
            raise HTTPException(status_code=400, detail=str(ve))
    except json.JSONDecodeError as je:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {je}")

    columns, rows, last_error, attempt = [], [], None, 0

    for attempt in range(2):
        try:
            if db_type == "sql":
                columns, rows = await exec_sql_async(_engines[user_id], generated_query)

            elif db_type == "mongo":
                q = json.loads(generated_query)
                forced_coll     = next(iter(matched.keys()))
                q["collection"] = forced_coll
                def _mongo_exec():
                    coll   = _mongos[user_id][forced_coll]
                    cursor = coll.find(q.get("filter") or {}, q.get("projection") or None)
                    if q.get("sort"):
                        cursor = cursor.sort(list(q["sort"].items()))
                    docs = list(cursor.limit(int(q.get("limit") or 50)))
                    for d in docs:
                        d.pop("_id", None)
                    cols = list(docs[0].keys()) if docs else []
                    rws  = [[serialize_value(d.get(c)) for c in cols] for d in docs]
                    return cols, rws
                columns, rows   = await run_in_thread(_mongo_exec)
                generated_query = json.dumps(q, indent=2)

            elif db_type == "cassandra":
                def _cql_exec():
                    result = _cassandras[user_id].execute(generated_query)
                    data   = list(result)
                    if data:
                        cols = list(data[0]._fields)
                        return cols, safe_rows(cols, data)
                    return [], []
                columns, rows = await asyncio.wait_for(
                    run_in_thread(_cql_exec), timeout=DB_QUERY_TIMEOUT
                )

            break

        except asyncio.TimeoutError:
            last_error = f"Query timed out after {DB_QUERY_TIMEOUT}s."
            break
        except Exception as e:
            last_error = str(e)
            if attempt == 0 and db_type == "sql":
                try:
                    generated_query = attempt_fix(generated_query, last_error, schema_str, db_flavor, client)
                    generated_query = validate_sql(generated_query, schema, db_flavor, req.question)
                except Exception:
                    break
            else:
                break

    latency_ms = int((time.monotonic() - t_start) * 1000)

    if last_error and not columns:
        background_tasks.add_task(
            storage.log_query,
            user_id=user_id, question=req.question, generated_query=generated_query,
            db_flavor=db_flavor, tables_used=list(matched.keys()),
            row_count=0, latency_ms=latency_ms, success=False, error=last_error,
        )
        raise HTTPException(status_code=400, detail=last_error)

    if ke and generated_query and rows:
        ke.learn_successful_query(req.question, generated_query)

    memory = update_memory(req.question, context, memory)
    session["memory"] = memory
    redis_cache.set_session(user_id, session)
    background_tasks.add_task(storage.save_memory, user_id, memory)

    redis_cache.push_turn(user_id, req.session_id, "user", req.question)
    redis_cache.push_turn(user_id, req.session_id, "assistant", f"Generated SQL: {generated_query[:200]}")
    background_tasks.add_task(storage.save_turn, user_id=user_id, session_id=req.session_id,
                               role="user", content=req.question)
    background_tasks.add_task(storage.save_turn, user_id=user_id, session_id=req.session_id,
                               role="assistant", content=f"Returned {len(rows)} rows.", query=generated_query)

    # ── Pagination ─────────────────────────────────────────────
    total      = len(rows)
    start      = (req.page - 1) * req.page_size
    paged_rows = rows[start:start + req.page_size]

    payload = {
        "success":       True,
        "generated_sql": generated_query,
        "columns":       columns,
        "rows":          paged_rows,
        "total_rows":    total,
        "page":          req.page,
        "page_size":     req.page_size,
        "total_pages":   -(-total // req.page_size),
        "business_type": context.get("business_type", ""),
        "tables_used":   list(matched.keys()),
        "retried":       attempt > 0,
        "from_cache":    False,
        "latency_ms":    latency_ms,
        "tokens":        token_usage,
    }

    # Cache the full (unpaginated) result so any page request is fast
    redis_cache.set_cached_query(user_id, db_flavor, req.question, {**payload, "rows": rows})
    background_tasks.add_task(
        storage.log_query,
        user_id=user_id, question=req.question, generated_query=generated_query,
        db_flavor=db_flavor, tables_used=list(matched.keys()),
        row_count=total, latency_ms=latency_ms, success=True, retried=attempt > 0,
    )

    return payload


# ─────────────────────────────────────────────────────────────
#  EXPLAIN — query execution plan
#
#  Use this to diagnose slow queries on large datasets:
#  - Sequential scans → missing index
#  - High cost estimate → consider partitioning or covering index
#  - Nested loops on large tables → rewrite as hash join
# ─────────────────────────────────────────────────────────────

@app.post("/api/query/explain")
async def explain_query(req: ExplainRequest, user_id: str = Depends(get_current_user)):
    session = redis_cache.get_session(user_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active session.")

    engine    = _engines.get(user_id)
    db_flavor = session.get("db_flavor", "")

    if not engine:
        raise HTTPException(status_code=400, detail="SQL engine not found — SQL databases only.")

    explain_map = {
        "PostgreSQL": f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {req.sql}",
        "MySQL":      f"EXPLAIN {req.sql}",
        "MariaDB":    f"EXPLAIN {req.sql}",
        "SQLite":     f"EXPLAIN QUERY PLAN {req.sql}",
        "SQL Server": f"SET SHOWPLAN_TEXT ON; {req.sql}",
    }
    explain_sql = explain_map.get(db_flavor, f"EXPLAIN {req.sql}")

    def _run_explain():
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            return [dict(row._mapping) for row in result]

    try:
        plan = await run_in_thread(_run_explain)
        return {"query": req.sql, "flavor": db_flavor, "plan": plan}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────
#  STREAM QUERY — SSE token-by-token
# ─────────────────────────────────────────────────────────────

@app.post("/api/query/stream")
async def query_stream(
    req: StreamQueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")

    allowed, count, ttl = redis_cache.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Retry in {ttl}s.")

    session = redis_cache.get_session(user_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active session.")

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})
    client    = _get_groq_client(session)
    schema    = redis_cache.get_schema(user_id) or {}

    retriever  = _get_retriever(user_id)
    matched    = retriever.retrieve(req.question, top_k=4, db_type=db_type) if retriever else {}
    matched    = compress_schema(matched, req.question, max_cols=10)
    schema_str = build_schema_str(matched, db_type)

    if db_type == "mongo":
        sys_prompt = make_mongo_prompt(schema_str, list(matched.keys()), context)
    elif db_type == "cassandra":
        sys_prompt = make_cassandra_prompt(schema_str, session.get("keyspace", ""), context)
    else:
        sys_prompt = make_sql_prompt(schema_str, db_flavor, context, memory)

    history  = redis_cache.get_turns(user_id, req.session_id, last_n=6)
    messages = build_llm_messages(sys_prompt, req.question, history)

    async def generate() -> AsyncGenerator[str, None]:
        generated_query = ""
        t_start = time.monotonic()
        try:
            def _stream():
                return client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    reasoning_effort="low",
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=512,
                    stream=True,
                )
            stream = await run_in_thread(_stream)
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    generated_query += token
                    yield _evt({"token": token})
        except Exception as e:
            yield _evt({"error": f"LLM stream error: {e}"})
            return

        generated_query = re.sub(r"^```[a-z]*\n?", "", generated_query.strip())
        generated_query = re.sub(r"\n?```$", "", generated_query).strip()

        try:
            if db_type == "sql":
                generated_query = validate_sql(generated_query, schema, db_flavor, req.question)
        except QueryValidationError as ve:
            yield _evt({"error": str(ve), "query": generated_query})
            return

        columns, rows, error = [], [], None
        try:
            if db_type == "sql":
                columns, rows = await exec_sql_async(_engines[user_id], generated_query)

            elif db_type == "mongo":
                q = json.loads(generated_query)
                forced_coll     = next(iter(matched.keys()))
                q["collection"] = forced_coll
                def _mongo_stream_exec():
                    coll   = _mongos[user_id][forced_coll]
                    cursor = coll.find(q.get("filter") or {}, q.get("projection") or None)
                    if q.get("sort"):
                        cursor = cursor.sort(list(q["sort"].items()))
                    docs = list(cursor.limit(int(q.get("limit") or 50)))
                    for d in docs:
                        d.pop("_id", None)
                    cols = list(docs[0].keys()) if docs else []
                    rws  = [[serialize_value(d.get(c)) for c in cols] for d in docs]
                    return cols, rws
                columns, rows = await run_in_thread(_mongo_stream_exec)

            elif db_type == "cassandra":
                def _cql_stream_exec():
                    result = _cassandras[user_id].execute(generated_query)
                    data   = list(result)
                    if data:
                        cols = list(data[0]._fields)
                        return cols, safe_rows(cols, data)
                    return [], []
                columns, rows = await asyncio.wait_for(
                    run_in_thread(_cql_stream_exec), timeout=DB_QUERY_TIMEOUT
                )

        except Exception as e:
            error = str(e)

        latency_ms = int((time.monotonic() - t_start) * 1000)

        if error:
            yield _evt({"error": error, "query": generated_query})
            background_tasks.add_task(
                storage.log_query,
                user_id=user_id, question=req.question, generated_query=generated_query,
                db_flavor=db_flavor, tables_used=list(matched.keys()),
                row_count=0, latency_ms=latency_ms, success=False, error=error,
            )
        else:
            payload = {
                "done": True, "query": generated_query,
                "columns": columns, "rows": rows,
                "latency_ms": latency_ms,
                "tables_used": list(matched.keys()),
            }
            yield _evt(payload)
            redis_cache.set_cached_query(user_id, db_flavor, req.question, {**payload, "from_cache": False})
            background_tasks.add_task(
                storage.log_query,
                user_id=user_id, question=req.question, generated_query=generated_query,
                db_flavor=db_flavor, tables_used=list(matched.keys()),
                row_count=len(rows), latency_ms=latency_ms, success=True,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
#  HISTORY
# ──────────────────────────────── ─────────────────────────────

@app.get("/api/history")
async def get_history(
    session_id: str = "default",
    limit: int = 20,
    user_id: str = Depends(get_current_user),
):
    history = storage.get_history(user_id, session_id, limit=min(limit, 100))
    return {"history": history, "count": len(history)}


@app.delete("/api/history")
async def clear_history(
    background_tasks: BackgroundTasks,
    session_id: str = "default",
    user_id: str = Depends(get_current_user),
):
    redis_cache.clear_conversation(user_id, session_id)
    # Also delete from Supabase so history doesn't reappear after Redis flush
    background_tasks.add_task(storage.delete_history, user_id, session_id)
    return {"success": True}


# ─────────────────────────────────────────────────────────────
#  CHART SUGGESTION
# ─────────────────────────────────────────────────────────────

@app.post("/api/chart")
async def suggest_chart(req: ChartRequest, user_id: str = Depends(get_current_user)):
    session = redis_cache.get_session(user_id)
    if not session:
        raise HTTPException(status_code=400, detail="Not connected.")
    if not req.rows:
        return {"chart": None}

    client = _get_groq_client(session)
    sample = [dict(zip(req.columns, row)) for row in req.rows[:8]]
    prompt = (
        "You are a data visualisation expert. Pick the single best chart type.\n\n"
        f"Columns: {req.columns}\nSample rows: {sample}\nTotal rows: {len(req.rows)}\n\n"
        "Rules: scatter=2 numeric cols; area=date+numeric cumulative; line=date+numeric;"
        " doughnut=2-6 categories proportion; pie=7+ categories; "
        "horizontalBar=top-N long labels; bar=category+numeric; none=single row/all text\n\n"
        'Return ONLY JSON: {"chart_type":"<type>","label_column":"<col>","value_column":"<col>",'
        '"x_column":"<col or empty>","title":"<title>"}'
    )
    try:
        def _chart_llm():
            return client.chat.completions.create(
                model="openai/gpt-oss-120b",
                reasoning_effort="low",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=256,
            )
        res = await run_in_thread(_chart_llm)
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        return {"chart": json.loads(raw)}
    except Exception as e:
        return {"chart": None, "error": str(e)}


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return {
        "auth0_domain":    os.getenv("AUTH0_DOMAIN", ""),
        "auth0_client_id": os.getenv("AUTH0_CLIENT_ID", ""),
        "auth0_audience":  os.getenv("AUTH0_AUDIENCE", ""),
        "dev_mode":        os.getenv("DEV_MODE", "false").lower() == "true",
    }


# ─────────────────────────────────────────────────────────────
#  STATUS
# ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status(user_id: str = Depends(get_current_user)):
    session = redis_cache.get_session(user_id)
    schema  = redis_cache.get_schema(user_id) or {}
    return {
        "connected":     session is not None,
        "db_type":       session.get("db_type")   if session else None,
        "db_flavor":     session.get("db_flavor") if session else None,
        "business_type": session.get("context", {}).get("business_type") if session else None,
        "tables":        list(schema.keys()),
        "memory_keys":   list((session or {}).get("memory", {}).keys()),
        "redis_ok":      redis_cache.ping(),
    }


# ─────────────────────────────────────────────────────────────
#  FEEDBACK
# ─────────────────────────────────────────────────────────────

@app.post("/api/feedback")
async def feedback(req: FeedbackRequest, user_id: str = Depends(get_current_user)):
    ke = _knowledge.get(user_id)
    if ke:
        ke.learn_user_feedback(req.question, req.sql, req.is_correct, req.correction)
    return {"success": True, "learned": True}


# ─────────────────────────────────────────────────────────────
#  KNOWLEDGE GRAPH STATS
# ─────────────────────────────────────────────────────────────

@app.get("/api/knowledge")
async def knowledge_stats(user_id: str = Depends(get_current_user)):
    ke = _knowledge.get(user_id)
    if not ke:
        raise HTTPException(status_code=400, detail="No knowledge graph. Connect first.")
    return ke.get_graph_stats()


# ─────────────────────────────────────────────────────────────
#  DISCONNECT
# ─────────────────────────────────────────────────────────────

@app.post("/api/disconnect")
async def disconnect(user_id: str = Depends(get_current_user)):
    redis_cache.delete_session(user_id)
    redis_cache.clear_user_query_cache(user_id)
    _engines.pop(user_id, None)
    _mongos.pop(user_id, None)
    _cassandras.pop(user_id, None)
    _retrievers.pop(user_id, None)
    _knowledge.pop(user_id, None)
    _docstores.pop(user_id, None)
    return {"success": True}


# ─────────────────────────────────────────────────────────────
#  DOCSTORE HELPERS
# ─────────────────────────────────────────────────────────────

def _get_docstore(user_id: str) -> DocStore:
    if user_id not in _docstores:
        _docstores[user_id] = DocStore(user_id)
    return _docstores[user_id]


_SYNTHESIS_PROMPT = """\
You are an analytical assistant. You have been given:
1. A structured data result from a database query.
2. Relevant excerpts from documents (policies, manuals, reports, etc.).

Synthesize both into a clear, concise answer. Where numbers from the database
can be compared against rules or thresholds from the documents, do so explicitly.
Cite the SQL result and the document source. If no documents are available,
answer from the database result alone.

Keep the answer factual and auditable — show your reasoning.\
"""


def _summarise_rows(columns: list, rows: list, limit: int = 10) -> str:
    if not rows:
        return "No rows returned."
    header = " | ".join(columns)
    lines  = [header, "-" * len(header)]
    for row in rows[:limit]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more rows")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  UPLOAD DOCUMENT
# ─────────────────────────────────────────────────────────────

@app.post("/api/docs/upload")
async def upload_doc(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    data   = await file.read()
    store  = _get_docstore(user_id)
    chunks = await run_in_thread(store.add_file, file.filename, data)
    return {"success": True, "filename": file.filename, "chunks_indexed": chunks}


@app.post("/api/docs/ingest_url")
async def ingest_url(req: HybridRequest, user_id: str = Depends(get_current_user)):
    if not req.url:
        raise HTTPException(status_code=400, detail="url field is required.")
    store  = _get_docstore(user_id)
    chunks = await run_in_thread(store.add_url, req.url)
    return {"success": True, "url": req.url, "chunks_indexed": chunks}


@app.get("/api/docs")
async def list_docs(user_id: str = Depends(get_current_user)):
    store = _get_docstore(user_id)
    return {"sources": await run_in_thread(store.list_sources), "total_chunks": await run_in_thread(store.count)}


# ─────────────────────────────────────────────────────────────
#  HYBRID QUERY  (SQL branch + document RAG + synthesis)
# ─────────────────────────────────────────────────────────────

@app.post("/api/hybrid")
async def hybrid_query(
    req: HybridRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    if not req.question:
        raise HTTPException(status_code=400, detail="Empty question.")

    session = redis_cache.get_session(user_id)
    if not session:
        raise HTTPException(status_code=400, detail="No active session. Please connect first.")

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})
    client    = _get_groq_client(session)
    schema    = redis_cache.get_schema(user_id) or {}

    store = _get_docstore(user_id)

    # Ingest on-the-fly URL if provided
    if req.url:
        await run_in_thread(store.add_url, req.url)

    # ── SQL branch ────────────────────────────────────────────
    sql_result: dict = {}
    try:
        ke = _knowledge.get(user_id)
        kg_joins = []
        if ke:
            try:
                kg_ctx   = ke.get_context_for_question(req.question, db_type)
                kg_joins = kg_ctx.get("join_paths", [])
            except Exception:
                pass

        plan = build_query_plan(req.question, schema, kg_joins, flavor=db_flavor)

        if plan["tables"] and db_type == "sql":
            sys_prompt = plan_to_prompt(plan, db_flavor)
            matched    = {t: schema[t] for t in plan["tables"] if t in schema}
            schema_str = build_schema_str(matched, db_type)
        else:
            retriever  = _get_retriever(user_id)
            matched    = retriever.retrieve(req.question, top_k=4, db_type=db_type) if retriever else {}
            matched    = compress_schema(matched, req.question, max_cols=10)
            schema_str = build_schema_str(matched, db_type)
            if db_type == "mongo":
                sys_prompt = make_mongo_prompt(schema_str, list(matched.keys()), context)
            elif db_type == "cassandra":
                sys_prompt = make_cassandra_prompt(schema_str, session.get("keyspace", ""), context)
            else:
                sys_prompt = make_sql_prompt(schema_str, db_flavor, context, memory)

        messages = build_llm_messages(sys_prompt, req.question, [])

        def _sql_llm():
            return client.chat.completions.create(
                model="openai/gpt-oss-120b",
                reasoning_effort="low",
                messages=messages,
                temperature=0.1,
                max_completion_tokens=512,
            )

        res = await run_in_thread(_sql_llm)
        generated_query = res.choices[0].message.content.strip()
        generated_query = re.sub(r"^```[a-z]*\n?", "", generated_query)
        generated_query = re.sub(r"\n?```$", "", generated_query).strip()
        generated_query = re.sub(r"^.*?(?=SELECT\b|WITH\b)", "", generated_query, flags=re.IGNORECASE | re.DOTALL)
        generated_query = generated_query.rstrip(";").strip()

        if db_type == "sql":
            generated_query = validate_sql(generated_query, schema, db_flavor, req.question)
            columns, rows   = await exec_sql_async(_engines[user_id], generated_query)
        else:
            columns, rows, generated_query = [], [], ""

        sql_result = {
            "generated_sql": generated_query,
            "columns":       columns,
            "rows":          rows[:50],
            "row_count":     len(rows),
        }
    except Exception as e:
        sql_result = {"error": str(e), "generated_sql": "", "columns": [], "rows": []}

    # ── Document RAG branch ───────────────────────────────────
    doc_hits = await run_in_thread(store.search, req.question, 4)

    # Determine route label
    has_sql  = bool(sql_result.get("columns"))
    has_docs = bool(doc_hits)
    route    = "hybrid" if (has_sql and has_docs) else ("sql_only" if has_sql else ("doc_only" if has_docs else "no_data"))

    # ── Synthesis ─────────────────────────────────────────────
    sql_text = (
        f"SQL: {sql_result.get('generated_sql', '')}\n\nResult:\n"
        + _summarise_rows(sql_result.get("columns", []), sql_result.get("rows", []))
        if has_sql else "No database result available."
    )
    doc_text = (
        "\n\n---\n".join(
            f"[Source: {h['source']}]\n{h['text']}" for h in doc_hits
        ) if has_docs else "No document context available."
    )
    synthesis_user = (
        f"Question: {req.question}\n\n"
        f"=== Database Result ===\n{sql_text}\n\n"
        f"=== Document Context ===\n{doc_text}"
    )

    def _synth_llm():
        return client.chat.completions.create(
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": _SYNTHESIS_PROMPT},
                {"role": "user",   "content": synthesis_user},
            ],
            temperature=0.2,
            max_completion_tokens=1024,
        )

    synth_res = await run_in_thread(_synth_llm)
    answer    = synth_res.choices[0].message.content.strip()

    background_tasks.add_task(
        storage.log_query,
        user_id=user_id, question=req.question,
        generated_query=sql_result.get("generated_sql", ""),
        db_flavor=db_flavor, tables_used=list(matched.keys() if "matched" in dir() else []),
        row_count=sql_result.get("row_count", 0), latency_ms=0, success=True,
    )

    return {
        "route":       route,
        "answer":      answer,
        "sql_result":  sql_result,
        "doc_context": doc_hits,
    }


if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    uvicorn.run("sql_agent.app:app", host="0.0.0.0", port=5000, reload=False)
