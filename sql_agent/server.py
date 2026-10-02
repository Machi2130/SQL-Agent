"""
mcp_server.py — MCP (Model Context Protocol) server for the AI SQL Agent.

Exposes the query engine as tools that any MCP-compatible AI agent
(Claude Desktop, Cursor, etc.) can call directly — no UI needed.

Run:
    python mcp_server.py

Add to Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "sql-agent": {
          "command": "python",
          "args": ["C:/path/to/sql/mcp_server.py"],
          "env": {
            "MCP_USER_ID": "claude-desktop",
            "GROQ_API_KEY": "your-key-here"
          }
        }
      }
    }

Auth note:
    MCP runs as a local stdio process — there is no HTTP layer,
    so JWT is not needed. We use MCP_USER_ID from env as the identity.
    For a remote/hosted MCP server you would add OAuth.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import time
import urllib.parse
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from sqlalchemy import text

# MCP hosts (Claude Code, Claude Desktop) launch this from an arbitrary cwd, while the
# engine uses project-relative paths (.chroma_data, audit.log, .env): work from the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from . import cache as redis_cache
from . import storage
from .docstore import DocStore

# Shared engine lives in app.py — no need to duplicate it
from .app import (
    SchemaRetriever,
    _make_engine,
    attempt_fix,
    build_llm_messages,
    build_schema_str,
    compress_schema,
    enrich_context_joins,
    fetch_sql_schema,
    filter_empty_tables,
    infer_joins,
    make_sql_prompt,
    run_schema_agent,
    safe_rows,
    update_memory,
    validate_sql,
    QueryValidationError,
)
from .knowledge import KnowledgeEngine
from .query_planner import build_query_plan, plan_to_prompt, reset_concept_cache

load_dotenv()

__version__ = "1.0.0"

# ── MCP identity — no JWT needed for local stdio ──────────────
MCP_USER_ID  = os.getenv("MCP_USER_ID", "mcp-agent")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Hosted-mode auth: MCP_AUTH_TOKENS="alice:<token>,bob:<token>" gives each person
# their own token (named in the audit log); MCP_AUTH_TOKEN is a single shared one.
_TOKEN_USERS: dict[str, str] = {
    t.strip(): n.strip()
    for n, _, t in (p.partition(":") for p in os.getenv("MCP_AUTH_TOKENS", "").split(",") if ":" in p)
}
if os.getenv("MCP_AUTH_TOKEN"):
    _TOKEN_USERS.setdefault(os.getenv("MCP_AUTH_TOKEN", ""), "default")

# Process-level caches keyed by CONNECTION key (session + host + port + database),
# so several databases stay open side by side and switching never reconnects.
# session_id defaults to MCP_USER_ID; pass an explicit one per call for
# multi-tenant/parallel agents. _current maps session -> its active connection.
_engines:    dict[str, Any] = {}
_mongos:     dict[str, Any] = {}
_cassandras: dict[str, Any] = {}
_retrievers: dict[str, Any] = {}
_knowledge:  dict[str, Any] = {}
_docstores:  dict[str, Any] = {}   # keyed by session (documents are per user, not per DB)
_fks:        dict[str, list] = {}  # real foreign keys per connection, as join dicts
_scope:      dict[str, dict] = {}  # tenant filter per connection: {column, value, label, tables}
_current:    dict[str, str] = {}

# In-process usage counters — keyed by session_id
_usage: dict[str, dict[str, int]] = {}


def _get_docstore(sid: str) -> DocStore:
    if sid not in _docstores:
        _docstores[sid] = DocStore(sid)
    return _docstores[sid]

DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT_SECONDS", 30))
MAX_ROWS     = int(os.getenv("MCP_MAX_ROWS", 5000))   # server-side cap on rows fetched per query
ALLOW_WRITES = os.getenv("MCP_ALLOW_WRITES", "").lower() in ("1", "true", "yes")
ALLOW_DDL    = os.getenv("MCP_ALLOW_DDL", "").lower() in ("1", "true", "yes")
AUDIT_LOG    = os.getenv("MCP_AUDIT_LOG") or os.path.join(PROJECT_ROOT, "audit.log")

server = Server("sql-agent")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

class ErrCode:
    NOT_CONNECTED = "NOT_CONNECTED"
    QUERY_FAILED  = "QUERY_FAILED"
    AUTH_FAILED   = "AUTH_FAILED"
    UNSUPPORTED_DB = "UNSUPPORTED_DB"
    INVALID_SQL   = "INVALID_SQL"
    TIMEOUT       = "TIMEOUT"
    UNKNOWN       = "UNKNOWN"


def _ok(data: Any) -> list[types.TextContent]:
    payload = {"ok": True, "data": data}
    return [types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]


def _err(msg: str, code: str = ErrCode.UNKNOWN) -> list[types.TextContent]:
    payload = {"ok": False, "error": {"code": code, "message": msg}}
    return [types.TextContent(type="text", text=json.dumps(payload, default=str))]


def _resolve_sid(args: dict) -> str:
    """Return the session_id for this call. Explicit session_id wins; over HTTP
    each client gets its own (from the transport's Mcp-Session-Id header) so
    colleagues don't share a 'current database'; stdio defaults to MCP_USER_ID."""
    if args.get("session_id"):
        return args["session_id"]
    try:
        request = server.request_context.request
    except LookupError:
        request = None
    headers = getattr(request, "headers", None)
    http_session = headers.get("mcp-session-id") if headers is not None else None
    return f"http-{http_session}" if http_session else MCP_USER_ID


def _conn_key(session_id: str, host: str, port: str, database: str) -> str:
    """Stable key for one open database connection (also used as a folder name)."""
    import hashlib
    raw = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{host}_{port}_{database}")
    if len(raw) > 60:  # e.g. a long SQLite file path — keep folder names within Windows path limits
        raw = raw[:30] + "_" + hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{session_id}__{raw}"


def _active(args: dict) -> str:
    """The connection key the session is currently working on."""
    sid = _resolve_sid(args)
    return _current.get(sid, sid)


def _get_session(sid: str = MCP_USER_ID) -> dict | None:
    return redis_cache.get_session(sid)


def _require_session(sid: str = MCP_USER_ID) -> dict:
    s = _get_session(sid)
    if not s:
        raise ValueError("Not connected. Call connect_database first.")
    return s


def _groq(session: dict | None = None) -> Groq:
    key = (session or {}).get("groq_key") or GROQ_API_KEY
    if not key:
        raise ValueError("No Groq API key. Use run_sql to execute SQL you wrote yourself, or pass groq_key to connect_database.")
    return Groq(api_key=key)


def _track_usage(sid: str, event: str, count: int = 1) -> None:
    bucket = _usage.setdefault(sid, {"queries": 0, "errors": 0, "charts": 0, "connects": 0})
    bucket[event] = bucket.get(event, 0) + count


def _request_user() -> str:
    """Who is calling: the name behind the bearer token over HTTP, else MCP_USER_ID."""
    try:
        request = server.request_context.request
    except LookupError:
        request = None
    headers = getattr(request, "headers", None)
    if headers is not None:
        token = headers.get("authorization", "").removeprefix("Bearer ").strip()
        return _TOKEN_USERS.get(token, "unknown-token")
    return MCP_USER_ID


def _audit(tool: str, sid: str, sql: str, ok: bool, rows: int | None = None,
           ms: int | None = None, error: str | None = None, extra: dict | None = None) -> None:
    """Append one JSON line per query/write: who, which connection, what, outcome."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": _request_user(), "connection": sid,
        "tool": tool, "sql": (sql or "")[:2000], "ok": ok, "rows": rows, "ms": ms,
        "error": (error or "")[:500] or None, **(extra or {}),
    }
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        print(f"[sql-agent] AUDIT WRITE FAILED ({e}): {json.dumps(entry, default=str)}", file=sys.stderr)


def _fetch_foreign_keys(conn, db_type: str, schema_dict: dict) -> list[dict]:
    """Real FK constraints as join dicts {from: 'a.col', to: 'b.col', source: 'fk'}.
    One metadata query per connect; bare table names to match schema_dict."""
    rows: list = []
    try:
        if db_type == "SQLite":
            for t in schema_dict:
                for r in conn.execute(text(f'PRAGMA foreign_key_list("{t}")')).fetchall():
                    rows.append((t, r[3], r[2], r[4]))
        elif db_type == "SQL Server":
            rows = conn.execute(text("""
                SELECT CASE WHEN sp.name = 'dbo' THEN tp.name ELSE sp.name + '.' + tp.name END,
                       cp.name,
                       CASE WHEN sr.name = 'dbo' THEN tr.name ELSE sr.name + '.' + tr.name END,
                       cr.name
                FROM sys.foreign_key_columns fkc
                JOIN sys.tables  tp ON tp.object_id = fkc.parent_object_id
                JOIN sys.schemas sp ON sp.schema_id = tp.schema_id
                JOIN sys.columns cp ON cp.object_id = tp.object_id AND cp.column_id = fkc.parent_column_id
                JOIN sys.tables  tr ON tr.object_id = fkc.referenced_object_id
                JOIN sys.schemas sr ON sr.schema_id = tr.schema_id
                JOIN sys.columns cr ON cr.object_id = tr.object_id AND cr.column_id = fkc.referenced_column_id
            """)).fetchall()
        elif db_type in ("MySQL", "MariaDB"):
            rows = conn.execute(text("""
                SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL
            """)).fetchall()
        elif db_type == "PostgreSQL":
            rows = conn.execute(text("""
                SELECT kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = current_schema()
            """)).fetchall()
    except Exception as e:
        print(f"[sql-agent] foreign key extraction failed ({db_type}): {e}", file=sys.stderr)
        return []
    known = {t.lower(): t for t in schema_dict}
    fks: list[dict] = []
    for ft, fc, tt, tc in rows:
        a, b = known.get(str(ft).lower()), known.get(str(tt).lower())
        if a and b:
            fks.append({"from": f"{a}.{fc}", "to": f"{b}.{tc}", "type": "INNER", "source": "fk"})
    return fks


# ── Glossary: human notes per database ("Account_Ref is the card number") ──

def _kb_dir(sid: str) -> str:
    """This connection's private knowledge folder (per user and database)."""
    kb = (_get_session(sid) or {}).get("kb") or ("db__" + sid.split("__", 1)[1])
    return os.path.join(".chroma_data", kb)


def _notes_path(sid: str) -> str:
    return os.path.join(_kb_dir(sid), "notes.json")


def _load_notes(sid: str) -> list[dict]:
    try:
        with open(_notes_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_notes(sid: str, notes: list[dict]) -> None:
    path = _notes_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=1)


# ── Table popularity: which tables this database's users actually query ──

_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([\[\]\w.`\"]+)", re.IGNORECASE)


def _tables_in_sql(sql: str, schema: dict) -> list[str]:
    """Schema table names referenced by a statement (handles [a].[b], db.schema.table, backticks)."""
    lookup = {t.lower(): t for t in schema}
    found: list[str] = []
    for raw in _TABLE_REF_RE.findall(sql):
        name = raw.replace("].[", ".").replace('"."', ".").strip('[]`"')
        parts = name.split(".")
        for cand in (name, parts[-1], ".".join(parts[-2:])):
            c = cand.strip('[]`"').lower()
            if c in lookup and lookup[c] not in found:
                found.append(lookup[c])
                break
    return found


def _usage_path(sid: str) -> str:
    return os.path.join(_kb_dir(sid), "usage.json")


def _load_usage(sid: str) -> dict:
    try:
        with open(_usage_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tables": {}, "statements": 0}


def _bump_usage(sid: str, tables: list[str], count: int = 1) -> None:
    if not tables:
        return
    usage = _load_usage(sid)
    for t in tables:
        usage["tables"][t] = usage["tables"].get(t, 0) + count
    usage["statements"] = usage.get("statements", 0) + 1
    path = _usage_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(usage, f)


def _fetch_query_history(conn, db_type: str, limit: int) -> list[tuple[str, int]]:
    """(statement, execution_count) from the engine's own statistics - what the app
    and the team actually run. Needs VIEW SERVER STATE / pg_stat_statements etc."""
    if db_type == "SQL Server":
        q = f"""SELECT TOP ({int(limit)}) st.text, qs.execution_count
                FROM sys.dm_exec_query_stats qs
                CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
                WHERE st.text LIKE '%FROM%' ORDER BY qs.execution_count DESC"""
    elif db_type in ("MySQL", "MariaDB"):
        q = f"""SELECT DIGEST_TEXT, COUNT_STAR FROM performance_schema.events_statements_summary_by_digest
                WHERE SCHEMA_NAME = DATABASE() AND DIGEST_TEXT LIKE 'SELECT%'
                ORDER BY COUNT_STAR DESC LIMIT {int(limit)}"""
    elif db_type == "PostgreSQL":
        q = f"""SELECT query, calls FROM pg_stat_statements
                WHERE query ILIKE 'select%' ORDER BY calls DESC LIMIT {int(limit)}"""
    else:
        return []
    return [(str(r[0]), int(r[1] or 0)) for r in conn.execute(text(q)).fetchall()]


# ── Tenant scope: one database, many clients (e.g. Tenant_Id) ──

def _scope_candidates(schema: dict) -> list[dict]:
    """Columns shared by most tables are how a multi-client database tells clients apart."""
    counts: dict[str, int] = {}
    for cols in schema.values():
        for c in {c.split(" ")[0] for c in cols}:
            counts[c] = counts.get(c, 0) + 1
    threshold = max(3, int(0.3 * len(schema)))
    hints = ("program", "client", "tenant", "company", "org", "merchant", "bank", "store")
    out = []
    for col, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        low = col.lower()
        if n >= threshold and (low.endswith("id") or any(h in low for h in hints)) and low != "id":
            stem = re.sub(r"_?id$", "", low)
            lookups = [t for t in schema if stem and stem in t.lower() and ("master" in t.lower() or t.lower().startswith(stem))]
            out.append({"column": col, "in_tables": n, "lookup_tables": lookups[:3]})
    return out[:5]


def _scope_violation(sid: str, sql: str, schema: dict) -> str | None:
    """A query on a scoped table must mention the tenant column; returns the message if not."""
    sc = _scope.get(sid)
    if not sc:
        return None
    touched = [t for t in _tables_in_sql(sql, schema) if t in sc["tables"]]
    if touched and not re.search(rf"\b{re.escape(sc['column'])}\b", sql, re.IGNORECASE):
        return (f"This database holds several clients; {', '.join(touched)} must be filtered by "
                f"{sc['column']} = {sc['value']!r}" + (f" ({sc['label']})" if sc.get("label") else "") +
                ". Add it to the WHERE clause (or call set_scope with an empty value to query across clients).")
    return None


# ═══════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
#  list_tools() tells the agent what it can call.
#  inputSchema is standard JSON Schema — agents use it to
#  understand what arguments to pass.
# ═══════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="connect_database",
            description=(
                "Connect to a database, or switch to one that is already open. Must be called before any query. "
                "Supports SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra. "
                "Several databases can be open at once: calling this for a database that is already "
                "connected just makes it the current one (instant, no reconnect); the others stay open. "
                "Pass session_id to run multiple independent agent sessions in parallel."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "db_type":    {"type": "string", "description": "One of: SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra"},
                    "groq_key":   {"type": "string", "description": "Optional. Groq API key for query_database (server-side NL→SQL). Omit when the calling AI writes SQL itself and uses run_sql.", "default": ""},
                    "host":       {"type": "string", "default": "localhost"},
                    "port":       {"type": "string", "default": ""},
                    "username":   {"type": "string", "default": ""},
                    "password":   {"type": "string", "default": ""},
                    "database":   {"type": "string", "default": ""},
                    "keyspace":   {"type": "string", "description": "For Cassandra: keyspace name", "default": ""},
                    "db_path":    {"type": "string", "description": "For SQLite: path to .db file", "default": ""},
                    "session_id": {"type": "string", "description": "Optional agent session token. Omit to use the default MCP_USER_ID session."},
                },
                "required": ["db_type"],
            },
        ),
        types.Tool(
            name="run_sql",
            description=(
                "Execute a read-only SQL query you wrote yourself against the connected database. "
                "No LLM or Groq key involved: call get_query_context(question) to get the relevant tables, joins "
                "and past queries, write SELECT/WITH SQL in the database's dialect, then call this. Pass the "
                "original question too so the SQL is remembered for next time. Use bare table names exactly as "
                "returned (no schema prefix such as dbo.). Writes (INSERT/UPDATE/DELETE/DDL) are rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string", "description": "A single SELECT or WITH statement"},
                    "question":   {"type": "string", "default": "", "description": "The user's question this SQL answers (remembered as a question->SQL example)"},
                    "page":       {"type": "integer", "default": 1,  "description": "Page number (1-based)"},
                    "page_size":  {"type": "integer", "default": 50, "description": "Rows per page (max 500)"},
                    "session_id": {"type": "string", "description": "Session token from connect_database. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="query_database",
            description=(
                "Ask a natural language question about the connected database. "
                "Returns columns, rows, the generated SQL, and usage metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string", "description": "Natural language question, e.g. 'top 10 customers by revenue'"},
                    "page":       {"type": "integer", "default": 1,  "description": "Page number (1-based)"},
                    "page_size":  {"type": "integer", "default": 50, "description": "Rows per page (max 500)"},
                    "session_id": {"type": "string", "description": "Session token from connect_database. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="execute_write",
            description=(
                "Developer mode only: run ONE data-changing statement (INSERT, UPDATE, DELETE, MERGE; DDL only if "
                "the server allows it) on the current connection. Requires the connection to have been opened with "
                "allow_writes=true and the server to allow writes; the user confirms every statement in a form. "
                "UPDATE/DELETE must have a WHERE clause. Prefer dry_run=true first: it runs inside a transaction, "
                "reports the affected row count and rolls back. Never use this for SELECT (use run_sql)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string", "description": "A single INSERT/UPDATE/DELETE/MERGE (or DDL) statement"},
                    "dry_run":    {"type": "boolean", "default": False, "description": "Execute inside a transaction and roll back, reporting rows affected"},
                    "reason":     {"type": "string", "default": "", "description": "Why this change is being made (recorded in the audit log)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="get_query_context",
            description=(
                "Start here before writing SQL. Uses the database's knowledge graph (built on connect, no LLM) to "
                "return ONLY what a question needs: the relevant tables with their columns, the join paths between "
                "them, business notes/metrics for those tables, and similar past question->SQL pairs. Far cheaper "
                "than get_schema, which lists every table."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string", "description": "The user's question in plain language"},
                    "max_tables": {"type": "integer", "default": 6, "description": "Max tables to return (1-12)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="get_schema",
            description=(
                "Return tables and their columns for the connected database. Without 'tables' it returns EVERY "
                "table (large); prefer get_query_context(question) and use this only to look up specific tables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tables":     {"type": "array", "items": {"type": "string"}, "description": "Only these tables (exact names)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="explain_query",
            description=(
                "Return the database execution plan for a SQL query. "
                "Use this to identify slow queries, missing indexes, or sequential scans."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql":        {"type": "string", "description": "The SQL SELECT to explain"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="suggest_chart",
            description="Given columns and rows from a query result, suggest the best chart type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {"type": "array",  "items": {"type": "string"}},
                    "rows":    {"type": "array",  "items": {"type": "array"}},
                },
                "required": ["columns", "rows"],
            },
        ),
        types.Tool(
            name="set_scope",
            description=(
                "For a database shared by several clients/programs: fix the tenant filter for this connection, "
                "e.g. column='Tenant_Id', value='7', label='Acme Retail'. Afterwards any query on a table that has "
                "that column MUST filter on it or it is rejected, so clients' data is never mixed. connect_database "
                "and get_query_context report 'scope_candidates' (columns shared by most tables); list the values "
                "with run_sql on a lookup table (e.g. TENANT_MASTER) and ask the user which one. Empty value clears."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "column":     {"type": "string", "description": "Tenant column, e.g. Tenant_Id"},
                    "value":      {"type": "string", "description": "The client's value; empty string clears the scope"},
                    "label":      {"type": "string", "default": "", "description": "Human name for the value, e.g. 'Acme Retail'"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["column", "value"],
            },
        ),
        types.Tool(
            name="import_query_history",
            description=(
                "Learn which tables matter from the database's own execution statistics (SQL Server plan cache, "
                "MySQL performance_schema, Postgres pg_stat_statements): counts how often each table is queried by "
                "the application and the team. Improves table ranking and 'hot_tables' in get_query_context. "
                "Needs VIEW SERVER STATE (or equivalent); reports if unavailable."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit":      {"type": "integer", "default": 300, "description": "Top-N statements to read (max 2000)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="import_queries",
            description=(
                "Import existing SQL the team already uses (paste a .sql file or several statements separated by ';'). "
                "Each SELECT becomes a remembered question->SQL example (a leading '-- comment' is used as the "
                "question) and counts toward table popularity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql_text":   {"type": "string", "description": "One or more SQL statements"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["sql_text"],
            },
        ),
        types.Tool(
            name="add_note",
            description=(
                "Teach the current database something a human knows and the schema doesn't, e.g. "
                "'Account_Ref is the card number' on ACCOUNT_LINK, or a database-wide rule like "
                "'amounts are in MUR'. Notes are stored once per database, shown to anyone whose question touches "
                "that table, and used to pick tables (a question mentioning 'card' will pull that table in)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "note":       {"type": "string", "description": "The fact, in one or two sentences"},
                    "table":      {"type": "string", "default": "", "description": "Table the note is about (omit for a database-wide note)"},
                    "column":     {"type": "string", "default": "", "description": "Column the note is about (requires table)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["note"],
            },
        ),
        types.Tool(
            name="list_notes",
            description="List the glossary notes for the current database (optionally only one table's, plus database-wide notes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "table":      {"type": "string", "default": ""},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="delete_note",
            description="Delete a glossary note by id (from list_notes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id":         {"type": "string"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["id"],
            },
        ),
        types.Tool(
            name="disconnect",
            description="Clear the current database session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="describe_capabilities",
            description=(
                "Return what databases are currently connected, available tools, "
                "supported DB types, and error codes. Call this first to bootstrap "
                "an agent session without human configuration."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="upload_document",
            description=(
                "Index an unstructured document (PDF, DOCX, TXT, MD, CSV) into the document store. "
                "Call this before search_documents. The document is chunked and stored per session. "
                "Use for policies, manuals, contracts, handbooks — any text that contains rules, "
                "thresholds, or context that cannot be queried with SQL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename":   {"type": "string", "description": "Original filename including extension, e.g. 'aml_policy.pdf'"},
                    "content_b64": {"type": "string", "description": "Base64-encoded file bytes"},
                    "url":        {"type": "string", "description": "Alternatively, a URL to fetch and index (PDF or web page). Provide either content_b64 or url, not both."},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
        types.Tool(
            name="search_documents",
            description=(
                "Search the indexed document store for content relevant to a question. "
                "Returns the top matching text chunks with their source filename. "
                "Call this to retrieve policy rules, thresholds, or guidelines before "
                "or after querying the database — then combine both results in your answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question":   {"type": "string", "description": "The question or topic to search for"},
                    "n_results":  {"type": "integer", "default": 4, "description": "Number of chunks to return (max 10)"},
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="list_documents",
            description="List all documents that have been uploaded to the document store for this session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session token. Omit to use default."},
                },
            },
        ),
    ]


# ═══════════════════════════════════════════════════════════════
#  TOOL DISPATCH
#  call_tool() routes each tool name to its implementation.
# ═══════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "connect_database":
            return await _connect(arguments)
        if name == "query_database":
            return await _query(arguments)
        if name == "run_sql":
            return await _run_sql(arguments)
        if name == "execute_write":
            return await _execute_write(arguments)
        if name == "get_query_context":
            return await _query_context(arguments)
        if name == "get_schema":
            return await _schema(arguments)
        if name == "add_note":
            return await _add_note(arguments)
        if name == "list_notes":
            return await _list_notes(arguments)
        if name == "delete_note":
            return await _delete_note(arguments)
        if name == "set_scope":
            return await _set_scope(arguments)
        if name == "import_query_history":
            return await _import_query_history(arguments)
        if name == "import_queries":
            return await _import_queries(arguments)
        if name == "explain_query":
            return await _explain(arguments)
        if name == "suggest_chart":
            return await _chart(arguments)
        if name == "disconnect":
            return await _disconnect(arguments)
        if name == "describe_capabilities":
            return await _describe_capabilities(arguments)
        if name == "upload_document":
            return await _upload_document(arguments)
        if name == "search_documents":
            return await _search_documents(arguments)
        if name == "list_documents":
            return await _list_documents(arguments)
        return _err(f"Unknown tool: {name}", ErrCode.UNKNOWN)
    except ValueError as e:
        return _err(str(e), ErrCode.NOT_CONNECTED)
    except Exception as e:
        return _err(f"Unexpected error: {e}", ErrCode.UNKNOWN)


# ═══════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

_CONN_FIELDS = {
    "SQLite":    (["db_path"], ["db_path"]),
    "MongoDB":   (["host", "port", "username", "password", "database"], ["host", "database"]),
    "Cassandra": (["host", "port", "username", "password", "keyspace"], ["host", "keyspace"]),
}
_SQL_CONN_FIELDS = (["host", "port", "username", "password", "database"],
                    ["host", "username", "password", "database"])
_DEFAULT_PORTS = {"MySQL": "3306", "MariaDB": "3306", "PostgreSQL": "5432",
                  "SQL Server": "1433", "MongoDB": "27017", "Cassandra": "9042"}


def _env_db_password(database: str) -> str:
    """Password from the environment: MCP_DB_PASSWORD_<DATABASE> (several DBs on
    different servers), falling back to the generic MCP_DB_PASSWORD."""
    key = re.sub(r"[^A-Z0-9]", "_", (database or "").upper())
    return (os.getenv(f"MCP_DB_PASSWORD_{key}") if key else "") or os.getenv("MCP_DB_PASSWORD", "")


def _missing_connection_fields(db_type: str, args: dict) -> tuple[list[str], list[str]]:
    """Return (all fields for this db_type, the required ones the caller didn't supply)."""
    fields, required = _CONN_FIELDS.get(db_type, _SQL_CONN_FIELDS)
    missing = [f for f in required if not args.get(f)]
    if "password" in missing and _env_db_password(args.get("database", "")):
        missing.remove("password")
    return fields, missing


async def _ask_connection_fields(db_type: str, fields: list[str], required: list[str],
                                 args: dict) -> dict | None:
    """Ask the user for connection details via MCP elicitation (a form in the
    client). The answers travel client -> server only; the model never sees them.
    Returns the answers, or None if the user declined/cancelled."""
    try:
        session = server.request_context.session
    except LookupError:  # called outside an MCP request (e.g. a script)
        raise ValueError(f"Missing connection fields: {', '.join(required)}")
    props: dict[str, dict] = {}
    for f in fields:
        props[f] = {"type": "string", "title": f.replace("_", " ").title()}
        default = args.get(f) or {"host": "localhost", "port": _DEFAULT_PORTS.get(db_type)}.get(f)
        if default:
            props[f]["default"] = str(default)
    try:
        result = await session.elicit(
            message=f"Connect to {db_type} — enter the connection details.",
            requestedSchema={"type": "object", "properties": props, "required": required},
        )
    except Exception as e:  # client cannot show forms (e.g. Claude Desktop, older clients)
        raise ValueError(f"Missing connection fields: {', '.join(required)}. "
                         f"Pass them as arguments (this client cannot show a form: {str(e)[:80]}).")
    if result.action != "accept" or not result.content:
        return None
    return {k: v for k, v in result.content.items() if v not in (None, "")}


async def _connect(args: dict) -> list[types.TextContent]:
    sid      = _resolve_sid(args)
    db_type  = args["db_type"]
    fields, missing = _missing_connection_fields(db_type, args)
    if missing:
        answers = await _ask_connection_fields(db_type, fields, missing, args)
        if answers is None:
            return _ok({"connected": False, "cancelled": True,
                        "message": "Connection cancelled by user."})
        args = {**args, **answers}
    groq_key = args.get("groq_key") or GROQ_API_KEY   # optional: only needed for query_database
    host     = args.get("host", "localhost")
    port     = args.get("port", "")
    username = args.get("username", "")
    # Fallback lets the calling agent connect without the secret ever appearing in chat.
    password = args.get("password") or _env_db_password(args.get("database", ""))
    database = args.get("database", "")
    db_path  = args.get("db_path", "")

    # One key per open database; everything below is stored under it, and the
    # session just points at it. Reconnecting to an open database = switch.
    session_id = sid
    sid = _conn_key(session_id, host, port, database or db_path or args.get("keyspace", ""))
    if (sid in _engines or sid in _mongos or sid in _cassandras) and _get_session(sid):
        _current[session_id] = sid
        schema = redis_cache.get_schema(sid) or {}
        return _ok({
            "connected":   True,
            "switched":    True,
            "session_id":  session_id,
            "connection":  sid,
            "db_flavor":   db_type,
            "table_count": len(schema),
            "message":     "Already open - switched to it without reconnecting.",
        })

    schema_dict: dict = {}
    db_flavor = db_type
    loop = asyncio.get_event_loop()

    # ── Connect and fetch schema (I/O-bound → thread) ──────────
    def _do_connect():
        if db_type == "SQLite":
            engine = _make_engine(f"sqlite:///{db_path}")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                sd = fetch_sql_schema(conn, "SQLite")
                _fks[sid] = _fetch_foreign_keys(conn, "SQLite", sd)
            _engines[sid] = engine
            return sd

        if db_type == "MongoDB":
            try:
                from pymongo import MongoClient as MC
            except ImportError:
                raise ValueError("pymongo not installed. Run: pip install pymongo")
            uri = (
                f"mongodb://{username}:{urllib.parse.quote_plus(password)}"
                f"@{host}:{port or 27017}/{database}?authSource=admin"
                if username else f"mongodb://{host}:{port or 27017}/"
            )
            mc       = MC(uri, serverSelectionTimeoutMS=5000)
            mc.server_info()
            mongo_db = mc[database]
            sd = {}
            for coll in sorted(mongo_db.list_collection_names()):
                sample = mongo_db[coll].find_one()
                if sample:
                    sample.pop("_id", None)
                sd[coll] = list(sample.keys()) if sample else ["(empty)"]
            _mongos[sid] = mongo_db
            return sd

        if db_type == "Cassandra":
            try:
                from cassandra.cluster import Cluster as C
                from cassandra.auth import PlainTextAuthProvider as PA
            except ImportError:
                raise ValueError("cassandra-driver not installed. Run: pip install cassandra-driver")
            auth         = PA(username, password) if username else None
            cluster      = C([host], port=int(port or 9042), auth_provider=auth)
            cass_session = cluster.connect(args.get("keyspace", ""))
            rows = cass_session.execute(
                "SELECT table_name,column_name,type FROM system_schema.columns "
                f"WHERE keyspace_name='{args.get('keyspace', '')}'"
            )
            sd = {}
            for row in rows:
                sd.setdefault(row.table_name, []).append(f"{row.column_name} ({row.type})")
            _cassandras[sid] = cass_session
            return sd

        default_ports = {"MySQL": 3306, "MariaDB": 3306, "PostgreSQL": 5432, "SQL Server": 1433}
        p   = port or default_ports.get(db_type, 3306)
        epw = urllib.parse.quote_plus(password)
        edb = urllib.parse.quote_plus(database)
        urls = {
            "MySQL":      f"mysql+pymysql://{username}:{epw}@{host}:{p}/{edb}",
            "MariaDB":    f"mysql+pymysql://{username}:{epw}@{host}:{p}/{edb}",
            "PostgreSQL": f"postgresql+psycopg2://{username}:{epw}@{host}:{p}/{edb}",
            "SQL Server": f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(
                f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host},{p};'
                f'DATABASE={database};UID={username};PWD={password};'
                f'Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'
            )}",
        }
        if db_type not in urls:
            raise ValueError(f"Unsupported db_type: {db_type}. Choose: SQLite, MySQL, MariaDB, PostgreSQL, SQL Server, MongoDB, Cassandra")
        engine = _make_engine(urls[db_type])
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            sd = fetch_sql_schema(conn, db_type)
            _fks[sid] = _fetch_foreign_keys(conn, db_type, sd)
        _engines[sid] = engine
        return sd

    schema_dict = await loop.run_in_executor(None, _do_connect)

    # ── Filter empty tables (SQL only) ────────────────────────
    if db_type not in ("MongoDB", "Cassandra"):
        try:
            def _filter():
                with _engines[sid].connect() as conn:
                    return filter_empty_tables(conn, schema_dict, db_flavor)
            schema_dict, _ = await loop.run_in_executor(None, _filter)
        except Exception:
            pass

    reset_concept_cache()

    # ── Build schema index ─────────────────────────────────────
    retriever = await loop.run_in_executor(None, SchemaRetriever, schema_dict)
    _retrievers[sid] = retriever

    # ── AI schema analysis (only when a Groq key is available) ─
    # Without a key the calling AI (e.g. Claude Code) reads get_schema and writes SQL for run_sql.
    # Knowledge graph + schema embeddings are built locally (no LLM) per USER and
    # DATABASE: nothing learned, noted or remembered is visible to anyone else, even
    # on a team instance (identity = token name over HTTP, MCP_USER_ID locally).
    # The Groq business summary is optional.
    owner = re.sub(r"[^A-Za-z0-9_.-]", "_", _request_user())
    kb = f"{owner}__" + sid.split("__", 1)[1]
    ke = KnowledgeEngine(user_id=kb)
    await loop.run_in_executor(None, ke.learn_schema, schema_dict, db_flavor)
    _knowledge[sid] = ke

    context: dict = {}
    db_name = database or db_path or "default"
    cached_ctx = ke.load_sia_context(db_name)
    if cached_ctx:
        context = enrich_context_joins(cached_ctx, schema_dict)
    elif groq_key:
        client = Groq(api_key=groq_key)
        ke.set_groq_client(client)
        top     = retriever.retrieve("business schema transactions orders", top_k=20)
        context = await loop.run_in_executor(None, run_schema_agent, top, db_type, client)
        context = enrich_context_joins(context, schema_dict)
        ke.save_sia_context(context, db_name)
    else:
        context = enrich_context_joins({}, schema_dict)   # heuristic joins only, no LLM
    ke.learn_context(context)

    # ── Save session ───────────────────────────────────────────
    memory = storage.load_memory(sid)
    redis_cache.set_session(sid, {
        "db_type":      "sql",
        "db_flavor":    db_flavor,
        "context":      context,
        "memory":       memory,
        "groq_key":     groq_key,
        "allow_writes": bool(args.get("allow_writes")) and ALLOW_WRITES,
        "kb":           kb,
    })
    redis_cache.set_schema(sid, schema_dict)
    _track_usage(sid, "connects")
    _current[session_id] = sid

    return _ok({
        "connected":     True,
        "session_id":    session_id,
        "connection":    sid,
        "db_flavor":     db_flavor,
        "table_count":   len(schema_dict),
        "tables":        list(schema_dict.keys()),
        "business_type": context.get("business_type", "unknown"),
        "foreign_keys":  len(_fks.get(sid, [])),
        "scope_candidates": _scope_candidates(schema_dict),
        "hint": ("If scope_candidates is not empty this database holds several clients: list the values "
                 "(e.g. run_sql on the lookup table), ask the user which client, then call set_scope."),
    })


async def _query(args: dict) -> list[types.TextContent]:
    sid       = _active(args)
    question  = args["question"]
    page      = max(1, int(args.get("page", 1)))
    page_size = min(500, max(1, int(args.get("page_size", 50))))

    session = _require_session(sid)
    schema  = redis_cache.get_schema(sid) or {}
    client  = _groq(session)
    loop    = asyncio.get_event_loop()

    db_type   = session["db_type"]
    db_flavor = session["db_flavor"]
    context   = session.get("context", {})
    memory    = session.get("memory", {})

    # ── Check cache ────────────────────────────────────────────
    cached = redis_cache.get_cached_query(sid, db_flavor, question)
    if cached:
        all_rows = cached.get("rows", [])
        total    = len(all_rows)
        start    = (page - 1) * page_size
        return _ok({
            "generated_sql": cached.get("generated_sql"),
            "columns":       cached.get("columns"),
            "rows":          all_rows[start:start + page_size],
            "total_rows":    total,
            "page":          page,
            "page_size":     page_size,
            "total_pages":   -(-total // page_size),
            "from_cache":    True,
            "retried":       False,
        })

    # ── Build prompt ───────────────────────────────────────────
    ke        = _knowledge.get(sid)
    kg_joins  = []
    if ke:
        try:
            kg_ctx   = ke.get_context_for_question(question, db_type)
            kg_joins = kg_ctx.get("join_paths", [])
        except Exception:
            pass

    plan = build_query_plan(question, schema, kg_joins, flavor=db_flavor)
    if plan["tables"] and db_type == "sql":
        sys_prompt = plan_to_prompt(plan, db_flavor)
        matched    = {t: schema[t] for t in plan["tables"] if t in schema}
        schema_str = build_schema_str(matched, db_type)
    else:
        retriever  = _retrievers.get(sid)
        matched    = retriever.retrieve(question, top_k=4) if retriever else {}
        matched    = compress_schema(matched, question, max_cols=10)
        schema_str = build_schema_str(matched, db_type)
        sys_prompt = make_sql_prompt(schema_str, db_flavor, context, memory)

    messages = build_llm_messages(sys_prompt, question, [])

    # ── LLM call (I/O-bound → thread) ─────────────────────────
    def _llm():
        return client.chat.completions.create(
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
            messages=messages,
            temperature=0.1,
            max_completion_tokens=512,
        )

    res   = await loop.run_in_executor(None, _llm)
    query = res.choices[0].message.content.strip()
    query = re.sub(r"^```[a-z]*\n?", "", query)
    query = re.sub(r"\n?```$",        "", query).strip()
    query = re.sub(r"^.*?(?=SELECT\b|WITH\b)", "", query, flags=re.IGNORECASE | re.DOTALL)
    query = query.rstrip(";").strip()

    try:
        query = validate_sql(query, schema, db_flavor, question)
    except QueryValidationError as ve:
        return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
    violation = _scope_violation(sid, query, schema)
    if violation:
        return _err(violation, ErrCode.INVALID_SQL)

    # ── Execute (I/O-bound → thread) ──────────────────────────
    columns, rows = [], []
    for attempt in range(2):
        try:
            def _exec():
                with _engines[sid].connect() as conn:
                    result = conn.execute(text(query))
                    cols   = list(result.keys())
                    return cols, safe_rows(cols, result.fetchmany(MAX_ROWS))  # cap: never pull a whole table
            columns, rows = await asyncio.wait_for(
                loop.run_in_executor(None, _exec),
                timeout=DB_QUERY_TIMEOUT,
            )
            break
        except asyncio.TimeoutError:
            return _err(f"Query timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
        except Exception as e:
            if attempt == 0:
                try:
                    query = attempt_fix(query, str(e), schema_str, db_flavor, client)
                    query = validate_sql(query, schema, db_flavor, question)
                except Exception:
                    return _err(str(e), ErrCode.QUERY_FAILED)
            else:
                return _err(str(e), ErrCode.QUERY_FAILED)

    # ── Learn + persist ────────────────────────────────────────
    if ke and rows:
        ke.learn_successful_query(question, query)
    memory = update_memory(question, context, memory)
    session["memory"] = memory
    redis_cache.set_session(sid, session)
    _track_usage(sid, "queries")
    _audit("query_database", sid, query, ok=True, rows=len(rows), extra={"question": question[:500]})
    if attempt > 0:
        _track_usage(sid, "retries")

    total      = len(rows)
    start      = (page - 1) * page_size
    paged_rows = rows[start:start + page_size]

    result = {
        "generated_sql": query,
        "columns":       columns,
        "rows":          paged_rows,
        "total_rows":    total,
        "page":          page,
        "page_size":     page_size,
        "total_pages":   -(-total // page_size),
        "from_cache":    False,
        "retried":       attempt > 0,
        "usage":         dict(_usage.get(sid, {})),
    }
    redis_cache.set_cached_query(sid, db_flavor, question, {**result, "rows": rows})
    return _ok(result)


async def _run_sql(args: dict) -> list[types.TextContent]:
    """Execute caller-written read-only SQL. No LLM involved."""
    sid       = _active(args)
    sql       = str(args.get("sql", "")).strip().rstrip(";")
    page      = max(1, int(args.get("page", 1)))
    page_size = min(500, max(1, int(args.get("page_size", 50))))

    session   = _require_session(sid)
    if session.get("db_type") != "sql" or sid not in _engines:
        return _err("run_sql supports SQL databases only.", ErrCode.UNSUPPORTED_DB)
    schema    = redis_cache.get_schema(sid) or {}
    db_flavor = session["db_flavor"]

    if ";" in sql:
        return _err("One statement per call (no ';').", ErrCode.INVALID_SQL)
    try:
        sql = validate_sql(sql, schema, db_flavor, question="")
    except QueryValidationError as ve:
        _audit("run_sql", sid, sql, ok=False, error=str(ve))
        return _err(f"Query validation failed: {ve}", ErrCode.INVALID_SQL)
    violation = _scope_violation(sid, sql, schema)
    if violation:
        _audit("run_sql", sid, sql, ok=False, error="scope filter missing")
        return _err(violation, ErrCode.INVALID_SQL)

    loop = asyncio.get_event_loop()

    def _exec():
        with _engines[sid].connect() as conn:
            result = conn.execute(text(sql))
            cols   = list(result.keys())
            # +1 so we can tell "exactly MAX_ROWS" from "more than MAX_ROWS"; never pulls a whole table.
            return cols, safe_rows(cols, result.fetchmany(MAX_ROWS + 1))

    t0 = time.monotonic()
    try:
        columns, rows = await asyncio.wait_for(
            loop.run_in_executor(None, _exec), timeout=DB_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        _audit("run_sql", sid, sql, ok=False, error=f"timeout after {DB_QUERY_TIMEOUT}s")
        return _err(f"Query timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except Exception as e:
        _track_usage(sid, "errors")
        _audit("run_sql", sid, sql, ok=False, error=str(e))
        return _err(str(e), ErrCode.QUERY_FAILED)

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    _track_usage(sid, "queries")
    _audit("run_sql", sid, sql, ok=True, rows=len(rows), ms=int((time.monotonic() - t0) * 1000))
    question = str(args.get("question", "")).strip()
    ke = _knowledge.get(sid)
    if question and rows and ke:
        ke.learn_successful_query(question, sql)   # feeds get_query_context's similar_queries
    _bump_usage(sid, _tables_in_sql(sql, schema))
    total = len(rows)
    start = (page - 1) * page_size
    return _ok({
        "sql":         sql,
        "columns":     columns,
        "rows":        rows[start:start + page_size],
        "total_rows":  total,
        "truncated":   truncated,
        "max_rows":    MAX_ROWS,
        "page":        page,
        "page_size":   page_size,
        "total_pages": -(-total // page_size),
    })


_WRITE_RE = re.compile(r"^\s*(INSERT|UPDATE|DELETE|MERGE)\b", re.IGNORECASE)
_DDL_RE   = re.compile(r"^\s*(CREATE|ALTER|DROP|TRUNCATE)\b", re.IGNORECASE)
_FORBIDDEN_WRITE_RE = re.compile(
    r"\b(xp_\w+|sp_\w+|OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|SHUTDOWN|RECONFIGURE|BACKUP|RESTORE|DBCC|"
    r"GRANT|REVOKE|DENY|EXEC|EXECUTE)\b", re.IGNORECASE)


async def _confirm_write(sid: str, sql: str, dry_run: bool) -> bool:
    """A human confirms every write through an elicitation form; the model cannot
    approve its own statement. No form available (or declined) -> not confirmed."""
    try:
        session = server.request_context.session
        result = await session.elicit(
            message=(("DRY RUN (rolled back) on " if dry_run else "EXECUTE WRITE on ") + sid
                     + ":\n\n" + sql[:1500] + "\n\nType YES to confirm."),
            requestedSchema={"type": "object",
                             "properties": {"confirm": {"type": "string", "title": "Type YES to run this statement"}},
                             "required": ["confirm"]},
        )
    except Exception as e:
        print(f"[sql-agent] write confirmation unavailable: {e}", file=sys.stderr)
        return False
    return result.action == "accept" and str((result.content or {}).get("confirm", "")).strip().upper() == "YES"


async def _execute_write(args: dict) -> list[types.TextContent]:
    """Developer mode: one INSERT/UPDATE/DELETE (DDL if allowed), confirmed by the user, audited."""
    sid     = _active(args)
    sql     = str(args.get("sql", "")).strip().rstrip(";")
    dry_run = bool(args.get("dry_run", False))
    reason  = str(args.get("reason", ""))[:500]
    session = _require_session(sid)

    if not ALLOW_WRITES:
        return _err("Writes are disabled on this server (set MCP_ALLOW_WRITES=true to enable).", ErrCode.UNSUPPORTED_DB)
    if not session.get("allow_writes"):
        return _err("This connection is read-only. Reconnect with allow_writes=true (developer mode) "
                    "using your own write-capable credentials.", ErrCode.UNSUPPORTED_DB)
    if session.get("db_type") != "sql" or sid not in _engines:
        return _err("execute_write supports SQL databases only.", ErrCode.UNSUPPORTED_DB)
    if ";" in sql:
        return _err("One statement per call (no ';').", ErrCode.INVALID_SQL)
    is_ddl = bool(_DDL_RE.match(sql))
    if not (_WRITE_RE.match(sql) or is_ddl):
        return _err("execute_write accepts INSERT, UPDATE, DELETE, MERGE"
                    + (", CREATE, ALTER, DROP, TRUNCATE" if ALLOW_DDL else "")
                    + " only. Use run_sql for SELECT.", ErrCode.INVALID_SQL)
    if is_ddl and not ALLOW_DDL:
        return _err("DDL is disabled on this server (set MCP_ALLOW_DDL=true to enable).", ErrCode.INVALID_SQL)
    if _FORBIDDEN_WRITE_RE.search(sql):
        return _err("Statement contains a forbidden keyword.", ErrCode.INVALID_SQL)
    if re.match(r"^\s*(UPDATE|DELETE)\b", sql, re.IGNORECASE) and not re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        return _err("UPDATE/DELETE without a WHERE clause is not allowed.", ErrCode.INVALID_SQL)

    if not await _confirm_write(sid, sql, dry_run):
        _audit("execute_write", sid, sql, ok=False, error="not confirmed by user",
               extra={"dry_run": dry_run, "reason": reason})
        return _ok({"executed": False, "cancelled": True, "message": "Write not confirmed by user."})

    loop = asyncio.get_event_loop()

    def _exec() -> int:
        with _engines[sid].connect() as conn:
            trans = conn.begin()
            try:
                affected = conn.execute(text(sql)).rowcount
            except Exception:
                trans.rollback()
                raise
            if dry_run:
                trans.rollback()
            else:
                trans.commit()
            return affected

    t0 = time.monotonic()
    try:
        affected = await asyncio.wait_for(loop.run_in_executor(None, _exec), timeout=DB_QUERY_TIMEOUT)
    except asyncio.TimeoutError:
        _audit("execute_write", sid, sql, ok=False, error=f"timeout after {DB_QUERY_TIMEOUT}s",
               extra={"dry_run": dry_run, "reason": reason})
        return _err(f"Statement timed out after {DB_QUERY_TIMEOUT}s.", ErrCode.TIMEOUT)
    except Exception as e:
        _track_usage(sid, "errors")
        _audit("execute_write", sid, sql, ok=False, error=str(e), extra={"dry_run": dry_run, "reason": reason})
        return _err(str(e), ErrCode.QUERY_FAILED)

    _audit("execute_write", sid, sql, ok=True, rows=affected, ms=int((time.monotonic() - t0) * 1000),
           extra={"dry_run": dry_run, "reason": reason})
    if not dry_run:
        redis_cache.clear_user_query_cache(sid)   # cached SELECT results may now be stale
    return _ok({"executed": not dry_run, "dry_run": dry_run, "rows_affected": affected, "sql": sql})


async def _query_context(args: dict) -> list[types.TextContent]:
    """Knowledge-graph lookup for the client's own LLM: only the tables, joins and
    examples a question needs, so it can write SQL without reading the whole schema."""
    sid        = _active(args)
    question   = str(args.get("question", "")).strip()
    max_tables = min(12, max(1, int(args.get("max_tables", 6))))
    session    = _require_session(sid)
    ke         = _knowledge.get(sid)
    if ke is None:
        return _err("No knowledge for this connection yet - reconnect to build it.", ErrCode.NOT_CONNECTED)

    loop   = asyncio.get_event_loop()
    schema = redis_cache.get_schema(sid) or {}
    fks    = _fks.get(sid, [])
    notes  = _load_notes(sid)
    tables: dict = {}

    def add(t: str) -> None:
        if t in schema and t not in tables and len(tables) < max_tables:
            tables[t] = schema[t]

    # 1) Glossary first: a human said "this table/column means X" — if the question
    #    uses those words, that table is in, whatever the schema names look like.
    words = {w for w in re.findall(r"[a-z0-9_]+", question.lower()) if len(w) > 3}
    for n in notes:
        if n.get("table") and any(w in n["note"].lower() for w in words):
            add(n["table"])
    # 2) Rank tables with the in-process FAISS/BM25 retriever (ChromaDB's vector index
    #    is unreliable for small collections across processes), 3) anything the
    #    knowledge engine found, 4) tables linked by REAL foreign keys, 5) graph neighbours.
    retriever = _retrievers.get(sid)
    if retriever:
        ranked = await loop.run_in_executor(None, retriever.retrieve, question, max_tables) or {}
        for t in (list(ranked) if isinstance(ranked, dict) else list(ranked)):
            add(t)
    ctx = await loop.run_in_executor(None, ke.get_context_for_question, question, session.get("db_type", "sql"))
    for t in ctx.get("tables_found") or []:
        add(t)
    for t in list(tables):
        for j in fks:
            a, b = j["from"].split(".")[0], j["to"].split(".")[0]
            if a == t:
                add(b)
            elif b == t:
                add(a)
    for t in list(tables):
        if ke.graph.has_node(t):
            for n in ke.get_related_tables(t, depth=1):
                add(n)

    # Popular tables first (what this database's users actually query), so an
    # ambiguous name resolves to the one everyone uses.
    usage = _load_usage(sid).get("tables", {})
    if usage:
        tables = dict(sorted(tables.items(), key=lambda kv: -usage.get(kv[0], 0)))
    hot = [t for t, _ in sorted(usage.items(), key=lambda kv: -kv[1])[:5]]
    sc = _scope.get(sid)
    scope_info = (
        {"column": sc["column"], "value": sc["value"], "label": sc.get("label", ""),
         "rule": f"Every query on tables that have {sc['column']} must include WHERE {sc['column']} = {sc['value']!r}."}
        if sc else None
    )
    scope_candidates = [] if sc else _scope_candidates(schema)

    biz = session.get("context") or {}
    biz_tables = biz.get("tables") if isinstance(biz.get("tables"), dict) else {}
    # Joins: real FKs first, then graph paths, then column-name inference (x_id -> x.id).
    joins: list = []
    seen: set = set()
    fk_here = [j for j in fks if j["from"].split(".")[0] in tables and j["to"].split(".")[0] in tables]
    for j in fk_here + list(ke.find_all_join_paths(list(tables)) or []) + infer_joins({t: schema[t] for t in tables}):
        key = (str(j.get("from")), str(j.get("to")))
        if key not in seen:
            joins.append(j)
            seen.add(key)
    glossary = [{k: v for k, v in n.items() if k in ("table", "column", "note")}
                for n in notes if not n.get("table") or n["table"] in tables][:20]
    return _ok({
        "dialect":          session.get("db_flavor"),
        "tables":           tables,
        "joins":            joins,
        "glossary":         glossary,
        "scope":            scope_info,
        "scope_candidates": scope_candidates,
        "hot_tables":       hot,
        "table_notes":      {t: n for t, n in biz_tables.items() if t in tables},
        "metrics":          biz.get("metrics") or {},
        "similar_queries":  ctx.get("similar_queries") or [],
        "business_type":    biz.get("business_type", "unknown"),
        "hint": ("Write ONE SELECT in this dialect using only these tables/columns, then call run_sql with the "
                 "sql and the question. Call get_schema(tables=[...]) only if a needed table is missing here."),
    })


async def _schema(args: dict) -> list[types.TextContent]:
    sid    = _active(args)
    schema = redis_cache.get_schema(sid)
    if not schema:
        return _err("Not connected. Call connect_database first.", ErrCode.NOT_CONNECTED)
    wanted = args.get("tables")
    if wanted:
        lookup = {t.lower(): t for t in schema}
        schema = {lookup[w.lower()]: schema[lookup[w.lower()]] for w in wanted if w.lower() in lookup}
    return _ok({"tables": schema, "table_count": len(schema)})


async def _add_note(args: dict) -> list[types.TextContent]:
    """Teach the database once: a note on a table, a column, or the whole database."""
    import hashlib
    sid    = _active(args)
    _require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    note   = str(args.get("note", "")).strip()
    table  = str(args.get("table", "")).strip()
    column = str(args.get("column", "")).strip()
    if len(note) < 3:
        return _err("note is required.", ErrCode.INVALID_SQL)
    if table:
        lookup = {t.lower(): t for t in schema}
        if table.lower() not in lookup:
            return _err(f"Table '{table}' not in schema.", ErrCode.INVALID_SQL)
        table = lookup[table.lower()]
        if column:
            cols = {c.split(" ")[0].lower(): c.split(" ")[0] for c in schema[table]}
            if column.lower() not in cols:
                return _err(f"Column '{column}' not in {table}.", ErrCode.INVALID_SQL)
            column = cols[column.lower()]
    elif column:
        return _err("column requires table.", ErrCode.INVALID_SQL)

    notes = _load_notes(sid)
    entry = {"id": hashlib.sha1(f"{table}|{column}|{note}".encode()).hexdigest()[:10],
             "table": table, "column": column, "note": note,
             "author": _request_user(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if any(n["id"] == entry["id"] for n in notes):
        return _ok({"added": False, "duplicate": True, "note": entry})
    notes.append(entry)
    _save_notes(sid, notes)
    _audit("add_note", sid, f"{table}.{column}: {note}" if table else note, ok=True)
    return _ok({"added": True, "note": entry, "total_notes": len(notes)})


async def _list_notes(args: dict) -> list[types.TextContent]:
    sid   = _active(args)
    _require_session(sid)
    table = str(args.get("table", "")).strip().lower()
    notes = _load_notes(sid)
    if table:
        notes = [n for n in notes if n.get("table", "").lower() == table or not n.get("table")]
    return _ok({"notes": notes, "total": len(notes)})


async def _delete_note(args: dict) -> list[types.TextContent]:
    sid   = _active(args)
    _require_session(sid)
    nid   = str(args.get("id", "")).strip()
    notes = _load_notes(sid)
    kept  = [n for n in notes if n["id"] != nid]
    if len(kept) == len(notes):
        return _err(f"No note with id '{nid}'.", ErrCode.INVALID_SQL)
    _save_notes(sid, kept)
    _audit("delete_note", sid, nid, ok=True)
    return _ok({"deleted": True, "id": nid, "total_notes": len(kept)})


async def _set_scope(args: dict) -> list[types.TextContent]:
    sid    = _active(args)
    _require_session(sid)
    schema = redis_cache.get_schema(sid) or {}
    column = str(args.get("column", "")).strip()
    value  = str(args.get("value", "")).strip()
    label  = str(args.get("label", "")).strip()
    if not value:
        _scope.pop(sid, None)
        _audit("set_scope", sid, f"cleared {column}", ok=True)
        return _ok({"scope": None, "message": "Scope cleared - queries now span all clients."})
    tables = [t for t, cols in schema.items() if any(c.split(" ")[0].lower() == column.lower() for c in cols)]
    if not tables:
        return _err(f"No table has a column named '{column}'.", ErrCode.INVALID_SQL)
    exact = next(c.split(" ")[0] for c in schema[tables[0]] if c.split(" ")[0].lower() == column.lower())
    _scope[sid] = {"column": exact, "value": value, "label": label, "tables": tables}
    _audit("set_scope", sid, f"{exact} = {value} ({label})", ok=True)
    return _ok({"scope": {"column": exact, "value": value, "label": label, "scoped_tables": len(tables)},
                "message": f"Every query on the {len(tables)} tables that have {exact} must now filter {exact} = {value}."})


async def _import_query_history(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    session = _require_session(sid)
    limit   = min(2000, max(10, int(args.get("limit", 300))))
    if session.get("db_type") != "sql" or sid not in _engines:
        return _err("Query history import supports SQL databases only.", ErrCode.UNSUPPORTED_DB)
    schema  = redis_cache.get_schema(sid) or {}
    flavor  = session.get("db_flavor", "")
    loop    = asyncio.get_event_loop()

    def _fetch():
        with _engines[sid].connect() as conn:
            return _fetch_query_history(conn, flavor, limit)
    try:
        history = await asyncio.wait_for(loop.run_in_executor(None, _fetch), timeout=DB_QUERY_TIMEOUT)
    except Exception as e:
        return _err(f"Could not read query statistics ({flavor}); the login may lack VIEW SERVER STATE. {str(e)[:200]}",
                    ErrCode.QUERY_FAILED)
    if not history:
        return _ok({"imported_statements": 0, "message": f"No query statistics available for {flavor}."})

    usage = _load_usage(sid)
    matched = 0
    for stmt, count in history:
        tables = _tables_in_sql(stmt, schema)
        if not tables:
            continue
        matched += 1
        for t in tables:
            usage["tables"][t] = usage["tables"].get(t, 0) + max(1, count)
    usage["statements"] = usage.get("statements", 0) + matched
    path = _usage_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(usage, f)
    top = sorted(usage["tables"].items(), key=lambda kv: -kv[1])[:10]
    _audit("import_query_history", sid, f"{matched} statements", ok=True, rows=matched)
    return _ok({"imported_statements": matched, "statements_read": len(history),
                "hot_tables": [{"table": t, "executions": n} for t, n in top]})


async def _import_queries(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    _require_session(sid)
    schema  = redis_cache.get_schema(sid) or {}
    ke      = _knowledge.get(sid)
    text_in = str(args.get("sql_text", ""))
    imported, skipped = 0, 0
    for chunk in re.split(r";\s*\n|;\s*$", text_in, flags=re.MULTILINE):
        chunk = chunk.strip()
        if not chunk:
            continue
        comments = [l[2:].strip() for l in chunk.splitlines() if l.strip().startswith("--")]
        sql = "\n".join(l for l in chunk.splitlines() if not l.strip().startswith("--")).strip()
        if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
            skipped += 1
            continue
        tables = _tables_in_sql(sql, schema)
        if not tables:
            skipped += 1
            continue
        question = " ".join(comments) or f"query over {', '.join(tables)}"
        if ke:
            ke.learn_successful_query(question, sql)
        _bump_usage(sid, tables)
        imported += 1
    _audit("import_queries", sid, f"{imported} imported, {skipped} skipped", ok=True, rows=imported)
    return _ok({"imported": imported, "skipped": skipped,
                "hint": "Imported SELECTs now appear as similar_queries in get_query_context and count toward hot_tables."})


async def _explain(args: dict) -> list[types.TextContent]:
    sid     = _active(args)
    session = _require_session(sid)
    engine  = _engines.get(sid)
    if not engine:
        return _err("No SQL engine found. Reconnect first.", ErrCode.NOT_CONNECTED)

    db_flavor = session.get("db_flavor", "")
    sql       = args["sql"]
    explain_map = {
        "PostgreSQL": f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}",
        "MySQL":      f"EXPLAIN {sql}",
        "MariaDB":    f"EXPLAIN {sql}",
        "SQLite":     f"EXPLAIN QUERY PLAN {sql}",
        "SQL Server": f"SET SHOWPLAN_TEXT ON; {sql}",
    }
    explain_sql = explain_map.get(db_flavor, f"EXPLAIN {sql}")

    def _run():
        with engine.connect() as conn:
            result = conn.execute(text(explain_sql))
            return [dict(row._mapping) for row in result]

    loop = asyncio.get_event_loop()
    try:
        plan = await loop.run_in_executor(None, _run)
        return _ok({"sql": sql, "flavor": db_flavor, "plan": plan})
    except Exception as e:
        return _err(str(e), ErrCode.QUERY_FAILED)


async def _chart(args: dict) -> list[types.TextContent]:
    sid = _active(args)
    session = _get_session(sid)
    client  = _groq(session)
    columns = args["columns"]
    rows    = args["rows"]

    if not rows:
        return _ok({"chart": None})

    sample = [dict(zip(columns, row)) for row in rows[:8]]
    prompt = (
        "You are a data visualisation expert. Pick the single best chart type.\n\n"
        f"Columns: {columns}\nSample rows: {sample}\nTotal rows: {len(rows)}\n\n"
        "Rules: scatter=2 numeric cols; line=date+numeric; bar=category+numeric; "
        "doughnut=2-6 category proportions; horizontalBar=top-N rankings; none=single row\n\n"
        'Return ONLY JSON: {"chart_type":"<type>","label_column":"<col>","value_column":"<col>","title":"<title>"}'
    )
    loop = asyncio.get_event_loop()

    def _llm():
        return client.chat.completions.create(
            model="openai/gpt-oss-120b",
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=128,
        )

    try:
        res = await loop.run_in_executor(None, _llm)
        raw = re.sub(r"^```[a-z]*\n?", "", res.choices[0].message.content.strip())
        raw = re.sub(r"\n?```$", "", raw).strip()
        chart = json.loads(raw)
        _track_usage(sid, "charts")
        return _ok({"chart": chart})
    except json.JSONDecodeError as e:
        _track_usage(sid, "errors")
        return _err(f"LLM returned invalid JSON for chart suggestion: {e}", ErrCode.QUERY_FAILED)
    except Exception as e:
        _track_usage(sid, "errors")
        return _err(str(e), ErrCode.UNKNOWN)


async def _describe_capabilities(args: dict) -> list[types.TextContent]:
    sid     = _resolve_sid(args)
    current = _current.get(sid)
    session = _get_session(current) if current else None
    connected_db = None
    if session:
        connected_db = {
            "connection":    current,
            "db_flavor":     session.get("db_flavor"),
            "business_type": session.get("context", {}).get("business_type", "unknown"),
            "has_schema":    bool(redis_cache.get_schema(current)),
        }
    prefix = f"{sid}__"
    open_connections = [k for k in (*_engines, *_mongos, *_cassandras) if k.startswith(prefix)]
    return _ok({
        "server":          "sql-agent",
        "version":         __version__,
        "session_id":      sid,
        "connected":       connected_db is not None,
        "current_session": connected_db,
        "open_connections": open_connections,
        "usage":           dict(_usage.get(sid, {})),
        "supported_databases": [
            "SQLite", "MySQL", "MariaDB", "PostgreSQL", "SQL Server", "MongoDB", "Cassandra"
        ],
        "tools": [
            {"name": "connect_database",    "requires_session": False},
            {"name": "query_database",      "requires_session": True,  "needs_groq_key": True},
            {"name": "run_sql",             "requires_session": True,  "sql_only": True},
            {"name": "get_query_context",   "requires_session": True},
            {"name": "get_schema",          "requires_session": True},
            {"name": "add_note",            "requires_session": True},
            {"name": "list_notes",          "requires_session": True},
            {"name": "delete_note",         "requires_session": True},
            {"name": "set_scope",           "requires_session": True},
            {"name": "import_query_history", "requires_session": True, "sql_only": True},
            {"name": "import_queries",      "requires_session": True},
            {"name": "explain_query",       "requires_session": True,  "sql_only": True},
            {"name": "suggest_chart",       "requires_session": False},
            {"name": "execute_write",       "requires_session": True,  "sql_only": True,
             "developer_mode": True, "enabled": ALLOW_WRITES, "ddl_enabled": ALLOW_DDL},
            {"name": "disconnect",          "requires_session": True},
            {"name": "describe_capabilities", "requires_session": False},
        ],
        "error_codes": {
            ErrCode.NOT_CONNECTED:  "No active session — call connect_database first",
            ErrCode.QUERY_FAILED:   "Database returned an error executing the query",
            ErrCode.AUTH_FAILED:    "Authentication / credential error",
            ErrCode.UNSUPPORTED_DB: "db_type is not one of the supported databases",
            ErrCode.INVALID_SQL:    "Generated SQL failed validation",
            ErrCode.TIMEOUT:        "Query exceeded the timeout limit",
            ErrCode.UNKNOWN:        "Unexpected server error",
        },
    })


async def _upload_document(args: dict) -> list[types.TextContent]:
    import base64
    sid   = _resolve_sid(args)
    store = _get_docstore(sid)
    loop  = asyncio.get_event_loop()

    url = args.get("url", "").strip()
    if url:
        try:
            chunks = await loop.run_in_executor(None, store.add_url, url)
            return _ok({"indexed": True, "source": url, "chunks_indexed": chunks})
        except Exception as e:
            return _err(f"Failed to fetch URL: {e}")

    filename   = args.get("filename", "document.txt")
    content_b64 = args.get("content_b64", "")
    if not content_b64:
        return _err("Provide either content_b64 (base64 file bytes) or url.")
    try:
        data   = base64.b64decode(content_b64)
        chunks = await loop.run_in_executor(None, store.add_file, filename, data)
        return _ok({"indexed": True, "source": filename, "chunks_indexed": chunks})
    except Exception as e:
        return _err(f"Failed to index document: {e}")


async def _search_documents(args: dict) -> list[types.TextContent]:
    sid      = _resolve_sid(args)
    question = args["question"]
    n        = min(10, max(1, int(args.get("n_results", 4))))
    store    = _get_docstore(sid)
    loop     = asyncio.get_event_loop()

    if store.count() == 0:
        return _ok({
            "chunks": [],
            "hint": "No documents indexed yet. Call upload_document first.",
        })

    hits = await loop.run_in_executor(None, store.search, question, n)
    return _ok({"chunks": hits, "total_chunks_in_store": store.count()})


async def _list_documents(args: dict) -> list[types.TextContent]:
    sid   = _resolve_sid(args)
    store = _get_docstore(sid)
    loop  = asyncio.get_event_loop()
    sources = await loop.run_in_executor(None, store.list_sources)
    return _ok({"sources": sources, "total_chunks": store.count()})


async def _disconnect(args: dict) -> list[types.TextContent]:
    session_id = _resolve_sid(args)
    sid = _current.pop(session_id, session_id)   # closes only the current connection
    redis_cache.delete_session(sid)
    redis_cache.clear_user_query_cache(sid)
    _engines.pop(sid, None)
    _mongos.pop(sid, None)
    _cassandras.pop(sid, None)
    _retrievers.pop(sid, None)
    _knowledge.pop(sid, None)
    _fks.pop(sid, None)
    _scope.pop(sid, None)
    _docstores.pop(sid, None)
    _usage.pop(sid, None)
    return _ok({"disconnected": True, "session_id": sid})


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _serve_http(bind: str) -> None:
    """Hosted mode: the same tools over Streamable HTTP at /mcp, protected by a
    bearer token (MCP_AUTH_TOKEN). Clients: claude mcp add --transport http ..."""
    import hmac

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount

    tokens = [t for t in _TOKEN_USERS if len(t) >= 16]
    if not tokens:
        print("[sql-agent] --http requires MCP_AUTH_TOKEN or MCP_AUTH_TOKENS (name:token,...) "
              "with 16+ character tokens in .env", file=sys.stderr)
        sys.exit(1)
    host, _, port = bind.rpartition(":")
    manager = StreamableHTTPSessionManager(app=server)

    async def guarded(scope, receive, send):
        # Plain ASGI (no Mount("/mcp"), no middleware): no slash redirects that
        # would bypass the check, and streaming (SSE) responses pass through untouched.
        if scope["type"] != "http":
            return
        if scope["path"].rstrip("/") != "/mcp":
            await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
            return
        auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
        if not any(hmac.compare_digest(auth, f"Bearer {t}") for t in tokens):
            await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    app = Starlette(routes=[Mount("/", app=guarded)], lifespan=lifespan)
    print(f"[sql-agent] HTTP mode: http://{host or '0.0.0.0'}:{port or 8000}/mcp", file=sys.stderr)
    uvicorn.run(app, host=host or "0.0.0.0", port=int(port or 8000), log_level="info")


def cli() -> None:
    """Entry point behind the root launcher: python mcp_server.py [--http host:port]."""
    if "--http" in sys.argv:
        i = sys.argv.index("--http")
        nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        _serve_http(nxt if nxt and not nxt.startswith("-") else "127.0.0.1:8000")
    else:
        asyncio.run(main())


if __name__ == "__main__":
    cli()
