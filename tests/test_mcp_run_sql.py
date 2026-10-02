"""MCP server: connect without a Groq key and run AI-written SQL via run_sql.

Run: python -m pytest test_mcp_run_sql.py -q
"""
import asyncio
import json
import sqlite3
import uuid

import pytest

from sql_agent import server as mcp_server


def _call(name: str, args: dict) -> dict:
    out = asyncio.run(mcp_server.call_tool(name, args))
    return json.loads(out[0].text)


@pytest.fixture(scope="module")
def sqlite_session(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "t.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("INSERT INTO items (id, name) VALUES (1, 'apple'), (2, 'pear')")
    con.commit()
    con.close()

    sid = f"pytest-{uuid.uuid4().hex[:8]}"
    saved = mcp_server.GROQ_API_KEY
    mcp_server.GROQ_API_KEY = ""          # simulate no key anywhere
    try:
        res = _call("connect_database", {
            "db_type": "SQLite", "db_path": str(db_path), "session_id": sid,
        })
        assert res["ok"], res
        yield sid
    finally:
        _call("disconnect", {"session_id": sid})
        mcp_server.GROQ_API_KEY = saved


def test_connect_without_groq_key_reports_tables(sqlite_session):
    res = _call("get_schema", {"session_id": sqlite_session})
    assert res["ok"]
    assert "items" in res["data"]["tables"]


def test_run_sql_returns_rows(sqlite_session):
    res = _call("run_sql", {"sql": "SELECT id, name FROM items ORDER BY id",
                            "session_id": sqlite_session})
    assert res["ok"], res
    assert res["data"]["columns"] == ["id", "name"]
    assert res["data"]["rows"] == [[1, "apple"], [2, "pear"]]
    assert res["data"]["total_rows"] == 2


def test_run_sql_paginates(sqlite_session):
    res = _call("run_sql", {"sql": "SELECT id FROM items ORDER BY id",
                            "page": 2, "page_size": 1, "session_id": sqlite_session})
    assert res["ok"], res
    assert res["data"]["rows"] == [[2]]
    assert res["data"]["total_pages"] == 2


def test_run_sql_rejects_writes(sqlite_session):
    res = _call("run_sql", {"sql": "DELETE FROM items", "session_id": sqlite_session})
    assert not res["ok"]
    assert res["error"]["code"] == mcp_server.ErrCode.INVALID_SQL


def test_run_sql_rejects_unknown_table(sqlite_session):
    res = _call("run_sql", {"sql": "SELECT * FROM nope", "session_id": sqlite_session})
    assert not res["ok"]
    assert res["error"]["code"] == mcp_server.ErrCode.INVALID_SQL


def test_run_sql_requires_session():
    res = _call("run_sql", {"sql": "SELECT 1", "session_id": "no-such-session"})
    assert not res["ok"]
    assert res["error"]["code"] == mcp_server.ErrCode.NOT_CONNECTED


def test_query_database_without_key_points_to_run_sql(sqlite_session):
    saved = mcp_server.GROQ_API_KEY
    mcp_server.GROQ_API_KEY = ""
    try:
        res = _call("query_database", {"question": "how many items", "session_id": sqlite_session})
    finally:
        mcp_server.GROQ_API_KEY = saved
    assert not res["ok"]
    assert "run_sql" in res["error"]["message"]
