"""
query_planner.py — Code-first query planning.

Does 90% of the work in Python so the LLM only generates SQL syntax.
Reduces token usage from ~6000 to ~800 per query.

Pipeline:
  1. Detect intent (aggregation, listing, time-series, count)
  2. Map business concepts to actual tables/columns
  3. Find join paths via Knowledge Graph
  4. Build a minimal, precise prompt for the LLM
"""

from __future__ import annotations

import re
from typing import Any


# ═══════════════════════════════════════════════════════════════
#  INTENT DETECTION — pure code, no LLM
# ═══════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "count": [
        r"\bhow many\b", r"\bcount\b", r"\btotal number\b", r"\bnumber of\b",
    ],
    "sum": [
        r"\btotal\b.*\b(sales|amount|revenue|value|price|cost)\b",
        r"\bsum\b", r"\bgrand total\b",
    ],
    "average": [
        r"\baverage\b", r"\bavg\b", r"\bmean\b",
    ],
    "time_series": [
        r"\bdaily\b", r"\bweekly\b", r"\bmonthly\b", r"\byearly\b",
        r"\bby (day|week|month|year|date)\b", r"\btrend\b", r"\bover time\b",
    ],
    "top_n": [
        r"\btop \d+\b", r"\bbest\b", r"\bhighest\b", r"\blargest\b",
        r"\bmost\b", r"\bmaximum\b",
    ],
    "bottom_n": [
        r"\bbottom \d+\b", r"\bworst\b", r"\blowest\b", r"\bsmallest\b",
        r"\bleast\b", r"\bminimum\b",
    ],
    "latest": [
        r"\blatest\b", r"\brecent\b", r"\blast \d+\b", r"\bnewest\b",
    ],
    "filter": [
        r"\bwhere\b", r"\bfor\b.*\b(store|customer|product|category)\b",
        r"\bonly\b", r"\bspecific\b",
    ],
    "list": [
        r"\bshow\b", r"\blist\b", r"\bdisplay\b", r"\bget\b", r"\bfetch\b",
    ],
}

_TIME_RANGES: dict[str, dict[str, str]] = {
    "SQL Server": {
        "today":        "CAST(GETDATE() AS date)",
        "yesterday":    "DATEADD(day, -1, CAST(GETDATE() AS date))",
        "this week":    "DATEADD(week, DATEDIFF(week, 0, GETDATE()), 0)",
        "last 7 days":  "DATEADD(day, -7, CAST(GETDATE() AS date))",
        "this month":   "DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0)",
        "last month":   "DATEADD(month, DATEDIFF(month, 0, GETDATE()) - 1, 0)",
        "last 30 days": "DATEADD(day, -30, CAST(GETDATE() AS date))",
        "this year":    "DATEADD(year, DATEDIFF(year, 0, GETDATE()), 0)",
        "last year":    "DATEADD(year, DATEDIFF(year, 0, GETDATE()) - 1, 0)",
    },
    "PostgreSQL": {
        "today":        "CURRENT_DATE",
        "yesterday":    "CURRENT_DATE - INTERVAL '1 day'",
        "this week":    "DATE_TRUNC('week', CURRENT_DATE)",
        "last 7 days":  "CURRENT_DATE - INTERVAL '7 days'",
        "this month":   "DATE_TRUNC('month', CURRENT_DATE)",
        "last month":   "DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'",
        "last 30 days": "CURRENT_DATE - INTERVAL '30 days'",
        "this year":    "DATE_TRUNC('year', CURRENT_DATE)",
        "last year":    "DATE_TRUNC('year', CURRENT_DATE) - INTERVAL '1 year'",
    },
    "MySQL": {
        "today":        "CURDATE()",
        "yesterday":    "DATE_SUB(CURDATE(), INTERVAL 1 DAY)",
        "this week":    "DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY)",
        "last 7 days":  "DATE_SUB(CURDATE(), INTERVAL 7 DAY)",
        "this month":   "DATE_FORMAT(CURDATE(), '%Y-%m-01')",
        "last month":   "DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m-01')",
        "last 30 days": "DATE_SUB(CURDATE(), INTERVAL 30 DAY)",
        "this year":    "DATE_FORMAT(CURDATE(), '%Y-01-01')",
        "last year":    "DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 YEAR), '%Y-01-01')",
    },
    "SQLite": {
        "today":        "date('now')",
        "yesterday":    "date('now', '-1 day')",
        "this week":    "date('now', 'weekday 0', '-7 days')",
        "last 7 days":  "date('now', '-7 days')",
        "this month":   "date('now', 'start of month')",
        "last month":   "date('now', 'start of month', '-1 month')",
        "last 30 days": "date('now', '-30 days')",
        "this year":    "date('now', 'start of year')",
        "last year":    "date('now', 'start of year', '-1 year')",
    },
}
_TIME_RANGES["MariaDB"] = _TIME_RANGES["MySQL"]

def detect_intents(question: str) -> list[str]:
    q = question.lower()
    found = []
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                found.append(intent)
                break
    return found or ["list"]


_MONTH_YEAR_FN: dict[str, str] = {
    "SQL Server": "YEAR(GETDATE())",
    "MySQL":      "YEAR(CURDATE())",
    "MariaDB":    "YEAR(CURDATE())",
    "PostgreSQL": "EXTRACT(YEAR FROM CURRENT_DATE)",
    "SQLite":     "strftime('%Y', 'now')",
}

def detect_time_range(question: str, flavor: str = "SQL Server") -> str | None:
    q = question.lower()
    ranges = _TIME_RANGES.get(flavor, _TIME_RANGES["SQL Server"])
    for phrase, sql_expr in ranges.items():
        if phrase in q:
            return sql_expr
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?", q
    )
    if month_match:
        months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                  "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
        m = months.get(month_match.group(1), 1)
        y = month_match.group(2) or _MONTH_YEAR_FN.get(flavor, "YEAR(GETDATE())")
        return f"'{y}-{m:02d}-01'"
    return None


def detect_top_n(question: str) -> int | None:
    m = re.search(r"\btop\s+(\d+)\b|\blast\s+(\d+)\b|\b(\d+)\s+rows?\b", question.lower())
    if m:
        return int(m.group(1) or m.group(2) or m.group(3))
    return None


def build_concept_map(schema_dict: dict) -> dict:
    """Auto-discover concepts from actual table/column names."""
    concepts = {}
    for tbl, cols in schema_dict.items():
        col_names = [c.split(" ")[0] for c in cols]
        col_lower = [c.lower() for c in col_names]

        tbl_lower = tbl.lower()
        keywords = set(re.findall(r'[a-z]+', tbl_lower))

        date_col = next((c for c, cl in zip(col_names, col_lower)
                         if cl in ("invoice_dt", "order_dt", "created_dt", "payment_dt",
                                   "transaction_dt", "invoice_date", "order_date", "created_at",
                                   "sale_date", "purchase_date", "entry_date", "bill_dt", "bill_date")), None)
        value_col = next((c for c, cl in zip(col_names, col_lower)
                          if cl in ("grand_total", "total_amount", "net_amount", "amount",
                                    "total", "net_total", "bill_amount", "invoice_amount",
                                    "line_total", "sub_total", "paid_amount")), None)

        for kw in keywords:
            if kw in ("tr", "ms", "tbl", "sys", "asp", "net", "log"):
                continue
            if kw not in concepts:
                concepts[kw] = []
            concepts[kw].append({
                "table": tbl,
                "columns": col_names,
                "date_column": date_col,
                "value_column": value_col,
            })

    return concepts


_concept_cache: dict | None = None


def get_concept_map(schema_dict: dict) -> dict:
    global _concept_cache
    if _concept_cache is None:
        _concept_cache = build_concept_map(schema_dict)
    return _concept_cache


def reset_concept_cache():
    global _concept_cache
    _concept_cache = None


def map_concepts(question: str, schema_dict: dict) -> dict:
    """Map natural language concepts to actual tables that exist in the schema."""
    q = question.lower()
    concepts = get_concept_map(schema_dict)
    mapped = {}
    scores: dict[str, int] = {}

    q_words = set(re.findall(r'\w+', q))

    for word in q_words:
        if word in concepts:
            for entry in concepts[word]:
                tbl = entry["table"]
                scores[tbl] = scores.get(tbl, 0) + 1
                if tbl not in mapped:
                    mapped[tbl] = {
                        "columns": entry["columns"],
                        "value_column": entry["value_column"],
                        "date_column": entry["date_column"],
                        "role": word,
                    }

    if mapped:
        top_tables = sorted(mapped.keys(), key=lambda t: scores.get(t, 0), reverse=True)[:4]
        return {t: mapped[t] for t in top_tables}

    return mapped


def auto_map_from_schema(question: str, schema_dict: dict) -> dict:
    """Fuzzy-match question words to actual table names in the schema."""
    q_words = set(re.findall(r'\w+', question.lower()))
    mapped = {}

    for tbl in schema_dict:
        tbl_words = set(re.findall(r'\w+', tbl.lower()))
        overlap = q_words & tbl_words
        if overlap:
            cols = [c.split(" ")[0] for c in schema_dict[tbl]]
            date_cols = [c for c in cols if any(d in c.lower() for d in ["_dt", "_date", "date", "created", "invoice_dt"])]
            value_cols = [c for c in cols if any(v in c.lower() for v in ["amount", "total", "grand", "price", "value", "qty", "quantity"])]
            mapped[tbl] = {
                "columns": cols,
                "value_column": value_cols[0] if value_cols else None,
                "date_column": date_cols[0] if date_cols else None,
                "role": "matched",
            }

    return mapped


# ═══════════════════════════════════════════════════════════════
#  QUERY PLAN — structured output for the LLM
# ═══════════════════════════════════════════════════════════════

def build_query_plan(question: str, schema_dict: dict, kg_joins: list = None, flavor: str = "SQL Server") -> dict:
    """
    Build a structured query plan from the question — all in code.
    Returns a compact dict that gets turned into a tiny LLM prompt.
    """
    intents = detect_intents(question)
    time_range = detect_time_range(question, flavor)
    top_n = detect_top_n(question)

    tables = map_concepts(question, schema_dict)
    if not tables:
        tables = auto_map_from_schema(question, schema_dict)

    joins = []
    if kg_joins:
        table_names = list(tables.keys())
        for j in kg_joins:
            from_tbl = j["from"].split(".")[0]
            to_tbl = j["to"].split(".")[0]
            if from_tbl in table_names or to_tbl in table_names:
                joins.append(j)

    return {
        "intents": intents,
        "tables": tables,
        "joins": joins,
        "time_range": time_range,
        "top_n": top_n or 50,
        "question": question,
    }


def plan_to_prompt(plan: dict, flavor: str) -> str:
    """
    Convert a query plan to a minimal LLM prompt.
    ~300-500 tokens instead of ~5000.
    """
    lines = []
    lines.append(f"Generate a {flavor} SELECT query.")

    if plan["intents"]:
        lines.append(f"Intent: {', '.join(plan['intents'])}")

    lines.append(f"Top N: {plan['top_n']}")

    if plan["time_range"]:
        lines.append(f"Time filter: >= {plan['time_range']}")

    lines.append("")
    lines.append("Tables and columns (use ONLY these exact names):")
    for tbl, info in plan["tables"].items():
        col_str = ", ".join(info["columns"][:15])
        meta = []
        if info.get("value_column"):
            meta.append(f"value={info['value_column']}")
        if info.get("date_column"):
            meta.append(f"date={info['date_column']}")
        meta_str = f" [{', '.join(meta)}]" if meta else ""
        lines.append(f"  {tbl}: {col_str}{meta_str}")

    if plan["joins"]:
        lines.append("")
        lines.append("Joins:")
        for j in plan["joins"]:
            lines.append(f"  {j['from']} = {j['to']}")

    lines.append("")
    if flavor == "SQL Server":
        lines.append("Use T-SQL: SELECT TOP N, square brackets for identifiers.")
    lines.append("Return ONLY raw SQL. No markdown. No explanation.")

    return "\n".join(lines)
