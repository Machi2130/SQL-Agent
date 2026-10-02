"""Quick test for DocStore hybrid search. Run: python test_docstore.py"""

from sql_agent.docstore import DocStore

store = DocStore("test-user")

# ── Ingest sample policy text ──────────────────────────────
sample = b"""
EMEA Travel Policy

Maximum quarterly travel spend per team: $18,000.
Exception: Teams may exceed limit by 20% with regional director approval.
All expenses above $5,000 require receipts and manager sign-off.

AML Compliance Rules

Transactions above $10,000 must be flagged for enhanced verification.
Cash transactions above $3,000 require source-of-funds documentation.
Structuring transactions below reporting thresholds is prohibited.

HR Leave Policy

Annual PTO allowance: 20 days per employee.
Sick leave: 10 days per year, non-transferable.
Unused PTO above 5 days cannot be carried forward to the next year.
"""

chunks = store.add_file("company_policies.txt", sample)
print(f"Indexed {chunks} chunks\n")

# ── Test queries ───────────────────────────────────────────
queries = [
    "What is the EMEA travel spending limit?",
    "AML threshold for transactions",
    "How many PTO days are employees allowed?",
    "cash transactions documentation required",
]

for q in queries:
    print(f"Q: {q}")
    results = store.search(q, n_results=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['source']} → {r['text'][:120]}...")
    print()
