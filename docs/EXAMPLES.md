# Worked examples

Abridged from real sessions against a multi-tenant SQL Server database (several dozen
populated tables, many tenants sharing one schema) and the test databases in the repo. All
identifiers and figures are generalised.

## 1. Finding the right tables without reading the schema

Question: *"total transaction amount for a given card"*

The schema never mentions "card": the card number lives in an opaquely named column,
`ACCOUNT_LINK.Account_Ref`. Before anyone taught the server that, the context came back
with `txn, member, branch` — plausible, wrong.

```
add_note("Account_Ref holds the card number", table="ACCOUNT_LINK", column="Account_Ref")
```

Same question afterwards:

```
tables : account_link, txn, member
joins  : account_link.member_id -> member.id        (source: fk)
         txn.account_link_id -> account_link.id     (source: fk)
glossary: "Account_Ref holds the card number…"
```

One note, written once, and every later question about cards resolves correctly — the
note's words steer table selection, and real foreign keys supply the joins.

## 2. One database, twenty clients

`connect_database` reported `scope_candidates: [{column: Tenant_Id, in_tables: 50}]`.

```
run_sql("SELECT SUM(amount) FROM transactions")            -> 1099   (two tenants mixed)
set_scope(column="Tenant_Id", value="7", label="Acme Retail")
run_sql("SELECT SUM(amount) FROM transactions")            -> REJECTED:
   "transactions must be filtered by Tenant_Id = '7' (Acme Retail). Add it to the WHERE clause…"
run_sql("SELECT SUM(amount) FROM transactions WHERE Tenant_Id = 7") -> 100
run_sql("SELECT key, value FROM system_parameters")        -> fine (no tenant column)
```

## 3. Several databases open at once

```
connect A  → connected
connect B  → connected            (A stays open)
run_sql    → hits B
connect A  → switched=True, 0.015 s   (no reconnect, no re-learn)
run_sql    → hits A
disconnect → closes A only; B still open
```

## 4. Developer write, three gates

```
execute_write("DELETE FROM members")                      -> rejected: no WHERE clause
execute_write("UPDATE members SET name='x' WHERE id<=3", dry_run=True)
   -> form: "DRY RUN (rolled back) on …: UPDATE … Type YES to confirm."  -> rows_affected=3, executed=False
   -> table unchanged
execute_write(same, dry_run=False) -> form -> YES -> rows_affected=3, executed=True
execute_write("DROP TABLE members")                        -> rejected: DDL disabled
on a connection opened without allow_writes                -> rejected: connection is read-only
```

## 5. The cost of the knowledge layer

Measured on the 70-table database: the full schema is ~24,000 characters (~6,000 tokens);
`get_query_context` returns ~600 tokens. Building the knowledge layer costs no LLM tokens
(local embeddings, FK metadata query); the optional Groq business summary is ~3,100 tokens
in / ≤4,100 out, once per database, cached.
