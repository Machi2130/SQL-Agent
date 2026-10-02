"""
cache.py — Redis layer with in-memory fallback.

If Redis is not configured or unreachable, all operations silently
fall back to a process-local dict so the app still works for dev/testing.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

# ── Connection ────────────────────────────────────────────────

_client = None
_redis_ok = None
_memory_store: dict[str, Any] = {}
_memory_expiry: dict[str, float] = {}


def _is_redis_configured() -> bool:
    pw = os.getenv("REDIS_PASSWORD", "")
    configured = bool(pw) and "PASTE" not in pw
    log.debug("Redis configured: %s", configured)
    return configured


def get_redis():
    global _client, _redis_ok
    if _redis_ok is False:
        return None
    if _client is None:
        if not _is_redis_configured() or redis_lib is None:
            _redis_ok = False
            log.info("Redis not configured — using in-memory fallback")
            return None
        try:
            _client = redis_lib.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                username=os.getenv("REDIS_USERNAME", "default"),
                password=os.getenv("REDIS_PASSWORD", ""),
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
            )
            _client.ping()
            _redis_ok = True
            log.info("Redis connected")
        except Exception as e:
            log.warning("Redis connection failed: %s — using in-memory fallback", e)
            _client = None
            _redis_ok = False
            return None
    return _client


def ping() -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        return r.ping()
    except Exception:
        return False


# ── Fallback helpers ─────────────────────────────────────────

def _mem_get(key: str) -> str | None:
    exp = _memory_expiry.get(key)
    if exp and time.time() > exp:
        _memory_store.pop(key, None)
        _memory_expiry.pop(key, None)
        return None
    return _memory_store.get(key)


def _mem_set(key: str, value: str, ttl: int = 3600) -> None:
    _memory_store[key] = value
    _memory_expiry[key] = time.time() + ttl


def _mem_delete(key: str) -> None:
    _memory_store.pop(key, None)
    _memory_expiry.pop(key, None)


# ── TTL constants ─────────────────────────────────────────────

SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 3600))
CACHE_TTL   = int(os.getenv("CACHE_TTL_SECONDS",   86400))
RATE_WINDOW = 60
RATE_LIMIT  = int(os.getenv("RATE_LIMIT_PER_MINUTE", 30))


# ═══════════════════════════════════════════════════════════════
#  SESSION
# ═══════════════════════════════════════════════════════════════

def get_session(user_id: str) -> dict | None:
    r = get_redis()
    if r:
        raw = r.get(f"session:{user_id}")
    else:
        raw = _mem_get(f"session:{user_id}")
    return json.loads(raw) if raw else None


def set_session(user_id: str, data: dict) -> None:
    payload = json.dumps(data, default=str)
    r = get_redis()
    if r:
        r.setex(f"session:{user_id}", SESSION_TTL, payload)
    else:
        _mem_set(f"session:{user_id}", payload, SESSION_TTL)


def refresh_session(user_id: str) -> None:
    r = get_redis()
    if r:
        r.expire(f"session:{user_id}", SESSION_TTL)


def delete_session(user_id: str) -> None:
    r = get_redis()
    if r:
        r.delete(f"session:{user_id}")
    else:
        _mem_delete(f"session:{user_id}")


# ═══════════════════════════════════════════════════════════════
#  QUERY CACHE
# ═══════════════════════════════════════════════════════════════

def _qcache_key(user_id: str, db_flavor: str, question: str) -> str:
    h = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"qcache:{user_id}:{db_flavor}:{h}"


def get_cached_query(user_id: str, db_flavor: str, question: str) -> dict | None:
    key = _qcache_key(user_id, db_flavor, question)
    r = get_redis()
    raw = r.get(key) if r else _mem_get(key)
    return json.loads(raw) if raw else None


def set_cached_query(user_id: str, db_flavor: str, question: str, result: dict) -> None:
    key = _qcache_key(user_id, db_flavor, question)
    payload = json.dumps(result, default=str)
    r = get_redis()
    if r:
        r.setex(key, CACHE_TTL, payload)
    else:
        _mem_set(key, payload, CACHE_TTL)


def clear_user_query_cache(user_id: str) -> int:
    r = get_redis()
    if r:
        keys = list(r.scan_iter(f"qcache:{user_id}:*", count=100))
        if keys:
            r.delete(*keys)
        return len(keys)
    else:
        to_del = [k for k in _memory_store if k.startswith(f"qcache:{user_id}:")]
        for k in to_del:
            _mem_delete(k)
        return len(to_del)


# ═══════════════════════════════════════════════════════════════
#  RATE LIMITING
# ═══════════════════════════════════════════════════════════════

_rate_counters: dict[str, tuple[int, float]] = {}


def check_rate_limit(user_id: str) -> tuple[bool, int, int]:
    r = get_redis()
    if r:
        key = f"ratelimit:{user_id}"
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()
        if ttl == -1:
            r.expire(key, RATE_WINDOW)
            ttl = RATE_WINDOW
        return count <= RATE_LIMIT, int(count), max(0, int(ttl))
    else:
        now = time.time()
        count, window_start = _rate_counters.get(user_id, (0, now))
        if now - window_start > RATE_WINDOW:
            count, window_start = 0, now
        count += 1
        _rate_counters[user_id] = (count, window_start)
        ttl = max(0, int(RATE_WINDOW - (now - window_start)))
        return count <= RATE_LIMIT, count, ttl


# ═══════════════════════════════════════════════════════════════
#  CONVERSATION TURNS
# ═══════════════════════════════════════════════════════════════

MAX_HOT_TURNS = 12
_conv_store: dict[str, list] = {}


def push_turn(user_id: str, session_id: str, role: str, content: str) -> None:
    key = f"conv:{user_id}:{session_id}"
    entry = json.dumps({"role": role, "content": content})
    r = get_redis()
    if r:
        r.rpush(key, entry)
        r.ltrim(key, -MAX_HOT_TURNS, -1)
        r.expire(key, SESSION_TTL)
    else:
        lst = _conv_store.setdefault(key, [])
        lst.append(entry)
        if len(lst) > MAX_HOT_TURNS:
            _conv_store[key] = lst[-MAX_HOT_TURNS:]


def get_turns(user_id: str, session_id: str, last_n: int = 6) -> list[dict]:
    key = f"conv:{user_id}:{session_id}"
    r = get_redis()
    if r:
        raw = r.lrange(key, -last_n, -1)
    else:
        raw = _conv_store.get(key, [])[-last_n:]
    return [json.loads(r) for r in raw]


def clear_conversation(user_id: str, session_id: str) -> None:
    key = f"conv:{user_id}:{session_id}"
    r = get_redis()
    if r:
        r.delete(key)
    else:
        _conv_store.pop(key, None)


# ═══════════════════════════════════════════════════════════════
#  SCHEMA CACHE
# ═══════════════════════════════════════════════════════════════

def get_schema(user_id: str) -> dict | None:
    key = f"schema:{user_id}"
    r = get_redis()
    raw = r.get(key) if r else _mem_get(key)
    return json.loads(raw) if raw else None


def set_schema(user_id: str, schema: dict) -> None:
    key = f"schema:{user_id}"
    payload = json.dumps(schema, default=str)
    r = get_redis()
    if r:
        r.setex(key, SESSION_TTL, payload)
    else:
        _mem_set(key, payload, SESSION_TTL)
