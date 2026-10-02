"""
auth.py — Auth0 JWT verification middleware.

Fixes:
  - No authentication (was the #1 critical gap)
  - Groq API key sent in plain text — now stored server-side per-user in Redis
  - No session isolation — g.user_id is set here and used everywhere
"""

import os

import httpx
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

# FastAPI dependency injection pattern
# Used as: user_id: str = Depends(get_current_user)
from fastapi import Depends, HTTPException
from fastapi import status as http_status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer()

AUTH0_DOMAIN   = os.getenv("AUTH0_DOMAIN", "")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "")
DEV_MODE       = os.getenv("DEV_MODE", "false").lower() == "true"

# In-process JWKS cache — refreshed on key-rotation mismatch.
_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache is None:
        resp = httpx.get(
            f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
            timeout=10,
        )
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache


def _find_rsa_key(token: str) -> dict:
    """Match the JWT kid header against the JWKS."""
    jwks = _get_jwks()
    header = jwt.get_unverified_header(token)
    for key in jwks.get("keys", []):
        if key.get("kid") == header.get("kid"):
            return {k: key[k] for k in ("kty", "kid", "use", "n", "e") if k in key}
    # Key not found — JWKS may have rotated; bust the cache and retry once.
    global _jwks_cache
    _jwks_cache = None
    jwks = _get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == header.get("kid"):
            return {k: key[k] for k in ("kty", "kid", "use", "n", "e") if k in key}
    raise ValueError("No matching public key found in JWKS — check AUTH0_DOMAIN.")


def verify_token(token: str) -> str:
    """
    Verify a Bearer JWT and return the user_id (sub claim).

    In DEV_MODE the token value itself is used as the user_id so you can
    test locally without a real Auth0 tenant:
        curl -H "Authorization: Bearer dev-user-123" ...
    """
    if DEV_MODE:
        return token

    if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
        raise RuntimeError(
            "AUTH0_DOMAIN and AUTH0_AUDIENCE must be set (or enable DEV_MODE=true)."
        )

    rsa_key = _find_rsa_key(token)
    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
    except ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except JWTClaimsError as e:
        raise ValueError(f"Token claims invalid: {e}")
    except JWTError as e:
        raise ValueError(f"Token invalid: {e}")

    return payload["sub"]  # e.g. "auth0|abc123"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    FastAPI Depends() pattern — injects user_id into any route as a parameter.

    Flask pattern:   @require_auth sets g.user_id (thread-local, not testable)
    FastAPI pattern: Depends(get_current_user) passes user_id as a typed argument.
                     Testable by overriding: app.dependency_overrides[get_current_user] = lambda: "test-user"
    """
    try:
        return verify_token(credentials.credentials)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail=f"Auth error: {e}")


