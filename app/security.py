from __future__ import annotations

from fastapi import Header, HTTPException


def require_api_key(configured_key: str | None, provided_key: str | None) -> None:
    if not configured_key:
        return
    if not provided_key or provided_key != configured_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def api_key_dependency(x_api_key: str | None = Header(default=None)) -> str | None:
    return x_api_key
