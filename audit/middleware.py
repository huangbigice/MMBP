"""
FastAPI middleware: audit key API calls (who, when, symbol, params, status, result summary).
Records client_ip, X-Client-Id, optional X-Api-Key hash; does not store full request/response.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from audit.audit_logger import AuditLogger, STOCK_PATH_PATTERN, is_audited_path

# Max symbols to store in body_summary for portfolio
MAX_SYMBOLS_IN_LOG = 10


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


def _body_summary_from_json(body_bytes: bytes, path: str, method: str) -> dict[str, Any]:
    """Extract safe, small summary from request body (POST only)."""
    if method != "POST" or not body_bytes:
        return {}
    try:
        data = json.loads(body_bytes)
    except (json.JSONDecodeError, TypeError):
        return {}
    out: dict[str, Any] = {}
    if path == "/api/v1/predict":
        if isinstance(data, dict):
            out["symbol"] = data.get("symbol")
            out["period"] = data.get("period")
    elif path == "/api/v1/backtest/portfolio":
        if isinstance(data, dict):
            syms = data.get("symbols")
            if isinstance(syms, list):
                out["symbols_count"] = len(syms)
                out["symbols"] = syms[:MAX_SYMBOLS_IN_LOG]
            out["start"] = data.get("start")
            out["end"] = data.get("end")
            out["target_vol_annual"] = data.get("target_vol_annual")
            out["vol_lookback"] = data.get("vol_lookback")
            out["max_leverage"] = data.get("max_leverage")
            out["target_portfolio_vol_annual"] = data.get("target_portfolio_vol_annual")
            out["max_single_weight"] = data.get("max_single_weight")
    return out


def _query_summary(request: Request, path: str) -> dict[str, Any]:
    """Safe query params for audited GET routes."""
    out: dict[str, Any] = {}
    if path == "/api/v1/backtest":
        q = request.query_params
        out["symbol"] = q.get("symbol")
        out["start"] = q.get("start")
        out["end"] = q.get("end")
        out["use_vol_targeting"] = q.get("use_vol_targeting")
        out["target_vol_annual"] = q.get("target_vol_annual")
        out["vol_lookback"] = q.get("vol_lookback")
        out["max_leverage"] = q.get("max_leverage")
    elif "/api/v1/stock/" in path and ("/indicators" in path or "/data" in path):
        out["period"] = request.query_params.get("period", "10y")
    return out


def _path_symbol(path: str) -> str | None:
    m = STOCK_PATH_PATTERN.match(path)
    return m.group(1) if m else None


def _result_summary_from_response(
    body_bytes: bytes, path: str, status_code: int
) -> dict[str, Any]:
    """Extract small result summary; never store full data/indicators."""
    out: dict[str, Any] = {}
    if status_code >= 400:
        try:
            data = json.loads(body_bytes)
            if isinstance(data, dict) and "detail" in data:
                out["detail"] = (
                    data["detail"][:500] if isinstance(data["detail"], str) else str(data["detail"])[:500]
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return out
    try:
        data = json.loads(body_bytes)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(data, dict):
        return out
    if path == "/api/v1/predict":
        out["symbol"] = data.get("symbol")
        out["recommendation"] = data.get("recommendation")
        out["system_score"] = data.get("system_score")
        out["proba_buy"] = data.get("proba_buy")
    elif path in ("/api/v1/backtest", "/api/v1/backtest/portfolio"):
        out["symbol"] = data.get("symbol")
        out["start"] = data.get("start")
        out["end"] = data.get("end")
        out["annualized_return"] = data.get("annualized_return")
        out["volatility"] = data.get("volatility")
        out["max_drawdown"] = data.get("max_drawdown")
        out["trade_count"] = data.get("trade_count")
        out["sharpe_ratio"] = data.get("sharpe_ratio")
    elif "/api/v1/stock/" in path and ("/indicators" in path or "/data" in path):
        out["symbol"] = data.get("symbol")
        out["period"] = data.get("period")
        out["rows"] = data.get("rows")
        if body_bytes:
            out["response_hash"] = hashlib.sha256(body_bytes).hexdigest()[:16]
    return out


class AuditMiddleware(BaseHTTPMiddleware):
    """Log audited API calls to JSONL: ip, client_id, symbol/params, status, latency, result summary."""

    def __init__(self, app: Any, audit_logger: AuditLogger):
        super().__init__(app)
        self._audit = audit_logger

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if not is_audited_path(path):
            return await call_next(request)

        request_id = str(uuid.uuid4())
        client_ip = _client_ip(request)
        client_id = request.headers.get("x-client-id") or ""
        api_key_raw = request.headers.get("x-api-key") or ""
        api_key_hash = self._audit.hash_api_key(api_key_raw) if api_key_raw else ""

        # For POST, read body once and re-inject so route still receives it
        body_bytes: bytes = b""
        if request.method == "POST" and path in ("/api/v1/predict", "/api/v1/backtest/portfolio"):
            body_bytes = await request.body()

            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request = Request(scope=request.scope, receive=receive)

        start_ts = time.time()
        start_perf = time.perf_counter()
        status_code = 500
        response_body: bytes = b""
        error_type = ""
        error_message = ""

        try:
            response = await call_next(request)
            status_code = response.status_code
            response_body = getattr(response, "body", b"") or b""
            return response
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)[:500]
            if isinstance(e, HTTPException):
                status_code = e.status_code
            raise
        finally:
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            ts = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()

            body_summary = _body_summary_from_json(body_bytes, path, request.method)
            query_summary = _query_summary(request, path) if not body_summary else {}
            symbol_from_path = _path_symbol(path)
            result_summary = _result_summary_from_response(response_body, path, status_code)

            entry: dict[str, Any] = {
                "ts": ts,
                "request_id": request_id,
                "client_ip": client_ip,
                "client_id": client_id,
                "api_key_hash": api_key_hash or None,
                "method": request.method,
                "path": path,
                "query_summary": query_summary or None,
                "body_summary": body_summary or None,
                "symbol": (
                    body_summary.get("symbol")
                    or query_summary.get("symbol")
                    or symbol_from_path
                    or result_summary.get("symbol")
                ),
                "symbols": body_summary.get("symbols"),
                "status_code": status_code,
                "latency_ms": latency_ms,
                "result_summary": result_summary or None,
                "error_type": error_type or None,
                "error_message": error_message or None,
            }
            # Prune None values for cleaner JSONL
            entry = {k: v for k, v in entry.items() if v is not None}
            try:
                self._audit.log(entry)
            except Exception:
                pass  # do not fail the request if audit write fails
