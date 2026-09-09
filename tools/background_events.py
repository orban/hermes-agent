"""Durable host wake events published by Hermes plugins.

Plugins use :class:`hermes_cli.plugins.PluginContext` to capture a serializable
return route while handling a tool call, then publish bounded semantic events
from supervised background work. Events enter the shared process notification
queue, which already creates fresh internal turns at safe conversation
boundaries.

This module deliberately owns a ledger separate from ``async_delegations``.
Plugin events and delegated-agent results share delivery consumers, but they
have different lifecycle and retention semantics and must not share storage.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EVENT_TOKEN = re.compile(r"^[A-Za-z0-9._:-]+$")
_MAX_MESSAGE_CHARS = 8_192
_MAX_PAYLOAD_BYTES = 16_384
_MAX_ROUTE_FIELD_CHARS = 4_096
_MAX_ROUTE_BYTES = 16_384
_MAX_DELIVERY_ATTEMPTS = 8
_MAX_REPLAY_AGE_S = 48 * 3600.0
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_RETAINED_EVENTS = 500
_MAX_PENDING_EVENTS = 1_000
_CLAIM_LEASE_SECONDS = 300.0
_ROUTE_FIELDS = (
    "session_key",
    "origin_ui_session_id",
    "origin_session_id",
    "parent_session_id",
    "platform",
    "chat_id",
    "chat_type",
    "thread_id",
    "message_id",
    "scope_id",
    "user_id",
    "user_name",
)
_DB_LOCK = threading.Lock()


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state_repair import apply_durability_barriers

    apply_durability_barriers(conn)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS background_events (
            ledger_id TEXT PRIMARY KEY,
            plugin_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            delivery_claim TEXT,
            delivery_claimed_at REAL,
            UNIQUE(plugin_id, event_id)
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_background_events_delivery
           ON background_events(delivery_state, created_at)"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Commit or roll back a short transaction and always close its connection."""
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _current_origin_session_id() -> str:
    """Return the originating api_server request id when one is bound."""
    try:
        from gateway.session_context import get_session_env

        if get_session_env("HERMES_SESSION_PLATFORM", "") != "api_server":
            return ""
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or ""
    except Exception:
        return ""


def capture_background_event_route(*, parent_session_id: str = "") -> Dict[str, str]:
    """Snapshot the current turn's serializable return address."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return {}

    route = {
        "session_key": get_session_env("HERMES_SESSION_KEY", ""),
        "origin_ui_session_id": get_session_env("HERMES_UI_SESSION_ID", ""),
        "origin_session_id": _current_origin_session_id(),
        "parent_session_id": (
            parent_session_id or get_session_env("HERMES_SESSION_ID", "")
        ),
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "chat_type": get_session_env("HERMES_SESSION_CHAT_TYPE", ""),
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
        "message_id": get_session_env("HERMES_SESSION_MESSAGE_ID", ""),
        "scope_id": get_session_env("HERMES_SESSION_SCOPE_ID", ""),
        "user_id": get_session_env("HERMES_SESSION_USER_ID", ""),
        "user_name": get_session_env("HERMES_SESSION_USER_NAME", ""),
    }
    return {key: str(value) for key, value in route.items()}


def _validate_token(
    field: str, value: str, limit: int, *, optional: bool = False
) -> None:
    if optional and value == "":
        return
    if not isinstance(value, str) or not value or len(value) > limit:
        qualifier = "" if optional else "non-empty "
        raise ValueError(
            f"{field} must be a {qualifier}string of at most {limit} characters"
        )
    if not _EVENT_TOKEN.fullmatch(value):
        raise ValueError(f"{field} contains unsupported characters")


def publish_background_event(
    *,
    plugin_id: str,
    event_id: str,
    kind: str,
    message: str,
    route: Dict[str, Any],
    producer_id: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist and enqueue one plugin-produced event idempotently."""
    _validate_token("plugin_id", plugin_id, 64)
    _validate_token("event_id", event_id, 128)
    _validate_token("kind", kind, 64)
    _validate_token("producer_id", producer_id, 128, optional=True)

    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if len(message) > _MAX_MESSAGE_CHARS:
        raise ValueError(f"message exceeds {_MAX_MESSAGE_CHARS} characters")
    if not isinstance(route, dict):
        raise ValueError("route must be a mapping")

    clean_route = {field: str(route.get(field) or "") for field in _ROUTE_FIELDS}
    if any(len(value) > _MAX_ROUTE_FIELD_CHARS for value in clean_route.values()):
        raise ValueError(f"route field exceeds {_MAX_ROUTE_FIELD_CHARS} characters")
    encoded_route = json.dumps(clean_route, ensure_ascii=False, sort_keys=True)
    if len(encoded_route.encode("utf-8")) > _MAX_ROUTE_BYTES:
        raise ValueError(f"route exceeds {_MAX_ROUTE_BYTES} bytes")
    if not any(
        clean_route[field]
        for field in ("session_key", "origin_ui_session_id", "origin_session_id")
    ):
        raise ValueError("route has no routable session identity")

    clean_payload = {} if payload is None else payload
    if not isinstance(clean_payload, dict):
        raise ValueError("payload must be a mapping")
    encoded_payload = json.dumps(clean_payload, ensure_ascii=False, sort_keys=True)
    if len(encoded_payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {_MAX_PAYLOAD_BYTES} bytes")

    ledger_id = f"plugin:{plugin_id}:{event_id}"
    now = time.time()
    event: Dict[str, Any] = {
        "type": "background_event",
        "delegation_id": ledger_id,
        "event_id": event_id,
        "producer": plugin_id,
        "producer_id": producer_id,
        "kind": kind,
        "message": message.strip(),
        "payload": clean_payload,
        "status": "completed",
        "dispatched_at": now,
        "completed_at": now,
        **clean_route,
    }
    encoded_event = json.dumps(event, ensure_ascii=False, sort_keys=True)

    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO background_events
               (ledger_id, plugin_id, event_id, event_json, created_at,
                updated_at, delivery_state, delivery_attempts)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0)""",
            (ledger_id, plugin_id, event_id, encoded_event, now, now),
        )
        inserted = cursor.rowcount == 1
        row = conn.execute(
            "SELECT delivery_state FROM background_events WHERE ledger_id=?",
            (ledger_id,),
        ).fetchone()

    _prune_records()
    if inserted:
        from tools.process_registry import process_registry

        process_registry.completion_queue.put(event)
        status = "published"
    else:
        status = str(row[0] if row else "pending")
    return {"ok": True, "event_id": event_id, "status": status}


def _prune_records() -> None:
    """Bound terminal history and the number of undelivered plugin events."""
    now = time.time()
    cutoff = now - _RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """DELETE FROM background_events
               WHERE delivery_state IN ('delivered', 'dropped') AND updated_at < ?""",
            (cutoff,),
        )
        terminal_count = conn.execute(
            """SELECT COUNT(*) FROM background_events
               WHERE delivery_state IN ('delivered', 'dropped')"""
        ).fetchone()[0]
        excess = max(0, terminal_count - _MAX_RETAINED_EVENTS)
        if excess:
            conn.execute(
                """DELETE FROM background_events WHERE ledger_id IN (
                     SELECT ledger_id FROM background_events
                     WHERE delivery_state IN ('delivered', 'dropped')
                     ORDER BY updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )

        pending_count = conn.execute(
            """SELECT COUNT(*) FROM background_events
               WHERE delivery_state='pending'"""
        ).fetchone()[0]
        overflow = max(0, pending_count - _MAX_PENDING_EVENTS)
        if overflow:
            conn.execute(
                """UPDATE background_events SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL,
                          updated_at=?
                   WHERE ledger_id IN (
                     SELECT ledger_id FROM background_events
                     WHERE delivery_state='pending'
                     ORDER BY created_at ASC LIMIT ?
                   )""",
                (now, overflow),
            )


def _claim_owner_is_live(claim_id: str, claimed_at: Any, now: float) -> bool:
    """Return whether a fresh delivery claim still belongs to a live process."""
    try:
        claimed_timestamp = float(claimed_at)
    except (TypeError, ValueError):
        return False
    if claimed_timestamp < now - _CLAIM_LEASE_SECONDS:
        return False

    parts = str(claim_id).rsplit(":", 2)
    if len(parts) != 3:
        return True
    try:
        owner_pid = int(parts[1])
    except ValueError:
        return True

    try:
        import psutil  # type: ignore
    except Exception:
        return True

    try:
        process = psutil.Process(owner_pid)
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except Exception:
        # A fresh claim whose owner cannot be inspected remains protected until
        # its short lease expires; this avoids stealing from another profile
        # process merely because process metadata is temporarily unreadable.
        return True


def restore_undelivered_events(target_queue) -> int:
    """Replay recent pending plugin events into the shared notification queue."""
    now = time.time()
    restored = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT ledger_id, event_json, created_at, delivery_claim,
                      delivery_claimed_at
               FROM background_events
               WHERE delivery_state='pending'
               ORDER BY created_at, ledger_id"""
        ).fetchall()
        for ledger_id, payload, created_at, claim_id, claimed_at in rows:
            if created_at and (now - created_at) > _MAX_REPLAY_AGE_S:
                conn.execute(
                    """UPDATE background_events SET delivery_state='dropped',
                              delivery_claim=NULL, delivery_claimed_at=NULL,
                              updated_at=?
                       WHERE ledger_id=? AND delivery_state='pending'""",
                    (now, ledger_id),
                )
                logger.warning(
                    "Background event %s is %.1fh old (cap %.1fh); dropping "
                    "restart replay while retaining the record.",
                    ledger_id,
                    (now - created_at) / 3600.0,
                    _MAX_REPLAY_AGE_S / 3600.0,
                )
                continue
            if claim_id:
                if _claim_owner_is_live(str(claim_id), claimed_at, now):
                    continue
                released = conn.execute(
                    """UPDATE background_events SET delivery_claim=NULL,
                              delivery_claimed_at=NULL, updated_at=?
                       WHERE ledger_id=? AND delivery_state='pending'
                         AND delivery_claim=?""",
                    (now, ledger_id, claim_id),
                )
                if released.rowcount != 1:
                    continue
            try:
                event = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                event = None
            if not isinstance(event, dict):
                logger.warning(
                    "Background event %s has an invalid payload; dropping it",
                    ledger_id,
                )
                conn.execute(
                    """UPDATE background_events SET delivery_state='dropped',
                              updated_at=? WHERE ledger_id=?""",
                    (now, ledger_id),
                )
                continue
            event["restored"] = True
            target_queue.put(event)
            restored += 1
    return restored


def claim_event_delivery(event: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a pending plugin event whose queued body matches the durable ledger."""
    ledger_id = str(event.get("delegation_id") or "")
    if not ledger_id:
        return ""
    claim_id = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state, event_json FROM background_events WHERE ledger_id=?",
            (ledger_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            durable_event = json.loads(row[1])
        except (TypeError, json.JSONDecodeError):
            durable_event = None
        if not isinstance(durable_event, dict):
            logger.warning(
                "Background event %s has an invalid durable payload; dropping it",
                ledger_id,
            )
            conn.execute(
                """UPDATE background_events SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL,
                          updated_at=? WHERE ledger_id=?""",
                (now, ledger_id),
            )
            return None

        queued_event = dict(event)
        if queued_event.get("restored") is True:
            queued_event.pop("restored")
        if queued_event != durable_event:
            logger.warning(
                "Refusing background event %s because its queued payload does not "
                "match the durable ledger",
                ledger_id,
            )
            return None
        cursor = conn.execute(
            """UPDATE background_events SET delivery_claim=?,
                      delivery_claimed_at=?, delivery_attempts=delivery_attempts+1,
                      updated_at=?
               WHERE ledger_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, ledger_id, now - _CLAIM_LEASE_SECONDS),
        )
        return claim_id if cursor.rowcount == 1 else None


def complete_event_delivery(event: Dict[str, Any], claim_id: str) -> bool:
    """Acknowledge delivery by the consumer that owns ``claim_id``."""
    if not claim_id:
        return False
    ledger_id = str(event.get("delegation_id") or "")
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE background_events SET delivery_state='delivered',
                      delivered_at=?, updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE ledger_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, now, ledger_id, claim_id),
        )
        return cursor.rowcount == 1


def release_event_delivery(event: Dict[str, Any], claim_id: str) -> bool:
    """Release a failed claim, dropping events that exhaust their retry budget."""
    if not claim_id:
        return False
    ledger_id = str(event.get("delegation_id") or "")
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        capped = conn.execute(
            """UPDATE background_events SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
               WHERE ledger_id=? AND delivery_state='pending'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, ledger_id, claim_id, _MAX_DELIVERY_ATTEMPTS),
        )
        if capped.rowcount == 1:
            logger.warning(
                "Background event %s exhausted %d delivery attempts; dropping it.",
                ledger_id,
                _MAX_DELIVERY_ATTEMPTS,
            )
            return True
        cursor = conn.execute(
            """UPDATE background_events SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=?
               WHERE ledger_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, ledger_id, claim_id),
        )
        return cursor.rowcount == 1


def defer_event_delivery(event: Dict[str, Any], claim_id: str) -> bool:
    """Return an unadmitted event to pending without spending an attempt."""
    if not claim_id:
        return False
    ledger_id = str(event.get("delegation_id") or "")
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE background_events SET delivery_claim=NULL,
                      delivery_claimed_at=NULL,
                      delivery_attempts=MAX(0, delivery_attempts-1),
                      updated_at=?
               WHERE ledger_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, ledger_id, claim_id),
        )
        return cursor.rowcount == 1


def drop_event_delivery(event: Dict[str, Any], claim_id: str) -> bool:
    """Terminally drop an event whose originating session no longer exists."""
    if not claim_id:
        return False
    ledger_id = str(event.get("delegation_id") or "")
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE background_events SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL,
                      updated_at=?
               WHERE ledger_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, ledger_id, claim_id),
        )
        return cursor.rowcount == 1


def get_background_event(ledger_id: str) -> Optional[Dict[str, Any]]:
    """Return one durable event record for diagnostics and tests."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT plugin_id, event_id, event_json, created_at, updated_at,
                      delivery_state, delivery_attempts, delivered_at
               FROM background_events WHERE ledger_id=?""",
            (ledger_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        event = json.loads(row[2])
    except (TypeError, json.JSONDecodeError):
        event = None
    return {
        "ledger_id": ledger_id,
        "plugin_id": row[0],
        "event_id": row[1],
        "event": event,
        "created_at": row[3],
        "updated_at": row[4],
        "delivery_state": row[5],
        "delivery_attempts": row[6],
        "delivered_at": row[7],
    }
