from __future__ import annotations

import queue
import threading
import time

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager
from tools import async_delegation, background_events
from tools.process_registry import ProcessRegistry, format_process_notification


def _context(name: str = "event-publisher") -> PluginContext:
    return PluginContext(PluginManifest(name=name), PluginManager())


@pytest.fixture
def isolated_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path):
    db_path = tmp_path / "state.db"
    target_queue: queue.Queue = queue.Queue()
    monkeypatch.setattr(async_delegation, "_db_path", lambda: db_path)
    monkeypatch.setattr(background_events, "_db_path", lambda: db_path)
    monkeypatch.setattr(
        "tools.process_registry.process_registry.completion_queue", target_queue
    )
    return target_queue


def test_capture_background_event_route_snapshots_current_session() -> None:
    tokens = set_session_vars(
        platform="telegram",
        source="gateway",
        chat_id="chat-1",
        chat_type="dm",
        thread_id="thread-1",
        user_id="user-1",
        user_name="Ryan",
        scope_id="scope-1",
        session_key="telegram:dm:chat-1",
        session_id="parent-1",
        ui_session_id="ui-1",
        message_id="message-1",
        profile="default",
    )
    try:
        route = _context().capture_background_event_route(parent_session_id="parent-1")
    finally:
        clear_session_vars(tokens)

    assert route == {
        "session_key": "telegram:dm:chat-1",
        "origin_ui_session_id": "ui-1",
        "origin_session_id": "",
        "parent_session_id": "parent-1",
        "platform": "telegram",
        "chat_id": "chat-1",
        "chat_type": "dm",
        "thread_id": "thread-1",
        "message_id": "message-1",
        "scope_id": "scope-1",
        "user_id": "user-1",
        "user_name": "Ryan",
    }


def test_publish_background_event_is_durable_idempotent_and_routable(
    isolated_ledger: queue.Queue,
) -> None:
    ctx = _context("claude-sessions")
    route = {
        "session_key": "telegram:dm:chat-1",
        "parent_session_id": "parent-1",
        "platform": "telegram",
        "chat_id": "chat-1",
        "chat_type": "dm",
    }

    first = ctx.publish_background_event(
        event_id="claude-a1b2c3d4-permission-1",
        kind="permission_required",
        message="Managed Claude session a1b2c3d4 needs a permission decision.",
        route=route,
        producer_id="a1b2c3d4",
        payload={"request_id": "f" * 32},
    )
    second = ctx.publish_background_event(
        event_id="claude-a1b2c3d4-permission-1",
        kind="permission_required",
        message="Managed Claude session a1b2c3d4 needs a permission decision.",
        route=route,
        producer_id="a1b2c3d4",
        payload={"request_id": "f" * 32},
    )

    assert first == {
        "ok": True,
        "event_id": "claude-a1b2c3d4-permission-1",
        "status": "published",
    }
    assert second == {
        "ok": True,
        "event_id": "claude-a1b2c3d4-permission-1",
        "status": "pending",
    }

    event = isolated_ledger.get_nowait()
    assert event["type"] == "background_event"
    assert event["producer"] == "claude-sessions"
    assert event["producer_id"] == "a1b2c3d4"
    assert event["kind"] == "permission_required"
    assert event["session_key"] == "telegram:dm:chat-1"
    assert event["parent_session_id"] == "parent-1"
    assert isolated_ledger.empty()

    ledger_id = "plugin:claude-sessions:claude-a1b2c3d4-permission-1"
    record = background_events.get_background_event(ledger_id)
    assert record is not None
    assert record["delivery_state"] == "pending"
    assert async_delegation.get_durable_delegation(ledger_id) is None

    claim = async_delegation.claim_event_delivery(event, "test")
    assert claim
    async_delegation.complete_event_delivery(event, claim)

    record = background_events.get_background_event(ledger_id)
    assert record is not None
    assert record["delivery_state"] == "delivered"

    restored: queue.Queue = queue.Queue()
    assert background_events.restore_undelivered_events(restored) == 0
    assert restored.empty()


def test_publish_background_event_keeps_manager_profile_in_background_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    profile_home = tmp_path / "profile-home"
    manager = PluginManager(scope_key=str(profile_home))
    ctx = PluginContext(PluginManifest(name="claude-sessions"), manager)
    observed_homes = []

    def publish(**_kwargs):
        observed_homes.append(get_hermes_home())
        return {"ok": True, "status": "published"}

    monkeypatch.setattr(background_events, "publish_background_event", publish)

    worker = threading.Thread(
        target=lambda: ctx.publish_background_event(
            event_id="event-1",
            kind="completed",
            message="done",
            route={"session_key": "session-1"},
        )
    )
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert observed_homes == [profile_home]


def test_restore_drops_corrupt_background_event_payload(
    isolated_ledger: queue.Queue,
) -> None:
    ctx = _context()
    ctx.publish_background_event(
        event_id="event-corrupt",
        kind="completed",
        message="done",
        route={"session_key": "session-1"},
    )
    isolated_ledger.get_nowait()

    ledger_id = "plugin:event-publisher:event-corrupt"
    with background_events._transaction() as conn:
        conn.execute(
            "UPDATE background_events SET event_json=? WHERE ledger_id=?",
            ("{not-json", ledger_id),
        )

    restored: queue.Queue = queue.Queue()
    assert background_events.restore_undelivered_events(restored) == 0
    assert restored.empty()
    record = background_events.get_background_event(ledger_id)
    assert record is not None
    assert record["delivery_state"] == "dropped"


def test_restore_replays_dead_claim_without_stealing_live_claim(
    isolated_ledger: queue.Queue, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context("claude-sessions")
    ctx.publish_background_event(
        event_id="claimed-event",
        kind="completed",
        message="done",
        route={"session_key": "session-1"},
    )
    event = isolated_ledger.get_nowait()
    claim = background_events.claim_event_delivery(event, "gateway")
    assert claim

    restored: queue.Queue = queue.Queue()
    monkeypatch.setattr(background_events, "_claim_owner_is_live", lambda *_: True)
    assert background_events.restore_undelivered_events(restored) == 0
    assert restored.empty()

    monkeypatch.setattr(background_events, "_claim_owner_is_live", lambda *_: False)
    assert background_events.restore_undelivered_events(restored) == 1
    replay = restored.get_nowait()
    assert replay["restored"] is True
    assert background_events.claim_event_delivery(replay, "gateway-restart")


def test_claim_owner_liveness_protects_fresh_unknown_owners() -> None:
    now = time.time()

    assert background_events._claim_owner_is_live("unknown", now, now) is True
    assert background_events._claim_owner_is_live("unknown", None, now) is False
    assert (
        background_events._claim_owner_is_live(
            "gateway:2147483647:claim",
            now - background_events._CLAIM_LEASE_SECONDS - 1,
            now,
        )
        is False
    )
    assert (
        background_events._claim_owner_is_live("gateway:2147483647:claim", now, now)
        is False
    )


def test_unledgered_background_event_cannot_claim_delivery(
    isolated_ledger: queue.Queue,
) -> None:
    event = {
        "type": "background_event",
        "delegation_id": "plugin:claude-sessions:missing",
    }

    assert background_events.claim_event_delivery(event, "gateway") is None
    assert isolated_ledger.empty()


def test_forged_background_event_cannot_reuse_pending_ledger_id(
    isolated_ledger: queue.Queue,
) -> None:
    ctx = _context("claude-sessions")
    ctx.publish_background_event(
        event_id="real-event",
        kind="completed",
        message="canonical message",
        route={"session_key": "session-1"},
        producer_id="real-producer",
    )
    canonical = isolated_ledger.get_nowait()
    forged = {
        **canonical,
        "message": "forged message",
        "producer_id": "forged-producer",
        "session_key": "attacker-session",
    }

    assert background_events.claim_event_delivery(forged, "gateway") is None
    record = background_events.get_background_event(canonical["delegation_id"])
    assert record is not None
    assert record["delivery_state"] == "pending"
    assert record["delivery_attempts"] == 0

    claim = background_events.claim_event_delivery(canonical, "gateway")
    assert claim
    assert background_events.complete_event_delivery(canonical, claim)


def test_publish_background_event_rejects_unbounded_or_unrouted_input(
    isolated_ledger: queue.Queue,
) -> None:
    ctx = _context()

    with pytest.raises(ValueError, match="route"):
        ctx.publish_background_event(
            event_id="event-1",
            kind="completed",
            message="done",
            route={},
        )
    with pytest.raises(ValueError, match="event_id"):
        ctx.publish_background_event(
            event_id="bad id",
            kind="completed",
            message="done",
            route={"session_key": "session-1"},
        )
    with pytest.raises(ValueError, match="message"):
        ctx.publish_background_event(
            event_id="event-1",
            kind="completed",
            message="x" * 8193,
            route={"session_key": "session-1"},
        )
    with pytest.raises(ValueError, match="route field"):
        ctx.publish_background_event(
            event_id="event-1",
            kind="completed",
            message="done",
            route={"session_key": "x" * 4097},
        )
    with pytest.raises(ValueError, match="route exceeds"):
        ctx.publish_background_event(
            event_id="event-1",
            kind="completed",
            message="done",
            route={
                "session_key": "a" * 3500,
                "origin_ui_session_id": "b" * 3500,
                "chat_id": "c" * 3500,
                "thread_id": "d" * 3500,
                "user_name": "e" * 3500,
            },
        )
    with pytest.raises(ValueError, match="payload"):
        ctx.publish_background_event(
            event_id="event-1",
            kind="completed",
            message="done",
            route={"session_key": "session-1"},
            payload=[],
        )

    assert isolated_ledger.empty()


def test_process_registry_formats_and_requires_ownership_for_background_event() -> None:
    registry = ProcessRegistry()
    registry.completion_queue.put({
        "type": "background_event",
        "delegation_id": "claude-sessions:event-1",
        "event_id": "event-1",
        "producer": "claude-sessions",
        "producer_id": "a1b2c3d4",
        "kind": "completed",
        "message": "Managed Claude session completed; inspect and verify it.",
        "session_key": "session-a",
    })

    assert registry.drain_notifications(session_key="session-b") == []
    event, text = registry.drain_notifications(session_key="session-a")[0]

    assert event["event_id"] == "event-1"
    assert "claude-sessions" in text
    assert "completed" in text
    assert "inspect and verify" in text
    assert format_process_notification(event) == text
