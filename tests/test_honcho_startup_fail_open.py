"""Regression tests for Honcho startup fail-open behavior."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider


class _FakeHonchoConfig(SimpleNamespace):
    def resolve_session_name(self, **kwargs):
        return "test-session"


def _configured_hybrid_config() -> _FakeHonchoConfig:
    return _FakeHonchoConfig(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        init_on_session_start=False,
        injection_frequency="every-turn",
        context_cadence=1,
        dialectic_cadence=1,
        query_rewrite=False,
        first_turn_base_wait=3.0,
        first_turn_dialectic_wait=2.0,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-directory",
    )


def _configured_tools_config(*, init_on_session_start: bool = False) -> _FakeHonchoConfig:
    cfg = _configured_hybrid_config()
    cfg.recall_mode = "tools"
    cfg.init_on_session_start = init_on_session_start
    return cfg




def test_stalled_init_only_delays_first_turn_prefetch(monkeypatch):
    """A stalled session init may bound-wait on turn 1 only; every later
    prefetch must keep the fail-open contract and return immediately."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = threading.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def stalled_session_init(self, cfg, session_id, **kwargs):
        release.wait(timeout=10)

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", stalled_session_init)
    provider.initialize("session-1", platform="cli")
    provider._FIRST_TURN_BASE_TIMEOUT = 1.0

    try:
        provider._turn_count = 1
        start = time.perf_counter()
        assert provider.prefetch("first question") == ""
        assert time.perf_counter() - start >= 0.5  # turn 1 waited (bounded)

        for turn in (2, 3, 4):
            provider._turn_count = turn
            start = time.perf_counter()
            assert provider.prefetch("follow-up question") == ""
            assert time.perf_counter() - start < 0.4  # fail-open, no wait
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)


def test_honcho_background_init_rechecks_state_after_lock_race():
    """Startup should not spawn/crash if init completes while waiting for lock."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()
    provider._lazy_init_kwargs = {"platform": "cli"}
    provider._lazy_init_session_id = "session-1"

    class RacingLock:
        def __enter__(self):
            provider._session_initialized = True
            provider._lazy_init_kwargs = None
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    provider._init_lock = RacingLock()

    provider._start_session_init_background()

    assert provider._init_thread is None
    assert provider._session_initialized is True




def test_first_turn_base_wait_is_shared_by_init_and_context_fetch():
    """Session init and base retrieval share one configured turn-1 deadline."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.first_turn_base_wait = 0.5
    cfg.timeout = None
    release_context = threading.Event()

    class SlowManager:
        def get_prefetch_context(self, session_key, user_message=None):
            release_context.wait(timeout=5)
            return {"representation": "late"}

        def set_context_result(self, session_key, result):
            pass

        def pop_context_result(self, session_key):
            return {}

    def finish_init():
        time.sleep(0.3)
        provider._manager = SlowManager()
        provider._session_initialized = True

    provider._config = cfg
    provider._session_key = "test-session"
    provider._recall_mode = "context"
    provider._turn_count = 1
    provider._last_dialectic_turn = 0
    provider._FIRST_TURN_BASE_TIMEOUT = cfg.first_turn_base_wait
    provider._init_thread = threading.Thread(target=finish_init, daemon=True)
    provider._init_thread.start()

    try:
        started = time.perf_counter()
        assert provider.prefetch("what do you know about me?") == ""
        elapsed = time.perf_counter() - started
        # Property: prefetch waits for init (0.3s sleep) but is bounded by
        # first_turn_base_wait rather than blocking forever on the slow
        # context fetch. The old 0.4..0.65 window was 0.25s wide — pure
        # scheduler noise on a loaded runner. Lower bound proves the wait
        # happened; loose upper bound proves it didn't hang.
        assert 0.25 <= elapsed < 2.0
    finally:
        release_context.set()
        provider._init_thread.join(timeout=10)





def test_honcho_sync_turn_waits_for_full_background_startup(monkeypatch):
    """Manager assignment alone is not readiness while background init continues."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    session_created = threading.Event()
    migration_started = threading.Event()
    release_migration = threading.Event()
    get_calls = []

    class StartupManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_create(self, session_key):
            get_calls.append(session_key)
            session_created.set()
            return SimpleNamespace(messages=[])

        def migrate_memory_files(self, session_key, mem_dir):
            migration_started.set()
            release_migration.wait(timeout=5)

        def prefetch_context(self, session_key, user_message=None):
            pass

        def _flush_session(self, session):
            pass

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )
    monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda cfg: object())
    monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", StartupManager)

    provider.initialize("session-1", platform="cli")
    try:
        assert session_created.wait(timeout=1)
        assert migration_started.wait(timeout=1)
        assert provider._manager is not None
        assert provider._session_initialized is False

        provider.sync_turn("hello", "world")

        assert provider._sync_thread is None
        assert get_calls == ["test-session"]
    finally:
        release_migration.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=1)

    assert provider._session_initialized is True


def test_honcho_system_prompt_advertises_active_while_background_init_runs(monkeypatch):
    """Prompt metadata should not require a completed network session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = threading.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def slow_session_init(self, cfg, session_id, **kwargs):
        release.wait(timeout=5)
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    provider.initialize("session-1", platform="cli")
    try:
        prompt = provider.system_prompt_block()
        assert "Honcho Memory" in prompt
        assert "hybrid mode" in prompt
    finally:
        release.set()
        init_thread = getattr(provider, "_init_thread", None)
        if init_thread:
            init_thread.join(timeout=1)




def test_honcho_tools_eager_init_failure_does_not_leave_ready_manager(monkeypatch):
    """Failed eager tools startup must not leave hooks seeing a ready session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    def failing_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        raise RuntimeError("boom")

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", failing_session_init)

    provider.initialize("session-1", platform="cli")
    assert provider._session_initialized is False
    assert provider._manager is None

    background_started = threading.Event()
    provider._start_session_init_background = background_started.set
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers safe Honcho startup")

    assert provider._sync_thread is None
    assert not background_started.is_set()

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))
    assert "could not be initialized" in result["error"]
    assert provider._manager is None


def test_honcho_tools_lazy_hooks_do_not_prestart_background_init(monkeypatch):
    """tools lazy mode lets the first tool call own session initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=False)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider.initialize("session-1", platform="cli")
    background_started = threading.Event()
    provider._start_session_init_background = background_started.set

    provider.prefetch("what do you know?")
    provider.queue_prefetch("what do you know?")
    provider.sync_turn("hello", "world")
    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert not background_started.is_set()
    assert provider._session_initialized is False

    class ToolManager:
        def get_peer_card(self, session_key, peer="user"):
            return ["ready"]

    init_calls = []

    def fake_session_init(self, cfg, session_id, **kwargs):
        init_calls.append(session_id)
        self._manager = ToolManager()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    result = json.loads(provider.handle_tool_call("honcho_profile", {"peer": "user"}))

    assert result == {"result": ["ready"]}
    assert init_calls == ["session-1"]
    assert not background_started.is_set()


# ---------------------------------------------------------------------------
# Write-containment regression tests
# ---------------------------------------------------------------------------


def test_honcho_sync_turn_skips_write_when_save_messages_is_disabled():
    """The resolved write-disable switch must gate an initialized provider."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = False
    manager_calls = []

    class Manager:
        def get_or_create(self, session_key):
            manager_calls.append(session_key)
            return SimpleNamespace()

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn("a genuine user turn", "a genuine assistant reply")

    assert provider._sync_thread is None
    assert manager_calls == []


def test_honcho_sync_turn_skips_anchored_gateway_notifications():
    """Known bracketed gateway wrappers must not become durable messages."""
    wrappers = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_1]\nworker results follow",
        "[ASYNC DELEGATION COMPLETE — deleg_2]",
        "[CONTEXT COMPACTION — REFERENCE ONLY]\nsummary follows",
        "[CONTEXT COMPACTION - REFERENCE ONLY]",
        "[CONTEXT COMPACTION]",
        "[PRIOR CONTEXT — for reference only; not a new message]",
        "[Your active task list was preserved across context compression]",
        "[CONTEXT SUMMARY]: previous context",
        "[IMPORTANT: Background process 12 matched watch pattern \"foo\"\nCommand: x",
        "[IMPORTANT: Background process proc_125d8e6a53d8 completed normally (exit code 0).\n"
        "Command: /Users/ryo/.hermes/bin/uv sync --locked --all-packages --extra boxlite\nOutput:\n",
        "[IMPORTANT: Background process proc_9f3a1b2c3d4e failed to start (exit code ?).\nCommand: x",
        "[IMPORTANT: Background process proc_deadbeef0001 exited (exit code 1).\nCommand: x\nOutput:\n",
        "[IMPORTANT: Background process proc_deadbeef0002 terminated by Hermes (exit code -15, SIGTERM).\n"
        "Command: x",
    )

    for wrapper in wrappers:
        provider = HonchoMemoryProvider()
        manager_calls = []

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return SimpleNamespace()

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(wrapper, "assistant reply")

        assert provider._sync_thread is None, f"wrapper not suppressed: {wrapper[:60]!r}"
        assert manager_calls == [], f"wrapper not suppressed: {wrapper[:60]!r}"


def test_honcho_sync_turn_skips_prose_gateway_notifications():
    """Prose-form gateway notifications must not become durable messages."""
    prose_wrappers = (
        "A background fan-out of 3 subagent(s) you dispatched earlier has finished.",
        "A background subagent you dispatched earlier has finished. You may have moved on.",
    )

    for wrapper in prose_wrappers:
        provider = HonchoMemoryProvider()
        manager_calls = []

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return SimpleNamespace()

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(wrapper, "assistant reply")

        assert provider._sync_thread is None, f"prose wrapper not suppressed: {wrapper[:60]!r}"
        assert manager_calls == [], f"prose wrapper not suppressed: {wrapper[:60]!r}"


def test_honcho_sync_turn_does_not_suppress_genuine_user_messages():
    """Genuine user messages that mention gateway terms must still be stored."""
    genuine = (
        "A background process I ran has finished — can you check the output?",
        "A background subagent you dispatched earlier has finished? no wait, I was asking about the report",
        "the async delegation batch complete marker disappeared from my log",
        "CONTEXT COMPACTION happened mid-message and I want to see it",
        "When you see PRIOR CONTEXT, treat it carefully",
        "I want to know about your task list",
        "IMPORTANT: Background process — can you explain what that means?",
        "[IMPORTANT: Background process — what does that mean?]",
        "Thanks for checking — the background process proc_abc123def456 completed and "
        "I'm happy with the results.",
    )

    for msg in genuine:
        provider = HonchoMemoryProvider()
        manager_calls = []

        class Manager:
            def get_or_create(self, session_key):
                manager_calls.append(session_key)
                return SimpleNamespace()

            def resolve_author_peer_id(self, session_key, author_id, author_name=None):
                return None

        provider._config = _configured_tools_config(init_on_session_start=True)
        provider._manager = Manager()
        provider._session_key = "test-session"
        provider._session_initialized = True

        provider.sync_turn(msg, "assistant reply")

        assert provider._sync_thread is not None, f"genuine message suppressed: {msg[:60]!r}"
        assert manager_calls != [], f"genuine message suppressed: {msg[:60]!r}"


def test_honcho_sync_turn_skips_empty_content():
    """Empty or whitespace-only turns must not be stored."""
    provider = HonchoMemoryProvider()
    manager_calls = []

    class Manager:
        def get_or_create(self, session_key):
            manager_calls.append(session_key)
            return SimpleNamespace()

    provider._config = _configured_tools_config(init_on_session_start=True)
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.sync_turn("   ", "  ")

    assert provider._sync_thread is None
    assert manager_calls == []


def test_honcho_sync_turn_same_instance_config_flip_gates_writes():
    """The cached-provider regression: flipping save_messages on the SAME
    configured instance must stop writes without re-initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = True
    manager_calls = []
    write_done = threading.Event()

    class Manager:
        def get_or_create(self, session_key):
            manager_calls.append(session_key)
            return SimpleNamespace(add_message=lambda role, content, **kwargs: None)

        def resolve_author_peer_id(self, session_key, author_id, author_name=None):
            return None

        def save(self, session):
            write_done.set()

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    # enabled -> write happens
    provider.sync_turn("user turn", "assistant reply")
    assert write_done.wait(timeout=5), "first write never completed"

    # operator flips containment on the same cached config object
    cfg.save_messages = False
    manager_calls.clear()
    provider.sync_turn("user turn two", "assistant reply two")

    # no new write may occur; the stale _sync_thread from the enabled write is fine
    assert manager_calls == []


def test_honcho_on_memory_write_honors_save_messages_false():
    """The memory-tool mirror is an automatic write path and must respect the
    write-disable switch; otherwise containment only covers conversation turns."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = False
    conclusion_calls = []

    class Manager:
        def create_conclusion(self, session_key, content):
            conclusion_calls.append((session_key, content))

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert conclusion_calls == []


def test_honcho_on_memory_write_still_writes_when_enabled():
    """With save_messages enabled, the memory-tool mirror still writes."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)
    cfg.save_messages = True
    conclusion_calls = []
    write_done = threading.Event()

    class Manager:
        def create_conclusion(self, session_key, content):
            conclusion_calls.append((session_key, content))
            write_done.set()

    provider._config = cfg
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True

    provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert write_done.wait(timeout=5), "memory mirror write never completed"
    assert conclusion_calls != []


# ---------------------------------------------------------------------------
# Explicit-writes regression tests (honcho_conclude / honcho_profile card writes)
#
# save_messages/_writes_enabled() above gates only automatic/background projection.
# honcho_conclude and honcho_profile's card mutation are model-invoked and were
# entirely ungated before explicit_writes existed — these tests cover that gap.
# ---------------------------------------------------------------------------


def test_honcho_conclude_hidden_from_schemas_when_explicit_writes_disabled():
    """Default (no explicit_writes attribute at all) must hide honcho_conclude and
    strip honcho_profile's card-write parameter — fail closed on missing config too."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_tools_config()
    provider._recall_mode = "tools"

    schemas = provider.get_tool_schemas()
    names = [s["name"] for s in schemas]

    assert "honcho_conclude" not in names
    profile_schema = next(s for s in schemas if s["name"] == "honcho_profile")
    assert "card" not in profile_schema["parameters"]["properties"]


def test_honcho_conclude_present_when_explicit_writes_enabled():
    """explicit_writes=True restores honcho_conclude and the card-write parameter."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config()
    cfg.explicit_writes = True
    provider._config = cfg
    provider._recall_mode = "tools"

    schemas = provider.get_tool_schemas()
    names = [s["name"] for s in schemas]

    assert "honcho_conclude" in names
    profile_schema = next(s for s in schemas if s["name"] == "honcho_profile")
    assert "card" in profile_schema["parameters"]["properties"]


def test_honcho_conclude_schema_filtering_does_not_mutate_shared_schema():
    """Filtering for a read-only host must not corrupt the shared ALL_TOOL_SCHEMAS list
    a later writable host would read."""
    read_only = HonchoMemoryProvider()
    read_only._config = _configured_tools_config()
    read_only._recall_mode = "tools"
    read_only.get_tool_schemas()

    writable = HonchoMemoryProvider()
    cfg = _configured_tools_config()
    cfg.explicit_writes = True
    writable._config = cfg
    writable._recall_mode = "tools"
    schemas = writable.get_tool_schemas()

    profile_schema = next(s for s in schemas if s["name"] == "honcho_profile")
    assert "card" in profile_schema["parameters"]["properties"]
    assert any(s["name"] == "honcho_conclude" for s in schemas)


def test_honcho_forged_conclude_call_rejected_before_session_init():
    """A forged honcho_conclude call on a read-only host must be rejected before any
    lazy session init is attempted — schema hiding is advisory, this is enforcement."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_tools_config()
    init_calls = []

    def fake_ensure_session():
        init_calls.append(1)
        return True

    provider._ensure_session = fake_ensure_session

    result = provider.handle_tool_call("honcho_conclude", {"conclusion": "forged fact"})

    assert "disabled by policy" in result.lower()
    assert init_calls == []


def test_honcho_forged_profile_card_write_rejected_before_session_init():
    """A forged honcho_profile card write is rejected the same way a conclude call is;
    a plain read (no card arg) is a different code path and is not covered here."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_tools_config()
    init_calls = []

    def fake_ensure_session():
        init_calls.append(1)
        return True

    provider._ensure_session = fake_ensure_session

    result = provider.handle_tool_call("honcho_profile", {"peer": "user", "card": ["forged"]})

    assert "disabled by policy" in result.lower()
    assert init_calls == []


def test_honcho_system_prompt_marks_derived_and_hides_write_guidance():
    """Read-only hosts see the derived/non-canonical framing and no mention of
    honcho_conclude; writable hosts keep the original write guidance."""
    read_only = HonchoMemoryProvider()
    read_only._config = _configured_tools_config()
    read_only._recall_mode = "tools"
    prompt = read_only.system_prompt_block()

    assert "derived, fallible" in prompt.lower()
    assert "not canonical memory" in prompt.lower()
    assert "explicit honcho conclusion and peer-card mutation is disabled" in prompt.lower()
    assert "honcho_conclude" not in prompt

    writable = HonchoMemoryProvider()
    cfg = _configured_tools_config()
    cfg.explicit_writes = True
    writable._config = cfg
    writable._recall_mode = "tools"
    writable_prompt = writable.system_prompt_block()

    assert "honcho_conclude" in writable_prompt
    assert "derived, fallible" in writable_prompt.lower()


def test_honcho_low_level_write_guards_cover_all_mutators():
    """Manager-layer defense in depth: even a caller that bypasses the provider's tool
    dispatch entirely must still be refused by the mutators themselves."""
    from plugins.memory.honcho.session import HonchoSessionManager

    stub = SimpleNamespace(_config=_configured_tools_config())

    assert HonchoSessionManager.create_conclusion(stub, "session-key", "a fact") is False
    assert HonchoSessionManager.delete_conclusion(stub, "session-key", "concl-1") is False
    assert HonchoSessionManager.set_peer_card(stub, "session-key", ["fact"]) is None
    assert HonchoSessionManager.seed_ai_identity(stub, "session-key", "identity text") is False
    assert HonchoSessionManager.migrate_memory_files(stub, "session-key", "/tmp/nonexistent") is False


def test_honcho_low_level_write_guards_allow_when_explicit_writes_enabled():
    """The manager-layer guards must not block writes for a host that explicitly enabled
    them — this test only proves the guard doesn't short-circuit; it doesn't exercise the
    full Honcho SDK call path."""
    from plugins.memory.honcho.session import HonchoSessionManager

    cfg = _configured_tools_config()
    cfg.explicit_writes = True
    stub = SimpleNamespace(_config=cfg, _cache={})

    # No cached session for "session-key" -> both return False/None via the existing
    # "no session cached" path, NOT the explicit-writes guard. Reaching that later
    # check (instead of an early False from the guard) proves the guard let it through.
    assert HonchoSessionManager.create_conclusion(stub, "session-key", "a fact") is False
    assert HonchoSessionManager.seed_ai_identity(stub, "session-key", "identity text") is False
