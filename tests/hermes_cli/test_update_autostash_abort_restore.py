"""Regression tests: an aborted ``hermes update`` must put the autostash back.

Live incident (2026-09-12, ~/.hermes/logs/update.log)::

    → Local changes detected — stashing before update...
    Saved working directory and index state On local: hermes-update-autostash-20260912-225556
    → Found 8 new commit(s)
    → Pulling updates...
      ⚠ Checkout is on custom branch 'local' — merging origin/main instead of resetting...
    ✗ Merge conflict between local commits and upstream — update stopped, nothing was changed.
      Then re-run the update. Local work is untouched.
      ℹ️  Local changes preserved in stash (ref: 351c811d78b0...)

``git merge --abort`` had already put the checkout back exactly as the update found
it, but ``_pull_updates`` treated "the update did not land" as "the tree state is
unknown" and refused every restore. 207 lines of uncommitted work sat in ``git stash``
for four days, on a run that was invoked with ``--yes`` and
``updates.non_interactive_local_changes: stash`` — the configuration that is supposed
to give the changes straight back.

The contract these tests pin down:

* abort that undid itself cleanly (``merge --abort`` / rollback ``reset``) -> restore,
  regardless of ``--keep-stash`` or the discard policy (nothing new landed, so neither
  "don't ride local edits onto new code" nor "discard before updating" applies);
* abort that left the tree in an unknown state -> keep the entry, and name the ref plus
  the exact recovery command (never silent);
* successful update -> unchanged behaviour (discard / park / restore).
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# Mocked-git harness for the pull phase
# ---------------------------------------------------------------------------

STASH_REF = "351c811d78b0804316a0652834133fc338bd36f4"


def _ok(stdout=""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _fail(stderr="boom", returncode=1):
    return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


def _install_pull_harness(
    monkeypatch, tmp_path, *, current_branch="local", ff_only_fails=True, merge_fails=True,
    merge_abort_fails=False, reset_fails=False, syntax_ok=True,
):
    """Drive ``_pull_updates`` against a scripted git, recording stash decisions.

    Defaults reproduce the incident: a custom branch whose merge with origin/main
    conflicts and is then cleanly aborted.
    """
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)

    def fake_git_run(git_cmd, args, cwd=None, *, check=False, network=False):
        joined = " ".join(args)
        if args[:2] == ["rev-parse", "HEAD"]:
            return _ok("1111111111111111111111111111111111111beef\n")
        if args[:2] == ["branch", "--show-current"]:
            return _ok(f"{current_branch}\n")
        if args[0] == "tag":
            return _ok()
        if "--ff-only" in args:
            return _fail("fatal: Not possible to fast-forward, aborting.\n") if ff_only_fails else _ok()
        if args[:2] == ["merge", "--abort"]:
            return _fail("fatal: There is no merge to abort\n") if merge_abort_fails else _ok()
        if args[0] == "merge":
            return _fail("CONFLICT (content): Merge conflict in agent/core.py\n") if merge_fails else _ok()
        if args[0] == "merge-base":
            return _ok("abc123\n")
        if args[:2] == ["reset", "--hard"]:
            return _fail("error: unable to write\n") if reset_fails else _ok()
        raise AssertionError(f"unexpected git call: {joined}")

    monkeypatch.setattr(update_cmd, "_git_run", fake_git_run)
    monkeypatch.setattr(
        update_cmd, "_validate_critical_files_syntax",
        lambda root: (True, None, None) if syntax_ok else (False, "agent/core.py", "SyntaxError: bad"))

    calls = {"restore": [], "park": [], "discard": [], "warn": []}
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda git_cmd, cwd, ref, prompt_user=False, input_fn=None: calls["restore"].append(ref) or True)
    monkeypatch.setattr(
        hermes_main, "_park_stashed_changes", lambda ref: calls["park"].append(ref))
    monkeypatch.setattr(
        hermes_main, "_discard_stashed_changes",
        lambda git_cmd, cwd, ref: calls["discard"].append(ref) or True)
    monkeypatch.setattr(
        hermes_main, "_warn_autostash_left_parked",
        lambda ref: calls["warn"].append(ref) or update_cmd._warn_autostash_left_parked(ref))
    return calls


def _run_pull(*, discard_local_changes=False, keep_stash=False, stash_ref=STASH_REF):
    return update_cmd._pull_updates(
        ["git"], "main", stash_ref, prompt_for_restore=False, gw_input_fn=None,
        discard_local_changes=discard_local_changes, keep_stash=keep_stash)


# ---------------------------------------------------------------------------
# The incident: merge conflict on a custom branch, cleanly aborted
# ---------------------------------------------------------------------------

def test_merge_conflict_abort_restores_the_autostash(monkeypatch, tmp_path):
    """``merge --abort`` succeeded, so the tree is as found — put the changes back."""
    calls = _install_pull_harness(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        _run_pull()

    assert excinfo.value.code == 1
    assert calls["restore"] == [STASH_REF]
    assert calls["park"] == [] and calls["discard"] == [] and calls["warn"] == []


def test_merge_conflict_abort_restores_even_with_keep_stash(monkeypatch, tmp_path):
    """``--keep-stash`` exists so local edits never ride onto NEW code. Nothing new
    landed here, so the desktop updater must not shelve the work either."""
    calls = _install_pull_harness(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        _run_pull(keep_stash=True)

    assert calls["restore"] == [STASH_REF]
    assert calls["park"] == []


def test_merge_conflict_abort_does_not_discard_under_discard_policy(monkeypatch, tmp_path):
    """``non_interactive_local_changes: discard`` means "don't block the update on my
    uncommitted edits" — not "delete them even when the update never happened"."""
    calls = _install_pull_harness(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        _run_pull(discard_local_changes=True)

    assert calls["discard"] == []
    assert calls["restore"] == [STASH_REF]


def test_merge_conflict_abort_no_longer_claims_local_work_is_untouched(monkeypatch, tmp_path, capsys):
    """The old copy said "Local work is untouched" while the work sat in a stash."""
    _install_pull_harness(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        _run_pull()

    assert "Local work is untouched" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Aborts that leave the tree in an unknown state keep the stash — loudly
# ---------------------------------------------------------------------------

def test_failed_merge_abort_keeps_the_stash_and_names_the_ref(monkeypatch, tmp_path, capsys):
    """A conflicted merge still in progress is NOT a known-good tree."""
    calls = _install_pull_harness(monkeypatch, tmp_path, merge_abort_fails=True)

    with pytest.raises(SystemExit):
        _run_pull()

    assert calls["restore"] == []
    assert calls["warn"] == [STASH_REF]
    out = capsys.readouterr().out
    assert "still in progress" in out
    # The ref and a copy-pasteable recovery command, not a bare `git stash apply`.
    assert STASH_REF in out
    assert f"git stash apply {STASH_REF}" in out


def test_failed_reset_keeps_the_stash_parked(monkeypatch, tmp_path, capsys):
    """Same-branch divergence whose ``reset --hard`` fails: tree state unknown."""
    calls = _install_pull_harness(
        monkeypatch, tmp_path, current_branch="main", merge_fails=False, reset_fails=True)

    with pytest.raises(SystemExit):
        _run_pull()

    assert calls["restore"] == []
    assert calls["warn"] == [STASH_REF]
    assert "preserved in stash" in capsys.readouterr().out


def test_no_stash_taken_is_a_no_op(monkeypatch, tmp_path, capsys):
    """Nothing was stashed -> nothing to restore, and no stash chatter."""
    calls = _install_pull_harness(monkeypatch, tmp_path)

    with pytest.raises(SystemExit):
        _run_pull(stash_ref=None)

    assert calls == {"restore": [], "park": [], "discard": [], "warn": []}
    assert "preserved in stash" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Post-pull syntax rollback
# ---------------------------------------------------------------------------

def test_syntax_error_rollback_restores_the_autostash(monkeypatch, tmp_path):
    """The rollback ``reset --hard <pre_pull_sha>`` succeeded, so the tree is as found."""
    calls = _install_pull_harness(
        monkeypatch, tmp_path, current_branch="main", ff_only_fails=False, syntax_ok=False)

    with pytest.raises(SystemExit):
        _run_pull()

    assert calls["restore"] == [STASH_REF]
    assert calls["warn"] == []


def test_failed_syntax_rollback_keeps_the_stash_parked(monkeypatch, tmp_path):
    """Rollback itself failed: the checkout holds unknown, non-compiling code."""
    calls = _install_pull_harness(
        monkeypatch, tmp_path, current_branch="main", ff_only_fails=False, syntax_ok=False,
        reset_fails=True)

    with pytest.raises(SystemExit):
        _run_pull()

    assert calls["restore"] == []
    assert calls["warn"] == [STASH_REF]


# ---------------------------------------------------------------------------
# A successful update keeps its existing behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "restore"),
        ({"keep_stash": True}, "park"),
        ({"discard_local_changes": True}, "discard"),
    ],
)
def test_successful_update_settles_the_stash_as_before(monkeypatch, tmp_path, kwargs, expected):
    calls = _install_pull_harness(monkeypatch, tmp_path, current_branch="main", ff_only_fails=False)

    _run_pull(**kwargs)

    assert calls[expected] == [STASH_REF]
    assert not any(v for k, v in calls.items() if k != expected)


# ---------------------------------------------------------------------------
# The pre-pull phase must not leak the stash either
# ---------------------------------------------------------------------------

def test_commit_count_failure_restores_the_autostash(monkeypatch, tmp_path, capsys):
    """``rev-list --count`` runs under ``check=True``; the CalledProcessError used to
    unwind past the stash into the ZIP-fallback handler without ever naming it."""
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: STASH_REF)
    monkeypatch.setattr(
        update_cmd, "_apply_parked_branch_guard", lambda *a, **kw: (False, False, None))
    restored = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda git_cmd, cwd, ref, prompt_user=False, input_fn=None: restored.append(ref) or True)

    def fake_git_run(git_cmd, args, cwd=None, *, check=False, network=False):
        if args[0] == "rev-list":
            raise subprocess.CalledProcessError(128, ["git", *args], output="", stderr="fatal: bad revision")
        return _ok()

    monkeypatch.setattr(update_cmd, "_git_run", fake_git_run)

    with pytest.raises(subprocess.CalledProcessError):
        update_cmd._prepare_checkout_for_update(
            ["git"], "main", "main", is_fork=False, assume_yes=True, gateway_mode=False,
            gw_input_fn=None, switch_branch=False, _windows_gateway_resume=None)

    assert restored == [STASH_REF]


def test_interrupt_during_commit_count_names_the_parked_stash(monkeypatch, tmp_path, capsys):
    """Ctrl-C must not silently swallow the stash either — but it also must not kick off
    a multi-step restore the user is interrupting."""
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: STASH_REF)
    monkeypatch.setattr(
        update_cmd, "_apply_parked_branch_guard", lambda *a, **kw: (False, False, None))
    restored = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restored.append(1) or True)

    def fake_git_run(git_cmd, args, cwd=None, *, check=False, network=False):
        if args[0] == "rev-list":
            raise KeyboardInterrupt
        return _ok()

    monkeypatch.setattr(update_cmd, "_git_run", fake_git_run)

    with pytest.raises(KeyboardInterrupt):
        update_cmd._prepare_checkout_for_update(
            ["git"], "main", "main", is_fork=False, assume_yes=True, gateway_mode=False,
            gw_input_fn=None, switch_branch=False, _windows_gateway_resume=None)

    assert restored == []
    assert STASH_REF in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End to end against real git: the file content actually comes back
# ---------------------------------------------------------------------------

def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
        env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
             "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
             "HOME": str(cwd)})


@pytest.fixture
def diverged_checkout(tmp_path):
    """A checkout on branch ``local`` whose merge with ``origin/main`` conflicts."""
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git not available")
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--initial-branch", "main")
    (origin / "shared.txt").write_text("base\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-m", "base")

    repo = tmp_path / "repo"
    _git(tmp_path, "clone", str(origin), str(repo))
    _git(repo, "checkout", "-b", "local")
    (repo / "shared.txt").write_text("local rewrite\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "local work")

    (origin / "shared.txt").write_text("upstream rewrite\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-m", "upstream work")
    _git(repo, "fetch", "origin", "main")
    return repo


def test_real_merge_conflict_abort_returns_the_working_tree_changes(
    monkeypatch, diverged_checkout, capsys,
):
    """The end-to-end guarantee, against real git: uncommitted edits are on disk again
    after the conflicted update aborts, and the stash entry is gone."""
    repo = diverged_checkout
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    # The import health probe spawns an interpreter per critical module; irrelevant here.
    monkeypatch.setattr(update_cmd, "_critical_module_import_failures", lambda *a, **kw: {})

    (repo / "shared.txt").write_text("local rewrite\nUNCOMMITTED EDIT\n")
    (repo / "brand-new.txt").write_text("untracked work\n")

    stash_ref = update_cmd._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref, "the fixture must produce a real autostash"
    assert (repo / "shared.txt").read_text() == "local rewrite\n"
    assert not (repo / "brand-new.txt").exists()

    with pytest.raises(SystemExit):
        update_cmd._pull_updates(
            ["git"], "main", stash_ref, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False)

    assert (repo / "shared.txt").read_text() == "local rewrite\nUNCOMMITTED EDIT\n"
    assert (repo / "brand-new.txt").read_text() == "untracked work\n"
    # Restored cleanly -> the entry is dropped rather than left to rot for days.
    stash_list = subprocess.run(
        ["git", "stash", "list"], cwd=repo, capture_output=True, text=True).stdout
    assert stash_list.strip() == ""
    # And the merge really was aborted: HEAD is still the local commit, tree has no markers.
    assert "<<<<<<<" not in (repo / "shared.txt").read_text()
    assert not (Path(repo) / ".git" / "MERGE_HEAD").exists()


def test_real_in_place_update_incident_end_to_end(monkeypatch, diverged_checkout, capsys):
    """The whole incident against real git: parked on ``local``, ``parked_branch_strategy:
    update_in_place``, ``--yes``, ``non_interactive_local_changes: stash`` — checkout phase
    stashes, pull phase conflicts and aborts, and the edits are back on disk.

    The guard is told the tree is clean (that is what it observed in the incident, one
    ``git status`` before the stash saw changes) so the in-place path is reached exactly as
    the live log shows.
    """
    repo = diverged_checkout
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(update_cmd, "_critical_module_import_failures", lambda *a, **kw: {})
    monkeypatch.setattr(
        hermes_main, "_assess_parked_branch_switch", lambda *a, **kw: (True, "unmerged:1"))
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {
        "parked_branch_strategy": "update_in_place", "non_interactive_local_changes": "stash"})

    (repo / "shared.txt").write_text("local rewrite\nUNCOMMITTED EDIT\n")

    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "local", is_fork=False, assume_yes=True, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, _windows_gateway_resume=None)

    assert plan.in_place_update is True
    assert plan.auto_stash_ref, "the checkout phase must have autostashed the edit"
    assert plan.prompt_for_restore is False, "--yes must not gate the restore on a prompt"
    assert plan.commit_count == 1

    with pytest.raises(SystemExit):
        update_cmd._pull_updates(
            ["git"], "main", plan.auto_stash_ref, prompt_for_restore=plan.prompt_for_restore,
            gw_input_fn=None, discard_local_changes=False, keep_stash=False)

    assert (repo / "shared.txt").read_text() == "local rewrite\nUNCOMMITTED EDIT\n"
    out = capsys.readouterr().out
    assert "Merge conflict between local commits and upstream" in out
    assert "Local changes were restored on top of the updated codebase." in out
