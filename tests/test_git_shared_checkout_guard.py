"""Unit tests for the shared-checkout git guard hook.

Denies index-sweeping / tree-destroying git ops so simultaneous sessions in one
working tree don't clobber each other; the explicit-path forms must pass. Loaded
by path like tests/test_reuse_first_reminder.py.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

_HOOK = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".claude" / "hooks" / "git_shared_checkout_guard.py"
)
_spec = importlib.util.spec_from_file_location("git_shared_checkout_guard", _HOOK)
assert _spec is not None and _spec.loader is not None, f"hook not found at {_HOOK}"
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _verdict(command):
    """Return 'deny' | 'warn' | None for a command."""
    result = hook.analyze(command)
    return result[0] if result else None


# ------------------------------ index-sweepers: DENY ------------------------------

def test_add_all_denied():
    assert _verdict("git add -A") == "deny"


def test_add_update_denied():
    assert _verdict("git add -u") == "deny"


def test_add_dot_denied():
    assert _verdict("git add .") == "deny"


def test_add_all_long_denied():
    assert _verdict("git add --all") == "deny"


def test_commit_am_denied():
    assert _verdict('git commit -am "message"') == "deny"


def test_commit_a_separate_denied():
    assert _verdict('git commit -a -m "message"') == "deny"


def test_bare_commit_denied():
    assert _verdict('git commit -m "message"') == "deny"


def test_bare_commit_no_message_denied():
    assert _verdict("git commit") == "deny"


# ------------------------------ tree destroyers: DENY ------------------------------

def test_reset_hard_denied():
    assert _verdict("git reset --hard") == "deny"


def test_reset_hard_ref_denied():
    assert _verdict("git reset --hard HEAD~1") == "deny"


def test_clean_fd_denied():
    assert _verdict("git clean -fd") == "deny"


def test_checkout_dot_denied():
    assert _verdict("git checkout .") == "deny"


def test_restore_dot_denied():
    assert _verdict("git restore .") == "deny"


# ------------------------------ safe forms: ALLOW ------------------------------

def test_add_explicit_paths_allowed():
    assert _verdict("git add CLAUDE.md docs/bar.md") is None


def test_add_patch_allowed():
    assert _verdict("git add -p") is None


def test_commit_only_pathspec_allowed():
    assert _verdict("git commit --only -- CLAUDE.md docs/bar.md -m 'msg'") is None


def test_commit_double_dash_pathspec_allowed():
    assert _verdict('git commit -- src/foo.py -m "msg"') is None


def test_commit_amend_allowed():
    assert _verdict("git commit --amend --no-edit") is None


def test_reset_paths_allowed():
    assert _verdict("git reset src/foo.py") is None


def test_checkout_file_allowed():
    assert _verdict("git checkout -- src/foo.py") is None


def test_read_only_git_allowed():
    assert _verdict("git status") is None
    assert _verdict("git diff --cached --name-only") is None
    assert _verdict("git log --oneline -5") is None


def test_global_option_before_subcommand_allowed():
    # `git -C <path> commit --only -- ...` must be parsed correctly
    assert _verdict("git -C /repo commit --only -- CLAUDE.md -m 'x'") is None


def test_quoted_message_with_dash_a_not_flagged_as_commit_a():
    # `-a` inside the commit message must not read as the -a flag; this is a
    # safe --only commit, so it must ALLOW (not deny).
    assert _verdict('git commit --only -- x.py -m "add -a and -A support"') is None


# ------------------------------ warn (allow + note) ------------------------------

def test_switch_branch_warns():
    assert _verdict("git switch feature-x") == "warn"


def test_checkout_new_branch_warns():
    assert _verdict("git checkout -b feature-x") == "warn"


# ------------------------------ merge/rebase exception ------------------------------

def test_bare_commit_allowed_during_merge(monkeypatch):
    monkeypatch.setattr(hook, "_merge_or_rebase_in_progress", lambda: True)
    assert _verdict("git commit") is None


def test_bare_commit_denied_when_not_merging(monkeypatch):
    monkeypatch.setattr(hook, "_merge_or_rebase_in_progress", lambda: False)
    assert _verdict('git commit -m "x"') == "deny"


# ------------------------------ compound commands ------------------------------

def test_compound_add_all_then_commit_denied():
    assert _verdict('git add -A && git commit -m "x"') == "deny"


def test_compound_safe_add_and_commit_allowed():
    assert _verdict("git add src/foo.py && git commit --only -- src/foo.py -m 'x'") is None


# ---------------------- multi-line / quoted-newline commits ----------------------
# Regression: segmentation split on raw newlines, severing `--only -- <paths>` from
# `git commit` when the message spanned lines (heredoc or multi-line -m), so a safe
# commit was mis-denied as "bare". Splitting must be quote-aware.

def test_multiline_quoted_message_only_commit_allowed():
    cmd = 'git commit -m "line one\nline two" --only -- src/foo.py'
    assert _verdict(cmd) is None


def test_heredoc_substitution_message_only_commit_allowed():
    cmd = (
        "git commit -m \"$(cat <<'EOF'\n"
        "feat: something\n"
        "\n"
        "body line\n"
        "EOF\n"
        ")\" --only -- src/foo.py tests/bar.py"
    )
    assert _verdict(cmd) is None


def test_operator_inside_quoted_message_not_a_new_command():
    # `&&` / `;` inside the message must not spawn a phantom segment.
    cmd = 'git commit --only -- x.py -m "fix a && drop b; done"'
    assert _verdict(cmd) is None


def test_newline_separated_compound_add_all_still_denied():
    # A REAL unquoted-newline compound must still be split so `git add -A` is caught.
    assert _verdict('git add -A\ngit commit --only -- x.py -m "y"') == "deny"


def test_multiline_bare_commit_still_denied():
    # A genuinely bare commit whose message spans lines is still bare (no pathspec).
    assert _verdict('git commit -m "line one\nline two"') == "deny"


# ------------------------------ end-to-end stdin -> stdout ------------------------------

def _run(payload):
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=15,
    )
    return proc.stdout.strip()


def test_e2e_deny_emits_permission_decision():
    out = _run({"tool_name": "Bash", "tool_input": {"command": "git add -A"}})
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_e2e_safe_commit_is_silent():
    out = _run(
        {"tool_name": "Bash", "tool_input": {"command": "git commit --only -- x.py -m 'y'"}}
    )
    assert out == ""


def test_e2e_non_git_command_ignored():
    out = _run({"tool_name": "Bash", "tool_input": {"command": "python -m pytest -q"}})
    assert out == ""


# ------------------------------ satellite mode ------------------------------
# In a satellite clone the index is private: sweepers downgrade to warnings,
# destroyers stay denied, branch-switch warnings disappear (spec 2026-07-06 §5).

def _sat_verdict(command):
    result = hook.analyze(command, satellite=True)
    return result[0] if result else None


def test_satellite_add_all_warns_not_denies():
    assert _sat_verdict("git add -A") == "warn"


def test_satellite_bare_commit_warns_not_denies():
    assert _sat_verdict('git commit -m "x"') == "warn"


def test_satellite_commit_a_warns_not_denies():
    assert _sat_verdict('git commit -am "x"') == "warn"


def test_satellite_checkout_dot_warns_not_denies():
    assert _sat_verdict("git checkout .") == "warn"


def test_satellite_reset_hard_still_denied():
    assert _sat_verdict("git reset --hard") == "deny"


def test_satellite_clean_f_still_denied():
    assert _sat_verdict("git clean -fd") == "deny"


def test_satellite_switch_is_silent():
    assert _sat_verdict("git switch feature-x") is None


def test_satellite_checkout_new_branch_is_silent():
    assert _sat_verdict("git checkout -b feature-x") is None


def test_satellite_safe_forms_still_silent():
    assert _sat_verdict("git add src/foo.py") is None
    assert _sat_verdict("git commit --only -- src/foo.py -m 'x'") is None


def test_default_analyze_unchanged_strict():
    # No-satellite callers keep canonical strictness (regression guard).
    assert _verdict("git add -A") == "deny"


# ------------- committing onto another session's branch (canonical only) -------------
# 2026-07-27: a session created a feature branch in the shared canonical checkout
# (with the user's go-ahead). Creating a branch moves HEAD for EVERY session, so a
# second session's commits silently landed on that branch instead of master --
# noticed only afterwards, and untangling it cost an afternoon. The existing
# branch-switch warning fires for the session that switches; nothing warned the
# session that merely committed later.

def test_safe_commit_warns_when_canonical_is_off_master(monkeypatch):
    monkeypatch.setattr(hook, "_current_branch", lambda: "fix/someone-elses-work")
    monkeypatch.setattr(hook, "_already_fired", lambda *a: False)
    assert _verdict("git commit --only -- src/foo.py -m 'x'") == "warn"


def test_safe_commit_silent_on_master(monkeypatch):
    monkeypatch.setattr(hook, "_current_branch", lambda: "master")
    assert _verdict("git commit --only -- src/foo.py -m 'x'") is None


def test_off_master_commit_warning_names_the_branch(monkeypatch):
    monkeypatch.setattr(hook, "_current_branch", lambda: "fix/someone-elses-work")
    monkeypatch.setattr(hook, "_already_fired", lambda *a: False)
    result = hook.analyze("git commit --only -- src/foo.py -m 'x'")
    assert result is not None and "fix/someone-elses-work" in result[1]


def test_off_master_commit_silent_in_satellite(monkeypatch):
    """A satellite's whole point is its own branch -- never warn there."""
    monkeypatch.setattr(hook, "_current_branch", lambda: "feat/whatever")
    assert hook.analyze("git commit --only -- src/foo.py -m 'x'", satellite=True) is None


def test_detached_head_commit_warns(monkeypatch):
    monkeypatch.setattr(hook, "_current_branch", lambda: None)
    monkeypatch.setattr(hook, "_already_fired", lambda *a: False)
    assert _verdict("git commit --only -- src/foo.py -m 'x'") == "warn"


def test_off_master_warning_does_not_mask_a_deny(monkeypatch):
    monkeypatch.setattr(hook, "_current_branch", lambda: "fix/someone-elses-work")
    monkeypatch.setattr(hook, "_already_fired", lambda *a: False)
    assert _verdict("git add -A && git commit -m 'x'") == "deny"


def test_repeat_commits_on_same_branch_warn_once(monkeypatch):
    """Per-branch, once. Three commits in a row shouldn't nag three times."""
    fired = set()

    def _fire(session_id, category):
        already = category in fired
        fired.add(category)
        return already

    monkeypatch.setattr(hook, "_current_branch", lambda: "fix/other")
    monkeypatch.setattr(hook, "_already_fired", _fire)
    first = _verdict("git commit --only -- a.py -m 'x'")
    second = _verdict("git commit --only -- b.py -m 'x'")
    assert first == "warn" and second is None
