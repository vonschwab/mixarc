"""PreToolUse hook: keep simultaneous sessions from clobbering each other in one tree.

This repo runs multiple Claude sessions in a single SHARED working checkout
(worktrees were retired — they broke on data/ symlinks + SQLite WAL aliasing).
One working tree means one shared index and one HEAD, so the classic failure is
an agent staging or committing another session's in-flight work. This guard
denies the index-sweeping and tree-destroying git ops and points at the safe,
explicit-path form the repo already uses (`git add <paths>`, `git commit --only
-- <paths>`). See CLAUDE.md "Session discipline" + memory feedback_shared_checkout_commit_only.

DENY (safe alternative always exists):
  - git add -A / --all / -u / --update / . / * / :/     -> stage explicit paths
  - git commit -a / --all / -am                          -> git commit --only -- <paths>
  - bare git commit (no `--`/`--only` pathspec)          -> git commit --only -- <paths>
    (allowed while a merge/rebase/cherry-pick/revert is in progress, or --amend/
     --squash/--fixup/--no-edit — those are legitimate conclusion commits)
  - git reset --hard / git clean -f / git checkout . / git restore .

WARN (allowed, but flagged):
  - git switch / git checkout -b|-B — changes the shared HEAD for every
    session; make sure no one else is mid-edit.
  - an otherwise-safe `git commit` while HEAD is off `master` — canonical's
    resting state is master, so any other branch means a session moved the
    shared HEAD and this commit is about to stack onto their work. Fires once
    per branch per session. This is the other half of the branch-switch warning
    above: that one fires for whoever switches, this one for whoever commits
    afterwards (2026-07-27, when only the first existed and a commit landed on
    another session's branch unnoticed).

SATELLITE MODE: a satellite clone's working tree is private (see
workspace_identity.is_satellite), so the sweeper DENYs above downgrade to a
once-per-session WARN there — nothing else can be clobbered, but the broad
forms are still a habit that corrupts the shared canonical checkout. The
truly destructive ops (reset --hard, clean -f) stay DENIED in both modes, and
the branch-switch WARN goes silent (switching HEAD only affects the satellite
itself).

Contract mirrors the repo's other PreToolUse hooks: read the tool-call JSON on
stdin; emit permissionDecision "deny" to block, additionalContext to warn, stay
silent to allow. FAIL OPEN — any error returns without blocking.
"""

import json
import os
import re
import shlex
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workspace_identity import is_satellite  # noqa: E402

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

# Set from the hook payload in main(); analyze_segment needs it for the
# once-per-branch off-master marker, and analyze() is also called directly by tests.
_SESSION_ID = ""

_SAFE_ADD = "Stage explicit paths instead: `git add <path1> <path2>` (never -A/-u/.)."
_SAFE_COMMIT = (
    "On a shared checkout a bare/`-a` commit sweeps other sessions' staged work. "
    "Commit explicit paths: `git commit --only -- <path1> <path2>` "
    "(verify first with `git diff --cached --name-only`)."
)
_SAT_WARN = (
    "Hook note (satellite clone): this tree is private so nothing else is at "
    "risk, but broad staging/committing is a habit that corrupts work in the "
    "SHARED canonical checkout. Prefer `git add <paths>` + `git commit --only "
    "-- <paths>` everywhere. One reminder per session."
)

_MASTER_BRANCH = "master"


def _off_master_message(branch):
    """Warn that this commit is landing somewhere other than canonical's master.

    2026-07-27: one session created a feature branch here (with the user's
    go-ahead). Creating a branch moves HEAD for EVERY session in the checkout,
    so a second session's commits silently stacked onto that branch instead of
    master, and were only noticed after the fact. The existing branch-switch
    warning fires for whoever switches; nobody warned the session that just
    committed afterwards. This is that missing half.
    """
    where = f"branch `{branch}`" if branch else "a DETACHED HEAD"
    return (
        f"Hook note: this canonical checkout is on {where}, not `{_MASTER_BRANCH}` — "
        "another session very likely moved the shared HEAD, and your commit is about "
        "to land on their branch. Check `git log --oneline master..HEAD` to see whose "
        "work is there. If this commit isn't part of that branch's work, coordinate "
        "before committing (or run this session from a satellite clone: "
        "`python tools/create_satellite.py --name PG3_SAT3 --port 8773`). "
        "Once per branch per session."
    )


_BROAD_ADD = {"-A", "--all", "-u", "--update", ".", "*", ":/", "-Au", "-uA", "-uAll"}
_MERGE_MARKERS = ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD", "REVERT_HEAD")
_COMMIT_CONCLUSION_FLAGS = {"--amend", "--squash", "--fixup", "--no-edit"}


def _merge_or_rebase_in_progress():
    """True if a git operation that legitimately ends in a bare `git commit` is live."""
    gitdir = os.path.join(PROJECT_DIR, ".git")
    return any(os.path.exists(os.path.join(gitdir, m)) for m in _MERGE_MARKERS)


def _current_branch():
    """Branch name from .git/HEAD, or None when detached/unreadable.

    Read directly rather than shelling out to `git`: this runs PreToolUse on
    every Bash call, same reason workspace_identity parses .git/config by hand.
    """
    try:
        with open(os.path.join(PROJECT_DIR, ".git", "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        ref = head[4:].strip()
        prefix = "refs/heads/"
        return ref[len(prefix):] if ref.startswith(prefix) else ref
    return None  # detached HEAD (raw sha)


def _segments(command):
    """Split a compound command (git add -A && git commit ...) into segments,
    evaluating each on its own.

    Quote-aware: shell operators and newlines INSIDE a quoted string must not
    split the command. A multi-line `-m "..."` or a `"$(cat <<'EOF' ... EOF)"`
    message otherwise gets severed from its trailing `--only -- <paths>`, and the
    `git commit` fragment reads as a bare commit and is wrongly denied. Only
    unquoted `; & | && ||` and newlines are real segment boundaries.
    """
    s = command or ""
    segments, buf = [], []
    quote = None  # None | "'" | '"'
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if quote is not None:
            buf.append(ch)
            # backslash escapes the next char inside double quotes (not single).
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch in ("\n", ";"):
            segments.append("".join(buf))
            buf = []
        elif ch == "&":
            segments.append("".join(buf))
            buf = []
            i += 2 if i + 1 < n and s[i + 1] == "&" else 1
            continue
        elif ch == "|":
            segments.append("".join(buf))
            buf = []
            i += 2 if i + 1 < n and s[i + 1] == "|" else 1
            continue
        else:
            buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return segments


def _tokens(segment):
    try:
        toks = shlex.split(segment, posix=False)  # posix=False keeps Windows backslashes
    except ValueError:
        toks = segment.split()
    return [t.strip("\"'") for t in toks]


def _git_args(tokens):
    """Return the args after `git`, skipping global options (-C <path>, -c k=v, ...)."""
    if "git" not in tokens:
        return None
    args = tokens[tokens.index("git") + 1:]
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c") else 1
    return args[i:]  # [subcommand, *subargs] or []


def _has_a_cluster(flag):
    # single-dash short-flag cluster containing 'a' (-a, -am, -ma) but not --amend
    return bool(re.fullmatch(r"-[a-z]*a[a-z]*", flag))


def analyze_segment(tokens, satellite=False):
    """Return ('deny', msg) | ('warn', msg) | None for one command segment."""
    args = _git_args(tokens)
    if not args:
        return None
    sub, sargs = args[0], args[1:]

    if sub == "add":
        if any(a in _BROAD_ADD for a in sargs):
            if satellite:
                return ("warn", _SAT_WARN)
            return ("deny", f"`git add` with a broad selector stages every session's changes. {_SAFE_ADD}")
        return None

    if sub == "commit":
        if any(a == "--all" or _has_a_cluster(a) for a in sargs):
            if satellite:
                return ("warn", _SAT_WARN)
            return ("deny", f"`git commit -a/--all` stages and commits all tracked changes. {_SAFE_COMMIT}")
        has_pathspec = "--" in sargs or "--only" in sargs or "--include" in sargs or "-i" in sargs
        conclusion = any(a in _COMMIT_CONCLUSION_FLAGS for a in sargs)
        if not has_pathspec and not conclusion and not _merge_or_rebase_in_progress():
            if satellite:
                return ("warn", _SAT_WARN)
            return ("deny", f"Bare `git commit` commits the shared index. {_SAFE_COMMIT}")
        # The commit form is safe -- but WHERE is it landing? Canonical's resting
        # state is master; any other branch means some session moved the shared
        # HEAD, and this commit is about to stack onto their work.
        if not satellite:
            branch = _current_branch()
            if branch != _MASTER_BRANCH:
                # Once per branch per session: enough to be seen before the first
                # commit lands, quiet enough not to nag through a commit series.
                if _already_fired(_SESSION_ID, "offmaster_" + (branch or "detached")):
                    return None
                return ("warn", _off_master_message(branch))
        return None

    if sub == "reset":
        if "--hard" in sargs:
            return ("deny", "`git reset --hard` discards every session's uncommitted work in the shared tree. Use `git reset <paths>` (unstage) or `git restore -- <paths>` for your own files only.")
        return None

    if sub == "clean":
        if any(a == "--force" or re.fullmatch(r"-[a-z]*f[a-z]*", a) for a in sargs):
            return ("deny", "`git clean -f` deletes untracked files, including other sessions' new files. Delete specific paths explicitly instead.")
        return None

    if sub in ("checkout", "restore"):
        if "." in sargs:
            if satellite:
                return ("warn", _SAT_WARN)
            return ("deny", f"`git {sub} .` discards all working-tree changes, including other sessions'. Restore specific files: `git {sub} -- <paths>`.")
        if sub == "checkout" and any(a in ("-b", "-B") for a in sargs):
            return None if satellite else ("warn", "Hook note: creating/switching a branch changes the shared HEAD for every session in this checkout — confirm no other session is mid-edit first.")
        return None

    if sub == "switch":
        return None if satellite else ("warn", "Hook note: `git switch` changes the shared HEAD for every session in this checkout — confirm no other session is mid-edit first.")

    return None


def analyze(command, satellite=False):
    """First deny wins; else first warn; else None."""
    warn = None
    for seg in _segments(command):
        result = analyze_segment(_tokens(seg), satellite)
        if result is None:
            continue
        if result[0] == "deny":
            return result
        warn = warn or result
    return warn


def _already_fired(session_id, category):
    """Once-per-session-per-category marker (copied from stale_state_reminder.py)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id) or "nosession"
    marker = os.path.join(
        tempfile.gettempdir(), f"claude_git_guard_sat_{safe}_{category}"
    )
    if os.path.exists(marker):
        return True
    try:
        with open(marker, "w", encoding="utf-8"):
            pass
    except OSError:
        pass
    return False


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    try:
        if (data.get("tool_name") or "") not in ("Bash", "PowerShell"):
            return
        command = (data.get("tool_input") or {}).get("command") or ""
        global _SESSION_ID
        _SESSION_ID = data.get("session_id") or ""
        satellite = is_satellite()
        result = analyze(command, satellite)
        if result is None:
            return
        kind, message = result
        if kind == "deny":
            out = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": message,
            }
        else:
            if satellite and _already_fired(data.get("session_id") or "", "discipline"):
                return
            out = {"hookEventName": "PreToolUse", "additionalContext": message}
        print(json.dumps({"hookSpecificOutput": out}))
    except Exception:
        return  # fail-open: never brick a session on a guard bug


if __name__ == "__main__":
    main()
