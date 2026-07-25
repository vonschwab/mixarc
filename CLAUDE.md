# Playlist Generator — project guide

For the listener-facing feature catalog, see `README.md`. Newest, most authoritative implementation doc is `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`. Active roadmap and audit findings: `audit/07-roadmap.md` (don't re-derive — evidence is cited as `[A#3]`, `[P#1]`, etc.).

## Environment

- **Python 3.11+** required (pinned in `pyproject.toml`; README's "3.8+" is stale).
- **Install:** `pip install -e .[web]` (users) or `pip install -e .[web,dev]` (contributors — adds pytest, ruff, mypy, pre-commit).
- **Test:** `pytest`. Markers: `smoke`, `integration`, `golden`, `slow`. Use `-m "not slow"` for fast feedback.
- **Lint / types:** `ruff check` (E, F rules) and `mypy`. The `extend-ignore` list and `[[tool.mypy.overrides]]` modules are intentional — each entry has a comment. Don't relax without flagging.
- **CLI:** `python main_app.py --artist "..." --tracks 30`. Full reference: `docs/GOLDEN_COMMANDS.md`.
- **GUI:** `python tools/serve_web.py` (browser GUI, default port 8770) — the only front-end. The PySide6 desktop GUI was removed 2026-06-10 (`docs/DEAD_CODE_AUDIT_2026-06-10.md`); `src/playlist_gui/` now holds only the worker process + the policy layer the web app shares.
- **Doctor:** `python tools/doctor.py`.

## Key paths

- `data/metadata.db` — SQLite track database
- `data/artifacts/beat3tower_32k/data_matrices_step1.npz` — pre-computed sonic + genre matrices
- `data/archive/mert_2026/mert_shards/` — MERT embedding shards + `manifest.json` — retired rollback data, superseded by MuQ; **ARCHIVED (never delete)** by SP-B Task 11 (2026-07-02); **irreplaceable** (~55h CPU to regenerate). See `data/archive/mert_2026/README.md`.
- `data/archive/mert_2026/mert_sidecar.npz` — merged MERT embeddings (+ historical `.bak.*`) — retired rollback data; **ARCHIVED (never delete)**; **irreplaceable** (regenerated from shards via `--merge-only`, but shards are the ground truth)
- `data/archive/mert_2026/mert_transform_calibration.npz` — fitted MERT transform params — retired rollback data, **ARCHIVED**; re-fittable from sidecar, but keep
- `config.yaml` — gitignored; copy from `config.example.yaml`

## Session discipline

Distilled from recurring session friction (insights review 2026-06-12). Short rules — the deep how-to lives in the `playlist-testing` and `web-gui` skills.

- **Check process state before debugging "broken" behavior.** Most "the fix didn't work" reports are stale state, not logic bugs: `serve_web.py` not restarted after a worker edit, `web/dist` not rebuilt after a front-end edit, `@lru_cache` holding an old artifact. Walk the `web-gui` skill's trap catalog before reading code.
- **To explain WHY a playlist came out the way it did, read the generation logs — never trust summary metrics alone.** Every wrong conclusion in the 2026-06-19 pace/energy investigation (a BPM-gate-disabled confound, a "narrow is inert" pool-starvation artifact, "sonic admission is the lever" when the onset band was the real one) survived until someone ran a real generation at INFO and read the gate-tally + per-segment pool lines. A metric like "0/12 tracks changed" can mean a true null, a starved pool where the beam never ran, *or* the knob silently not applying — only the log distinguishes them. The `playlist-testing` skill's "Diagnosing a generation outcome" section lists exactly what to grep.
- **Pytest: never pipe through `tail`/`head`, always bound the run.** Piped pytest output has hung sessions. Run `python -m pytest -q -m "not slow"` directly and use the tool's timeout parameter, not a shell pipe.
- **Don't claim "failures are isolated to X" until the full suite has run.** A green subset is not a green suite. Quote real pass/fail counts from output you actually saw.
- **Generation tests must mirror production.** Multi-pier seeds through the `gui_fidelity` harness — never hand-built overrides, never single-seed topology. The `playlist-testing` skill is mandatory reading before writing one.
- **After a fix, exercise the real path before declaring success.** Restart the worker, run the feature end-to-end, and look for regressions adjacent to the change — a past scan-crash fix shipped with a `job_id` leak that cost a second debugging round.
- **GUI design work follows `docs/UI_UX_DISCIPLINE.md`.** Hard, verifiable UI rules — touch targets, `dvh`, tokens/contrast, type scale, states/copy, iOS lifecycle — plus the `mobile-audit` screenshot sweep (`web/tests/mobile-audit.spec.ts`, evidence under `web/test-results/`). Run the sweep and *look at the screenshots* before and after layout changes. Current violations: `docs/UI_AUDIT_2026-07-09.md`. (The `web-gui` skill covers wiring; the discipline doc covers design.)
- **Project concepts: search before answering.** Asked what something project-specific does, grep code/docs first — never infer from the name.
- **Prior art before design.** Before designing any fix, feature, or investigation for a subsystem, check the auto-memory index (MEMORY.md) and `docs/` (HANDOFF_*, INCIDENT_*, `superpowers/specs/`, `superpowers/plans/`) for prior decisions on it — design from prior art, not first principles. First-principles redesigns of already-decided subsystems are a recurring interrupt cause (insights review 2026-07-06). The `reuse-first` skill covers the code half of this rule.
- **Scope check before long work.** Before committing to a long execution path, restate the intended deliverable in one sentence and confirm any genuinely ambiguous boundary (e.g., web-enriched vs offline-only). One question, not five.
- **Git stays goal-focused.** Shortest safe path to the user's actual goal; if a push/merge is blocked, one fix attempt, then ask before going deeper.
- **Simultaneous sessions: three workspaces — canonical + standing satellite clones (worktrees stay retired).** The canonical checkout (this directory) sits on `master` and owns ALL data writes. Satellites `C:\Users\Dylan\Desktop\PG3_SAT1` (GUI port 8771) and `PG3_SAT2` (8772) are full clones with their own branches and full dev powers — real generations included — reaching canonical `data/` via absolute paths in their gitignored `config.yaml`. One active session per workspace; ALWAYS launch the session with cwd already in its workspace (mid-session switching breaks hook/subagent anchoring). Bootstrap a new satellite with `python tools/create_satellite.py --name PG3_SAT3 --port 8773`. Spec: `docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md`.
- **Satellite rules.** Work on feature branches off `origin/master` (`git fetch origin` first — staleness is the main satellite risk); land by `git push origin <branch>` then merging to `master` in canonical. Data-writing pipeline stages (`analyze_library.py`, folds, MuQ extraction) are canonical-only — `satellite_data_write_guard` denies them in satellites. GUI genre publishing (Genre Review "Publish decided") is canonical-only by policy (port 8770). If a satellite behaves oddly, run `python tools/doctor.py` there first — it validates the absolute data paths (the clone's own `data/metadata.db` is a 0-byte stub; config must point away from it).
- **Canonical (shared) checkout: stage and commit explicit paths ONLY — hook-enforced.** Never `git add -A`/`-u`/`.` and never a bare `git commit` (both sweep other sessions' in-flight work from the shared index). Use `git add <paths>` then `git commit --only -m "msg" -- <paths>`, and verify with `git diff --cached --name-only` first. **`-m` and the message must precede `--`** — everything after `--` is a pathspec, so the `-- <paths> -m "msg"` ordering fails with a confusing pathspec error. For multi-line messages, or any message containing backticks or `$`, use a single-quoted heredoc (`git commit --only -F - -- <paths> <<'EOF'`): backticks inside a double-quoted bash string are command-substituted and vanish silently. The `git_shared_checkout_guard` hook denies the sweeping/destroying forms here (in satellites it downgrades to a once-per-session reminder — their trees are private). Treat unexpected modified files or commits that appear under you as another session's work: leave them out, leave them alone, and re-derive groupings from the live diff, never a remembered snapshot. Keep your uncommitted pile small — commit your own paths frequently.

---

## Design principles

Four layers: **destination → architectural commitments → current best methods → engineering discipline**. The first two are durable. The third is replaceable when something better emerges. The fourth governs how we change anything without breaking the rest.

### Layer 1 — North star: what a playlist should *be*

The destination. These don't change if we rewrite the engine tomorrow.

1. **A playlist should feel intentional, like someone curated it.** Not a shuffle, not a random walk through "similar tracks." There's a hand on it.
2. **A playlist has an arc.** It moves somewhere over its duration — energy, mood, texture, era, density. A flat line is a failure mode even if every track is individually good.
3. **A playlist reflects this listener's taste, not music in general.** The user's seeds, library, and history are the gravity. Generic "popular and similar" is the failure mode.
4. **A playlist surprises without disorienting.** Familiar enough to feel like home, fresh enough to discover something. Whiplash and monotony are the twin failure modes.
5. **The worst edge defines the experience.** A single jarring transition breaks a 30-track playlist. Quality means floor quality, not average. Optimize for "no broken moments," not "good on average."
6. **The user should feel listened-to.** What they just heard, what they own, what they seeded — these are respected. The system serves the user; it doesn't impose on them.

### Layer 2 — Architectural commitments

What we believe is *necessary* to reach the north star. Stronger than any one algorithm.

7. **Sonic ⊗ genre fusion is necessary.** Sonic alone misses cultural context; genre alone misses sonic feel. The fusion is the value prop.
8. **Sonic feel is multi-dimensional.** Rhythm, timbre, and harmony each carry independent musical meaning. A *naive* collapse of hand-built axes loses signal — the tower era proved this — but a *learned contrastive* embedding (MuQ) can capture the multi-dimensional feel holistically, without explicit decomposition. The commitment is to preserving the feel, not to any one representation of it.
9. **Multi-genre signatures must be preserved.** A track tagged "shoegaze + dreampop + slowcore" is not "indie rock." Reducing to a single label destroys taste fidelity.
10. **Identity is computed, not raw.** Ensemble suffixes ("Trio", "Quartet"), collaborations ("X feat. Y"), and "The"-prefixes are normalized before any counting. Underpins diversity, dedup, and seed-artist exclusion.
11. **Diversity is part of the experience, enforced as a hard constraint.** Min-gap, per-artist cap, seed-artist disallowed in bridge interiors. Enforce; don't recommend.
12. **Rare > common when expressing taste.** Niche signals (shoegaze, slowcore) outweigh popular labels (indie rock). The user's taste is the ground truth, not the global average.
13. **Recency awareness.** Respect "I just heard this." Recency exclusion lives in candidate-pool construction (pre-order), never post-order.
14. **Local-first.** Desktop app on the user's library. External APIs (Last.fm, MusicBrainz, Discogs, Plex) enrich offline and export at the end — they never gate runtime generation.

### Layer 3 — Current best methods (so far)

What we believe works *today*. Replaceable in principle, but they encode hard-won lessons — don't replace them casually.

15. **Pier-bridge with beam search** is our current best playlist topology. Seeds anchor structure; bridges are beam-searched per segment; progress is monotonic in sonic space.
16. **Taxonomy-graph genre steering** is our current best genre-arc method: the beam routes a per-segment genre arc through the SP3a taxonomy graph (shortest path between pier genres + smoothed targets, hub-guarded), on the in-artifact `X_genre_raw` (rebuild-robust). The earlier vector-mode + IDF + coverage-bonus system (the `dj_bridging` lever — it solved hub-genre collapse and saturation) is now **off by default and superseded** by taxonomy steering; it survives as an opt-in code path (`dj_bridging_enabled`).
17. **HISTORICAL (superseded by MuQ, SP-B 2026-07-01):** the tower-weighted blend (rhythm 0.20 / timbre 0.50 / harmony 0.30, harmony via 2DFTM) was our sonic decomposition through v6. Lesson kept: independent rhythm/timbre/harmony axes carried real signal — worth remembering if a future embedding needs re-decomposing.
18. **HISTORICAL (superseded by MuQ, SP-B 2026-07-01):** `transition_weights` had to align with `tower_weights` — the beam's per-edge score and the reporter's post-hoc T must share one feature balance, or the beam approves edges the reporter scores poorly. Lesson kept: any future multi-component sonic space needs the same alignment discipline between its admission and transition weights.
19. **Four independent mode axes** are our current best UX for cohesion-vs-discovery: `cohesion_mode` (beam tightness: strict/narrow/dynamic/discover), `genre_mode` (genre pool gating: strict/narrow/dynamic/discover/off), `sonic_mode` (sonic pool gating: strict/narrow/dynamic/off), `pace_mode` (rhythm gating: strict/narrow/dynamic/off). Per-mode pier-bridge knobs (`bridge_floor_<mode>`, `weight_bridge_<mode>`, `soft_genre_penalty_*_<mode>`) are keyed by `cohesion_mode`.

### Layer 4 — Engineering discipline

How we evolve the system without breaking the layers above.

19. **Continuous gradients beat hard cliffs.** Centered baselines, smooth squashing, weighted (not binary) scoring. Saturation hides influence; ties hide signal.
20. **Diagnostic logging is part of the feature.** Per-step values, winner-changed counts, saturation indicators, opt-in audit reports under `docs/run_audits/`. If a scoring component can't be measured, it doesn't ship.
21. **Quality metrics are first-class output.** Every generation produces transition stats (min / mean / p10 / p90), weakest-edge report, distinct-artist count.
22. **Activate fixes; never default to legacy.** When a problem is fixed with sound, tested R&D, the fix becomes the **live default** — keep it configurable in `config.yaml` for tuning and an explicit rollback, but never default to the old path. This is a single-user app, so "backward-compatible by default" protects nothing; it just leaves fixes off — *the #1 failure mode here* (2026-06-10 audit: beam widths ran at half config for months, the pace gate was dead in the live path, the web policy silently disabled dj_bridging). **Never merge an inactive fix:** if it can't be activated and validated yet (blocked on a dependency), hold the branch — a built-but-off scaffold is not shipped, it's the failure mode. **Clean up deprecated code as you find it:** once a fix is the default, delete the legacy path; don't leave dead/superseded branches lingering. (Pairs with the "a configured knob that can't act is a startup error" gotcha.)
23. **Tunability over hardcoded behavior.** Knobs go in `config.yaml`. When a new knob is added, document the tuning recipe.
24. **Pre-compute heavy work; warm path stays fast.** N+1 SQL in generation (audit `[P#1]`) and uncached artifact decodes (`[P#3]`) are bugs, not perf nits.
25. **Edge cases get graceful fallbacks.** <4 artist tracks → error with suggested mode. Infeasible bridge → progressively relax (`bridge_floor` → pool size → beam width → `min_gap`). Don't crash.

---

## Hotspots

Oversized monoliths from earlier phases. Read carefully before editing; prefer extracting helpers over adding to them.

| File | Size | Role |
|------|------|------|
| `src/playlist/pier_bridge_builder.py` | ~5.3k LOC | DJ Bridge beam search + pool union |
| `src/playlist_generator.py` | ~4.1k LOC | Top-level orchestrator |
| `src/playlist/pipeline.py` | ~1.9k LOC | Single-seed DS pipeline (treated as load-bearing) |
| `src/playlist_gui/worker.py` | ~2.5k LOC | NDJSON IPC worker (spawned by the web bridge) |

GUI/backend coupling is already clean (audit `[A#5]`). The architecture problem is god-classes within layers, not coupling between them.

## Project-specific gotchas

- **`data/metadata.db` is irreplaceable — treat it like production.** A full re-analysis takes days. Never write to, migrate, or alter the database without explicit user instruction followed by a second confirmation. Back it up **atomically** before any write — `python tools/copy_db_safe.py <dest>` or `src.db_backup.backup_database` (online-backup snapshot + `integrity_check`); **never `cp`/`Copy-Item` an open DB** — a torn WAL copy silently corrupts an index (the 2026-07 incident propagated a desynced index into *every* backup, undetected for a month). The `data_safety_guard` hook now blocks raw copies of the live DB. When in doubt, stop and ask.
- **`data/archive/mert_2026/mert_shards/` and `mert_sidecar.npz` are irreplaceable — ~55h CPU to regenerate.** Retired to the archive by SP-B (Task 11, 2026-07-02) but STILL never delete or overwrite. The shards are the ground truth; the sidecar is derived from them via `--merge-only`. The live sonic space is now MuQ (`muq_sidecar.npz` in `data/artifacts/beat3tower_32k/`); its extraction backs up before writing (see `src/analyze/muq_runner.py`).
- **Music library files are permanently read-only.** The audio files on disk are never written, moved, renamed, or deleted — ever. Read access only, no exceptions.
- **A configured knob that can't act is a startup error, not a silent no-op.** This codebase's recurring failure mode is config that looks wired but isn't (2026-06-10 audit: beam widths ran at half config for months; the pace gate was dead in the live path; the web policy silently disabled dj_bridging). When adding a gate or knob, make the missing-data path warn loudly or raise — never fall back silently. See `docs/DEAD_CODE_AUDIT_2026-06-10.md`.
- **Don't re-introduce post-order recency filtering.** Recency lives pre-order, in pool construction. The v3.4 fix exists for a reason — seed tracks at pier positions may be recently played but are explicitly requested.
- **Sonic space is MuQ (`X_sonic_muq`, 512-dim, `sonic_variant_override: muq`); the tower/MERT code paths were removed by SP-B (2026-07-01/02).** The Task-10 artifact rebuild is done — the live artifact carries only `X_sonic_muq*` sonic keys — and the MERT source data is archived (`data/archive/mert_2026/`); future embeddings slot in via the variant seam (extract a sidecar, add a `fold_<variant>` + auto-fold, register its transition calibration). (`tower_weights`/`transition_weights` alignment and the 0.20/0.50/0.30 tower blend are historical — see Layer 3 items 17/18.)
- **Pace is a separate axis from the sonic embedding.** `pace_mode` controls rhythm/tempo via DB features (BPM + onset-rate bands + soft penalty), *independently* of the (MuQ) sonic space — so it survived the MERT→MuQ migration unchanged. `docs/PLAYLIST_ORDERING_TUNING.md` Knob 5. `dynamic` preserves current behavior; `strict`/`narrow` engage rhythm admission and bridge gates.
- **Genre compatibility gating (`candidate_pool.genre_compatibility_enabled`) is OFF by default.** When enabled it scores candidate-vs-seed genre compatibility on the 410-dim raw genre vocabulary via `genre_compatibility_compatible_threshold` (0.35) / `genre_compatibility_conflict_threshold` (0.15) and demotes off-axis candidates by `genre_compatibility_penalty_strength`. Enabling genre gating against the raw vocabulary with identity affinity can reject legitimate candidates — keep it off / prefer the soft penalty. (These were the old `genre_conflict_*` keys, since renamed; the `_min_confidence` reject threshold no longer exists.)
- **Segment-pool one-per-artist collapse is OFF by default in `config.yaml`** (`pier_bridge.collapse_segment_pool_by_artist: false`). The beam enforces per-segment artist diversity on its own; the upstream pool collapse only starves the projection of bridging candidates. Don't re-enable without understanding the trade-off.
- **`[[tool.mypy.overrides]]` is for clean modules that should stay clean.** Don't add new modules there to silence errors — type the code instead.
- **`cohesion_mode` drives the beam; the other three slider axes drive pool composition.** All four axes (`cohesion_mode`, `genre_mode`, `sonic_mode`, `pace_mode`) live at `playlists.<axis>` in `config.yaml`. The pier-bridge per-mode knobs (`bridge_floor_<mode>`, `weight_bridge_<mode>`, `soft_genre_penalty_*_<mode>`) are keyed by `cohesion_mode`. The old `playlists.ds_pipeline.mode` key is gone — use `cohesion_mode` instead.
- **`sonic_only` mode no longer exists.** The closest equivalent is `genre_mode: off` (disables genre pool gating) combined with `cohesion_mode: discover` (relaxed beam).
- **Genre mode in the GUI:** the web GUI sends `genre_mode` through the API (`GenerateRequestBody`); the old "CLI-only" limitation applied to the removed PySide6 GUI.

---

## SQLite schema reference

**Rule:** Always run `SELECT name FROM sqlite_master WHERE type='table'` and `PRAGMA table_info(<table>)` before writing any query against an unfamiliar table. Never guess column names.

### `data/metadata.db`

Primary key convention: TEXT keys everywhere, no integer surrogate IDs.

> ⚠️ **GENRE AUTHORITY — read this before any genre query.** The published genre authority is
> **`release_effective_genres`** (album-keyed; graph-resolved + user overrides + layers), read via
> **`src/genre/authority.py`** — see the `genre-data-authority` skill. The `track_genres` /
> `album_genres` / `artist_genres` / **`track_effective_genres`** tables below are **RAW PIPELINE
> INPUTS**, not the authority: they are unreconciled per-source tags, MusicBrainz-polluted, and
> **missing file-sourced tags that the authority has**. Reading them as "the genre for a track" is a
> real bug that shipped (2026-07-08: tag steering read `track_effective_genres`, so user-tagged
> Hauntology/Ghost-Box records were invisible while `release_effective_genres` had them correctly).
> For per-track genre membership, join `tracks.album_id → release_effective_genres` (filter
> `assignment_layer NOT LIKE 'inferred%'` for observed tags). Never wire a consumer to a raw `*_genres`
> table.

| Table | Key columns |
|-------|-------------|
| `tracks` | `track_id TEXT` (PK), `artist TEXT`, `album TEXT`, `album_id TEXT`, `artist_key TEXT`, `norm_artist TEXT`, `norm_title TEXT`, `file_path TEXT` |
| `release_effective_genres` | **THE AUTHORITY** — `album_id TEXT`, `release_key TEXT`, `genre_id TEXT`, `assignment_layer TEXT` (`observed_leaf`/`legacy`/`inferred_*`), `confidence REAL`, `source TEXT`. Read via `src/genre/authority.py` |
| `genre_graph_canonical_genres` | `genre_id TEXT`, `name TEXT` — canonical id↔name map (`genre_id` uses underscores, `name` uses spaces); `genre_graph_aliases(alias, canonical_genre_id)` for aliases |
| `track_genres` *(raw input)* | `track_id TEXT`, `genre TEXT`, `source TEXT`, `weight REAL` — per-source file/MB/Discogs tags |
| `track_effective_genres` *(raw input, NOT the authority)* | `track_id TEXT`, `genre TEXT`, `source TEXT`, `priority INTEGER`, `weight REAL` — partial/stale; do not use as the genre source |
| `albums` | `album_id TEXT` (PK), `title TEXT`, `artist TEXT` |
| `album_genres` *(raw input)* | `album_id TEXT`, `genre TEXT`, `source TEXT` |
| `artists` | `artist_name TEXT` (PK — **not** `artist_id` or `id`) |
| `artist_genres` *(raw input)* | `artist TEXT`, `genre TEXT`, `source TEXT` — plain name string, **not** a FK integer |

Common patterns:
```sql
-- Genres for a track/album — THE AUTHORITY (prefer src/genre/authority.py helpers)
SELECT reg.genre_id, reg.assignment_layer, reg.confidence
FROM release_effective_genres reg JOIN tracks t ON t.album_id = reg.album_id
WHERE t.track_id = ? AND reg.assignment_layer NOT LIKE 'inferred%';

-- Tracks carrying a genre (chip name -> canonical id via the canonical/alias tables)
SELECT DISTINCT t.track_id
FROM tracks t JOIN release_effective_genres reg ON reg.album_id = t.album_id
WHERE reg.genre_id = 'jangle_pop' AND reg.assignment_layer NOT LIKE 'inferred%';

-- Raw per-source tags (INSPECTION/DEBUG ONLY — never as the genre source):
-- SELECT genre, source FROM track_genres WHERE track_id = ?;
```

### `data/ai_genre_enrichment.db`

| Table | Key columns |
|-------|-------------|
| `enriched_genre_signatures` | `release_key TEXT` (PK), `normalized_artist TEXT`, `normalized_album TEXT`, `album_id TEXT`, `signature_json TEXT` |
| `enriched_genres` | `enriched_genre_id INTEGER` (PK), `release_key TEXT`, `genre TEXT`, `confidence REAL`, `status TEXT` |
| `ai_genre_user_overrides` | `release_key TEXT` (PK), `genres_add_json TEXT`, `genres_remove_json TEXT` |
| `ai_genre_source_pages` | `source_page_id INTEGER` (PK), `release_key TEXT`, `source_url TEXT`, `source_domain TEXT` |
| `ai_genre_suggestions` | `suggestion_id INTEGER` (PK), `check_id INTEGER`, `genre TEXT`, `confidence REAL` |

`release_key` format: `"normalized_artist::normalized_album"` where normalization strips Unicode punctuation (P/S categories → space, lowercased). Example: `"acetone::york blvd"` (period stripped).

**Open bug:** `genre_resolver.py::_release_key` uses `normalize_source_tag` (no punctuation stripping), producing `"acetone::york blvd."` — mismatches the stored key. Albums with trailing punctuation silently miss enrichment lookup. Fix: align `_release_key` to use `normalize_release_name` from `normalization.py`.
