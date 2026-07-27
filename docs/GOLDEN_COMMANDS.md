# Golden Commands

The canonical command reference for the Playlist Generator. For *what the system is*, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for *why it's shaped this way*, see
[`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md); for the full config key list, see
[`CONFIG.md`](CONFIG.md).

## Prerequisites

1. Python 3.11+ (pinned in `pyproject.toml`).
2. Install dependencies:
   - `pip install -e .[web]` — GUI only, no genre enrichment.
   - `pip install -e .[web,ai]` — adds `claude-agent-sdk`, needed for the `adjudicate` /
     `apply` / `enrich` stages (genre production path). **The `web` extra alone is not enough
     to run enrichment** — a common gap the README glosses over.
   - `pip install -e .[web,ai,muq]` — adds `muq` + `torch`, needed to *build* the MuQ sonic
     embedding (the `muq` stage). Not required to generate playlists against an artifact that
     already has MuQ folded in.
   - Add `,dev` to any of the above for contributors (pytest, ruff, mypy, pre-commit).
3. Copy `config.example.yaml` to `config.yaml` and fill in your library path (`config.yaml` is
   gitignored — never commit it, it can carry API keys).
4. Ensure `data/metadata.db` exists (created by the `scan` stage of the analyze pipeline).

---

## 0. Doctor — environment verification

```bash
python tools/doctor.py
```

Checks, in order: Python version (3.11+), required dependencies importable, module imports,
`config.yaml` present and valid, `data/metadata.db` schema + row counts, artifact `.npz` files.
Exit is a pass/warn/fail summary with actionable fixes, not a bare pass/fail.

```bash
python tools/doctor.py --verbose      # per-check detail
python tools/doctor.py --no-color     # plain output (CI / log capture)
```

---

## 1. The analyze pipeline — `scripts/analyze_library.py`

**One command, one ordered stage list.** The single source of truth is
`ANALYZE_LIBRARY_STAGE_ORDER` in `src/playlist/request_models.py` — the CLI, the worker, and the
GUI Tools panel all drive this same list. Each stage is fingerprint-gated (`analyze_state`
table): an unchanged stage is skipped on re-run unless `--force` is given. `scan` never skips.

### Full run (default — runs every stage below, in order)

```bash
python scripts/analyze_library.py
```

### Verified stage order (15 stages)

| # | Stage | Does |
|---|-------|------|
| 1 | `scan` | Filesystem scan (incremental; `--force` = full) + orphan cleanup. |
| 2 | `genres` | MusicBrainz artist/release genres for tracks still missing them. |
| 3 | `discogs` | Discogs release/master genres + styles per album (needs `DISCOGS_TOKEN`). |
| 4 | `lastfm` | Last.fm top tags → enrichment sidecar (deterministic classification only, no LLM). |
| 5 | `sonic` | Legacy beat3tower hand-built features (rhythm/timbre/harmony towers — the sonic *rollback* space, no longer the runtime default). |
| 6 | `muq` | MuQ-MuLan contrastive sonic embedding → `muq_sidecar.npz`. **No-ops unless `muq` is the active sonic variant** (variant-gated, resumable/incremental otherwise). |
| 7 | `adjudicate` | Album-grain Claude (Sonnet) genre adjudication — the production genre path. |
| 8 | `apply` | Deterministic (no-LLM) materialize of adjudications; escalations → human review queue. |
| 9 | `publish` | Writes `release_effective_genres` — the genre authority. First-ever publish backs up `metadata.db`. |
| 10 | `genre-sim` | Builds the graph-based genre similarity matrix. |
| 11 | `artifacts` | Builds `data_matrices_step1.npz`, then auto-folds the MuQ (and any other sonic) sidecar back in. |
| 12 | `energy` | Essentia arousal/valence/danceability sidecar (WSL-only; the pace/energy axis). |
| 13 | `popularity` | Last.fm top-tracks popularity sidecar (for popular-seeds / bangers). |
| 14 | `genre-embedding` | Dense PMI-SVD genre embedding sidecar (legacy steering source). |
| 15 | `verify` | Post-build sanity: manifest fingerprint, row-count parity, `X_sonic_variant` must equal the configured variant or it errors loudly. |

**There is no separate `mert` stage.** MERT and the tower sonic space were removed from the
codebase (SP-B, 2026-07-01/02) — MuQ is the sole learned sonic embedding, and `sonic` now only
produces the legacy rollback towers. Don't add `--stages ...,mert,...`; it isn't a registered
stage and `analyze_library.py` will reject it as unknown.

**Two stages exist in code but are excluded from the default order** — pass them explicitly via
`--stages` if you need them:
- `mbid` — MusicBrainz ID backfill/repair (`--force-no-match` / `--force-error` /
  `--force-reject` / `--force-all` target its retry buckets).
- `enrich` — the legacy tag-grain enrichment path (Haiku-default, chunked). Superseded by
  `adjudicate`+`apply`; kept for one-off tag review, not part of the production run.

### Common invocations

```bash
# Run a subset of stages (comma-separated, order is normalized to the canonical order
# regardless of how you list them)
python scripts/analyze_library.py --stages scan,muq,artifacts,verify

# Re-run a stage even if its fingerprint says nothing changed
python scripts/analyze_library.py --stages publish --force

# Dry-run: print the plan and exit, no writes
python scripts/analyze_library.py --dry-run

# Limit scope for a quick smoke test
python scripts/analyze_library.py --stages scan,sonic --limit 100

# Genre adjudication with an explicit model (default: sonnet)
python scripts/analyze_library.py --stages adjudicate --adjudicate-model sonnet

# Legacy tag-grain enrichment, custom chunk size
python scripts/analyze_library.py --stages enrich --enrich-chunk-size 25

# Point at a non-default config or DB
python scripts/analyze_library.py --config config.yaml --db-path data/metadata.db
```

### Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--config` | `config.yaml` | |
| `--db-path` | from config | Override DB path. |
| `--stages` | all 15, canonical order | Comma-separated; unknown stage names raise. |
| `--workers` | `auto` | Workers for the `sonic` stage. |
| `--energy-workers` | config `analyze.energy.workers` | Workers for the WSL `energy` stage. |
| `--limit` | none | Caps tracks processed for `sonic`/`artifacts`. |
| `--max-tracks` | `0` (all) | Caps tracks for the artifact build specifically. |
| `--force` | off | Bypass the fingerprint gate; re-run requested stages fully. |
| `--force-no-match` / `--force-error` / `--force-reject` / `--force-all` | off | `mbid` stage: reprocess specific retry buckets. |
| `--out-dir` | default artifact dir | Output directory for artifacts. |
| `--dry-run` | off | Print the plan, make no writes. |
| `--progress` / `--no-progress` | on | Toggle progress logging. |
| `--progress-interval` | `15.0` s | |
| `--progress-every` | `500` items | |
| `--verbose` | off | Per-item DEBUG logging. |
| `--lastfm-api-key` | env `LASTFM_API_KEY` / config | For the `lastfm` stage. |
| `--model` | provider default | Model override for the legacy `enrich` stage. |
| `--adjudicate-model` | `sonnet` | Model for the `adjudicate` stage. |
| `--enrich-chunk-size` | `50` | Tags per adjudication chunk, `enrich` stage only. |
| `--beat-sync` | — | **Deprecated**: legacy sonic mode is disabled; flag is a no-op. |

Genre enrichment (`adjudicate`, `apply`, and the legacy `enrich`) runs through the Claude Agent
SDK against your Claude Max subscription — **no per-call API billing**, but it does require the
`ai` extra installed and Claude Code authenticated on the machine running the pipeline.

---

## 2. Playlist generation — `main_app.py`

Three mutually-exclusive seed sources; if neither `--artist` nor `--genre` is given, it generates
from listening history (multiple playlists, one per recent-history bucket). There is **no
`--seeds` flag on the CLI** — arbitrary multi-track seed mode is a GUI/worker-only path
(`GenerateMode = "seeds"` in `request_models.py`); the CLI only exposes artist and genre seeding.

```bash
# Generate for a specific artist (recommended starting point)
python main_app.py --artist "Radiohead" --tracks 30

# Anchor on a specific seed track within that artist
python main_app.py --artist "David Bowie" --track "Life On Mars" --tracks 30

# Generate for a genre
python main_app.py --genre "ambient" --tracks 30

# Dry run — preview without writing M3U files
python main_app.py --artist "Radiohead" --dry-run

# No --artist/--genre: generate from listening history
python main_app.py
```

### The four mode axes

`cohesion_mode` drives the beam; the other three gate the candidate pool. All default to
`dynamic` (from `config.yaml` `playlists.<axis>` if not passed on the CLI).

```bash
python main_app.py --artist "Radiohead" --cohesion-mode strict     # strict|narrow|dynamic|discover
python main_app.py --artist "Radiohead" --genre-mode narrow        # strict|narrow|dynamic|discover|off
python main_app.py --artist "Radiohead" --sonic-mode discover      # strict|narrow|dynamic|discover|off
python main_app.py --artist "Radiohead" --pace-mode off            # strict|narrow|dynamic|off (no --discover)

# Sonic-only, no genre gating: genre_mode off + a relaxed beam
python main_app.py --artist "Radiohead" --genre-mode off --cohesion-mode discover
```

### Quick presets — `--mode`

A shorthand that sets `genre_mode`/`sonic_mode` together (individual `--genre-mode`/
`--sonic-mode` flags, if also given, override the preset):

```bash
python main_app.py --artist "Radiohead" --mode tight          # strict + strict
python main_app.py --artist "Radiohead" --mode exploratory    # discover + discover
python main_app.py --artist "Radiohead" --mode sonic_only      # sonic-heavy, genre relaxed
python main_app.py --artist "Radiohead" --mode genre_only      # genre-heavy, sonic relaxed
```
Choices: `balanced` (default) | `tight` | `exploratory` | `sonic_only` | `genre_only` |
`varied_sound` | `sonic_thread`.

### Other useful flags

| Flag | Notes |
|------|-------|
| `--tracks N` | Playlist length (default 30). |
| `--anchor-seed-ids "id1,id2"` | Fix pier seeds by `rating_key` (artist mode only). |
| `--artist-only` | Restrict the pool to the requested artist — no discovery. |
| `--verbose` | DEBUG logging incl. transition metrics + constraint enforcement — **the first thing to reach for when a generation looks wrong; see the `playlist-testing` skill.** |
| `--audit-run` / `--audit-run-dir DIR` | Write a per-run pier-bridge markdown audit (default `docs/run_audits`). |
| `--pb-experiment-*` | Experimental/diagnostic pier-bridge scoring knobs — not for normal use. |

There is no `--sonic-variant` flag — sonic-space selection is a `config.yaml`
(`artifacts.sonic_variant_override`) concern now that MuQ is the sole learned embedding; the CLI
no longer exposes a per-run override.

**Note on the CLI vs. the GUI:** the CLI sets `genre_mode`/`sonic_mode`/`pace_mode` directly and
does **not** go through the policy layer (`src/playlist_gui/policy.py::derive_runtime_config`)
that the web GUI uses. A few policy-owned translations (recency filtering, artist-spacing →
`min_gap`, popular-seeds/bangers) are GUI/worker-only. For faithful mode testing, use the GUI or
route a harness through the policy layer — see the `playlist-testing` skill.

---

## 3. Web GUI

```bash
python tools/serve_web.py                 # build web/dist, serve, open browser — port 8770
python tools/serve_web.py --no-build       # skip the frontend rebuild (iterating via `npm run dev`, or fast restart)
python tools/serve_web.py --no-browser     # don't auto-open a browser tab
python tools/serve_web.py --port 8080 --host 0.0.0.0
```

`serve_web.py` rebuilds `web/dist` on every launch unless `--no-build` is passed — the FastAPI
app spawns one long-lived worker subprocess at startup (`src/playlist_gui/worker.py`); it does
**not** hot-reload on a `worker.py` edit, restart the server for those to take effect. See the
`web-gui` skill for the full trap catalog (stale dist, worker restart, silently-dropped results).

---

## Quick-start workflow (new library)

```bash
# 1. Verify environment
python tools/doctor.py

# 2. Copy and edit config
cp config.example.yaml config.yaml   # then set library.music_directory, etc.

# 3. Run the full analyze pipeline (scan through verify — this is the long step)
python scripts/analyze_library.py

# 4. Generate a test playlist
python main_app.py --artist "Your Favorite Artist" --tracks 20 --dry-run

# 5. Launch the GUI for everyday use
python tools/serve_web.py
```

## Typical maintenance workflow (after adding new music)

```bash
# Re-run the full pipeline — fingerprint gating makes this cheap: unchanged
# stages/tracks are skipped automatically, only new/changed material is processed.
python scripts/analyze_library.py

# Or scope to just what changed if you know it (e.g. only new tracks need genres re-published)
python scripts/analyze_library.py --stages scan,genres,discogs,lastfm,muq,adjudicate,apply,publish,genre-sim,artifacts,verify

# Spot-check with a dry run before trusting a generation
python main_app.py --artist "Newly Added Artist" --dry-run
```

## Troubleshooting quick-reference

- **`analyze_library.py` says "Unknown stage: mert"** — there is no `mert` stage anymore (SP-B
  removed it); use `muq` for sonic embedding extraction.
- **Enrichment stages fail with an import error** — you installed `.[web]` but not `.[ai]`
  (`claude-agent-sdk` is required for `adjudicate`/`apply`/`enrich`).
- **`muq` stage says skipped/no-op** — it's variant-gated: it only runs when `muq` is the active
  sonic variant (`artifacts.sonic_variant_override` / artifact-declared `X_sonic_variant`).
- **A playlist looks wrong** — re-run with `--verbose` and read the log, don't trust summary
  metrics alone; see the `playlist-testing` skill's "Diagnosing a generation outcome" section.
- **GUI shows stale behavior after a code edit** — rebuild (`npm --prefix web run build`) for
  `web/src` changes, restart `serve_web.py` for `worker.py`/backend changes.
