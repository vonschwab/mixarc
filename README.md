# MixArc

Build intentional, arc-shaped playlists from your own music library — a learned sonic-similarity
embedding fused with a real genre taxonomy graph. Not a shuffle, not a "similar artists" API.

## Install & run

```bash
pipx install mixarc      # or: pip install mixarc
mixarc                   # opens a browser wizard to set up your library
```

The first run opens a setup wizard: pick your music folder, optionally connect Last.fm / Discogs /
Plex, choose a genre backend, and MixArc analyzes your library and starts building playlists. It's a
local, single-user app — your library never leaves your machine.

## What it does

Point it at a folder of music and a seed (an artist, a genre, or a set of tracks in the GUI). It
builds a playlist that:

- **Sounds like a DJ set, not a shuffle.** Sonic similarity comes from **MuQ**, a learned
  contrastive audio-text embedding — not hand-built rhythm/timbre/harmony features. Seeds anchor
  the playlist as fixed "piers"; a beam search builds a smoothly-transitioning bridge between each
  pair, so the whole thing moves somewhere instead of wandering.
- **Preserves your taste, not the algorithm's average.** Genres are resolved against a real
  taxonomy graph, not free-text tags — multi-genre signatures ("shoegaze + dreampop + slowcore")
  survive instead of collapsing to "indie rock."
- **Doesn't sag into generic filler.** Long bridges are actively kept from drifting into the
  dense, "sounds like everything" middle of the sound space, and any transition that still comes
  out weak gets one more pass to fix it before the playlist ships.
- **Respects diversity as a hard rule.** Per-artist caps, minimum spacing, and collaboration-aware
  identity resolution (so "Bill Evans Trio" and "Bill Evans feat. X" count as the same artist) are
  enforced, not just recommended.

## Requirements

- Python 3.11+
- A local folder of music files
- ~8 GB RAM for sonic analysis; SSD recommended for feature extraction
- A GPU is not required for MuQ extraction but speeds it up considerably
- Optional: Last.fm / Discogs / Plex accounts enrich genres & popularity (never required)

## Run from source (developers)

```bash
pip install -e ".[web,dev]"
python tools/serve_web.py
pytest -m "not slow"
```
Deeper design docs: `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`. For the full install-extras
matrix, the 15-stage analyze pipeline, and CLI generation examples, see **Quick Start** below.

## Features

- **Learned sonic similarity (MuQ).** The sole sonic embedding is `OpenMuQ/MuQ-MuLan-large`, a
  512-dimensional contrastive audio-text model — it beats an earlier acoustic embedding (MERT) and
  the original hand-built towers on trusted soundalike triplets. There's no runtime variant switch
  anymore; the old rhythm/timbre/harmony towers and MERT were removed from the codebase (archived,
  not deleted) once MuQ replaced them outright.
- **Pier-bridge beam search with collapse prevention.** Seeds become fixed piers; a constrained
  beam search fills each segment between them. Two levers keep long bridges from sagging into a
  generic local average instead of representing the seeds' character: an **anti-center** scoring
  penalty (demotes candidates that drift toward the pool's centroid instead of the piers), and
  **mini-piers** (splits an over-long segment by pinning a high-character waypoint as an extra
  pier, so the beam structurally can't drift past it).
- **Weak-edge recovery cascade.** After the beam finishes, a fixed four-pass cascade — lengthen the
  bridge, re-optimize the last couple of interior slots, swap one interior track (break-glass
  repair), or delete one interior track as a last resort — lifts any transition that's still weak,
  escalating from least- to most-destructive.
- **Genre graph authority.** Genres come from `release_effective_genres`, a single published table
  written only by the enrichment pipeline's `publish` stage and resolved against a living taxonomy
  graph (`data/layered_genre_taxonomy.yaml`). Multi-genre signatures are preserved end to end, and
  a hub guard keeps broad genres ("rock", "indie") from gluing the whole similarity matrix
  together.
- **Four independent axes.** `cohesion_mode` controls beam tightness; `genre_mode`, `sonic_mode`,
  and `pace_mode` independently gate what's allowed into the candidate pool. Mix and match — e.g.
  same genre with varied sonic texture, or tight sound across genre boundaries.
- **Tag-steering (artist mode).** When seeding from an artist, the GUI offers chips of that
  artist's own published genres — pick up to three to softly lean the playlist toward that facet of
  their catalog. It's an additive nudge, never a hard filter, and with no tags picked the run is
  byte-identical to not using it.
- **Multi-artist blend (artist mode).** Name two or more artists and the playlist finds the sonic
  and genre ground they share, instead of traveling from one artist's catalog to the other's.
  Jointly-credited records get seated as their own pier when they exist. The second and later
  artist chips share the same name autocomplete as the primary artist field, so a name typed in
  the wrong form doesn't silently fail to resolve.
- **Pace as tempo + rhythmic density.** Pace is gated independently of the sonic embedding, using
  BPM and onset-rate log-distance bands plus a soft rhythm penalty — so it keeps working the same
  way regardless of what the sonic embedding is doing, and survives beatless/ambient seeds where
  BPM alone is meaningless.
- **Artist identity resolution.** Ensemble suffixes ("Trio", "Quartet"), collaborations ("X feat.
  Y"), and "The"-prefixes are normalized before anything counts diversity, dedups, or excludes the
  seed artist from bridge interiors.
- **Manual artist links (aliases & sibling projects).** Beyond automatic normalization, you can
  hand-link names no algorithm can reconcile: an **alias** (same act, different spelling — "Alex G"
  / "(Sandy) Alex G") merges into one identity everywhere, while a **sibling** link (one person,
  distinct projects — "Smog" / "Bill Callahan", "Mount Eerie" / "The Microphones") keeps each
  project independent but never places them within `min_gap` of each other. Curated in the GUI's
  "Artist Links" tab, stored in a git-tracked `data/artist_aliases.yaml`; empty file = no effect,
  no database or artifact rebuild.
- **Multisource genre enrichment.** `scripts/analyze_library.py` runs the full pipeline —
  MusicBrainz/Discogs/Last.fm collection, Claude-based album-grain genre adjudication, publish —
  as one resumable, fingerprint-gated command. Claude calls run through the Agent SDK against a
  Claude subscription, no separate API billing.
- **Browser GUI.** Generate playlists, run library analysis/enrichment, review and adjudicate
  genres, and edit the taxonomy graph — all from a local web app. No desktop GUI dependency.

## What's new

- **MuQ is now the only sonic embedding.** The earlier MERT embedding and the original hand-built
  towers are gone from the runtime path (code removed, artifacts archived under
  `data/archive/mert_2026/`) — there's no `--sonic-variant` flag or config switch to pick between
  them anymore.
- **Collapse prevention shipped on.** Anti-center scoring and mini-piers (structural waypoint
  insertion) both ship active in `config.example.yaml`, along with variable-length bridges that
  flex a segment to land on a better edge instead of forcing a rigid even split.
- **Tag-steering.** Artist-mode playlists can now be softly steered toward specific genre facets of
  the seed artist's own catalog, picked from chips in the GUI.
- **Pace rebuilt on tempo + rhythmic density.** BPM and onset-rate bands replaced an old
  rhythm-cosine floor that was near-noise and unsatisfiable for ambient/beatless artists.
- **Manual artist links.** A new "Artist Links" GUI tab lets you declare aliases (full identity
  merge) and sibling projects (independent catalogs, kept spaced apart) for names automatic
  normalization can't catch — a runtime resolution layer with no `metadata.db` or artifact changes.
- **Rebuilt candidate pooling fixes broken transitions.** Each segment's candidates are now drawn
  from the whole library relative to that segment's two anchor tracks, instead of a fixed
  neighborhood decided before the playlist's shape was known — this closes the small number of
  cases where a genuinely bad, "sticks out" transition used to ship even though a much better
  connecting track existed in your library. The weak-edge recovery pass now also catches an edge
  that's merely *worse than what the rest of the playlist achieved*, not only edges that are
  outright broken. A seed artist's own outlier/one-off tracks are now less likely to be picked as
  the playlist's very first or last track when a more representative track from their catalog is
  available.

## Quick Start

```bash
# 1. Install (Python 3.11+ required)
pip install -e .[web]         # browser GUI + generation (recommended baseline)
pip install -e .[web,ai]      # + Claude Agent SDK, needed for genre enrichment
pip install -e .[web,ai,muq]  # + muq/torch, needed to BUILD the MuQ embedding (analyze only —
                               # not needed just to generate against an artifact that already has it)
pip install -e .[web,dev]     # contributors: + pytest, ruff, mypy, pre-commit (combine with any of the above)

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml: music_directory, database_path, API keys

# 3. Verify environment
python tools/doctor.py

# 4. Scan + enrich + build the artifact — one command, 15 ordered stages
python scripts/analyze_library.py
#   or a subset:  python scripts/analyze_library.py --stages scan,muq,artifacts,verify
#   see the stage list:  python scripts/analyze_library.py --help

# 5. Generate playlists
python main_app.py --artist "Slowdive" --tracks 30
python main_app.py --genre "shoegaze" --tracks 30
# Note: there is no --seeds flag. Multi-track seed playlists ("DJ Bridge" mode) are GUI-only —
# pick several tracks in the browser GUI and generate from there.

# 6. Launch the browser GUI (http://127.0.0.1:8770)
python tools/serve_web.py
```

See [docs/GOLDEN_COMMANDS.md](docs/GOLDEN_COMMANDS.md) for the full command reference and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how it all fits together.

## Playlist modes

Four independent axes control cohesion vs. discovery. Each can be set in the GUI or via CLI
flags. `cohesion_mode` drives the beam search itself; the other three shape what's allowed into
the candidate pool before the beam ever runs.

### Cohesion mode (`--cohesion-mode`)
Overall beam tightness — how strictly the bridge search holds to a coherent path.

| Mode | Behaviour |
|---|---|
| `strict` | Tightest beam — minimal drift between piers |
| `narrow` | Cohesive |
| `dynamic` | Balanced (default) |
| `discover` | Loosest — allows more exploratory bridges |

### Genre mode (`--genre-mode`)
How closely candidates must match the seed's genre profile, blended against sonic similarity:

| Mode | Genre weight | Behaviour |
|---|---|---|
| `strict` | 0.80 | Ultra-tight genre coherence — stay within the seed genre |
| `narrow` | 0.65 | Close to seed genre with some flexibility |
| `dynamic` | 0.50 | Balanced genre exploration (default) |
| `discover` | 0.35 | Venture into related genres |
| `off` | 0.00 | Sonic-only — ignore genre completely |

### Sonic mode (`--sonic-mode`)
Sonic-similarity admission floor and blend weight in the MuQ space:

| Mode | Sonic weight | Behaviour |
|---|---|---|
| `strict` | 0.85 | Very similar sound, minimal variation |
| `narrow` | 0.70 | Familiar sound, strict coherence |
| `dynamic` | 0.50 | Balanced (default) |
| `discover` | 0.35 | Broader sonic palette, varied textures |
| `off` | 0.00 | No sonic floor — genre-only |

### Pace mode (`--pace-mode`)
Tempo + rhythmic-density fidelity, independent of the sonic embedding — two hard log-distance
bands (BPM, onset rate) plus a soft penalty that widens on backoff so pace never blows the
generation budget. No `discover` level on this axis (only `strict` / `narrow` / `dynamic` / `off`).

| Mode | BPM band (admission / bridge, log₂) | Onset band (admission / bridge, log₂) | Soft penalty strength |
|---|---|---|---|
| `strict`  | 0.30 / 0.40 | 0.30 / 0.40 | 0.50 |
| `narrow`  | 0.50 / 0.60 | 0.50 / 0.60 | 0.40 |
| `dynamic` | 0.75 / 0.85 | 0.75 / 0.85 | 0.30 (default) |
| `off`     | ∞ / ∞ | ∞ / ∞ | 0 |

Because pace reads BPM/onset-rate database features rather than the sonic embedding, it works
identically no matter what the sonic space is doing — including on beatless/ambient seeds where a
raw BPM comparison would be meaningless (handled via an onset-rate trust gate).

### Examples

```bash
# Tight shoegaze/slowcore — stays slow, stays cohesive
python main_app.py --artist "Slowdive" \
    --cohesion-mode narrow --genre-mode narrow --pace-mode strict --tracks 30

# Same genre, explore varied sonic textures
python main_app.py --artist "Bill Evans" \
    --genre-mode narrow --sonic-mode discover --tracks 30

# Pure sonic similarity, no genre constraint
python main_app.py --genre "ambient" \
    --genre-mode off --sonic-mode dynamic --tracks 30
```

## Multi-seed ("DJ Bridge") playlists

Every playlist — single-artist or multi-seed — is built on the same pier-bridge topology: seeds
become fixed piers, and a beam search fills the bridge between each adjacent pair with a smooth
transition and a genre arc routed across the taxonomy graph. Multi-track seed lists are currently
**GUI-only** — select several tracks in the browser GUI to build a playlist that bridges between
all of them in sequence. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pier-bridge
walkthrough.

## Project structure

```
.
├── main_app.py                  # CLI entry point
├── config.example.yaml          # Configuration template (copy to config.yaml)
├── src/
│   ├── playlist/                # Generation pipeline
│   │   ├── pipeline/            # DS pipeline orchestration
│   │   ├── pier_bridge/         # Beam search, anti-sag scoring, pace gate, recovery cascade
│   │   ├── repair/              # Post-beam edge repair (weak-edge cascade)
│   │   ├── candidate_pool.py    # Admission filtering (sonic, genre, pace, IDF, tag-steering)
│   │   ├── tag_steering.py      # Artist-mode soft genre lean
│   │   ├── genre_idf.py         # IDF computation for genre weighting
│   │   └── mode_presets.py      # Cohesion/genre/sonic/pace mode presets
│   ├── analyze/                 # MuQ extraction runner + resumable shard build
│   ├── features/                # Audio feature extraction + artifact resolution
│   ├── genre/                   # Genre authority, taxonomy graph adapter, granularity
│   ├── ai_genre_enrichment/     # Multisource enrichment (collection + adjudication)
│   ├── playlist_web/            # FastAPI web app (Generate / Tools / Genre Review / Taxonomy)
│   └── playlist_gui/            # Generation worker (NDJSON) + shared policy layer
├── web/                         # React + TypeScript + Vite browser front-end
├── scripts/                     # analyze_library orchestrator, artifact/fold scripts
├── tools/
│   ├── doctor.py                # Environment validator
│   └── serve_web.py             # Browser GUI launcher
├── tests/                       # pytest suite (smoke / integration / golden / slow markers)
└── docs/                        # See docs/ARCHITECTURE.md for the current map
```

## Diagnostics

```yaml
# config.yaml — enable per-edge audit
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true   # per-edge transition/sonic/genre/bridge breakdown in logs
```

- **Edge audit** (`emit_selected_edge_audit: true`): logs the transition, sonic, genre, and bridge
  score for every transition, plus BPM distance and title flags.
- **Weakest-edge report**: always on — shows the lowest-transition edges with artist names.
- **Quality metrics**: every generation reports transition stats (min / mean / p10 / p90) and
  distinct-artist count.

### Track replacement (GUI)
Right-click any non-pier track in the playlist table → **Replace this track…**. The dialog offers
Search, Best Match, Different Pace, Different Genre, and Different Sound — the auto modes require
the replacement to clear the transition floor against both neighbours.

### Genre Review & Taxonomy (GUI)
The **Genre Review** and **Taxonomy** sub-tabs (under the Advanced panel) let you adjudicate
enrichment suggestions per release and grow/edit the taxonomy graph itself, respectively —
decisions persist immediately and feed back into the next generation.

### Artist Links (GUI)
The **Artist Links** sub-tab (under the Advanced panel) links artist names that should count as one
act. Pick a type — **Alias** (same act, different spelling → treated as one artist for seeds,
diversity, dedup, and Fire popularity) or **Same artist** (distinct projects → kept independent but
never placed within `min_gap` of each other) — add two or more members via typeahead over your
library's artists, and Save. Links take effect on the next generation (`data/artist_aliases.yaml`);
no re-scan or artifact rebuild.

## License

MIT — see [LICENSE](LICENSE).
