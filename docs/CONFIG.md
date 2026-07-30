# Configuration Reference

Key-by-key reference for `config.yaml`. For the system-level picture (what each subsystem *is*
and how the pieces fit), see [`ARCHITECTURE.md`](ARCHITECTURE.md); for tuning recipes and worked
examples, see [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md).

> **Reading the defaults in this doc.** Three layers set behavior, and they deliberately differ.
> (1) **Dataclass defaults** (mostly `src/playlist/pier_bridge/config.py`) are the *rollback*
> baseline — an experimental lever defaults `False`/`off` here so "no config at all" is always
> safe. (2) **`config.example.yaml`** is the *shipped default* — the validated stack turned **on**
> (what a fresh install copies, per the project rule "activate fixes, never default to legacy").
> (3) **`config.yaml`** (gitignored) is the *live* config on one machine, which may have moved
> further than the template. This doc documents the **shipped default** (`config.example.yaml`)
> for every key, and calls out the dataclass rollback wherever a key is a collapse-prevention or
> safety lever worth knowing how to disable. Two places where shipped and live have drifted apart
> in a way that matters are called out explicitly below — see **"Two known shipped-vs-live gaps."**

---

## `library`

| Key | Shipped default | What it does |
|---|---|---|
| `music_directory` | `/path/to/music` (placeholder) | Root directory the `scan` stage walks. |
| `database_path` | `data/metadata.db` | SQLite track DB — the irreplaceable authority store (see `ARCHITECTURE.md` "Key data stores"). |

---

## `analyze`

Configuration for the offline analyze pipeline (`scripts/analyze_library.py`). Stage order and
list are canonical in `src/playlist/request_models.py::ANALYZE_LIBRARY_STAGE_ORDER` — this
section only covers stages with tunable config.

### `analyze.muq`

MuQ is the sole sonic embedding (see `artifacts.sonic_variant_override` below and
`ARCHITECTURE.md`'s Sonic feature space section). `stage_muq` → `src/analyze/muq_runner.py` →
`muq_sidecar.npz` → auto-fold at the end of `artifacts`.

| Key | Shipped default | What it does |
|---|---|---|
| `device` | `cpu` | `cpu` \| `cuda` — device for MuQ-MuLan extraction. |
| `torch_threads` | `0` | `0` = use all available CPU threads. |
| `fold_into_artifact` | `true` | After every `artifacts` rebuild, fold the MuQ sidecar back in and set `X_sonic_variant=muq`. Set `false` to keep whatever variant the artifact already declares. |

### `analyze.energy`

Feeds the Essentia arousal/valence/danceability sidecar (WSL-only — Essentia has no native
Windows wheel). This is the data source for the `pace_mode` axis's energy-rescue and (currently
off) energy-arc levers.

| Key | Shipped default | What it does |
|---|---|---|
| `distro` | `Ubuntu-22.04` | WSL distro hosting the Essentia venv. |
| `python` | `/opt/ess/bin/python` | Interpreter inside that distro. |
| `models_dir` | `/opt/ess/models` | Essentia model directory (preflight-checked before the stage runs). |
| `workers` | `14` | Parallel decode workers; `--energy-workers` CLI flag overrides. |

### `analyze.pace`

| Key | Shipped default | What it does |
|---|---|---|
| `energy_features` | `["arousal_p50"]` | Which energy sidecar columns feed pace/energy scoring. `danceability`, `arousal_p10`, `arousal_p90` are also available. |

---

## `discogs` / `lastfm` / `plex`

Offline enrichment + export credentials. None of these gate runtime generation (local-first —
Layer 2 commitment #14).

| Key | Shipped default | What it does |
|---|---|---|
| `discogs.token` | `''` | Discogs API token (`discogs.com/settings/developers`); needed for the `discogs` analyze stage. |
| `lastfm.api_key` | `''` | Last.fm API key; feeds the `lastfm` analyze stage (tag fetch), the `popularity` stage (top-tracks), and recency filtering. |
| `lastfm.username` | `''` | Last.fm username for scrobble-history recency filtering. |
| `lastfm.history_days` | `90` | Days of Last.fm scrobble history to fetch. |
| `lastfm.artist_top_tracks_limit` | `50` | Top tracks fetched per artist (`popularity` stage). |
| `lastfm.recheck_miss_days` | `30` | An album with no Last.fm tags is skipped this many days before one retry (tags accrue over time); `0` disables the miss-cache (re-fetch every miss, every run). |
| `plex.enabled` | `false` | Turns on the Plex exporter. |
| `plex.base_url` | `http://localhost:32400` | Plex server URL. |
| `plex.token` | `''` | Plex auth token. |
| `plex.music_section` | `Music` | Plex library section name. |
| `plex.verify_ssl` | `true` | TLS verification for the Plex API. |
| `plex.replace_existing` | `true` | Overwrite a same-named Plex playlist instead of erroring. |
| `plex.path_map` | `[]` | Path-translation pairs (e.g. Docker container paths vs host paths). |

---

## `ai_genre`

Backend for the Claude-driven genre enrichment/adjudication stages (`adjudicate`, `apply`, and
the legacy opt-in `enrich`). All calls go through the Agent SDK — no API billing, uses the Claude
Max subscription (`ARCHITECTURE.md` "Offline: the analyze pipeline").

| Key | Shipped default | What it does |
|---|---|---|
| `provider` | `claude_code` | `claude_code` (Agent SDK, local subscription) \| `openai` (legacy, needs `OPENAI_API_KEY`). |
| `claude_model` | `haiku` | Model alias for `claude_code` calls: `haiku` \| `sonnet` \| `opus`. Note: the *production* genre path (`stage_adjudicate`) uses its own `--adjudicate-model` CLI flag, default `sonnet` — this config key governs the legacy `enrich` stage and other ad hoc Claude calls, not album adjudication. |

---

## `playlists` — top-level scalars

| Key | Shipped default | What it does |
|---|---|---|
| `count` | `8` | Playlists generated per batch run. |
| `tracks_per_playlist` | `30` | Target track count. |
| `seed_count` | `5` | Seeds drawn per playlist (multi-seed mode). |
| `name_prefix` | `'Auto:'` | Prefix for generated playlist names. |
| `name_format` | `artists` | Naming scheme. |
| `export_m3u` | `true` | Write `.m3u` files. |
| `m3u_export_path` | `/path/to/playlists` (placeholder) | M3U output directory. |
| `history_days` | `14` | General history lookback window. |
| `max_age_days` | `14` | Max age for "recently generated" bookkeeping. |
| `min_duration_minutes` | `90` | Minimum total playlist duration. |
| `min_track_duration_seconds` / `max_track_duration_seconds` | `46` / `720` | Per-track duration admission window. **`min_track_duration_seconds` is also a HARD gate on PIER candidacy** (2026-07-30), not just bridge/candidate-pool fill — see the callout below. |
| `max_tracks_per_artist` | `3` | Hard per-artist cap. |
| `artist_window_size` / `max_artist_per_window` | `8` / `1` | Sliding-window artist-diversity constraint. |
| `min_seed_artist_ratio` | `0.125` | Minimum fraction of the playlist that must be seed-artist tracks (artist mode). |
| `pipeline` | `ds` | Pipeline selector. **`ds` (pier-bridge) is the only live path** — the legacy greedy `constructor.py` is dead code, unconditionally bypassed (`core.py`'s `if True:`). This key is a vestigial selector, not a real fork. |
| `recently_played_filter.enabled` | `true` | Recency exclusion switch. |
| `recently_played_filter.lookback_days` | `30` | Recency window. Applied **pre-order**, during candidate-pool construction, never post-order — see the "Don't re-introduce post-order recency filtering" gotcha in the project `CLAUDE.md`. |
| `recently_played_filter.min_playcount_threshold` | `0` | Minimum scrobble count before a track counts as "recently played." |

**Pier duration floor (2026-07-30).** `min_track_duration_seconds` (shipped `46`) was a bridge/
candidate-pool filter only until a real case slipped through it: `Brian Eno & David Byrne - Vocal
Outtakes` (0:36) seated as a **pier** — the structural anchor a whole bridge segment is built
around — because pier selection (`cluster_artist_tracks`) had no duration gate of its own. It is
now a hard pre-clustering filter inside `cluster_artist_tracks` itself (`src/playlist/artist_style.py`),
exposed as that function's `min_pier_duration_seconds` parameter and applied at all four call sites
that select piers: single-artist (`create_playlist_for_artist`), multi-artist
(`select_multi_artist_piers`'s per-group call), genre mode (`create_playlist_for_genre`), and
history mode (`_create_playlists_from_single_artists`) — one choke point, so a future fifth caller
can't bypass it by forgetting its own pre-filter. The fallback when the config key is absent is
`src/config_loader.py::DEFAULT_MIN_TRACK_DURATION_SECONDS` (`46`) — a single source of truth added
because the same fallback had quietly drifted to three different literals (47/46/90) across
readers, inert only because the shipped config always pins the key explicitly. Unknown/zero
duration is **kept, not excluded** (mirrors `genre_mode.filter_member_indices_by_duration` — missing
metadata must never silently shrink the pier pool). If the gate starves an artist below the minimum
cluster size, it logs a **WARNING naming the artist and the removed-track count**, then falls
through to that path's existing thin-artist handling (single-artist: propagates to the artist-style
fallback; multi-artist: the per-group relaxation catch) — it never crashes and never silently
relaxes the floor.

---

## The four mode axes

`cohesion_mode` drives the beam; `genre_mode` / `sonic_mode` / `pace_mode` gate the candidate
pool. All four default to `dynamic`. Full behavioral description in `ARCHITECTURE.md`; this is
the config-key view.

| Axis | Key | Levels | What it maps to |
|---|---|---|---|
| Cohesion | `playlists.cohesion_mode` | `strict` / `narrow` / `dynamic` / `discover` | Per-mode `bridge_floor_<mode>`, `weight_bridge_<mode>` / `weight_transition_<mode>` under `pier_bridge`. |
| Genre | `playlists.genre_mode` | `strict` / `narrow` / `dynamic` / `discover` / `off` | `genre_similarity.weight`/`sonic_weight`/`min_genre_similarity`, plus `candidate_pool.min_sonic_similarity_<mode>` fallback and the `genre_admission_percentile_<mode>` / `genre_arc_floor_<mode>` pier-bridge floors. |
| Sonic | `playlists.sonic_mode` | `strict` / `narrow` / `dynamic` / `off` | `candidate_pool.similarity_floor`, `sonic_admission_percentile_<mode>`. |
| Pace | `playlists.pace_mode` | `strict` / `narrow` / `dynamic` / `off` | BPM/onset bridge-band widths + soft-penalty strengths; energy-rescue `k`. Pace is embedding-independent — it reads BPM/onset from the DB directly, so it survives any sonic-embedding change. |

All four keys are commented out in `config.example.yaml` (the mode-preset block is documented
inline at `playlists:` lines 44-98) — uncomment to pin a mode instead of the `dynamic` code
default. **Only the web GUI path routes mode strings through the policy layer**
(`src/playlist_gui/policy.py::derive_runtime_config`); the CLI sets these strings directly. A
test/harness that bypasses policy will see modes as inert — the `playlist-testing` skill's
"mirror real use case" rule exists because of this trap.

**GUI dial layer (2026-07-04).** The browser GUI doesn't expose these four axis strings
directly — it exposes three intent dials, Range / Flow / Pace, which compile down to the four
engine axis modes via `src/playlist_gui/policy.py::DIAL_TO_AXES` (`resolve_dial_axes`). Range
drives `sonic_mode` + `genre_mode` together, Flow drives `cohesion_mode`, and the dial named
Pace drives `pace_mode`. The dials are GUI-only — there is no `config.yaml` key or CLI flag for
them; `config.yaml` and the CLI keep speaking the raw axis vocabulary (`strict` / `narrow` /
`dynamic` / `discover` / `off`) documented above.

---

## `playlists.ds_pipeline`

The pier-bridge engine's configuration root (`playlists.ds_pipeline` in the file; every key
below is nested under it unless stated otherwise).

| Key | Shipped default | What it does |
|---|---|---|
| `artifact_path` | `data/artifacts/beat3tower_32k/data_matrices_step1.npz` | The generation artifact (sonic + genre + energy matrices). |
| `genre_source` | `legacy` | `legacy` (raw track/album/artist genre tables) \| `enriched` \| `graph` (published `release_effective_genres` authority — **the current production genre source**, live config runs `graph`) \| `hybrid_shadow`. Shipped default is `legacy` because a genuinely fresh clone has no publish history yet; flip to `graph` once `publish` has run. Changing this requires an `artifacts` + `genre-embedding` rebuild. |
| `genre_similarity.source` | `cooccurrence` | `cooccurrence` (Jaccard over library tag co-occurrence) \| `graph` (derived from `data/layered_genre_taxonomy.yaml`). Same bootstrap reasoning as `genre_source` — live runs `graph`. Flipping forces a similarity-matrix rebuild. |
| `random_seed` | `0` | `0` = non-deterministic; set an int for reproducible runs. |
| `enable_logging` | `true` | Per-run DS pipeline logging. |
| `embedding.sonic_components` / `.genre_components` | `32` / `32` | PCA dims for the legacy hybrid-embedding path (vestigial alongside `pipeline: ds`; not the pier-bridge beam's sonic space). |
| `embedding.sonic_weight` / `.genre_weight` | `0.60` / `0.40` | Same legacy path. |

### `candidate_pool`

| Key | Shipped default | What it does |
|---|---|---|
| `similarity_floor` | `0.20` | Minimum hybrid similarity to seed track. |
| `min_sonic_similarity_narrow` / `_dynamic` | `0.10` / `0.00` | Mode-specific hard sonic floor before genre/hybrid blending. |
| `max_pool_size` | `1200` | Candidate cap before construction. |
| `max_artist_fraction` | `0.125` | Max fraction of the pool from one artist. |
| `duration_penalty_enabled` / `_weight` | `true` / `0.60` | Soft geometric penalty on candidates much longer than the median seed duration. |
| `duration_cutoff_multiplier` | `2.5` | Hard exclusion multiplier vs. median seed duration. |
| `title_exclusion_enabled` / `_words` | `true` / `["interlude", "skit"]` | Hard-drop candidates whose title is (or contains, per the matcher) one of these standalone words. |
| `genre_rescue_k` | `40` | Re-admit the top-K sonic-nearest tracks rejected only by the genre hard gate — keeps sonic connectors so tight genre modes cannot crater the worst edge (2026-07-04). `0` = off. |
| `broad_filters` | `["rock", "indie", "alternative", "pop"]` | Tags ignored for narrow-mode genre gating/similarity — the "IDF lesson," applied at the candidate-pool level. |

### `scoring` and `constraints`

`scoring` (`alpha`/`beta`/`gamma` + an `alpha_schedule: arc`) and `constraints` (`min_gap: 6`,
`hard_floor: true`, per-mode `transition_floor_*`, `artist_identity.*`) belong to the legacy
greedy scoring math. They're still read by shared helpers (artist-identity normalization, in
particular, underpins `min_gap` enforcement everywhere — Layer 2 commitment #10), but the
per-edge score that actually drives selection today is the pier-bridge beam's, documented below.
See `config.example.yaml`'s inline comments for the full field list (identity `split_delimiters`,
ensemble-suffix stripping, etc.).

### `repair`

Legacy post-construction repair pass (`enabled: true`, `max_iters: 5`, `max_edges: 5`,
`objective: gap_penalty`) — distinct from `pier_bridge.edge_repair` below, which is the
pier-bridge-native break-glass repair. Both exist; `edge_repair` is the one that matters for the
`ds` pipeline.

---

## `playlists.ds_pipeline.pier_bridge`

The beam search that builds every segment between piers. Per-segment score = transition + bridge
(harmonic-mean similarity to both piers) + soft genre/pace/energy penalties, kept to the top
`initial_beam_width` (`40`, doubling toward `max_beam_width` `200` on infeasibility — a global
setting, not per-mode).

### The collapse-prevention stack

Long bridges tend to **sag** into the dense, genre-blurred "average" region instead of
representing the seeds' own character (see `DESIGN_RATIONALE.md` §"Collapse prevention" for the
full experimental history). `config.example.yaml` turns the structural/scoring core **on**, with
the dataclass rollback **off** for two of the three:

| Lever | Key(s) | Shipped value | Dataclass rollback | What it does |
|---|---|---|---|---|
| Variable bridge length | `variable_bridge_length` | `true` | `False` | A segment flexes its interior length (deterministic flex cap `variable_bridge_max_flex_segments`, soft total band) to lift the worst edge instead of padding to a rigid count. ~2.5-3x generation cost on a playlist with a weak segment. |
| Anti-center (SP2) | `seed_character_mode`, `seed_character_strength` | `anti_center`, `2.0` | `"off"`, `0.0` | Demotes bridge-interior candidates that sit closer to the local pool centroid than to their own piers — the direct anti-sag *score*. The alternative `hubness` mode was deleted (weaker, non-scaling); only `off`\|`anti_center` exist now. |
| Mini-piers (SP3) | `mini_pier_enabled`, `mini_pier_max_interior`, `mini_pier_smoothness_margin` | `true`, `5`, `0.12` | `False`, `5`, `0.12` | Splits an over-long segment by pinning a high-character waypoint as an extra pier — a *structural* guarantee the beam can't sag past it. Fixes the residual sag anti-center alone plateaus on (dreampop 103%→63%). |

> Anti-center is a scoring fix; mini-piers is a structural fix — they compose. `DESIGN_RATIONALE.md`
> also records the abandoned "density-floor" lever (6 formulations tried, none worked): dense is
> on-character for roughly half of seeds, so a pre-gen proxy can't tell sag from legitimate
> convergence.

Two more collapse-adjacent levers ship **on** in the template but are **last-mile / endgame**
passes rather than beam-time scoring:

| Lever | Key(s) | Shipped value | Dataclass default | What it does |
|---|---|---|---|---|
| Tail-DP | `tail_dp.enabled`, `.epsilon`, `.floor` | `true`, `0.02`, `0.3` | `True` (already on — one of the few levers whose code-level rollback is *not* off) | After a segment's beam + var-bridge finalizes, re-opens the last `min(2, interior)` slots and exactly maximizes the window min-edge over the segment's own pool. Never-worse; falls back to the original tail on any internal error. |
| Edge-delete | `edge_delete.enabled`, `.floor`, `.max_deletions` | `true`, `0.3`, `4` | `True` (also already on) | Remove-only last resort: after break-glass repair, if an edge is *still* below floor, deletes the interior track whose removal merges the two edges — only if the merge strictly lifts the worst edge, and never a pier/seed. |

Two more live **only** in one machine's `config.yaml`, not in the shipped template:

| Lever | Key(s) | Live value | Dataclass default | What it does |
|---|---|---|---|---|
| Roam corridors | `roam.enabled`, `.width_sonic`, `.worst_edge_minimax`, `.genre_gate_percentile`, `.genre_pair_floor` | `true`, `1.0`, `true`, `0.5`, `0.30` | `roam_corridors_enabled: False` | On-manifold kNN geodesic corridor construction + minimax/min-bottleneck ordering, with a light dense-genre gate and a taxonomy-level pairwise genre-edge floor. |
| Edge repair | `edge_repair.enabled`, `.centered_cos_floor`, `.margin`, `.t_floor`, `.variety_guard.*` | `true`, `-0.5`, `0.05`, `0.3`, `enabled: false` | `edge_repair_enabled: False` | Break-glass single-interior-track swap when an edge's raw centered cosine is catastrophically anti-aligned (`< -0.5`) — essentially never fires except on outright pathological edges. Accepts a swap only if the worst-T improves by at least `margin`. |

**Generation budget.** `generation_budget_s` bounds a whole generation with one shared deadline
computed at generation start and threaded into every segment loop + relaxation tier. Dataclass
default `60.0`; **shipped default `0`** = disabled — quality-first, every lever (beam / var-bridge
/ mini-piers) runs to completion with no fallback bail. A positive value re-arms the budget when
optimizing for speed instead. There's a 90s hard ceiling as a design target, not a separately
enforced code path (see `feedback_generation_time_budget` — a playlist must never take >90s in
practice, even with the soft budget disabled).

> **Two known shipped-vs-live gaps.**
>
> 1. **`edge_repair:` is entirely absent from `config.example.yaml`.** The live `config.yaml` runs
>    it (`enabled: true`), but a fresh clone gets the dataclass rollback (`edge_repair_enabled:
>    False`) — the break-glass repair pass is off out of the box. This is a template gap, not a
>    deliberate design choice.
> 2. **`artist_style.enabled: false` in `config.example.yaml`, but `true` in the live
>    `config.yaml`.** A fresh clone running `--artist` playlists gets the **legacy seed-selection
>    pier path** (no medoid clustering of the artist's catalog into style-representative piers) —
>    not the medoid-clustered pier path this doc's `ARCHITECTURE.md` describes as the default
>    behavior for artist mode. It also means the **tag-steering PIER lever is dormant** on a fresh
>    clone: `tag_steering_pier_weight` only has an effect through `ArtistStyleConfig.medoid_tag_weight`,
>    which is only consulted when `artist_style.enabled` is true. See `tag_steering_*` below.

### Tag steering

Artist-mode GUI feature: soft genre lean toward chips the user picks from the artist's own
published genres — never a gate, and inert (byte-identical to legacy) when no tags are supplied.

| Key | Shipped default | What it does |
|---|---|---|
| `tag_steering_pool_blend` | `0.5` | POOL lever: blends the tag target into the dense admission centroid (`0` = ignore tags / legacy, `1` = pool centroid is the pure tag target). Forces `genre_admission_aggregate: centroid` when tags are supplied. Active regardless of `artist_style.enabled` — it operates on the candidate pool, not on pier selection. |
| `tag_steering_pier_weight` | `0.3` | PIER lever: on-tag bonus added to medoid (pier) scoring in artist-style clustering. **Gated by `artist_style.enabled`** — see the gap above. If `artist_style.enabled` is `false`, this key is read into `ArtistStyleConfig` but the medoid-clustering code path that would consume it never runs. |

A per-request tag list arrives via the policy override `tag_steering_tags` (web-only, capped at
3 chips); the resolver `src/playlist/tag_steering.py::resolve_tag_steering_target` treats an
unmapped tag / missing dense-genre sidecar / zero-norm result as a loud warning, inert for that
run — never a silent no-op.

### `artist_style`

Style-aware pier selection for `--artist` playlists: clusters the artist's catalog into `k`
style groups (`cluster_k_min`/`cluster_k_max`, heuristic-picked unless disabled) and takes one
medoid per cluster as a pier, instead of picking piers by the legacy heuristic.

| Key | Shipped default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch — see the gap callout above. Live `config.yaml` runs `true`. |
| `cluster_k_min` / `cluster_k_max` | `3` / `6` | Cluster count bounds. |
| `cluster_k_heuristic_enabled` | `true` | Auto-pick `k` within bounds vs. a fixed value. |
| `piers_per_cluster` | `1` | Piers drawn per style cluster. |
| ~~`per_cluster_candidate_pool_size`~~ | ~~`400`~~ | **RETIRED 2026-07** (corridor Phase 1 Task 8): fed `build_balanced_candidate_pool` (deleted); corridor pooling's library-wide eligible universe replaces the artist-mode external pool. Warns loudly if still set. |
| ~~`pool_balance_mode`~~ | ~~`equal`~~ | **RETIRED 2026-07** (corridor Phase 1 Task 8), same reason. `proportional_capped` was never implemented anyway. |
| `internal_connector_priority` / `_max_per_segment` | `true` / `2` | Prefer the artist's own tracks as bridge connectors, up to this cap. |
| `bridge_floor.{strict,narrow,dynamic}` | `0.10` / `0.05` / `0.02` | Per-cohesion-mode floor for style-cluster bridges. |
| `bridge_score_weights.{dynamic,narrow}` | `{bridge:0.6,transition:0.4}` / `{bridge:0.7,transition:0.3}` | Per-mode edge-score weights for artist-style bridges. **Shadows** the global `pier_bridge.weight_bridge_<mode>` / `weight_transition_<mode>` keys for artist playlists — two sources, one wins silently. |
| `medoid_energy_weight` | `0.0` | Off. Biases each cluster's medoid toward an evenly-spaced arousal slot (artist energy-spread; validated as the winning lever over "popularity," still opt-in). |
| `energy_feature` | `arousal_p50` | `arousal_p10` \| `arousal_p50` \| `arousal_p90` \| `danceability`. |
| `energy_slot_lo_pct` / `_hi_pct` | `10.0` / `90.0` | Robust-percentile span for arousal slot targets. |
| `dedupe_versions` | `true` | Collapse to one canonical version per song before clustering (studio/remaster preferred over live/demo) so duplicate releases don't multiply as seeds. |
| `medoid_popularity_weight` | `0.0` | Off. Within-slot bias toward Last.fm-popular tracks. |
| `toptracks_min_artist_tracks` | `8` | Minimum local tracks before fetching Last.fm top-tracks for popularity scoring — independent of `enabled`. |

### `multi_artist`

Artist mode with 2+ chips: every pier is drawn from the sonic/genre territory the named artists
share (Brian Eno + Harold Budd skews ambient; Brian Eno + David Bowie skews art rock), instead of
the playlist traveling from one artist's territory to the other's. Inert below two chips —
single-artist requests take the unmodified legacy path. Soft bias only: it re-ranks pier
candidates, it never gates or excludes, and a pairing with no shared ground still generates (see
`PLAYLIST_ORDERING_TUNING.md`'s multi-artist blend knob for the full mechanism and tuning recipe).

| Key | Shipped default | What it does |
|---|---|---|
| `enabled` | `true` | Master switch — live by default (principle 22). `false` with 2+ chips requested warns loudly and generates from the first artist alone, never a silent no-op. |
| `overlap_weight` | `0.6` | Strength of the pull toward the other artist(s)' prototype in per-group medoid ranking. Raise it for piers to sit deeper in the shared region; `0.0` is a valid, explicit rollback to per-artist-characteristic piers (each group clusters as if solo). |
| `genre_share` | `0.25` | Split within the overlap-affinity term — `(1 - genre_share)` MuQ cosine + `genre_share` dense-genre similarity. Sonic-dominant by default, consistent with genre being a nudge everywhere else in the engine. Renormalizes to pure-sonic (with a loud WARNING) if `X_genre_dense` is unavailable. |
| `max_artists` | `4` | Ceiling on chips accepted; extra names are dropped with a WARNING naming them. Uncalibrated — revisit after listening to a 3+ chip blend. |
| `joint_pier_min_budget` | `3` | Minimum total pier budget before the jointly-credited group (tracks credited to 2+ chips) is guaranteed a seat. Below this the joint group still exists but competes for surplus like any other group. |
| `low_overlap_threshold` | `0.15` | Mean seated-pier affinity below this reports a relaxation notice ("these artists share little ground; piers stayed characteristic") — cosmetic only, never blocks generation. Placeholder pending a read of real affinity distributions across known-distant pairings. |

**Pier ordering (2026-07-30 forced-interleaving rewrite).** `order_with_alternation` FORCES the
theoretical maximum artist alternation the group sizes allow (not `n - 1` — a majority group forces
`2*m - n - 1` repeats), picking the minimax-best order among those that reach it. There is no
config key for this any more — `alternation_bonus` was a scored preference over a candidate set
(`order_clusters`' own greedy walks) that structurally could never contain an interleaved order in
the first place, so the bonus weight had nothing to buy; it is deleted, and a leftover value in
config.yaml is caught and warned on loudly at startup
(`src.playlist_gui.worker._RETIRED_MULTI_ARTIST_KEYS`).

Alternation yields by a **ladder**, not a cliff: the search takes the best order at each alternation
level from the theoretical maximum downward, and keeps the HIGHEST level whose pier-to-pier worst
edge clears `PIER_ADJACENCY_ALTERNATION_FLOOR` (`0.08`, in `src/playlist/multi_artist.py`). Only if
no level clears it does it fall back to the unconstrained best order — which by construction can
never be less alternating than that same order, since every level above was tried first. An earlier
all-or-nothing valve regressed a real pairing (Eno+Bowie) to *less* interleaving than before the
rewrite; the ladder exists because of that.

Note that `0.08` measures **pier-to-pier adjacency**, a different and naturally much lower quantity
than `ABSOLUTE_MIN_TRANSITION_FLOOR` (`0.40`), which measures the **final playlist's**
`min_transition` after the beam has bridged the gaps between piers. Conflating the two fired the
valve far too eagerly. Both floors are live and independent. See `PLAYLIST_ORDERING_TUNING.md`
Knob 13 for the full mechanism and measured examples.

**Pier budget.** Total piers for an N-group blend scale as `N * base` (each group gets a
single-artist-equivalent share), clamped by `track_count // 5` so bridge segments stay long enough
to be bridgeable — **not `// 3`**: that looser clamp measurably starved the beam (10 piers on a
30-track/3-group blend, ~2.2-track bridges) and cratered the worst edge to 0.4645; `// 5` (~6
piers, ~4.8-track bridges) recovered it to 0.6135. This clamp has no config surface — it's a fixed
formula in `src/playlist/multi_artist.py::total_pier_budget`.

**Per-artist cap scaling.** The single-artist per-artist cap (`candidate_pool.max_artist_fraction`)
is scaled by the number of surviving artist groups (**the joint group counts** — Eno+Budd is 3
groups, not 2) whenever the multi-artist branch produces piers, via
`multi_artist_group_count` threaded into `max_artist_fraction_final`. Without this, a legitimate
N-artist pier allocation self-reported as `artist_cap_violation` / `degraded` in the GUI on every
successful blend, because the cap was sized for one seed artist while the blend deliberately gives
each group its own allocation.

**Worst-edge acceptance bar.** A blend is not held to single-artist smoothness — it spans two
artists' territory by design. The guard is an absolute `min_transition` floor of `0.40`
(`multi_artist.ABSOLUTE_MIN_TRANSITION_FLOOR` — no config key, but a single named constant shared by
`order_with_alternation`'s own safety valve and the acceptance test
`test_worst_edge_stays_above_an_absolute_floor`, so the two can never drift apart), well clear of
ordinary blend roughness (~0.55) and well above this project's break-glass edge-repair trigger
(`T < 0.30`). See `PLAYLIST_ORDERING_TUNING.md` Knob 13 for the full calibration table (re-measured
2026-07-30 after the forced-interleaving rewrite) and per-pairing `mean_affinity`.

**Known limitations.** Full detail, with measured examples, lives in
`PLAYLIST_ORDERING_TUNING.md` Knob 13 — summary:
- **Popular Seeds (`popular_seeds_mode: on`/`fire`) has no effect on blend pier selection.** The
  blend owns pier selection outright once 2+ groups survive; it warns loudly rather than silently
  no-opping, but the dial does nothing to blend piers.
- **Tag steering's pier-allocation and anchor injection are skipped on blends**, for the same
  reason (the blend owns pier selection). The candidate-pool tag lever is unaffected.
- **A chameleon artist (diffuse catalog, low prototype cohesion) blends poorly, and the system
  says so** rather than silently producing a bad blend — see the Eno+Bowie example in the tuning
  doc.
- **Pier interleaving (A/B/A/B) is FORCED to the highest level that keeps a healthy pier edge.**
  `order_with_alternation` computes the true max achievable alternation for the group sizes, then
  walks a ladder down from it, keeping the highest level whose pier-to-pier worst edge clears
  `PIER_ADJACENCY_ALTERNATION_FLOOR` (`0.08`). Measured 2026-07-30, alternation before the rewrite →
  after: Vegyn + Black Moth Super Rainbow 1/5 → **5/5**; Brian Eno + Harold Budd → **4/5**;
  Brian Eno + David Bowie 2/5 → **5/5**. See `PLAYLIST_ORDERING_TUNING.md` Knob 13.

### Genre steering (`genre_steering_*`)

The beam routes a per-segment genre arc toward the next pier, in addition to (not instead of) the
sonic beam objective.

| Key | Shipped default | What it does |
|---|---|---|
| `genre_steering_enabled` | `true` | Master switch. |
| `genre_steering_source` | `taxonomy` | `taxonomy` (route through `data/layered_genre_taxonomy.yaml`, scored with hub-damped taxonomy similarity on in-artifact `X_genre_raw` — rebuild-robust, needs no per-track taxonomy assignments) \| `dense` (legacy 64-dim PMI-SVD co-occurrence embedding; **raises at generation time** if its sidecar is unavailable rather than silently steering on nothing — a knob that can't act is an error). |
| `genre_admission_aggregate` | `centroid` | `centroid` (one mean-of-seeds centroid for the global admission gate) \| `per_seed` (union of per-seed neighborhoods — fixes pool starvation on diverse seed sets). Live config runs `per_seed`. |
| `segment_pool_genre_weight` | `0.0` | `0.0` = pure sonic re-ranking of the segment pool (default); `>0` blends genre harmonic-mean into the re-rank without adding a new hard gate. |
| `weight_genre_{strict,narrow,dynamic,discover}` | `0.30`/`0.20`/`0.12`/`0.06` | Per-mode weight of the genre-arc vote inside the edge score (renormalized with `weight_bridge`/`weight_transition`). |
| `genre_arc_floor_{strict,narrow,dynamic,discover}` | `0.50`/`0.40`/`0.25`/`0.10` | Absolute-fallback floor for the genre-arc vote. |
| `genre_arc_floor_percentile_{strict,narrow,dynamic,discover}` | `0.90`/`0.85`/`0.70`/`0.50` | Distribution-relative floor version (used when calibrated). |
| `genre_admission_percentile_{strict,narrow,dynamic,discover}` | `0.92`/`0.90`/`0.85`/`0.70` | Adaptive percentile admission floor for the candidate pool. |
| `genre_pair_floor_{strict,narrow,dynamic,discover}` | `0.15`/`0.10`/`0.10`/`0.05` | Per-mode floor for the taxonomy-graph pairwise genre-edge check (adjacent tracks). Below floor → **soft demotion**, never a hard reject (a hard gate here detonates the relaxation cascade). |
| `genre_pair_penalty` | `0.5` | Score subtracted from a below-`genre_pair_floor` edge (global, not per-mode). |
| `soft_genre_penalty_threshold_{strict,narrow,dynamic,discover,off}` | `0.82`/`0.78`/`0.73`/`0.20`/`0.20` | A second, independent soft-penalty mechanism: below this per-mode genre-similarity threshold, `final_edge_score *= (1 - strength)`. |
| `soft_genre_penalty_strength_{strict,narrow,dynamic,discover,off}` | `0.40`/`0.30`/`0.15`/`0.10`/`0.10` | Strength for the penalty above. |
| `genre_tiebreak_weight` | `0.05` | Small genre tie-breaker; never rescues a hard-gated candidate. |
| `dj_route_shape` | `ladder` | Legacy DJ-bridging route shape default (see `dj_bridging` below). |

> **`max` is the only genre-edge metric shipped.** A soft-cosine alternative
> (`SoftGenrePairSimProvider`) was built and evaluated, then **rejected** — once the sonic space
> dominates selection, `max`'s coarse "share a close tag?" catches egregious disjoint edges just
> as well, and the soft metric's wins didn't replicate. There is no `metric: max|soft` config
> switch anywhere in the code; don't add one without re-reading
> `docs/archive/findings/GENRE_SOFT_METRIC_FINDINGS_2026-06-26.md` first (archived, local-only).

### `dj_bridging` (legacy, opt-in, default off)

An older genre-waypoint-routing pooling strategy (S1+S2+S3 union pooling, IDF-weighted ladder
routing through the genre graph, coverage bonuses). Shipped default `enabled: false` — the
`segment_scored` strategy (`segment_pool_strategy: segment_scored`, the shipped default) has
superseded its pooling role for ordinary generation. The full nested config (vector-mode target,
IDF weighting, coverage bonus, waypoint squashing, connector bias, relaxation) is extensively
documented inline in `config.example.yaml` and in
[`DJ_BRIDGE_ARCHITECTURE.md`](DJ_BRIDGE_ARCHITECTURE.md) — not reproduced here field-by-field.
Leave `enabled: false` unless you're specifically reviving this path.

### Other pier-bridge knobs worth knowing

| Key | Shipped default | What it does |
|---|---|---|
| `bridge_floor_{strict,narrow,dynamic}` | `0.10`/`0.05`/`0.02` | Bridge-local gate: `min(sim_to_pier_A, sim_to_pier_B) >= floor`. |
| `weight_bridge_{dynamic,narrow}` / `weight_transition_{dynamic,narrow}` | `0.6`/`0.4` and `0.7`/`0.3` | Edge-score weights: bridge (whole-track fit toward the next pier) vs. transition (the immediate end-of-A/start-of-B "gel"). |
| `segment_pool_strategy` | `segment_scored` | Builds each segment's candidate pool by scoring jointly against **both** endpoints, rather than a neighbor-list union. Recommended / shipped default. |
| `segment_pool_max` / `max_segment_pool_max` | `400` / `1200` | Per-segment candidate cap and its ceiling under relaxation. |
| `disallow_pier_artists_in_interiors` | `true` | Pier/seed artists never appear inside their own bridge segments. |
| `disallow_seed_artist_in_interiors` | `false` (file default) — **artist mode always disallows regardless of this key** | Whether the seed artist can appear in bridge interiors for multi-seed mode. |
| `progress.enabled` / `.monotonic_epsilon` / `.penalty_weight` | `true` / `0.05` / `0.15` | Penalizes "teleporting"/bouncing via projection onto the pier-A→pier-B direction. |
| `progress_arc.enabled` | `false` | Optional stronger progress-arc shaping (linear/arc target curve) — off by default, layered on top of `progress`. |
| `duration_penalty.enabled` / `.weight` | `true` / `0.30` | Geometric penalty on bridge candidates much longer than the pier tracks (separate from the candidate-pool duration penalty, which runs earlier in the pipeline). |
| ~~`infeasible_handling.{enabled,strategy,min_bridge_floor,backoff_steps,max_attempts_per_segment,widen_search_on_backoff,extra_*,transition_floor_relaxation_enabled,min_transition_floor,genre_arc_relaxation_enabled,min_genre_arc_percentile}`~~ | — | **RETIRED 2026-07** (corridor Phase 1 Task 8): the legacy per-segment relaxation ladder was deleted — the corridor widening ladder (`corridor_widen_step`/`corridor_widen_attempts`/`corridor_widen_improvement_epsilon`) is the sole segment-level recovery mechanism now. Warns loudly if still set. `infeasible_handling.guarantee_feasible` (terminal last-resort placement) and `.greedy_genre_weight` remain live, NOT retired. |
| `audit_run.enabled` | `false` | Opt-in markdown run-audit report under `docs/run_audits/` (`--audit-run` CLI flag also enables it). |
| `emit_selected_edge_audit` | not set (file default `false`) | Diagnostic per-edge log block (`T`, `T_centered_cos`, `bridge`, `trans_beam`, etc.) — see `PLAYLIST_ORDERING_TUNING.md` for how to read it. |

---

## `playlists.genre_similarity` (top level)

Distinct from `ds_pipeline.genre_similarity.source` above — this block feeds the mode presets
and the legacy hybrid-embedding weighting.

| Key | Shipped default | What it does |
|---|---|---|
| `enabled` | `true` | Master switch. |
| `weight` / `sonic_weight` | `0.50` / `0.50` | Genre vs. sonic weight in the hybrid score; overwritten by `genre_mode`/`sonic_mode` presets when those are set. |
| `min_genre_similarity` | `0.35` | Absolute floor — rollback path; inert while `admission_percentile` > 0. |
| `admission_percentile` | preset per `genre_mode` (strict 0.75 / narrow 0.60 / dynamic 0.40 / discover 0.20) | Adaptive floor over the POSITIVE genre-sim mass (admit top 1-p of genre-affine tracks); the live gate since 2026-07-04. Set explicitly to override the preset. |
| `method` | `ensemble` | Similarity method. |
| `broad_filters` | long list (decades, `rock`/`pop`/`indie`/..., vocalist/nationality tags, `live`/`demo`/`remix`/... quality tags) | Tags excluded from genre matching/gating — the corpus-wide "too common to be signal" list (Layer 2 commitment #12, "rare > common"). |

---

## `logging`

| Key | Shipped default | What it does |
|---|---|---|
| `level` | `DEBUG` | Console/file log level. |
| `file` | `logs/playlist_generator.log` | Main log file path. |
| `playlist_logs.enabled` | `true` | Per-playlist DEBUG log files under `playlist_logs.dir`, one file per generation, full DEBUG detail regardless of console level, cleaned up asynchronously after each run. Set `false` to restore the pre-2026-07-02 single-log-file behavior byte-for-byte. |
| `playlist_logs.dir` | `logs/playlists` | Per-playlist log directory. |
| `playlist_logs.retention_days` | `30` | Cleanup horizon for per-playlist logs. |
| `playlist_logs.level` | `DEBUG` | Level captured in the per-playlist file (independent of the console level). |

---

## `artifacts.sonic_variant_override`

Top-level section, absent by default (commented out in `config.example.yaml`).

MuQ is now the **sole** sonic embedding — SP-B removed the MERT and hand-built tower
(`rhythm`/`timbre`/`harmony`) code paths from `src/` entirely (archived, not deleted:
`data/archive/mert_2026/`). There is consequently **no `tower_weights` / `transition_weights`
config surface anymore** — both keys were deleted from `config.example.yaml` and the code
(`core.py` hardcodes the transition-weight lookup to `None`, and `TRANSITION_CALIB_BY_VARIANT`
now has a single `"muq"` entry). If you see either key referenced in an older doc or a stale
`config.yaml`, it's dead — don't re-add it.

```yaml
# artifacts:
#   sonic_variant_override: muq
```

- `sonic_variant_override` **wins over** the artifact-declared `X_sonic_variant` key when set.
  `analyze_library.py`'s own default is also `muq` now (`... or "muq"`), so leaving this key
  entirely absent already resolves to MuQ — the commented-out example is a seam for a **future**
  sonic variant (auditioning a new embedding) without re-baking the artifact's other keys, not a
  live rollback to something else.
- **A configured-but-missing `X_sonic_<variant>` key raises at load** — never a silent fallback
  (Layer 4's "a configured knob that can't act is a startup error").
- See `ARCHITECTURE.md`'s Sonic feature space section and `DESIGN_RATIONALE.md`'s "MuQ" entry for
  why MuQ replaced MERT (contrastive objective beats a bigger acoustic SSL model on fine-grained
  soundalike triplets: 86–89% vs MERT's 73%).

---

## See also

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system map; the three-layer defaults framing referenced
  throughout this doc is defined there.
- [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md) — tuning recipes and how to read
  the selected-edge audit log.
- [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md) — why each lever above exists, with the
  experiments and rejected alternatives.
