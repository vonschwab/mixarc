# Playlist Ordering Tuning Recipe

A practical, knob-by-knob guide for playlist ordering and track-quality problems in the
pier-bridge engine. For orientation on *what the pieces are*, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for *why* a default landed where it did, see
[`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md) — this doc states the knob and the trade-off, not
the experiment behind it.

> **Reading the defaults below.** Same three layers as `ARCHITECTURE.md`: (1) **dataclass
> defaults** (`src/playlist/pier_bridge/config.py`) are the rollback baseline — every
> experimental lever is off/legacy here; (2) **`config.example.yaml`** is the shipped default —
> what a clean install actually runs; (3) **`config.yaml`** is the live, gitignored config on one
> machine. Each knob below names its config key and states which of the three layers it ships
> at, so you can always see the rollback path.

**Scope note:** sonic similarity is a single MuQ contrastive embedding — there is no
rhythm/timbre/harmony tower to reweight and no variant switch to document. See
`ARCHITECTURE.md` §"Sonic feature space" for the embedding itself; the knobs below are all about
pool composition, beam scoring, and post-beam repair, not sonic-space choice.

---

## Symptoms that suggest these knobs may help

- High-`T` transitions still feel jarring (texture/density mismatch despite 0.9+ scores)
- Demo, live, or medley tracks appearing unprompted in playlists
- A few catastrophically bad edges (`T < 0.20`) per playlist despite a high mean `T`
- `min_transition` in stats is much lower than `mean_transition`
- A long bridge's interior tracks feel like generic filler, not like either pier ("sag")
- Ambient/beatless seeds strand a segment, or a playlist takes far longer than others to build

---

## Step 1: Enable the diagnostic audit (always do this first)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true
```

Dataclass default: `False`. Not set in `config.example.yaml` (diagnostic-only, zero behavior
change, opt in per debugging session).

Generate a playlist. The log will include a **"Selected-edge audit"** section
(`src/playlist/reporter.py::emit_selected_edge_audit`) with per-edge:

- `T`, `T_centered_cos`, `S` (sonic cosine), `G` (genre similarity)
- `bridge_score` (harmonic mean sim to both piers)
- `trans_beam` (beam's internal transition score — comparable to `T`)
- `progress_t`, `progress_jump`
- `local_sonic_cos` (raw cosine of the edge, uncentered)
- `local_pen`, `genre_pen` (penalties applied)
- `title_flags` (artifact flags for the destination track)
- `bpm=A->B dist=D` (BPM values + log-distance, when both tracks have trusted BPM)
- `⚠` prefix on edges below the transition floor

Also watch for `WARNING: T-mismatch edge` lines
(`src/playlist/reporter.py::diagnose_t_mismatch`). These are regression signals: the beam and
final reporter share the same transition metric, so a mismatch usually means stale audit data or
a missing-data fallback, not normal tuning drift.

---

## Knob 0 (removed): sonic is a single MuQ embedding

**There is nothing to align here anymore.** SP-B (merged 2026-07-01/02) deleted the hand-built
rhythm/timbre/harmony towers and the MERT/tower-variant switch entirely — sonic similarity now
runs on one contrastive **MuQ** embedding (`OpenMuQ/MuQ-MuLan-large`, 512-d, `center_l2`
post-processed). `tower_weights` and `transition_weights` are gone from `config.example.yaml`,
and the runtime call site hardcodes the diagnostic field to `None`
(`src/playlist/pipeline/core.py:1242`, kept only for downstream consumers that still read the
key). There's no rhythm/timbre/harmony split left to reweight, so the beam and the reporter score
plain cosine in the same MuQ space by construction — not because two config blocks happen to
match. If an old `config.yaml` still sets `tower_weights` / `transition_weights`, they're inert
and safe to delete; nothing reads them.

This heading stays numbered "Knob 0" so older cross-references in this doc ("do Knob 0 first")
still resolve, but there's nothing to do here anymore — go straight to Knob 1.

See `ARCHITECTURE.md` §"Sonic feature space" for the current single-embedding picture and
`DESIGN_RATIONALE.md` for the towers → MERT → MuQ removal arc.

---

## Pool-level artist cap + never-starve backstop (retired 2026-07, corridor Phase 0)

The `candidates_per_artist`, `target_artists`, `max_pool_size`, `min_pool_size`, and `seed_artist_bonus` knobs (under `playlists.ds_pipeline.candidate_pool` and `playlists.ds_pipeline.pier_bridge.*_pool_size`) are retired. The beam now enforces diversity natively; pre-beam pool artist caps and min-pool backstops are no longer needed. See `docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md` for the rationale.

---

## Corridor pooling (Phase 1, 2026-07): the segment admission mechanism

`docs/superpowers/specs/2026-07-12-corridor-first-pooling-design.md`, flipped to the sole
pier-bridge pooling path (2026-07-17); see
`docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` §5.0 for the full universe → corridor → widening →
reseat walkthrough. This section is the knob-by-knob tuning reference.

**Per-mode widths + escape hatch.** `corridor_width_percentile` (`Optional[float]`, default
`None`) is the plain global override — set it to force one width regardless of `sonic_mode`, the
same "tuning escape hatch" pattern as every other per-mode knob in this codebase. Left `None` (the
default), the active `sonic_mode` resolves a width via the per-mode fields below
(`src/playlist/pier_bridge/config.py:415-419`, `resolve_corridor_width_percentile` in
`corridor.py:293`):

| Config key | Default | `sonic_mode` | Effect of raising it |
|---|---|---|---|
| `corridor_width_percentile_strict` | 0.985 | `strict` | Tighter corridor (fewer, closer candidates) — home/strict cells |
| `corridor_width_percentile_narrow` | 0.975 | `narrow` | Midway; **pinned by direct probe** (Phase 2 Task 4 — 3 artists × 3 widths on the `close` detent; 0.975 beat both the old provisional 0.9675 interpolation AND the bracketing widths on mean and worst-case min_T) |
| `corridor_width_percentile_dynamic` | 0.95 | `dynamic`, unset, or unrecognized | The historical flat pin; open/dynamic cells |
| `corridor_width_percentile_discover` | 0.94 | `discover` | Widest of the four named modes — **pinned by direct probe** (Phase 2 Task 4 — same method on the `wander` detent; 0.94 won decisively, e.g. Alex G/wander +0.074, over the old 0.93 interpolation). Confounded in part by `segment_pool_max=800` cap saturation on 2 of 3 probe artists — the win partly traces to pre-cap ranking, not post-cap size (documented, not resolved) |
| *(hardcoded, no field)* | 0.0 | `off` | Whole eligible universe qualifies — no sonic narrowing at all |

**Tuning:** raising a mode's percentile tightens that mode's corridor (fewer members, higher
min-sim floor) — use when a mode feels too loose/generic. Lowering it widens (more members, lower
floor) — use when a mode craters (a segment can't find a path, or the widening ladder fires on
nearly every segment). `narrow`/`discover` are now pinned from real 18-generation probes (Phase 2
Task 4, `.superpowers/sdd/p2-task-4-report.md`), not interpolation — but if you retune them further,
re-run the 12-cell corpus to confirm the change
before trusting it.

**The widening ladder.** Three knobs govern the sole segment-level recovery mechanism (replaces
every legacy relaxation tier for corridor segments — see the flow doc §5.0 point 4 for the full
state machine):

| Config key | Default | Meaning |
|---|---|---|
| `corridor_widen_step` | 0.05 | Percentile subtracted from `width_percentile` per widen attempt (lower percentile = wider corridor) |
| `corridor_widen_attempts` | 2 | Max widen attempts after the initial (un-widened) build, once the quality trigger (`min_edge_T < transition_floor`) fires |
| `corridor_widen_improvement_epsilon` | 0.02 | Attempt ≥2 only widens further if the *previous* attempt improved the best-seen `min_edge_T` by more than this; otherwise the ladder stops early and hands the best-seen path to the repair stack |

**Tuning:** a corpus with frequent `CorridorWiden[seg N] EXHAUSTED` warnings (see `LOGGING.md`)
either needs a wider starting `corridor_width_percentile_<mode>` (fewer widen cycles needed at
all) or a larger `corridor_widen_attempts`/smaller `corridor_widen_step` (finer-grained widening,
more attempts to find a working width) — the latter costs wall-clock (each attempt re-builds the
corridor + re-runs the beam). `corridor_widen_improvement_epsilon` trades quality-chasing against
wall-clock: lower it to keep widening on marginal gains (slower, occasionally better); raise it to
stop sooner (faster, accepts a merely-adequate path sooner). The Task 6 remediation report
(`.superpowers/sdd/p1-task6-remediation-report.md`) has the empirical basis for 0.02 — an earlier
*predictive* gate (widen only if anchor support looked promising) was tried and **falsified** by a
real cell (Alex G/home) that needed a widen the predictor said wouldn't help; the current gate is
purely empirical (did the last attempt actually help) for that reason.

**`segment_pool_max` / `max_segment_pool_max`.** Unchanged in *meaning* from the pre-corridor
world (still the cap on a segment's ranked candidate count after `build_corridor`'s harmonic-mean
ranking, with `force_include` ids exempt from the cap), but now capping a **corridor**'s ranked
output rather than a KNN-union pool's. `segment_pool_max` defaults to 400, escalating (doubling,
capped at `max_segment_pool_max`=1200) via the separate, pre-existing expansion-attempts loop
inside `_run_segment_backoff_attempts` — this is an orthogonal axis from the widening ladder's
`width_percentile`: widening controls *which* candidates qualify as corridor members; the pool-max
expansion controls *how many* of the ranked, qualifying members the beam gets to see. Both can fire
in the same segment build.

**Small-fixture note (Task 8 finding, keep in mind for new unit tests).** Corridor's
percentile-based membership degenerates on very small (2-4 candidate) synthetic fixtures — a fixed
floor and a percentile behave identically only in the large-N limit. Several existing unit tests
needed an explicit `corridor_width_percentile=0.0` override to restore the admission behavior
their fixed-floor-era fixtures assumed. If you write a new tiny-fixture unit test against the
corridor path, expect the same: either build a fixture with enough rows for percentile math to be
meaningful, or override `corridor_width_percentile=0.0` explicitly rather than relying on a
per-mode default.

**Retired keys.** Corridor's Task 8 flip retired 24 warning conditions (`pooling`,
`collapse_segment_pool_by_artist`, the 14-key legacy `infeasible_handling.*` relaxation ladder, the
6-key `dj_bridging.pooling.*` KNN-union family + its `strategy: dj_union` value, and Artist mode's
`per_cluster_candidate_pool_size`/`pool_balance_mode`) — every one now fires a loud startup warning
via `_warn_retired_keys` rather than silently no-op-ing. Full list + rationale per key:
`docs/CONFIG.md` (struck-through rows) and `.superpowers/sdd/p1-task-8-report.md` §4; don't
re-derive it from scratch, it's already enumerated there.

---

## Pier quality: support-aware demotion + arc-aware terminal ordering (Phase 2 Task 3, 2026-07)

Artist-mode medoid pier selection (`docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` §3.1/3.3). Two
mechanisms, both gated on `ArtistStyleConfig` fields (`src/playlist/artist_style.py:124–136`):

```yaml
playlists:
  ds_pipeline:
    artist_style:
      pier_support_enabled: true              # master switch
      pier_support_k: 10                      # neighbors-by-same-artist for the support estimate
      pier_support_demotion_strength: 1.0     # 0.0 = inert, byte-identical to pre-Task-3 behavior
      pier_support_floor: 0.50                # diagnostic-only; logs, never filters
      pier_support_terminal_avoidance: true   # arc-aware ordering tie-break
```

**`pier_support_demotion_strength`** — continuous rank demotion (`max(0, 1-support) * strength`)
applied inside medoid scoring, where `support` is `compute_within_artist_support`'s mean cosine
similarity to a candidate's `pier_support_k` nearest tracks BY THE SAME ARTIST, normalized by the
artist's own median. This is what stops a sonically off-character track from a seed artist's own
catalog (an outlier EP cut, a noise experiment) from winning a pier slot when a more representative
alternative is competitive — never a hard exclusion, and never promotes an above-typical candidate.
**Tuning:** raise toward `2.0`+ to demote outliers more aggressively (costs pier diversity on
artists whose whole catalog is stylistically scattered); lower toward `0.0` to disable. **Known
behavior change at the shipped default (1.0):** this DOES mildly reorder some healthy artists' piers
even when nothing is a true outlier (e.g. a live/remaster take losing to a studio version at
support 0.987 vs 1.017) — directionally consistent with this file's separate `dedupe_versions`
preference, not a regression, but worth knowing before you attribute a pier-list change to a
different cause.

**`pier_support_terminal_avoidance`** — once piers are ordered and capped, the pier with the lowest
`support_by_index` is checked against both terminal seats (opening/closing position); if seated
there, every alternate `order_clusters(start=...)` walk over the same pier set is tried until one
seats it in the interior instead (`reorder_avoiding_low_support_terminal`,
`artist_style.py:1184`). Pure tie-break — never invents a topology, never fails, no-op with `<3`
piers or when the sonic layout genuinely has no alternate walk. **Known trade-off (Alex G/open,
disclosed in `.superpowers/sdd/p2-task-3-report.md`):** moving the true worst pier off a terminal
seat can force an OTHER, better-but-imperfect pier into that seat instead — its realized terminal
edge can be weaker than the outlier's would have been (support is a proxy for "off-character," not
a guarantee about the specific beam-realized edge). Bounded, disclosed, not a systemic pattern in
the 12-cell corpus.

**Not wired into mini-pier waypoint selection** (`pier_bridge/mini_pier_select.py`) — deliberately.
A bridge waypoint isn't the seed artist's own catalog, so within-artist support has no meaning
there; see `docs/DESIGN_RATIONALE.md` for the full reasoning.

---

## Knob 1: Title-artifact penalty

**Use when:** the audit shows `title_flags` like `demo`, `live`, `medley` on bad-feeling tracks.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      title_artifact_penalty:
        enabled: true
        weights:
          demo: 0.10
          live: 0.05
          medley: 0.20
          remix: 0.10
          instrumental: 0.08
          version: 0.05
          take: 0.10
          outtake: 0.15
          alternate: 0.10
```

Dataclass default: `title_artifact_penalty_enabled: False`, no weights. Not set in
`config.example.yaml` — opt-in, add it yourself once the audit shows a real problem.

**Tuning:** higher weights demote more strongly. `0.10` demotes roughly on the order of a
moderate bridge-score difference; `0.20` (medley) strongly discourages medleys unless there's no
alternative. Recognized flags are detected by `src/playlist/title_quality.py::detect_title_artifacts`
(word-boundary, case-insensitive); a flag missing from the `weights` dict contributes nothing.

**Warning:** weights above `0.30` can strand long narrow-style segments (artist mode). If
generation starts failing, dial weights down by half.

---

## Knob 2: Scaled local-sonic-edge penalty

**Use when:** the audit shows many `local_sonic_cos` values below 0.10 and `local_pen` is tiny
(< 0.03) — the penalty is decorative in `legacy` mode.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      local_sonic_edge_penalty_enabled: true
      local_sonic_edge_penalty_threshold: 0.15
      local_sonic_edge_penalty_mode: scaled
      local_sonic_edge_penalty_scale: 2.0
```

Dataclass default: `enabled: False`, `mode: legacy`. Not set in `config.example.yaml`.

**Tuning:**
- `threshold: 0.15` flags edges with raw sonic cosine below 0.15 (sonically anti-correlated or
  orthogonal).
- `scale: 2.0` produces penalties of 0.05–0.30 — comparable to bridge-score differences.
  `legacy` mode's `strength * (threshold - edge_cos)` math is preserved for anyone who explicitly
  wants it, but it under-penalizes; `scaled` is the recommended mode.
- Verify in the audit that `local_pen` values are now non-trivial (0.05+) on triggering edges.
- Start at `scale: 1.5`, watch `min_transition` in stats, raise to 2.0 if edges are still jarring.

---

## Knob 3: Worst-edge lexicographic beam objective

**Use when:** `min_transition` in stats is dramatically lower than `mean_transition`, and a few
single bad edges ruin an otherwise good playlist.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      min_edge_objective: min_edge   # default: total_score
```

Dataclass default: `"total_score"`. Not set in `config.example.yaml`.

**Effect:** the beam picks the path that maximizes the worst edge, breaking ties by total score
— the direct implementation of Layer-1 principle 5 ("the worst edge defines the experience").
Expected: `min_transition` rises noticeably; `mean_transition` may drop slightly.

**Warning:** if playlists start feeling "safe but flat," revert to `total_score`. Consider
**variable bridge length (Knob 7)** first — it targets the same symptom by flexing segment length
rather than changing what the beam optimizes for, and is the shipped default lever for this
symptom.

---

## Knob 4: The weak-edge recovery cascade (edge repair · tail-DP · edge-delete)

After the beam assembles a playlist, a fixed **four-pass cascade** lifts weak or broken transition
edges, escalating from least-destructive (swap) to most-destructive (delete). It runs **once, top
to bottom — it is not a retry loop:** each pass hands its (possibly mutated) playlist to the next
and no earlier pass re-runs, so a late deletion is never re-optimized by tail-DP.

| # | Pass | Scope | What it does | Trigger | Config (shipped default) |
|---|------|-------|--------------|---------|--------------------------|
| 1 | **variable bridge length** (add-only) | per-segment, pre-assembly | lengthens a weak segment's interior (never shortens) to land more smoothly — documented as **Knob 7** | worst edge `< variable_bridge_min_edge` (0.30) | `variable_bridge_length: true` |
| 2 | **tail-DP** | per-segment, pre-assembly | re-opens the last ≤2 interior slots of each just-finalized segment and exactly maximizes the landing-window min-edge over the segment pool (never-worse) | window min-edge below the **relative trigger floor** (below); absolute component `tail_dp.floor` (0.30; 0 = always) | `tail_dp: {enabled: true, floor: 0.3, epsilon: 0.02, relative_epsilon: 0.25}` |
| 3 | **edge repair** (break-glass) | global, post-assembly | swaps ONE interior track for a pool candidate that lifts the local `min(T_in, T_out)` by ≥ `margin` (never changes length) | `T <` the **relative trigger floor** (below); absolute component `t_floor` (0.30) **or** catastrophic `T_centered_cos < centered_cos_floor` (−0.5) | `edge_repair: {enabled: true, t_floor: 0.3, centered_cos_floor: -0.5, margin: 0.05, relative_epsilon: 0.25}` |
| 4 | **edge delete** (remove-only, last resort) | global, post-repair | deletes ONE interior endpoint of the worst edge — only if the merged edge strictly beats it (never-worse) and it won't breach a bystander artist's `min_gap` | worst `T < edge_delete.floor` (0.30), up to `max_deletions` (4) | `edge_delete: {enabled: true, floor: 0.3, max_deletions: 4}` |

Each pass is independently disableable via its `*_enabled` key. Dataclass rollback defaults are
**off** for variable-bridge and edge-repair, **on** for tail-DP and edge-delete; shipped
`config.example.yaml` turns all four on, including edge-repair — like tail-DP and edge-delete, it
ships live. Design intent (swap → add → remove) and the reorder reasoning live in
`DESIGN_RATIONALE.md`.

**When to reach for these:** the cascade is a post-beam safety net for weak edges the beam couldn't
avoid — prefer fixing the cause upstream first (Knobs 0–3, variable bridge length). The
`edge_repair.variety_guard` sub-knob (default off) rejects a repair swap whose new edge is *too*
similar to its neighbours — enable only for dynamic/discover playlists.

**Relative repair triggers (Phase 2 Task 2, `tail_dp_relative_epsilon` / `edge_repair_relative_epsilon`, both default `0.25`).**
Tail-DP and edge repair used to gate on the fixed absolute floor alone — but an edge that merely
*clears* 0.30 while sitting well below what the rest of the segment/playlist actually achieved got
no fixer attention (Task 1's mechanism probes: Parquet Courts segment 4's 0.394 cleared
`tail_dp_floor=0.3` with a ~0.7–0.8-class connector sitting unused in the same pool). The effective
floor is now `max(base_floor, reference_mean - relative_epsilon)` — `reference_mean` is the
segment's own pre-swap mean `T` for tail-DP, the whole playlist's pre-repair mean `T` for edge
repair. Both call sites log the resolved floor + source (`absolute`/`relative`) at DEBUG and state
which trigger fired in their INFO summary lines (see `LOGGING.md`).

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      tail_dp:
        relative_epsilon: 0.25    # 0.0 = legacy absolute-only rollback
      edge_repair:
        relative_epsilon: 0.25    # 0.0 = legacy absolute-only rollback
```

**Tuning:** raise `relative_epsilon` to fire less often (only truly bad relative gaps trigger a
fix); lower it to catch subtler underperformance relative to the achievable level. `0.25` sits
above healthy segments' normal internal spread (~0.10–0.15, per the Task 1 probe doc) and below the
observed pathological gaps (~0.24–0.5) — tuned to fire on real problems, not normal variance.
**Escape hatch:** `relative_epsilon: 0.0` is the full legacy-absolute-only rollback for either
mechanism — a tie (`relative_threshold == base_floor`) always resolves to `"absolute"`,
deterministically. **Gotcha:** setting `edge_repair_t_floor: 0` no longer fully disables edge
repair's weak-`T` arm by itself once `edge_repair_relative_epsilon > 0` (the relative arm doesn't
know about that intent) — the true "fully legacy, anti-alignment-only" rollback is
`edge_repair_relative_epsilon: 0.0`, not `t_floor: 0` alone.

**Known limits (documented, not yet resolved):**
- **The fixer deadzone (0.30 – ~0.75).** Every trigger floor sits at 0.30, so an *ugly-but-legal*
  edge (e.g. a rough pier on-ramp at T ≈ 0.46) gets no fixer attention. If your worst edges cluster
  in this band, the deliberate levers are raising the floors or making them percentile-relative to
  the run — a policy choice, not a bug.
- **Edge-repair vs reporter T can disagree.** Edge-repair has been seen flagging edges the final
  reporter scores as healthy (T ≈ 0.66–0.79 against a 0.30 floor). Until that's root-caused, don't
  retune the floors against reporter numbers, and treat a `WARNING: T-mismatch edge` as a reason to
  distrust that run's quality metrics.
- **90 s ceiling can be breached** on large multi-seed runs (e.g. 10-seed / 50-track), since the
  shipped `generation_budget_s: 0` disables the time cap and the cascade adds work (see Knob 10).

---

## Knob 5: `pace_mode` (rhythm/tempo control)

**Use when:** a playlist has the right color/texture but drifts too far in pace or energy,
especially with slow/meditative seeds, deliberately high-energy seeds, or a beatless
(drone/ambient) pier.

```yaml
playlists:
  pace_mode: narrow   # strict | narrow | dynamic | off
```

Shipped default: `dynamic` (`config.example.yaml`, top-level `playlists.pace_mode`, commented
example; effective default even when absent). Presets live in
`src/playlist/mode_presets.py::PACE_MODE_PRESETS`.

**Effect:** `pace_mode` is **embedding-independent** — it gates on BPM and onset-rate log-distance
bands plus a soft rhythm penalty read from DB features, so it survives any sonic-embedding
change (this is why the old rhythm-cosine tower penalty is inert now — there's no rhythm tower
left to slice on the MuQ embedding, and why pace exists as its own axis rather than folding into
`sonic_mode`). It applies an admission floor when building the candidate pool, then a per-step
moving target inside the pier-bridge beam.

**Starting values (from the live presets):**

| Mode | BPM adm/bridge log-dist | Onset adm/bridge log-dist | Bridge soft-penalty strength | Use case |
|---|---|---|---|---|
| `strict` | 0.30 / 0.40 | 0.30 / 0.40 | 0.50 | Tight tempo fidelity — slow/meditative seeds |
| `narrow` | 0.50 / 0.60 | 0.50 / 0.60 | 0.40 | Moderate anchoring |
| `dynamic` | 0.75 / 0.85 | 0.75 / 0.85 | 0.30 | Gentle — catches double-time, allows drift (default) |
| `off` | ∞ / ∞ | ∞ / ∞ | 0.0 | No pace gate — rhythm still influences via the sonic embedding |

Bands are **soft penalties, not hard gates** (`bpm_bridge_soft_penalty_strength` /
`onset_bridge_soft_penalty_strength` > 0) — a hard gate on an outlier pier (e.g. a near-silent
ambient track) can strand a segment and detonate the relaxation cascade into minutes. The hard
gate still exists at `strength: 0` for anyone who wants the old behavior back.

**Beatless piers:** `bpm_trust_min_onset_rate` (preset 0.5 for strict/narrow/dynamic; dataclass
default 0.0/off) disables a segment's BPM band when its pier's onset rate is below this
threshold — BPM is meaningless on drone (a beatless track can read a confident, wrong BPM). The
onset band, which is reliable on beatless audio, stays active.

**Interaction with `sonic_mode`:** orthogonal. `sonic_mode` controls overall sonic similarity;
`pace_mode` constrains rhythm specifically. In multi-seed runs the beam's rhythm target
interpolates between adjacent piers, so a slow-to-fast route can still arc naturally when the
piers themselves differ.

**Energy admission-rescue** (`pace_rescue_k_energy`, part of the same preset table): re-admits
candidates rejected *only* by the BPM/onset bands, evenly spaced across sorted arousal — a relief
valve so pace gating doesn't starve a segment of arc range. Shipped `0` (off) in `dynamic`;
`strict`/`narrow` presets set 20/5. Off by default because the broader energy arc/contour feature
is still unvalidated — see `DESIGN_RATIONALE.md` "Pace / energy."

**Diagnostics:** watch for `Pace admission floor applied: floor=X rejected=N` in logs and the
`bpm=A->B dist=D` line in the selected-edge audit. If strict mode rejects too much of the pool or
makes segments infeasible, drop to `narrow` or `dynamic`.

**All these bands are now yaml-tunable (Phase 2 Task 4 fix).** Every field in the starting-values
table above, plus `bpm_trust_min_onset_rate`, the soft-penalty strengths, the energy-arc fields, and
`pace_rescue_k_energy`, previously could NOT be overridden from `config.yaml` — `resolve_pace_mode`
was called with no `overrides` param, a dead outlet regardless of what the config said. Fixed via
`_resolve_pace_overrides` (`src/playlist/pipeline/core.py:246`; see
`docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` §4.4 for the mechanism). To override one field for the
active `pace_mode` without redefining the whole preset:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      bpm_bridge_max_log_distance: 0.20   # bridge-side + energy fields (pier_bridge.<key>)
    candidate_pool:
      bpm_admission_max_log_distance: 0.20   # admission-side fields (candidate_pool.<key>)
```

Admission-side fields (`pace_admission_floor`, `pace_bridge_floor`,
`bpm_admission_max_log_distance`, `onset_admission_max_log_distance`) go under `candidate_pool.*`;
everything else (bridge-side bands, soft-penalty strengths, energy-arc fields,
`pace_rescue_k_energy`) goes under `pier_bridge.*`. Only numeric values are forwarded — a
non-numeric override is silently ignored in favor of the preset default, never a crash.

---

## Knob 6: Local genre continuity (`soft_genre_penalty_*`)

**What it does.** Penalizes any beam edge whose candidate-to-previous-track genre similarity
drops below a per-mode threshold. The penalty multiplies the edge's combined beam score by
`(1 - strength)`, demoting (not gating) genre-jarring transitions — what suppresses a single-track
genre detour like a folk-punk track in the middle of a dream-pop run.

**Where it lives.** `src/playlist/pier_bridge/beam.py` (penalty application);
`src/playlist/config.py` (per-mode resolution). Genre similarity here is the taxonomy-graph
**`max`** metric (the max canonicalized-tag-pair similarity over the two tracks) — a soft-cosine
alternative was built and evaluated but rejected; see `DESIGN_RATIONALE.md` §"Genre metric."

**Note:** per-mode knobs (`soft_genre_penalty_threshold_narrow`, etc.) are resolved by
`playlists.cohesion_mode`. With the default `cohesion_mode: dynamic`, only `*_dynamic` keys
apply. Set `cohesion_mode` to `strict`/`narrow`/`discover` (or use `--cohesion-mode` on the CLI
for one run) to activate those per-mode values.

**Per-mode defaults, shipped in `config.example.yaml`:**

| Mode      | threshold | strength | Role                                    |
|-----------|-----------|----------|------------------------------------------|
| strict    | 0.82      | 0.40     | Hard enforcement of local continuity     |
| narrow    | 0.78      | 0.30     | Suppress single-track detours            |
| dynamic   | 0.73      | 0.15     | Light continuity nudge                   |
| discover  | 0.20      | 0.10     | Safety net only — allow variety          |
| off       | 0.20      | 0.10     | Safety net only — allow variety          |

**How to diagnose.** Run with `--log-level DEBUG` and look for per-segment
`Segment N: soft_genre_penalty_hits=H edges_scored=E threshold=T strength=S` lines
(`pier_bridge_builder.py`). The post-generation summary also reports total
`soft_genre_penalty_hits`.

- If `hits == 0` across all segments in a non-discover mode, the threshold is too low to be
  doing anything — raise it toward the observed `G genre` median in the summary.
- If `hits > 50%` of `edges_scored` in narrow or strict mode, the threshold is too high — you're
  penalizing the median edge, not just outliers. Lower toward the `G genre` p25–p33 range.
- If bridge relaxation warnings (`Segment N attempt 2: widened=True`) appear in narrow mode after
  recalibration, the penalty plus the gate is starving segments — lower `strength` first, then
  `threshold`.

**Caution.** A flat threshold of 0.80 was tried and reverted — it introduced worse worst-case
edges by shifting the beam onto paths with low `T` but high `G`, which the penalty can't see.
Raise only after confirming `min_T` and `below_floor` hold.

**Related, independent mechanism — raw-tag pool compatibility.** A second, separate soft
demotion (`genre_compatibility_enabled` / `genre_compatibility_penalty_strength`, live-only —
dataclass default off, not in `config.example.yaml`) applies during *candidate-pool
construction*, not at the beam edge: it demotes a candidate whose raw-tag overlap with the seed
is below `genre_compatibility_conflict_threshold` (0.15) — a pool-level filter, complementary to
this beam-level continuity penalty. Both are soft; a hard genre gate anywhere in the pipeline
detonates the relaxation cascade. **CLAUDE.md gotcha caveat:** the project doc's
`genre_conflict_min_confidence` / `genre_conflict_penalty_strength` key names are stale — the
real keys are `genre_compatibility_*` as above.

**Relationship to `genre_tiebreak_weight`.** The tiebreaker (default 0.05) nudges near-tied
edges; the penalty actively demotes below-threshold edges. They're independent — leave the
tiebreaker at 0.05 unless you have a specific reason to change it.

---

## Knob 6b: Tag steering (artist-mode soft genre lean)

**Use when:** an artist-mode playlist should lean toward one facet of the seed artist's own
catalog (e.g. their shoegaze-tagged tracks over their dreampop-tagged tracks) without hard
filtering out everything else.

**What it is.** A per-request, GUI-only feature — not a config default anyone tunes once and
forgets. The artist-mode generate screen shows up to 3 chips drawn from the seed artist's *own*
published genres (`GET /api/genres/for_artist` → `src/genre/authority.py::resolved_genres_for_artist`
— excludes `inferred_family` hub genres, top-12 by weight). Selected chips travel as
`steering_tags` on the request → `policy.derive_runtime_config` (`src/playlist_gui/policy.py:325-334`,
capped at 3, **web-only** like the other policy-owned knobs — there's no CLI flag) →
`playlists.ds_pipeline.pier_bridge.tag_steering_tags` override.

**It is always a soft lean, never a gate.** `src/playlist/tag_steering.py::resolve_tag_steering_target`
maps the chosen tags to a unit-norm mean of their dense-genre-embedding vocabulary rows. With no
tags selected it returns `None` silently — **byte-identical to legacy generation**, zero overhead.

Two independent, additive levers, both soft:

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      tag_steering_pool_blend: 0.5    # 0 = ignore tags (legacy), 1 = pool centroid is pure target
      tag_steering_pier_weight: 0.3   # on-tag bonus in artist medoid (pier) scoring
```

- **Pool lever** (`tag_steering_pool_blend`, shipped `0.5`): blends the tag target into the
  dense genre-admission centroid used for candidate-pool admission
  (`candidate_pool.py:922-933`) — `(1-blend) * seed_centroid + blend * tag_target`, re-normalized.
  Selecting tags **forces `genre_admission_aggregate=centroid`** even if the config asked for
  `per_seed` (`candidate_pool.py:882-884`, logged), since the blend only makes sense against a
  single centroid.
- **Pier lever** (`tag_steering_pier_weight`, shipped `0.3`): adds an on-tag affinity bonus to
  each candidate's medoid score during artist-style pier clustering (`artist_style.py:923-939`,
  wired from config as `medoid_tag_weight` at `playlist_generator.py:1759-1761`). This lever is
  active in the shipped template — `artist_style.enabled` ships `true` in `config.example.yaml`,
  so the medoid-clustering code path it lives in runs by default for style-aware artist
  playlists.

**Degenerate cases (both loud, not silent):**
- No tags picked → resolver returns `None`, nothing logged beyond the normal path — this is the
  intentional legacy-identical no-op.
- Tags picked but the artifact has **no dense genre-embedding sidecar** (`X_genre_dense` /
  vocabulary missing) → `resolve_tag_steering_target` logs a **WARNING** ("dense genre sidecar
  has no vocabulary embedding — steering disabled for this run") and proceeds inert for that run.
  Same for tags that don't map to the vocabulary (per-tag WARNING) or that all map but net to a
  zero-norm target.

**Status:** pool + pier levers are shipped and live behind the GUI chips (`config.example.yaml`
lines ~292-295). A designed-but-not-built third lever — a beam-stage (per-edge) tag bonus — does
not exist yet; don't look for a `tag_steering_beam_weight` key. Tests:
`test_tag_steering*`, `test_authority_artist_genres`.

---

## Knob 6c: On-tag anchor placement (Phase B)

**Use when:** artist-mode tag steering should GUARANTEE a peripheral on-tag clique appears (e.g. a
Ghost Box record under Boards of Canada + hauntology), not just lean the pool/pier scoring toward
it. Phase A (Knob 6b) can't guarantee this — a sonically-peripheral on-tag track can fall below the
seed-artist genre gate and never get bridged to.

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      tag_steering_anchor_max: 3            # on-tag tracks injected as piers (0 = Phase-A-only / off)
      tag_steering_anchor_min_bridge: 0.35  # min sonic cosine to a seed pier for an anchor to qualify
      tag_steering_anchor_per_artist: 1     # per-artist cap within the anchor set
      tag_steering_anchor_gap_insertion: true
```

**What the anchors are.** Up to `tag_steering_anchor_max` on-tag tracks by OTHER artists are
injected as un-droppable PIERS (not just bridge candidates) — selected to be bridgeable to a seed
pier, tag-central, and cross-artist-diverse. Inert (byte-identical to legacy) unless tag steering
resolved on-tag ids for the run.

**Placement (`tag_steering_anchor_gap_insertion`, spec 2026-07-27).** The naive approach — append
anchors to the artist piers and let the normal co-equal seed-ordering step
(`_order_seeds_by_bridgeability`) re-sequence everything — let anchors cluster adjacent to each
other and land on a terminal seat (the reported defect: a Sonic Youth playlist that opened or
closed on an injected anchor, and a Boards of Canada run with three anchors bunched at consecutive
positions). Gap insertion fixes this structurally with two rules:

- **Never terminal** — an anchor can never occupy position 0 or the last position. The artist piers
  alone are ordered first (`_order_seeds_by_bridgeability` runs over artist piers only, anchors
  excluded), which also fixes their terminals; anchors are then inserted into the **interior gaps**
  between them.
- **Non-adjacent** — at most one anchor lands per gap, so two anchors can never sit back-to-back.

Assignment is a minimax objective: for anchor `a` and gap `g` between ordered artist piers
`(p_i, p_{i+1})`, `score(a, g) = min(bridge(a, p_i), bridge(a, p_{i+1}))` — the weaker flanking
edge is what's scored, since that's the edge that will actually get built. The assignment is
exhaustive over injective anchor→gap maps (cheap at `K<=3`, `P-1<=9`), maximizing the worst
assigned score, ties broken by total score then lowest gap index for determinism.

**Feasibility clamp.** P artist piers create P−1 interior gaps, one anchor per gap max:
`K <= P-1`. `target_pier_count` floors at 3 and `anchor_max` defaults to 3, so the P=3 case
(K=3 > P-1=2) is a real, reachable infeasibility — the excess anchor(s) are dropped by selection
rank (least tag-central first), logged at INFO naming the dropped track_ids (never a silent cap).
An anchor can also be dropped if its best gap scores below `tag_steering_anchor_min_bridge`
(unbridgeable to either flanking pier) or if it loses its best gap to a higher-scoring anchor in
the exhaustive assignment (displaced) — both logged separately.

**Rollback.** `tag_steering_anchor_gap_insertion: false` reproduces the legacy behavior: anchors
appended to the artist piers as co-equal seeds, then the whole set re-ordered together by
`_order_seeds_by_bridgeability` — the path that let anchors cluster/terminate. Kept as an escape
hatch, not the shipped default.

**Measured (2026-07-27, live DB + artifact, `random_seed=0`, same-session anchor_max=0
baselines — see `docs/run_audits/2026-07-27-anchor-placement-validation.md`):** Boards of Canada +
hauntology: 2/3 anchors placed into gaps [1,2] (1 dropped, unbridgeable), worst edge improved
0.651→0.711. Brian Eno + neoclassical: 3/3 anchors placed into gaps [0,1,2], worst edge unchanged
to four decimals (0.7714 both arms) — notably this is the case that previously (pre-gap-insertion,
2026-07-09) regressed hard under Phase B (0.133 with anchors vs 0.528 baseline, because
bridgeability was checked only against the nearest seed pier, not the two piers an anchor would
actually sit between); the gap-insertion minimax objective scores exactly the pair an anchor will
bridge to, which is consistent with fixing that failure mode (one seed, one day — not proven
across seeds/configs). See the evidence doc for the full case-by-case numbers, the Alvvays + twee
result, and the §B seed-ordering side-effect finding.

**Status:** shipped default (`config.example.yaml`). Tests: `test_phase_b_anchors_*`,
`test_anchor_never_terminal_sonic_youth`, `test_no_two_anchors_adjacent_sonic_youth`,
`test_anchor_guarantee_survives_placement_boc`, `test_gap_insertion_rollback_reproduces_legacy_placement`.

---

## The collapse-prevention stack (Knobs 7–9)

Long bridges tend to **sag**: interior tracks drift toward the dense, genre-blurred "average" of
the local pool rather than representing the seeds' actual character. This is now understood as
two related failure modes — cross-seed convergence and within-bridge sag — and three levers
target it, in the order you should reach for them. See `ARCHITECTURE.md` §"Anti-sag scoring
(collapse prevention)" for the map and `DESIGN_RATIONALE.md` for the evidence (including the
density-floor approach that was tried and abandoned).

### Knob 7: Variable bridge length

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      variable_bridge_length: true
      # variable_bridge_flex: 2
      # variable_bridge_band: 5
      # variable_bridge_min_edge: 0.30
      # variable_bridge_max_flex_segments: 3
```

Dataclass default: `False` (even split, rollback). **Shipped `true` in `config.example.yaml`** —
part of the collapse-core.

**What it does.** A rigid even split pads every segment to the same interior length regardless
of whether the pier-to-pier distance actually calls for it. Variable length instead builds the
nominal even split, and — only for a segment whose worst edge is below `variable_bridge_min_edge`
(0.30) — greedily tries every interior length in `[nominal-flex, nominal+flex]` (flex=2), keeps
the one with the best worst-edge (the "bottleneck," which includes the pier-return edge), and
prefers the nominal length unless a flexed length beats it by more than `variable_bridge_epsilon`
(0.02 — a prefer-nominal-first anti-crutch so it doesn't flex when it doesn't need to). The total
track count is allowed to land within `±variable_bridge_band` (5) of the requested playlist
length. `variable_bridge_max_flex_segments` (3) is a **deterministic cap** on how many segments
may actually flex per generation — it replaced an earlier wall-time guard so the cost is
predictable, not time-boxed.

**When to reach for it:** this is the first lever for "worst-edge is bad but average is fine" —
it's structurally cheaper than Knob 3's global objective change because it only perturbs the
specific segment that needs it.

**Trade-off:** flexing costs roughly 2.5–3x the generation time of the segments that actually
flex (every candidate length in the flex range gets a full trial build). It's inert on a seed set
where every segment nominal length is already the best fit (about half of real seed sets in
validation).

### Knob 8: Seed-character anti-collapse (SP2, `anti_center`)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      seed_character_mode: anti_center   # off | anti_center
      seed_character_strength: 2.0
```

Dataclass default: `seed_character_mode: "off"`, `strength: 0.0`. **Shipped
`anti_center` @ `2.0` in `config.example.yaml`.**

**What it does.** A *scoring* fix for within-bridge sag: it subtracts
`strength * max(0, cos(candidate, pool_centroid) - bridge_score)` from a candidate's combined
score — directly penalizing a candidate for sitting closer to the local segment's generic pool
center than to its own piers. The centroid is the mean of the segment's L2-normalized candidates,
**excluding the piers themselves**, so a candidate can't get credit just for being near the
seeds.

**Note — hubness was deleted.** An earlier second selector (`hubness`, a kNN in-degree
deflation) was evaluated alongside `anti_center` and lost (weaker effect, didn't scale) — it has
been **removed from the codebase**. `seed_character_mode` now accepts only `off` or
`anti_center`; don't look for a `hubness` option, and don't reintroduce it without new evidence.

**When to reach for it:** the default always-on lever against sag. `strength: 2.0` is the
validated sweet spot (measured sag reduction: electronic 60%→46%, dreampop 117%→101% on the
collapse harness, with playlist quality held). Turning it up further hasn't been validated —
treat this as a tuned constant, not a dial to crank.

**Trade-off / known limit:** it's a **partial** dent, not a fix. On some seed pairs (dreampop) it
plateaus around 101% — scoring alone can't fully prevent a long, genre-blurred segment from
sagging. That's what Knob 9 (mini-piers) exists to close.

### Knob 9: Mini-piers (SP3, structural anti-sag)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      mini_pier_enabled: true
      mini_pier_max_interior: 5
      mini_pier_smoothness_margin: 0.12
```

Dataclass default: `mini_pier_enabled: False`. **Shipped `true` / `max_interior: 5` /
`margin: 0.12` in `config.example.yaml`.**

**What it does.** Where Knob 8 nudges scoring, this is a *structural* fix: it splits any segment
whose interior would exceed `mini_pier_max_interior` (5) tracks by picking a high-character
waypoint — selected via a smoothness floor plus the same anti-center scoring as Knob 8 — and
pinning it as an **extra pier**. The beam then can't drift past that anchor; it's structurally
forced to route through a high-character point rather than the pool's generic average.
`mini_pier_smoothness_margin` trades some worst-edge quality for character (a smoother, more
generic waypoint is easier to bridge to but defeats the purpose; the margin bounds how much
smoothness you'll sacrifice character for).

**When to reach for it:** the residual-sag case Knob 8 plateaus on — long segments (interior >
5) with a diffuse local pool. Measured effect: dreampop within-bridge sag 103%→63%.

**Trade-off:** more piers means more segments, each independently beam-searched, so cost scales
with how many segments actually split. It composes with Knob 8 rather than replacing it — anti-
center still scores every candidate; mini-piers just adds structure the scoring alone can't
enforce.

### Roam corridors — live-only, advanced

Roam corridors (on-manifold kNN geodesic routing + minimax worst-edge ordering,
`playlists.ds_pipeline.pier_bridge.roam.*`) are a real, tested lever that is **live in the local
`config.yaml` but not in `config.example.yaml`** — genuinely advanced/opt-in, outside the
validated default bundle. Copy the relevant block from `config.yaml`'s comments if you want to try
it, and expect to tune `roam.width_sonic` / `genre_pair_floor` per library. (Edge repair, which an
earlier version of this note lumped in here, is part of the weak-edge recovery cascade in Knob 4
above — its absence from `config.example` is a template gap to fix, not an intentional exclusion.)

---

## Knob 10: `generation_budget_s` (time budget)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      generation_budget_s: 0   # 0 = disabled
```

Dataclass default: `60.0`. **Shipped `0` (disabled) in `config.example.yaml`.**

**What it does.** A single shared deadline computed once at generation start, threaded into
every segment loop and relaxation tier, so no combination of retries can collectively exceed the
budget. `0` **disables the time limit entirely** — every lever (beam, variable bridge length,
mini-piers) runs to completion regardless of wall-clock cost.

**Trade-off.** This is a deliberate quality-first choice while the collapse-prevention stack
(Knobs 7–9) is still being dialed in — none of those levers currently bail early to protect a
time budget, so a positive budget interacts with them by potentially truncating a flex/mini-pier
search mid-way rather than letting it finish. There's still a **90s hard ceiling as a design
target** (not separately enforced when the budget is 0) — a full generation should never
approach it in practice, but nothing currently kills a run that does. Set a positive value (the
dataclass default `60` is a reasonable starting point) once you want a hard speed guarantee back;
expect variable-bridge-length and mini-pier segments to be the first casualties of a tight
budget, since they're the ones that do extra trial work per segment.

---

## Knob 11: `progress_arc` (experimental monotonic-progress beam term, off by default)

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      progress_arc:
        enabled: false     # default OFF -- see calibration note below
        weight: 0.25
```

**What it does.** An optional beam-scoring term that penalizes a candidate for being
sonically "behind" or "ahead of" where it should sit along the pier-to-pier interpolation
line for its segment position (`beam.py`'s `_progress_arc_loss`), shaped by
`progress_arc_shape`/`tolerance`/`loss` and scaled by pier-to-pier `ab_distance` (and
optionally by segment length via `autoscale.per_step_scale`, off by default).

**Calibration note (Phase 2 Task 4, 2026-07-18):** Task 7's contract sweep flagged the shipped
`weight: 0.25` default as too weak against corridor's tighter beam ("effectively runs at 3.0" in
that sweep's own experiment). A dedicated probe (`progress_arc` forced on across weight `{0.25, 0.5, 1.0, 2.0,
3.0}` on the project's two canonical sweep cells — Bill Evans Trio/open, Swirlies/home — vs an
arc-disabled baseline) found the "audible" threshold is **cell-dependent, not a single number**:
because the effective weight scales with pier-to-pier `ab_distance`, one cell (Swirlies/home) was
already fully saturated at the shipped `0.25` (a real +0.0144 min_T improvement, unchanged through
3.0), while the other (Bill Evans Trio/open) was barely perturbed at `0.25` (jaccard 0.818 vs
baseline) and only became clearly audible at `weight=0.5` (jaccard drops to ~0.43, a genuine
track-set reshuffle), with a second, transition-quality-costing step at `weight=2.0` (-0.047 min_T
vs baseline). Recommended experimentation floor if you want to deliberately try this lever:
**`weight=0.5`** is reliably audible on both tested cells; `0.25` is not.

**No default change shipped.** The two probed cells disagree on direction (one improves, one
degrades as weight increases past 0.5) — not the "clear win" bar the calibration task required to
flip a default. `progress_arc_enabled` stays `False`; `weight: 0.25` is unchanged. This section is
a tuning recipe for opting in deliberately, not evidence the feature should ship on.

---

## Knob 12: genre-mode pool breadth (`playlists.genre_playlist`)

**Use when:** a genre-picked playlist (spec
`docs/superpowers/specs/2026-07-24-genre-mode-design.md`) feels too diffuse (drifting into
adjacent genres that don't feel like the one you asked for) or too repetitive (the same handful
of artists/tracks every regeneration, or generation frequently hits the relaxation ladder on a
thin genre).

```yaml
playlists:
  genre_playlist:
    similarity_threshold_strict: 0.50
    similarity_threshold_narrow: 0.42
    similarity_threshold_dynamic: 0.35
    similarity_threshold_discover: 0.20
    relaxation_steps: [0.35, 0.20, 0.10]
```

**What it does.** Genre mode's bridge pool is every OTHER genre scored by hub-damped taxonomy
similarity (`TaxonomySteering.similarity`, the same scorer the beam's genre-arc router uses) to
the genre FAMILY you picked, admitted above `similarity_threshold_<cohesion_mode>`.

The **family** is the genre plus its transitive `is_a` descendants, read from
`data/layered_genre_taxonomy.yaml` (never `genre_graph_edges` — the published tables lag the
YAML). A track tagged `classic soul` IS soul, so it is a full member: eligible to be a pier, and
pinned in the pool at similarity 1.0 regardless of its name-similarity score. That pinning is also
what guarantees seeds are a subset of the pool. Seeds/piers remain observed-only
(`release_effective_genres`, observed_leaf/legacy — never `inferred%`); this knob only widens or
narrows the BRIDGE material connecting those piers. Leaf genres have no descendants, so the family
is a no-op for them. There is no separate beam steering term for
genre mode (spec §3.5) — every pier is the same genre, so the arc target collapses to one node and
contributes nothing; **pool composition is the only lever**, which is why this single threshold is
also the entire genre-mode differentiation surface between `strict`/`narrow`/`dynamic`/`discover`.

**Measured pool sizes** (distinct tracks, `assignment_layer NOT LIKE 'inferred%'`, spec §3.1,
re-measured 2026-07-24 against the shipped `genre_mode.pool_track_ids`):

| Genre | exact (piers only) | thr 0.50 | thr 0.35 | thr 0.20 |
|---|---|---|---|---|
| `shoegaze` | 2,842 | 6,545 | 7,322 | 17,021 |
| `post_punk` | 3,930 | 5,196 | 5,588 | 8,461 |
| `hauntology` | 174 | 910 | 8,080 | 11,633 |
| `dub_techno` | 71 | 71 | 576 | 1,760 |

**Family sizes** (member tracks, taxonomy `0.53.3`) — how much the `is_a` walk adds. Umbrella
genres are where it matters; the four genres above are all leaves or near-leaves, which is why
their pool table is unchanged:

| Genre | exact | family (transitive `is_a`) | descendant genres |
|---|---|---|---|
| `rock` | 126 | 15,712 | 44 |
| `pop` | 237 | 12,546 | 32 |
| `punk` | 91 | 6,577 | 23 |
| `jazz` | 128 | 4,211 | 34 |
| `folk` | 237 | 3,781 | 21 |
| `soul` | 214 | 2,639 | 21 |
| `metal` | 0 | 212 | 22 |
| `reggae` | 206 | 336 | 3 |
| `shoegaze` / `hauntology` / `dub_techno` | unchanged | unchanged | 1 / 0 / 0 |

**Which direction to move it:**
- **Playlist doesn't feel like the genre you picked (too diffuse)** → raise the active mode's
  threshold (e.g. `similarity_threshold_dynamic: 0.35 -> 0.45`). Fewer neighbor genres qualify for
  the bridge pool, so more of the playlist stays sonically/taxonomically close to the exact genre.
- **Same few artists every regeneration, or generation keeps hitting the relaxation ladder (too
  repetitive / too tight)** → lower the threshold, or widen `relaxation_steps` downward (add a
  lower rung, e.g. `0.05`). More neighbor genres qualify, giving the beam and the cluster-then-rank
  seed selector (spec §3.2) more material to route through and pick piers from.
- `dub_techno`-shaped genres (few neighbors even at 0.50) are the case relaxation exists for —
  don't chase them by permanently lowering the base threshold for every genre; let the ladder do
  its job per-genre instead.

**Starvation relaxation is the same dial, stepped down** — not a separate mechanism. If the
pool at the mode's threshold doesn't support `max(track_count * 8, 200)` tracks, the code steps
down through `relaxation_steps` (only rungs below the starting threshold; already-below rungs are
skipped) until a rung clears the bar, or falls back to the widest rung tried on a genuinely thin
genre (never-fail on this soft axis). Every rung this fires is logged, so a starved or
unexpectedly narrow genre run is diagnosable from the log alone:

```
stage=genre_seeds | genre=<id> family=F genre(s): <descendant ids>
stage=genre_pool | genre=<name> threshold=0.35 family=F genres=N tracks=T
stage=genre_pool | relaxation step threshold=0.350 tracks=T < NEEDED — widening
stage=genre_pool | relaxation settled threshold=0.200 tracks=T (needed NEEDED)
```

(or, if every rung was exhausted: `stage=genre_pool | genre=<name> starved: widest threshold=...
yields N tracks (< NEEDED). Generating from the widest pool.`) — grep `stage=genre_pool` in the
per-playlist log (`logs/playlists/`) to see exactly which rung a given generation settled on.
`stage=genre_seeds` (same log) shows the cluster count and the piers realized after the
min-cluster-size + pier-bridgeability veto, if the seed side (not the bridge pool) is the thing to
diagnose instead.

**`min_cluster_size` and the composite seed-score weights** (`seed_weight_prominence` /
`_canonicity` / `_centrality` / `_typicality` / `_investment`) live in the same config block —
see the spec's §3.2 for what each weight represents; they are a retune, not a rewrite, if the
piers a genre selects feel wrong (e.g. raise `seed_weight_investment` to bias piers toward artists
already in your library, or lower `seed_weight_prominence` if generation over-indexes on
Last.fm-famous artists at the expense of niche ones you actually own).

---

## Knob 13: Multi-artist blend (`playlists.ds_pipeline.multi_artist`)

**Use when:** artist mode is given 2+ artist chips and the playlist should sit in the sonic/genre
territory those artists *share* for its whole duration (spec
`docs/superpowers/specs/2026-07-29-multi-artist-blend-design.md`) — e.g. Brian Eno + Harold Budd
should skew ambient, not travel from Eno's catalog to Budd's and back. Inert below two chips.

```yaml
playlists:
  ds_pipeline:
    multi_artist:
      enabled: true
      overlap_weight: 0.6          # strength of the pull toward the other artist(s)
      genre_share: 0.25            # split within the affinity term; the rest is MuQ
      max_artists: 4
      joint_pier_min_budget: 3     # min total piers before a jointly-credited seat is reserved
      alternation_bonus: 0.15      # ordering preference for A/B/A/B; never overrides sonic cost
      low_overlap_threshold: 0.15  # mean pier affinity below this -> report it to the user
```

**What it does.** Bundle rows are partitioned into one exclusive group per chip plus a joint group
(tracks credited to 2+ chips, e.g. "Harold Budd and Brian Eno"). Each group gets a centered,
L2-normalized MuQ prototype (`tag_steering.sonic_prototype_from_rows`); every candidate in a group
is scored by a soft **overlap affinity** toward the *other* groups' prototype(s)
(`(1 - genre_share)` MuQ cosine + `genre_share` dense-genre similarity), which biases —
never gates — `cluster_artist_tracks`'s medoid (pier) selection for that group. The merged piers
are then reordered with an alternation preference (A/B/A/B over A...A/B...B) that is bounded by a
minimum-edge floor: any candidate ordering whose worst edge is below the default walk's worst edge
is discarded before the alternation bonus is applied, so the preference can never trade a smooth
default ordering for a "tidier" but worse one.

**Pier budget and per-artist cap.** An N-group blend gets `N * base` piers (each group earns a
single-artist-equivalent share), clamped to `track_count // 5` so bridge segments stay long enough
to bridge — this replaced an earlier `// 3` clamp that starved a 3-group 30-track blend down to
~2.2-track bridges and cratered its worst edge to 0.4645 (`// 5` recovered it to 0.6135; see
`total_pier_budget` in `src/playlist/multi_artist.py` for the full account). The joint group claims
a *minimum* of one seat once the total reaches `joint_pier_min_budget`, but may absorb more of the
budget if the exclusive groups can't use it — it isn't capped at exactly one. The single-artist
per-artist cap (`candidate_pool.max_artist_fraction`) is scaled by the surviving group count
(joint group included) whenever a blend fires, so a legitimate N-way pier split doesn't
self-report as `artist_cap_violation` in the GUI.

**Which direction to move it:**
- **Piers feel too characteristic of one artist, not the shared middle ground** → raise
  `overlap_weight` (default `0.6`); `0.0` is the explicit rollback to per-artist-characteristic
  piers.
- **A blend's worst edge is at or below the absolute floor (`0.40` — see below) — first check
  whether the alternation preference actually changed the ordering** (the log line below says so
  explicitly); if it did, **lower `alternation_bonus`** before touching anything else. Sweeping
  `overlap_weight` down also reduces how far medoid selection drifts from each artist's own
  character, at the cost of a smaller "shared ground" effect.
- **Playlist reports "these artists share little ground"** — that's `low_overlap_threshold`
  firing; it's cosmetic (never blocks generation), so only retune it after reading real affinity
  numbers across a few known-distant pairings, not preemptively.

**Acceptance bar.** A blend is NOT held to single-artist smoothness — it deliberately spans two
artists' territory, which is rougher by nature than either solo playlist. The guard is an absolute
`min_transition` floor of `0.40`, calibrated 2026-07-30 from three real pairings at
`track_count=30`: Eno+Budd 0.6135, Eno+Bowie 0.5490, Budd+Foxx 0.6478 (lowest observed 0.5490,
comfortably above the floor and well clear of this project's break-glass edge-repair trigger,
`T < 0.30`).

**What to watch in the log** — every stage of the blend logs at INFO/WARNING under one prefix:

```
grep "Multi-artist" logs/playlists/<run>.log
```

Look for: `Multi-artist prototype: <label> n=<n> cohesion=<c>` (per-group prototype quality —
low cohesion means that group's own catalog is sonically scattered, so its "pull" on the others is
weak); `Multi-artist pier budget: total=<n> allocation=<dict>` (confirms the `// 5` clamp and the
even/remainder split actually landed where expected); `Multi-artist ordering: alternation
preference changed the pier walk (...)` (only logged when the alternation search found a legal,
never-worse winner — its absence means the default greedy walk already won, and `alternation_bonus`
is provably a no-op for that run); and `Multi-artist: <artist> share little sonic ground (mean pier
affinity ...)` for the `low_overlap_threshold` notice. A dropped chip or a too-thin group logs its
own named WARNING rather than silently vanishing from the pairing.

**Status:** shipped default (`config.example.yaml`). Tests: `tests/unit/test_multi_artist.py`,
`tests/integration/test_multi_artist_generation.py` (live-artifact acceptance, including the
absolute worst-edge floor above).

---

## The four mode axes and per-cohesion-mode knobs

Cohesion-vs-discovery is exposed as four independent axes (`ARCHITECTURE.md` §"The four mode
axes"): `cohesion_mode` drives the beam; `genre_mode`, `sonic_mode`, `pace_mode` gate the
candidate pool. All default to `dynamic`.

The pier-bridge per-mode knobs — `bridge_floor_<mode>`, `weight_bridge_<mode>` /
`weight_transition_<mode>`, `soft_genre_penalty_threshold_<mode>` /
`soft_genre_penalty_strength_<mode>`, `genre_pair_floor_<mode>`, `genre_arc_floor_<mode>`,
`sonic_admission_percentile_<mode>`, etc. — are **keyed by `cohesion_mode`**, not by
`genre_mode`/`sonic_mode`. With `cohesion_mode: dynamic` (the default), only the `_dynamic` suffix
of each of these keys is read; setting `cohesion_mode: strict` activates the `_strict` values
across all of them at once. Use `--cohesion-mode` on the CLI to override for a single run without
editing config.

Two gotchas worth restating here (full detail in `ARCHITECTURE.md`):
- **Only the web GUI goes through the policy layer** (`src/playlist_gui/policy.py`). The CLI
  sets mode strings directly. A test/harness that bypasses policy will see modes as inert — the
  `playlist-testing` skill's harness routes through policy for exactly this reason.
- **Beam width is global, not per-mode**: `initial_beam_width` / `max_beam_width` (shipped
  default 20/100; `config.example.yaml` doesn't override the dataclass), doubling toward the cap
  on infeasibility, regardless of `cohesion_mode`. Raising both (e.g. to 40/200) widens the
  search at a roughly linear cost in generation time — a reasonable knob to reach for if
  infeasibility retries are common on your library, independent of any mode setting.

---

## Reading the audit table

Example bad edge entry:
```
⚠ Edge #15: Hideous Sun Demon - Gimmicks -> Stove - Nightwalk
  T=0.092 T_centered_cos=-0.817 S=0.306 G=1.000 | bridge=0.55 trans_beam=0.25 title_flags=-
  progress_t=0.850 progress_jump=0.100 local_sonic_cos=0.030 local_pen=0.021 genre_pen=0.000 below_floor=True
  bpm=118->131 dist=0.152
```

Interpretation:
- `⚠` + `below_floor=True` — this edge fell below the transition floor in the final emitted
  playlist.
- `T_centered_cos=-0.817` — the underlying centered cosine is strongly anti-correlated; the
  calibrated `T=0.092` reflects that correctly (this is the current sigmoid rescale, not the
  crushed linear rescale it replaced — see `DESIGN_RATIONALE.md` "Centered-transition sigmoid").
- `bridge=0.55` — candidate was moderately positioned between both piers.
- `trans_beam=0.25` vs `T=0.092` would be a metric regression in a healthy run — `trans_beam`
  should match `T` for the same edge unless the row is stale after repair or an input was
  missing.
- `local_sonic_cos=0.030` — very low raw cosine; scaled local-sonic penalty (Knob 2) would demote
  this significantly.
- `local_pen=0.021` — current legacy-mode penalty is tiny; confirms `scaled` mode would help.
- `bpm=118->131 dist=0.152` — within the `dynamic` pace band (0.75); pace is not the cause of
  this edge's badness.

Likely fix for this edge: check `T` vs `trans_beam` first. If they match and are both low, tune
upstream scoring (`local_sonic_edge_penalty_mode: scaled`, `min_edge_objective: min_edge`, or
variable bridge length if this segment's nominal length is a poor fit) before reaching for edge
repair as a fallback.

---

## Track replacement as post-generation refinement

Single-track replacement is a GUI refinement tool, not a pre-generation tuning knob. It uses the
most recent generation's in-memory artifact bundle and transition metric to search for one local
substitute between the previous and next playlist tracks. **Best Match** ranks by transition
quality; **Different Pace**, **Different Genre**, and **Different Sound** first keep
high-transition candidates, then re-rank the top 50 by BPM/rhythm, genre-vector, or
timbre+harmony divergence from the current track. If replacement suggestions are consistently
weak, tune candidate admission, transition weights, and pace/genre/sonic modes upstream before
relying on manual replacement.
