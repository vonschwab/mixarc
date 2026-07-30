# The Beam Contract

**What this is.** The authoritative, enforced contract for the pier-bridge beam search — the engine
that fills each segment between two piers with an interior of bridge tracks. The beam is this
codebase's most delicate hotspot (`src/playlist/pier_bridge/beam.py` + the segment loop and post-beam
cascade in `src/playlist/pier_bridge_builder.py`, ~5.3k LOC). Read this **before** changing anything in
the beam or its scoring.

**Why a contract.** The beam's per-candidate score is a sum of ~20 optional terms, and it is trivially
easy to add a term in the wrong place and silently break transition quality — the one thing the whole
system optimizes for (north-star #5: *the worst edge defines the experience*). This document states the
invariants that must hold, and — the "live" part — names the test that enforces each one. If you change
the beam, keep these tests green or update this contract in the same commit. A claim here without a
green test next to it is a bug in this document.

Line references are anchors (they drift); the cited **function names and test files** are stable. When
in doubt, grep the function, not the line.

---

## The load-bearing invariants (each has an enforcing test)

### I1 — A ranking bonus never contaminates the transition score `T`.
The per-candidate **ranking** score (`combined_score`) and the per-edge **transition-quality** score
(`T` / `trans_score_in_beam`) are two different numbers. Every "prefer this candidate" term — genre
tie-break, waypoint, coverage, popularity, roam, seed-character, progress, `dest_pull`, and the tag-
steering term (Task 6) — is added to `combined_score` only. `T` comes from a single function,
`transition_metrics.score_transition_edge` (`transition_metrics.py:183`), called once per edge via the
memoized `beam._score_shared_transition` (`beam.py:591`), and the recorded `edge_component["T"]` reads
straight from that untouched result (`beam.py:1469`, tie-break path `beam.py:1614`). This is what lets
the beam prefer on-tag / on-genre candidates *without* being able to degrade the edge quality the
worst-edge machinery reads.
**Enforced by:** `tests/unit/test_transition_metric_alignment.py` (asserts `trans_score_in_beam == reporter T`
both via a direct `_beam_search_segment` call and via the full `build_pier_bridge_playlist` → `result.stats["edge_scores"]`),
and by `tests/.../test_beam_contract.py::test_ranking_bonus_does_not_change_edge_T`.

### I2 — Opt-in features are byte-identical no-ops at their (in-code) defaults.
A default-constructed `PierBridgeConfig()` run and one with any single opt-in flag left at its default
produce identical `track_ids` and `edge_scores`. Adding a new scoring term MUST preserve this: gate it
so that "feature off" changes nothing. The tag-steering beam term (`sonic_tag_beam_weight=0.0` /
`sonic_tag_affinity=None`) follows this — see Extension Points.
**Caveat (critical for tests):** the *shipped* `config.example.yaml` overrides several in-code defaults
(`variable_bridge_length: true`, `seed_character_mode: anti_center`, `mini_pier_enabled: true`,
`genre_steering_enabled: true`). "Default" in this contract means the explicit `PierBridgeConfig()`
dataclass default, **not** the live runtime config. A byte-identical test that loads `config.yaml` is
testing the wrong thing.
**Enforced by:** `tests/unit/test_pier_bridge_smoke_golden.py` (golden with opt-in flags pinned off) and
`tests/.../test_beam_contract.py::test_tag_steering_beam_off_is_byte_identical`.

### I3 — Every post-beam repair/reorder stage is "never-worse" on the metric it protects.
The cascade that runs after the beam only ever accepts a change that improves (or ties) its target
metric; on any exception it is a no-op:
- **Variable bridge length** (`var_bridge.choose_segment_length`): chosen length's segment bottleneck
  (pure `T`) ≥ the nominal length's.
- **Tail-DP** (`tail_dp.optimize_segment_tail`): accepts a tail swap only if `new_min ≥ old_min + eps`,
  else returns `None`.
- **Edge-delete** (`repair/edge_delete.py`): deletes an endpoint only if the merged edge strictly beats
  the broken edge; never a pier; respects `min_gap`; capped at `edge_delete_max_deletions`.
- **Edge-repair** (`repair/edge_repair.py`): accepts a swap only if the touched edge-pair's *minimum*
  improves by `margin`. **Nuance:** this is a *local-pair* minimax, not global monotonicity — one of the
  two touched edges can end lower while their min rises. Do not describe it as "worst edge never drops."
**Enforced by:** `tests/unit/test_tail_dp.py`, `test_var_bridge_add_only.py`, `test_edge_delete.py`,
`test_edge_repair_break_glass.py`.

### I4 — Piers are never removed or reordered.
Piers are fixed anchors. Var-bridge and tail-DP operate only on the mutable interior; edge-repair
refuses to replace a pier position (`edge_repair.py`); edge-delete protects all seed indices (piers ⊂
seeds). The seed track IDs appear, in order, at `PierBridgeResult.seed_positions` in every result.
**Enforced by:** `tests/.../test_beam_contract.py::test_piers_preserved_through_cascade` (runs with
edge-repair + edge-delete on and asserts seed IDs at `seed_positions` are unchanged).

### I5 — The two selection objectives are independently wired.
Per-step beam pruning sorts by total `score` **unless** `worst_edge_minimax_enabled`, which switches
pruning *and* final selection to lexicographic `(min_edge, score)`. Independently, `min_edge_objective="min_edge"`
changes only the **final** `_select_best_beam_state` choice, not per-step pruning. Changing one must not
silently change the other.
**Enforced by:** `tests/unit/test_minimax_worst_edge.py`.

---

## Interface (what the beam consumes)

`_beam_search_segment(pier_a, pier_b, interior_length, candidates, X_full, X_full_norm, X_start, X_mid,
X_end, X_genre_norm, cfg, beam_width, **optional)` → `(interior_indices | None, genre_penalty_hits,
edges_scored, failure_reason | None)` (`beam.py:159`).

- **Two sonic spaces, deliberately:** `X_full`/`X_start/mid/end` are the (optionally centered)
  *transition-space* matrices that feed `T`; `X_full_norm` is the plain L2-normalized sonic matrix used
  for bridge-floor gating and progress geometry. Do not conflate them.
- **Optional scoring inputs**, by subsystem: genre (`X_genre_*`, `g_targets_override`); identity/diversity
  (`artist_key_by_idx`, `seed_artist_keys`, `recent_global_artists`, `artist_identity_cfg`, `bundle`);
  pace (`perceptual_bpm`, `tempo_stability`, `onset_rate`, `energy_matrix`); transition (`transition_metric_context`);
  roam (`roam_detour_sonic`, `roam_dev_energy`); tag-steering (`sonic_tag_affinity`, `sonic_tag_beam_weight`).

`build_pier_bridge_playlist(seed_track_ids, total_tracks, bundle, candidate_pool_indices, **optional)`
(`pier_bridge_builder.py:437`) owns the segment loop and the post-beam cascade. It pulls sonic/genre
matrices off `bundle`, not as raw arrays. It has exactly one production caller (`pipeline/core.py:919`).

---

## Candidate ranking score (`combined_score`)

Built per candidate, in execution order (`beam.py:1250-1425`; the tie-break path `beam.py:1500-1643`
re-applies the same arithmetic, only gating the multiplicative genre penalty on the tie-break band):

```
base         = weight_bridge*bridge_score + weight_transition*trans_score   # 0.6/0.4 default
  - anti_center_penalty            (opt-in: seed_character_mode != off)
  - pace_penalty                   (opt-in: BPM/onset soft bands + energy)
  - progress_penalty               (DEFAULT ON: progress_penalty_weight 0.15)
  - progress_arc / max_step        (opt-in)
  +/- genre term                   (DEFAULT ON legacy tiebreak 0.05, OR opt-in steering arc vote)
  + layered_delta                  (opt-in)
  - roam penalties                 (opt-in)
  - local_sonic / title_artifact / genre_pair penalties   (opt-in / inert at default floor)
  *= (1 - genre_penalty_strength)  (DEFAULT ON: 0.10 when genre_sim < 0.20)
  *= popularity_factor             (opt-in)
  + waypoint_delta                 (opt-in: dj_bridging_enabled)
  + coverage_bonus                 (opt-in: dj_bridging + coverage)
  + sonic_tag_beam_weight * sonic_tag_affinity[cand]   (opt-in: tag steering — Task 6)
new_score    = state.score + combined_score + dest_pull    # dest_pull (0.10) is OUTSIDE combined_score
```

`trans_score` here is one *input* to the base blend; it is not the reported edge `T` and mutating
`combined_score` never mutates `T` (see I1).

---

## Gates vs. soft penalties

**Hard gates** drop a candidate for the step. Active by default: `used`, artist diversity / `min_gap`
(no soft variant — the diversity floor is the one hard taste constraint), `bridge_floor` (0.03),
progress monotonicity, segment-pool-too-small. Opt-in / inert by default: BPM & onset bands (both
`inf`), anti-alignment `is_broken_transition` (only with `center_transitions=True`), genre-arc floor,
local-sonic hard floor. **Note:** the legacy `transition_floor` hard-reject has been *removed*
(`transition_metrics.py:243-251`); it no longer gates.

**Soft penalties** subtract/multiply `combined_score` and never reject — see the block above for which
are on by default.

---

## Beam mechanics

Width starts at 20, doubles per expansion attempt to a max of 100, over at most 4 attempts. Per-step
pruning keeps the top `beam_width` states by `score` (or by `(min_edge, score)` under minimax). `used`
tracks placed track indices; `used_artists` tracks identity keys for the diversity gate; both propagate
immutably into child states. `_score_shared_transition` and `_get_genre_sim` memoize per-pair within a
segment (bit-identical, read-only sharing).

---

## Post-beam cascade (order, inside `build_pier_bridge_playlist`)

1. **Variable bridge length** — evaluate several interior lengths, pick the max-bottleneck (I3).
2. **Beam search** — the primary `_beam_search_segment` call.
3. **Micro-pier fallback** — only if the beam returns no path; two half-segment beams around a waypoint.
   *This is the second caller of `_beam_search_segment` (`micro_pier.py`).*
4. **Greedy terminal fill** — last resort under `guarantee_feasible`; not a beam.
5. **Tail-DP** (default ON) — re-optimize the last ≤2 slots' min-T (I3).
6. Segment concatenation (piers de-duplicated across boundaries).
7. **Edge-repair** (opt-in) — whole-playlist break-glass swaps (I3, local-pair; I4).
8. **Edge-delete** (default ON) — remove-only worst-edge fix (I3; I4; `min_gap`-safe; cap 4).
9. **Final edge-score recompute** — fresh `score_transition_edge` over the emitted playlist; this is the
   authoritative `T` in `stats`.

**Roam corridors is NOT here** — it is an *in-beam* mechanism: detours are precomputed and passed into
the beam as soft penalties (and can flip the beam's objective via `worst_edge_minimax_enabled`).

---

## Extension points — how to add a per-candidate preference term (worked example: tag steering)

The tag-steering beam term (Task 6, 2026-07-07) is the reference pattern for "make the beam prefer a
class of candidate." To add such a term:

1. **Compute a bundle-aligned `(N,)` preference vector** upstream (in `pipeline/core.py`, next to where
   the pool/prototype is already resolved) — do not compute per-candidate inside the beam loop.
2. **Thread it + its weight** through `build_pier_bridge_playlist` → `_beam_search_segment` as optional
   params defaulting to `None`/`0.0` (so every other caller is unaffected — I2).
3. **Add it to `combined_score`, next to `coverage_bonus`, in BOTH passes** (normal `beam.py:~1397` and
   tie-break `beam.py:~1524`). Never add it to `trans_score`/`edge_metric` (I1).
4. **Gate on the feature being active** (`vector is not None and weight > 0`) and **log once per segment**
   at INFO when active (diagnostic logging is part of the feature).
5. **Keep the weight small and soft** — the post-beam cascade (I3) is your safety net, but a term large
   enough to force sonically-broken bridges will just make the repair work harder and can still lower
   average quality. Validate worst-edge min-T before/after on a real generation.

**Status of the tag-steering beam term (2026-07-08):** it ships **off by default**
(`tag_steering_sonic_beam_weight=0.0`). A weight sweep proved the beam term cannot raise the on-tag lean
at any weight — the beam only reorders within the candidate pool it is handed, so the **pool** (its
centered sonic-admission lever) is the binding lever for tag adherence, not the beam. The term remains
wired, tested (contract I1), and documented here for opt-in experimentation; treat it as a demonstration
of the extension pattern, not a live default.

**Known coverage gap:** the tag-steering term is threaded into the primary beam call but **not** into the
micro-pier fallback (`micro_pier.py`). Mini-pier interior segments therefore do not get the on-tag bonus.
This is byte-identical/safe, but if a future task needs mini-pier segments to lean on-tag, thread the two
params through `_attempt_micro_pier_split` as well.

---

## Config defaults reference

The single source of truth for every knob and its in-code default is `PierBridgeConfig`
(`src/playlist/pier_bridge/config.py`). Remember I2's caveat: `config.example.yaml` turns several opt-in
features ON for the shipped product, so "what runs live" ≠ "the dataclass default." Tests that mean to
exercise the no-op path must construct `PierBridgeConfig()` explicitly.
