# Tag-steering anchor placement — live regression validation (2026-07-27)

Gate evidence for Task 5 of `docs/superpowers/specs/2026-07-27-tag-steering-anchor-placement.md`
(gap insertion for tag-steering on-tag anchor piers, Tasks 1-4). This is the record that
decided whether the change ships. Measured in the `PG3_SAT2` satellite, branch
`feat/tag-steering-anchor-placement`, against the live DB + artifact via absolute paths
(`config.yaml`), `random_seed=0`, `popular_seeds_mode="off"`, `track_count=30`, through the
exact production harness (`_artist_generator` / `create_playlist_for_artist`,
`tests/integration/test_gui_fidelity_regressions.py`) — not hand-built overrides.

**Caveat carried into every number below (per the task brief):** every historical baseline in
the spec and in the regression test file predates the corridor-wide-universe recalibration
(`test_gui_fidelity_regressions.py` ~line 660) and, separately, the Eno + neoclassical case is on
record as regressing hard under Phase B pre-gap-insertion (0.133 with anchors vs 0.528 baseline,
2026-07-09) — which does not match the spec's stated 0.68. None of those stale absolutes are the
bar here. Every number below is a **same-session** `anchor_max=3` (default) vs `anchor_max=0`
pair, both generated today, in the same run of the harness.

## Summary table

| Case | With anchors (min_T / below_floor) | Baseline anchor_max=0 (min_T / below_floor) | Anchors placed | Verdict |
|---|---|---|---|---|
| Boards of Canada + hauntology | 0.7108 / 0 | 0.6508 / 0 | 2/3 (1 dropped, unbridgeable) | **PASS — anchors improve the worst edge by +0.060** |
| Brian Eno + neoclassical | 0.7714 / 0 | 0.7714 / 0 | 3/3 | **PASS — worst edge unchanged to 4 decimals; consistent with the gap-insertion fix addressing the documented 2026-07-09 regression (see below)** |
| Alvvays + twee pop | 0.8805 / 0 | 0.7038 / 0 | 3/3 | **PASS — anchors improve the worst edge by +0.177** |

All three cases clear the transition floor (0.20) by a wide margin and show **zero** below-floor
edges in either arm. **BoC — the case flagged as "the one to watch" — does not regress; it
improves.** No tuning, no objective changes were made as a result of this measurement.

---

## Case 1: Boards of Canada + hauntology

**With anchors** (`anchor_max=3`, default): `min_transition=0.7108`, `mean_transition=0.858`,
`below_floor=0`, `distinct_artists=27`.

Log (`Tag steering anchor placement:`):
```
2/3 anchor(s) placed into gaps [1, 2] (artist piers=4)
dropped 1 anchor(s) with no gap clearing min_bridge=0.35: ['afacfe5ac9cd8ec5785c0cd889928b40']
```
`afacfe5a...` = The Focus Group — "Stringed Winds" (dropped: unbridgeable to either flanking
artist pier at the min_bridge=0.35 floor — the same floor that already gated it at *selection*
time in earlier Phase-B measurements; this is expected, not a regression).

Realized pier sequence (`Pier+Bridge: seed order`), terminals and gaps annotated:
```
[0] Boards of Canada — All Reason Departs           (artist pier, TERMINAL)
[1] Boards of Canada — Blood In The Labyrinth        (artist pier)
[2] Plone — The Greek Alphabet                       (ANCHOR, gap 1)
[3] Boards of Canada — Memory Death                  (artist pier)
[4] Belbury Poly — Caermaen                          (ANCHOR, gap 2)
[5] Boards of Canada — Somewhere Right Now In The Future (artist pier, TERMINAL)
```
Both placement rules hold: anchors are non-terminal, non-adjacent (separated by a BoC pier at
position 3).

**Baseline** (`anchor_max=0`): `min_transition=0.6508`, `mean_transition=0.818`, `below_floor=0`,
`distinct_artists=27`. Artist-pier order is byte-identical to the with-anchors run's artist
subsequence (`All Reason Departs → Blood In The Labyrinth → Memory Death → Somewhere Right Now In
The Future`) — confirms artist-pier ordering is independent of `anchor_max`, as designed (anchors
are excluded from `_order_seeds_by_bridgeability` entirely).

**Delta: +0.060 on the worst edge, with anchors present.** This is the case the spec explicitly
named as "the one to watch" because its prior (append + global re-order) worst-edge win was
partly a product of where the anchors happened to land; gap insertion **preserves** that win.

---

## Case 2: Brian Eno + neoclassical

**With anchors**: `min_transition=0.77143`, `mean_transition=0.896`, `below_floor=0`,
`distinct_artists=16`.

Log:
```
Tag steering on-tag anchors: injected 3 on-tag pier(s) across 3 artist(s):
  ['335ce5420bb56fc73c8e53a0d0690e67', '4a639a2e8a6c0d4ff2a7da9ffdcbf37a', '0ed2c6efdac59e8b3ce240dce963358b']
Tag steering anchor placement: 3/3 anchor(s) placed into gaps [0, 1, 2] (artist piers=4)
```
Realized pier sequence:
```
[0] Brian Eno — Five Light Paintings                              (artist pier, TERMINAL)
[1] Harold Budd — Valse Pour Le Fin Du Temps                       (ANCHOR, gap 0)
[2] Brian Eno — 77 Million Paintings                               (artist pier)
[3] Roger Eno & Brian Eno — Obsidian                               (ANCHOR, gap 1)
[4] Brian Eno — The Ritan Bells                                    (artist pier)
[5] Mary Lattimore & Walt Mcclements — The Poppies, the Wild
     Mustard, the Blue-Eyed Grass                                  (ANCHOR, gap 2)
[6] Brian Eno — Discreet Music                                     (artist pier, TERMINAL)
```
Every one of the 4 gaps between the 4 artist piers is filled (`P-1=3` gaps, `K=3` anchors — exact
fit, no clamping needed). Non-terminal and non-adjacent hold.

**Baseline** (`anchor_max=0`): `min_transition=0.77145`, `mean_transition=0.905`, `below_floor=0`,
`distinct_artists=26`. Artist-pier order matches the with-anchors artist subsequence exactly
(`Five Light Paintings → 77 Million Paintings → The Ritan Bells → Discreet Music`).

**Delta: -0.00002 on the worst edge — unchanged to four decimals.** `distinct_artists` drops from
26 (baseline) to 16 (with anchors) — the anchor-anchored bridges route through a narrower
ambient/neoclassical-adjacent neighborhood rather than the baseline's more eclectic Eno-adjacent
pool. That's a real compositional difference, but it's a variety trade-off, not a quality
regression — the floor-relevant metrics (min_T, below_floor) are unaffected.

**This is the headline finding.** `test_phase_b_anchors_no_regression_real_estate_jangle_pop`
(this repo, ~line 688) has this case on record as a genuine Phase-B regression **before** gap
insertion: worst-edge 0.133 with anchors vs 0.528 baseline (measured 2026-07-09), with the
documented cause being that "the injected neoclassical anchors are sonically distant from the Eno
pier chain once they must connect to their SEQUENTIAL pier neighbours (bridgeability is only
checked against the nearest seed pier, not the ordered chain)." What was measured today —
worst-edge unchanged to four decimals, 3/3 anchors placed — is **consistent with gap insertion
having fixed that failure mode**: the §C minimax gap-assignment objective scores each anchor
against `min(bridge(a, p_i), bridge(a, p_{i+1}))` — exactly the two piers it will actually sit
between in the final sequence — rather than only its nearest seed pier under the old append +
global-re-order path. This is stated carefully: **one seed, one config, one day.** It is not
proof the fix generalizes across seeds/configs, only that the previously-documented failure mode
does not reproduce here, and the mechanism-level reason (minimax over the actual flanking pair) is
a plausible, traceable explanation for why. Nothing here overrides the spec's original 0.68 figure
either way — that figure predates the corridor-wide-universe recalibration and was never
reproduced live.

---

## Case 3: Alvvays + twee pop

**A tag-string correction was required.** The task brief and the spec's Task 5 table both say
"Alvvays + twee." Measured with the literal tag `"twee"`, the anchor mechanism **never engages,
in either arm**: `resolve_tag_steering_target` logs `Tag steering: 1/1 selected tags not in the
artifact genre vocabulary: ['twee']` / `no selected tags mapped — steering disabled for this
run`, and the anchor-injection guard in `playlist_generator.py` (~line 2213,
`if steering_target is not None and _on_tag_track_ids and _anchor_max > 0:`) requires that same
`steering_target`. Since it is `None` for `"twee"`, the guard fails **regardless of
`anchor_max`** — `anchor_max=3` and `anchor_max=0` take a byte-identical code path. Confirmed
independently: the artifact's dense genre vocabulary contains `"twee pop"` but not `"twee"`
(checked directly against `bundle.genre_vocab`), and a GUI chip would send the artist's own
published genre *name* (`"twee pop"`), not the shorthand. The `"twee"` with-anchors arm did run
to completion (`min_transition=0.6714`, `below_floor=0`, `distinct_artists=26`) — but only after
~6.9 minutes (16:34:05→16:40:57) of `CorridorWiden` widen/exhausted grinding (see the cost note
below); its `anchor_max=0` counterpart was killed mid-run rather than waited out, since the
code-path argument above already settles the comparison (verified by reading the guard condition,
not inferred from timing) and the arm was ~10.5 minutes into an apparent stall with no new
information to gain. **This is reported as a named gap, not silently absorbed**: the brief's
literal tag string does not exercise the feature under test.

**The measurement below uses `"twee pop"`** — the actual vocabulary/authority term — which does
engage the anchor mechanism on both the dense-vocab path (steering) and the authority path
(anchor candidate universe).

**With anchors** (`anchor_max=3`): `min_transition=0.8805`, `mean_transition=0.936`,
`below_floor=0`, `distinct_artists=24`.

Log:
```
Arc-aware ordering: moved the lowest-support pier off the terminal seat (tag-steering allocation path)
Tag steering on-tag anchors: injected 3 on-tag pier(s) across 3 artist(s):
  ['fdfadbd4...', '5587e4f8...', 'db9bf0df...']
Tag steering anchor placement: 3/3 anchor(s) placed into gaps [0, 1, 2] (artist piers=4)
```
Realized pier sequence:
```
[0] Alvvays — The Agency Group          (artist pier, TERMINAL)
[1] The Umbrellas — Never Available     (ANCHOR, gap 0)
[2] Alvvays — Adult Diversion           (artist pier)
[3] Cub — Someday                       (ANCHOR, gap 1)
[4] Alvvays — Archie, Marry Me          (artist pier)
[5] Go Sailor — Bigger Than an Ocean    (ANCHOR, gap 2)
[6] Alvvays — Party Police              (artist pier, TERMINAL)
```
All 3 gaps filled, non-terminal and non-adjacent hold.

**Baseline** (`anchor_max=0`): `min_transition=0.7038`, `mean_transition=0.905`,
`below_floor=0`, `distinct_artists=22`. Artist-pier order matches the
with-anchors artist subsequence exactly (`The Agency Group → Adult Diversion → Archie, Marry Me →
Party Police`).

**Delta: +0.177 on the worst edge, with anchors present** — well clear of the 0.20 transition
floor in both arms (baseline itself sits at 0.7038, far above the floor; the brief's expected
"just above 0.20" baseline is from a pre-corridor-recalibration measurement and does not describe
current behavior — consistent with the general recalibration caveat above).

**Generation cost note:** the with-anchors `"twee"` arm (which never engages anchors, per above)
itself took ~6.9 minutes — well over the ~60s/generation estimate — grinding through repeated
`CorridorWiden[seg 0]` widen/exhausted cycles on a narrow candidate pool. Its `anchor_max=0`
counterpart ran for ~10.5 minutes (started 16:40:57, killed ~16:51:30) with **~8 minutes of
complete log silence** on the same segment, before being killed as scientifically redundant (see
above) rather than waited out. No `time.sleep` exists in this code path — this is genuine, if
slow, beam-search grinding under an
**unbounded relaxation budget**: `create_playlist_for_artist` called through this harness passes
no `deadline`, and `pier_bridge_builder.py` (~line 2453-2457) explicitly disables the 40s
per-segment relaxation cap when `deadline is None`, "so 'no time limit' must mean BOTH wall-clock
cutoffs are off." **This stall is a property of near-floor / narrow candidate pools for this artist+tag pairing
under an unbounded harness deadline, not something this task's change introduced** — the
corrected `"twee pop"` pair, which has a materially different (larger, more central) candidate
pool, completed in ~65s total for both arms, back in line with the ~60s/generation estimate.
Future harness-driven measurements of near-floor cases should consider passing an explicit
`generation_budget_s` to avoid this failure mode.

---

## §B: does the upstream terminal-avoiding order survive into the builder's sequence?

The spec (§B, second bullet) infers — flagged explicitly as "not yet instrumented" — that
`reorder_avoiding_low_support_terminal` (Corridor Phase 2 Task 3), which computes a
terminal-avoiding order over the artist piers upstream (in `_cap_order` or the tag-weighted pier
allocation path), used to have its result **unconditionally discarded** by the builder's
co-equal re-order (`_order_seeds_by_bridgeability` over piers+anchors together). With artist
piers alone now defining the sequence, the spec asks whether that upstream computation
"becomes load-bearing again."

**Observation: not confirmed — and the evidence points the other way for the common case.**

1. **`python main_app.py --artist "Sonic Youth" --tracks 30`** (no tags; log:
   `logs/playlists/2026-07-27_162954_Sonic_Youth_000001.log`): `_cap_order` capped 6 medoids to
   `target_pier_count=4`; **no** `Arc-aware ordering: moved...` line fired (the tie-break ran —
   `pier_support_terminal_avoidance: true` is the config.example.yaml default — but made no
   change, i.e. the lowest-support pier wasn't at a terminal to begin with for this run). The
   builder then logged `Seed ordering: evaluated 24 permutations, best_score=0.5409` (4! = 24,
   the **exhaustive** path) and `Pier+Bridge: seed order = [...]`.

2. **The Alvvays + twee pop run above DID fire the tie-break**: `Arc-aware ordering: moved the
   lowest-support pier off the terminal seat (tag-steering allocation path)`, confirming the
   mechanism is live in this feature's actual call path. The builder still evaluated **24
   permutations** (exhaustive, 4 artist piers) before producing `Pier+Bridge: seed order`.

3. **Neither BoC nor Eno fired `Arc-aware ordering`** (both went through the tag-first-piers path,
   `_cap_order`, and made no change either) — both also landed on 24-permutation exhaustive
   search (4 artist piers each).

4. **The mechanics of `_order_seeds_by_bridgeability`'s exhaustive branch** (`n<=6`,
   `src/playlist/pier_bridge/seeds.py:130-144`) make the answer structural, not just empirical:
   `itertools.permutations(seed_indices)` enumerates **every** ordering of the pier *set*,
   independent of the order that set arrived in, and keeps whichever permutation scores highest
   (`total_score > best_score`, strict). The upstream order only matters as a tie-break on an
   **exact** floating-point score tie — never observed in any run here. So for every pier count
   actually measured (Sonic Youth 4, BoC 4, Eno 4, Alvvays 4 — the target-pier-count floor of 3
   plus one, the typical case), whatever `reorder_avoiding_low_support_terminal` computes upstream
   is **provably discarded** by the builder's own optimum, exactly as before this task's change.
   This was already true pre-fix too: a pre-existing log from 2026-07-19
   (`logs/playlists/2026-07-19_124325_Gaussian_Curve_21bdee.log`, unrelated generation, 6 piers)
   shows the SAME pattern — `Arc-aware ordering: moved...` fired (support=0.957), and the builder
   still independently ran `Seed ordering: evaluated 720 permutations` (6! = 720, still exhaustive)
   to produce its own order.

**Where the claim could hold, unverified here:** `_order_seeds_by_bridgeability`'s **greedy**
branch (`n>6`) literally starts its walk at `seed_indices[0]` — the first element of whatever
order it was handed — so the upstream computation genuinely would influence (not fully determine)
the final terminal there. None of the four cases measured reach that regime with artist piers
alone (all have exactly 4). Removing anchors from the ordering step, this task's actual fix,
*reduces* the node count fed into `_order_seeds_by_bridgeability` relative to the pre-fix
append-everything path — which, if anything, pushes MORE cases onto the exhaustive path where
this claim doesn't apply, not fewer. The first side-effect the spec names (Sonic Youth's 9-pier
greedy path becoming a 4-pier exhaustive path) is real and reproduced structurally by this same
mechanism, but it's a separate claim from the "load-bearing again" one, and the two point in
opposite directions.

**Conclusion:** the spec's §B second inferred claim is **not confirmed** by this measurement and
is contradicted by the structural mechanics of the exhaustive-search path for every pier count
actually exercised (the common case, `target_pier_count` 3-6). It was explicitly flagged in the
spec as inferred rather than instrumented, and this write-up settles it: false for n≤6, untested
for n>6.

---

## Full fast suite

```
python -m pytest -q -m "not slow"
2581 passed, 7 skipped, 12 failed, 62 deselected, 2 warnings in 161.36s (0:02:41)
```

All 12 failures are pre-existing and environment-specific — **zero relation to this branch's
diff** (`git diff origin/master...HEAD --stat` against the failing files is empty):

- 11 in `tests/test_git_shared_checkout_guard.py` — canonical-vs-satellite hook-decision
  behavior (this checkout is a satellite, `PG3_SAT2`; per this repo's own CLAUDE.md, hook
  behavior deliberately differs in satellites — "in satellites it downgrades to a once-per-session
  reminder").
- 1 in `tests/test_workspace_identity.py::test_this_repo_is_canonical` — the test's own docstring
  states the assumption it depends on: "The canonical checkout's origin is GitHub — detection must
  say canonical." This checkout genuinely is a satellite, so `is_satellite(...)` correctly
  returns `True` and the test (written to only pass in the canonical checkout) fails here by
  design.

Anchor-specific unit tests spot-checked directly and pass:
`python -m pytest -q tests/unit/test_anchor_ids_threaded.py -k anchor` → `4 passed in 1.03s`.

**Count discrepancy vs. the brief's stated `origin/master` baseline (2726 passed / 2 skipped / 0
failed):** total collected here is 2662 (`2600/2662 tests collected (62 deselected)`, confirmed
via `pytest --collect-only`, no collection errors) — fewer than the stated 2728-total baseline,
not more, despite this branch adding tests. No collection errors were found and the internal
counts are self-consistent (2581+7+12 = 2600 run, +62 deselected = 2662 collected). This gap
is not explained by anything in this branch's diff (no test files were deleted) and is reported
as-is rather than assumed benign — possibly explained by a different measurement environment
(canonical vs. satellite) or a stale baseline figure, neither confirmed here.

## Lint and type-check

```
ruff check .
Found 2 errors. (F401 unused-import, both fixable)
```
Both in `tests/unit/test_genre_mode_pier_duration.py` and `tests/unit/test_genre_mode_topology.py`
— pre-existing, untouched by this branch (`git diff origin/master...HEAD --stat` against both is
empty). Not attributed to this work; not fixed here per the task brief's scope.

```
mypy src/    # the CI-canonical invocation (.github/workflows/ci.yml); `mypy .` errors out on an
             # unrelated tools/ module-path collision unrelated to any package config this task
             # touches
Success: no issues found in 194 source files
```

## Files touched by this task

- `src/playlist/pier_bridge/anchor_placement.py` (Task 1) — pure gap-insertion placement function.
- `src/playlist/pier_bridge_builder.py` (Task 2) — wires gap insertion into the seed-ordering step,
  orders artist piers only, inserts anchors after.
- `src/playlist_generator.py`, `src/playlist/pipeline/core.py` (Task 2-3) — thread anchor identity
  from selection through to the builder.
- `config.example.yaml` — `tag_steering_anchor_gap_insertion: true` (Task 3).
- This doc + `docs/PLAYLIST_ORDERING_TUNING.md` (Task 5).

## Verdict

**Ship.** All three cases pass; BoC (the case flagged as highest-risk) improves rather than
regresses; the previously-documented Eno Phase-B regression does not reproduce and the mechanism
gives a plausible causal account of why, stated with appropriate one-day/one-seed caution. The
§B side-effect claim was investigated as instructed and found not to hold for the common
(exhaustive-search) case — recorded here as a plain finding, not something requiring a code
change. No tuning or objective changes were made to produce these results.
