# TODO — live backlog

Open, actionable items that aren't yet a spec or a plan. Newest first.

For the frozen 2026-04-30 audit synthesis see `audit/07-roadmap.md`; for prior decisions on a
subsystem check the auto-memory index (MEMORY.md) and `docs/superpowers/specs/` **before**
designing anything here — first-principles redesigns of already-decided subsystems are a
recurring interrupt cause.

---

## Reggae tail starvation — worst edge 0.232 (pool-breadth lever)

**Opened:** 2026-07-26 · **Status:** open, known regression, deliberately shipped · **Area:** genre mode / candidate pool

Genre mode's transitive `is_a` family seeding (`98bd6b4`, merged `6f1c6d1`) lifted soul's worst
transition 0.275 → 0.882, but **reggae regressed 0.504 → 0.232**.

**What is actually happening — the failure moved head→tail, it did not appear from nowhere:**

- The *old* run opened badly: position 1 was a Sonny & The Sunsets indie-rock track.
- The *new* run opens correctly on Augustus Pablo and holds **24 straight roots/dub tracks**,
  then drifts out over the final ~6 slots.
- Reggae's bridge pool is only **590 tracks**, so the last segment starves and the beam has
  nothing in-genre left to reach for.

**Why this is a pool-breadth problem, not a seeding problem.** The measured A/B (one process,
only `genre_family_ids` swapped) showed the bridge pool **byte-identical in both arms** —
soul 4703, reggae 590. Family seeding changes *piers*, never the pool. So the lever here is
pool construction, not the taxonomy walk. Do not "fix" this by touching family seeding.

**Prior art to design from — read before proposing anything:**

- `project_pool_starvation_research` (memory, 2026-07-12) — starvation is **manufactured, not
  scarcity**: pier-centric provisioning (M1), artist-cap double-enforce (M2), size-only
  relaxation (M3). A fix was never designed. **Start here.**
- `project_genre_mode` (memory) — the A/B methodology and the veto/neighbour-set fix.
- Corridor widening ladder (`corridor_widen_*`) is the sole segment-level recovery mechanism
  now; the legacy relaxation ladder was deleted (Phase 1 Task 8).

**First diagnostic step — do not skip it.** Per the `playlist-testing` skill, run one real
reggae generation at INFO and read the gate tally + per-segment `pool_after_gate` lines.
"0.232" alone cannot distinguish a true in-genre scarcity from a starved final segment where
the beam never had candidates to rank. Confirm which before designing.
