# TODO — live backlog

Open, actionable items that aren't yet a spec or a plan. Newest first.

For the frozen 2026-04-30 audit synthesis see `audit/07-roadmap.md`; for prior decisions on a
subsystem check the auto-memory index (MEMORY.md) and `docs/superpowers/specs/` **before**
designing anything here — first-principles redesigns of already-decided subsystems are a
recurring interrupt cause.

---

## Artist aliases retroactively change storage keys — decouple them

**Opened:** 2026-07-27 · **Status:** open, scoped, not started · **Area:** identity / genre authority

`make_release_key` → `normalize_release_artist` → `normalize_primary_artist_key`, and that
last one applies `resolve_alias`. So **adding a line to `data/artist_aliases.yaml` changes the
storage key of every release by that artist**, retroactively orphaning rows written under the
old key. See memory `project_artist_alias_graph_key_coupling` for the damage that causes: the
next genre edit rebuilds the album from a key that finds nothing and deletes its genres.

**Cost of adding one alias today** (measured for Jimi Hendrix, 2026-07-27):

- 15 sidecar tables are keyed by `release_id` or `release_key`.
- Hendrix alone has **15 distinct release_ids across three spellings**, and at least three
  releases (`live in maui`, `axis bold as love …`, `live at berkeley 2nd show`) already exist
  under **two** keys — so a re-key is a MERGE with conflict resolution, not a rename.
- Then a publish, then verification.

That is the price of every future alias, for a modest benefit each time.

**Better investment — make storage identity alias-independent.** `identity_keys` already has
`_primary_artist_key_raw` (the pre-alias key; it exists to stop `build_artist_link_map`
recursing). Point `make_release_key` at that, and aliases go back to doing what they are for:
playlist-runtime identity — artist diversity, min-gap, seed matching — without touching stored
keys. One migration instead of one per alias.

Not free: ~90 albums already have a stored `release_key` that disagrees with the derived one,
so the change needs a re-key pass of its own. Worth specing before the next alias is added.

**Blocked on this:** the Jimi Hendrix alias (`Jimi Hendrix` / `Jimi Hendrix Experience` /
`The Jimi Hendrix Experience` are one act here, and their editions' genre evidence is split
across all three today).

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

---

### DIAGNOSED 2026-07-27 — it is CONTAMINATION, not scarcity. The stated lever is wrong.

Ran it (`logs/playlists/2026-07-27_141058_reggae_000001.log`), reproduced `min_T=0.2318`,
and read the log. The observation above is exactly right — positions 1–24 are genuine
roots/dub, 25–30 drift out. The *cause* is not pool breadth.

**All three weakest edges are one contiguous run of non-reggae tracks:**

```
T=0.232  Gigi – Kahn              -> Bill Callahan – Ride My Dub
T=0.349  Strange Garden           -> Tortoise – The Equator
T=0.533  Bill Callahan            -> Strange Garden – An Islamic Boat Song
```

Every one of them entered the pool through the `dub` tag:
`Tortoise = dub + jazz_fusion + krautrock + post_rock`,
`Peaking Lights = dub + chillwave + dream_pop`, `Gigi = dub + dance + trance`.

**The numbers that settle it:**

| Measure | Value |
|---|---:|
| Albums tagged `dub` | 76 |
| ...also carrying a core reggae genre | 27 |
| ...**`dub` with NO reggae genre** | **49** |
| `dub` share of the 590-track reggae pool | 497 |
| Tracks with a core reggae genre (no `dub`) | **378 across 40 artists** |
| Distinct artists the 30-track playlist used | 22 |

378 tracks across 40 artists is ample for 30 slots at `min_gap=6`, `max_artist=4`. Roughly
23 in-genre artists went **unused** while the beam reached for Tortoise. The tail is not
empty — it is full of the wrong tracks.

**So do NOT widen the pool.** Adding more `dub`-tagged tracks makes this worse. This is the
same shape as `project_pool_starvation_research`'s "manufactured, not scarcity", with a
specific mechanism: the transitive `is_a` family walk admits `dub`, and in *this library*
`dub` is 64% non-reggae, because the tag marks a production style as often as a Jamaican
genre.

**Design directions (not yet chosen — needs a decision):**

1. **Corroboration rule for family members.** A track admitted only via an umbrella family
   genre joins the pool only if its album (or artist) also carries a core genre. Cheapest,
   and targets the mechanism directly.
2. **Rank core above family.** Keep family members admissible but sort them below core-genre
   members, so the tail exhausts real reggae first. Softer; may still drift when core runs out.
3. **Split `dub` in the taxonomy** (Jamaican dub vs dub-as-production-style). Correct at the
   root and fixes every consumer, but it is taxonomy surgery and needs the growth loop.

Prefer (1) or (2); (3) is the honest long-term fix. Whichever is chosen, re-run this exact
generation and compare `min_T` — the log above is the baseline.
