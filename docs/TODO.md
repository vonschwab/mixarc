# TODO — live backlog

Open, actionable items that aren't yet a spec or a plan. Newest first.

For the frozen 2026-04-30 audit synthesis see `audit/07-roadmap.md`; for prior decisions on a
subsystem check the auto-memory index (MEMORY.md) and `docs/superpowers/specs/` **before**
designing anything here — first-principles redesigns of already-decided subsystems are a
recurring interrupt cause.

---

## ROOT CAUSE — album ↔ release_key identity is computed by several disagreeing rules

**Opened:** 2026-07-27 · **Status:** open, diagnosed, not designed · **Area:** identity / genre authority

Every genre-authority defect found on 2026-07-27 traces to one thing: **there is no single
album → release_key mapping.** At least four rules are in play and they disagree.

| Rule | Where | Used for |
|---|---|---|
| `make_release_key(artist, album)` — alias-resolved | `normalization.py` | edits, live lookups |
| stored `release_key` | `release_effective_genres` | published rows |
| stored `release_key` | sidecar `enriched_genre_signatures` | **publish's album→key map** |
| stamped `album_id` | `genre_graph_release_genre_assignments` | (was) graph membership |

**Measured symptoms, all the same root:**

| Symptom | Count | Status |
|---|---:|---|
| Albums an edit would DESTROY (graph membership judged by stamped album_id, not key) | 31 | FIXED `a035346` |
| Albums whose stored signature key ≠ derived key | 74 of 1585 mapped | open |
| ...of those: `Various` compilations keyed to a *contributor* | 29 | open |
| ...collaborations keyed to the non-primary artist | 3 | open |
| ...alias drift (`(Sandy) Alex G` vs `Alex G`) | ~42 | open |
| Albums stuck on the LEGACY layer despite having graph genres under their derived key | 5 | open |
| Albums whose stored `release_effective_genres` key ≠ derived | 90 | open |

Worked example — the 5 legacy-stuck albums, all compilations/collabs:

```
Joseph Shabason & Ben Gunning — Ample Habitat
   derived: joseph shabason::ample habitat      (primary artist)
   stored : ben gunning::ample habitat          (the OTHER artist)
Various — Boddie Recording Company … Disc 1
   derived: various::boddie …                   (the albums row)
   stored : angela alexander::boddie …          (a track contributor)
```

Enrichment stored the release under an individual contributor; the album row says `Various`.
The keys can never meet, so publish never maps the album into the graph and it falls to the
legacy layer with 1–3 raw tags instead of its full graph genre set.

**Alias resolution is one instance of this, not a separate problem.**
`make_release_key` → `normalize_release_artist` → `normalize_primary_artist_key` applies
`resolve_alias`, so adding a line to `data/artist_aliases.yaml` changes the storage key of
every release by that artist and orphans rows written under the old one. See memory
`project_artist_alias_graph_key_coupling`: the next genre edit then rebuilds the album from a
key that finds nothing, and deletes its genres.

**Cost of adding one alias today** (measured for Jimi Hendrix, 2026-07-27):

- 15 sidecar tables are keyed by `release_id` or `release_key`.
- Hendrix alone has **15 distinct release_ids across three spellings**, and at least three
  releases (`live in maui`, `axis bold as love …`, `live at berkeley 2nd show`) already exist
  under **two** keys — so a re-key is a MERGE with conflict resolution, not a rename.
- Then a publish, then verification.

That is the price of every future alias, for a modest benefit each time.

**Direction — UNTESTED, do not build without checking it first.** Single-source the mapping:
one function, one table, every consumer reads it. `identity_keys._primary_artist_key_raw`
already exists (the pre-alias key, added to stop `build_artist_link_map` recursing), so
storage identity can stop depending on the alias map. Disproof attempt run 2026-07-27: alias
resolution currently merges only 3 release keys library-wide, and 2 of those merge on
punctuation (`Godspeed You!` vs `Godspeed You`), so removing it from the key splits almost
nothing. That is one failure mode ruled out, not a validated design — the compilation and
collaboration cases above are untouched by it and need their own answer.

**Superseded framing (kept for the numbers):** `identity_keys` already has
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

### CORRECTED 2026-07-27 (Dylan) — the pier is legitimate; the TAGS are wrong

Dylan: *The Diving Kind* was recorded with a largely African band and **is** reggae plus
afrobeat/highlife. So the terminal pier is a correct choice and the "contaminated pier"
framing below is wrong on the music. What is wrong is the **genre data**, in three
compounding ways — none of which is the compression:

**1. Artist-name collisions inject another artist's genres.** `Gigi — Illuminated Audio` (the
Bill Laswell dub reworking of the Ethiopian singer) published as `dance, dub, trance`. Its
ALBUM tags said `african, ambient, dub, electronic`; its ARTIST tags said `dance, dream
trance, italo dance, trance` — a **different Gigi**, an Italo-dance act. The authority took the
artist tags and dropped `african`. Same failure as the 2026-06-12 Last.fm collision
("Green-House" → a Ukrainian hip-hop act, ~76 artists), still live via MusicBrainz artist
lookups.

**2. Artist-level tags are not down-weighted.** `layered_assignment.SOURCE_RELIABILITY` has one
`musicbrainz: 0.75` bucket, so `musicbrainz_artist` (a career tag) carries the same authority
as `musicbrainz_release` (this record). `legacy_genres_by_album` gets it right —
track 1.0 / album 0.8 / artist 0.5 — and the enrichment path does not. This is how
`indie_rock` (from `surf rock`/`indie` artist tags) landed on an African-band reggae record.

**3. World/African records are under-tagged.** *The Diving Kind* carries no `afrobeat` or
`highlife`; the taxonomy has both. Gigi's `african` was dropped. Nothing in Sonny's raw
sources said reggae at all — the AI enrichment inferred it unaided, which is a point in its
favour, but it stopped short.

**Compression is NOT the problem — measured.** Over 40k random pairs: Pearson r(raw, dense)
= 0.728; of 36,744 genuinely-unrelated pairs (raw < 0.20) only **0.4%** exceed dense 0.6 and
**0.0%** exceed 0.8. The dense embedding is not flattening distinctions. Bill Callahan's
raw 0.562 → dense 0.949 is likely correct, not an artifact: *Have Fun With God* is a dub
record and carries `reggae` tags.

**Directions — UNTESTED.** Split `musicbrainz` into release/artist reliability tiers so album
evidence outranks career tags; and treat artist-level tags for short/generic artist names as
unverified. Neither is validated. The name-collision half has prior art — see the
genre-data-authority skill's trap catalog.

---

### SUPERSEDED — "a non-reggae track is a structural PIER" (wrong on the music, kept for the mechanics)

The tail does not drift. It is **pulled**. The five anchor seeds were:

```
Augustus Pablo – Young Generation Dub     Bad Brains – Jah Calling
Sonny & The Sunsets – Letters from…  <--  The Slits – Liebe And Romanze
The Upsetters – Croaking Lizard
```

`Sonny & The Sunsets` is the **terminal pier** — it is track 30. Piers are mandatory
waypoints, so the final segment is *required* to bridge from roots reggae into an indie-rock
track. Positions 25–30 are that bridge. Its genre similarity to the other four seeds is 0.662
(the others sit at 0.74–0.83), and its album is tagged `indie_rock + reggae +
rhythm_and_blues` — so it entered through the **core `reggae` genre itself**, not through
`dub`. Note the 2026-07-26 observation above: this same track was position 1 in the old run.
Family seeding moved where it sits; it never stopped being a pier.

**Two rejected fixes — measured, do not retry:**

1. *"Admit a family/neighbour genre only if the album also carries the core genre."*
   Co-tag rates are ~0% across the board — 18 of soul's 18 family genres, 12 of hip-hop's 12 —
   because enrichment assigns specific leaves (`neo_soul`), never the umbrella (`soul`),
   exactly as principle 12 wants. Would have deleted soul's entire family and undone the
   0.275 → 0.882 win.
2. *"`pool_track_ids` returns a genre→similarity map that `playlist_generator.py:2825`
   discards into `_sims`; carry it into scoring."* True that it is discarded — but the signal
   is **not missing**. Measured against a reggae core centroid:

   | | raw 442-dim | dense 64-dim | vs actual seeds |
   |---|---:|---:|---:|
   | core reggae | 0.931–0.935 | 0.997–0.999 | 0.864 |
   | Tortoise | 0.286 | 0.426 | 0.357 |
   | Bill Callahan | 0.562 | **0.949** | 0.872 |

   The beam already has this. Adding pool affinity would be redundant.

**Unresolved, worth a look:** the log says `Genre hard gate applied: 0 candidates excluded
(floor=0.652)`, yet Tortoise/Gigi/Peaking Lights measure 0.36–0.51 against the seeds — below
that floor. Either they never reach that gate (segment pools may be built separately from the
gated candidate pool) or the gate's similarity is not what was measured here. Resolve before
designing anything.

**No fix proposed.** Three hypotheses died on contact with a measurement; the honest next step
is to find where pier selection admits a genre-atypical anchor, and whether the existing
sonic pier-bridgeability veto (`project_pier_bridgeability`) has a genre analogue. Baseline to
beat: `min_T = 0.2318` in the log above.

Splitting `dub` in the taxonomy (Jamaican dub vs dub-as-production-style) remains the honest
root fix for the tag itself, but it is taxonomy surgery and the affinity fix does not need it.

Re-run this exact generation and compare `min_T` — the log above is the baseline (0.2318).
