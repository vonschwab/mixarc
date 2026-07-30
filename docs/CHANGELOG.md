# Playlist Generator Changelog

## Unreleased

### Artist mode

- **Multi-artist blend.** Artist mode accepts 2+ artists and builds every pier from the
  sonic/genre territory they share, rather than traveling from one artist's catalog to the
  other's (Brian Eno + Harold Budd skews ambient; Brian Eno + David Bowie skews art rock).
  Bundle rows split into one exclusive group per artist plus a joint group of jointly-credited
  tracks; each group gets a centered MuQ prototype, and a soft "overlap affinity" (75% sonic /
  25% genre by default) biases pier selection toward the other artists' territory — never a
  gate, so a pairing with no shared ground still generates. New config block:
  `playlists.ds_pipeline.multi_artist` (`docs/CONFIG.md`, `PLAYLIST_ORDERING_TUNING.md` Knob 13).

## v6.0.0 - Learned Sonic Embedding, Genre Authority, Browser GUI

**Release Date:** 2026-07-11
**Focus:** Replace the perceptually-unreliable hand-built sonic towers with a learned
*contrastive* audio embedding (MuQ); make published graph-resolved genres the authority;
rebuild pace on tempo + rhythmic density; add weak-edge recovery, collapse prevention, and
artist-mode tag steering; consolidate on the browser GUI; and remove deprecated code.

### Sonic

- **Learned contrastive MuQ embedding is the sole sonic similarity space.** The hand-built
  rhythm/timbre/harmony towers were perceptually coarse (the dominant timbre tower rated
  Metallica ≈ Yeah Yeah Yeahs), capping playlist quality regardless of tuning. v6.0 replaces
  them with **MuQ-MuLan-large** — a self-supervised *contrastive* audio embedding (512-d) baked
  into the artifact as `X_sonic_muq{,_start,_mid,_end}` with `sonic_variant_override: muq`.
  Post-processing is `center_l2` (mean-center → L2) fitted on the full library. On trusted
  soundalike triplets MuQ screens ~84% correct vs MERT's ~73% (and the towers far lower).
- **A short intermediate MERT era preceded MuQ and has been fully removed.** MERT (768-d,
  `whiten_l2`) briefly replaced the towers before MuQ superseded it; the MERT/tower code paths
  and the `tower_weights`/`transition_weights` knobs were deleted. The MERT shards + sidecars are
  archived under `data/archive/mert_2026/` (irreplaceable — ~55h CPU to regenerate; never deleted).
- **Single live sonic space — no tower/MERT rollback.** Future embeddings slot in via the variant
  seam (extract a sidecar, add a `fold_<variant>` + auto-fold, register transition calibration).
- **sonic_mode floors recalibrated** to MuQ cosine percentiles, replacing the tower-era 0.3–0.5
  floors (MuQ cosines occupy a different range).

### Ordering & structure

- **Weak-edge recovery cascade.** After the beam, the weakest transitions are recovered
  least-to-most destructive: variable-bridge (add-only) → tail-DP segment endgame → break-glass
  edge repair (`T < 0.30`, never-worse) → edge delete (never a pier, never-worse).
- **Collapse prevention.** Anti-center scoring plus structural mini-piers counter
  bridge-interior sag (genre-blur / sonic collapse in long bridges).
- **Artist-mode tag steering.** A soft pool + pier lean toward the seed artist's own published
  genres, so an artist seed's bridges keep that artist's genre character.
- **Manual artist links.** `data/artist_aliases.yaml` (edited via the GUI "Artist Links" tab)
  merges aliases into one computed identity and spaces sibling projects ≥ `min_gap` apart — a
  runtime layer, no DB/artifact write.

### Pace

- **Rebuilt on tempo + rhythmic density.** The rhythm-cosine hard floor (near-noise,
  unsatisfiable for beatless/ambient artists) is replaced by two embedding-independent hard
  bands — **BPM log-distance** and **onset-rate log-distance** — plus a **soft** rhythm-cosine
  penalty (tower variants only). Bands widen on segment backoff so pace never blows the 90 s
  budget. Because the bands read DB features, pace is unchanged by the tower→MERT→MuQ migration,
  and `pace_mode: narrow` is now usable for an ambient seed.

### Genre

- **Enriched genres are the authority.** `release_effective_genres` (metadata.db), written
  only by the enrichment **publish** stage and read via `src/genre/authority.py`, is the single
  source consumers read. The generation artifact bakes it in (`genre_source: graph`).
- **Layered taxonomy graph** (`data/layered_genre_taxonomy.yaml`, ~455 genres) resolves and
  orders genres; artifact genre vectors exclude inferred hub-families (which had saturated
  genre similarity).
- **Multisource enrichment via Claude** (Agent SDK, no API billing), run as resumable
  `analyze_library.py` stages. Fusion rebalanced to weight artist-page over label-storefront
  evidence, never drop correct local tags, and avoid self-corroboration; a surgical delta
  migration repaired legacy mislabels (e.g. storefront/last.fm wrong-identity contamination).
  Transient rate-limit failures **pause** the enrich stage cleanly before publish, so partial
  enrichment never reaches metadata.db.
- **Genre Review GUI panel** queues hybrid-evidence review terms per release for human
  accept/reject (persisted as user overrides); genre chips are graph-canonical, ordered
  most-specific → broadest.

### GUI & pipeline

- **Browser GUI is the only front-end** (`tools/serve_web.py`): Generate, Tools (library
  analyze/enrich), and Genre Review tabs over a FastAPI + NDJSON worker. The PySide6 desktop
  GUI was removed.
- **`analyze_library.py`** orchestrates the full pipeline as resumable, fingerprint-skipping
  stages including the new `muq` extraction stage.

### Repo cleanup

- Removed deprecated/dead code: `src/genre/vocab_normalization.py`; dead A/B-sweep and
  one-off scripts (`sweep_pier_bridge_dials`, `run_dj_*_ab`, `fix_compound_genres`,
  `rebuild_sonic_tower_weighted` + `src/features/sonic_rebuild.py`); the PyInstaller
  `build_windows.ps1`; and their tests.
- Added `tools/dead_code_audit.py` (static reachability audit) to keep dead wiring from
  re-accumulating.
- `config.yaml` is no longer tracked; generated test DBs / backups / web test output are
  gitignored.

## v4.3.0 - 2DFTM Harmony Tower Rebuild

**Release Date:** 2026-06-03
**Branch:** `sonic-harmony-keyinvariant`
**Focus:** Replace the key-sensitive harmony tower with a key-invariant representation,
validated by blind A/B audition against the full library.

### Highlights

- **Key-invariant harmony tower.** Replaced the 20-dim chroma-median harmony tower
  with a 96-dim **2D Fourier Transform Magnitude (2DFTM)** representation. The old
  tower encoded absolute pitch class (musical key), which is noise for harmonic-character
  similarity — tracks in the same key ranked as harmonically similar regardless of actual
  musical relationship. 2DFTM captures chordal texture and voice-leading patterns as
  spatial frequencies; transposition becomes a circular shift on the pitch axis → phase
  change → discarded by the magnitude operation. Tested: cosine similarity between a
  track and itself pitch-shifted 1–7 semitones stays 0.99+ under 2DFTM vs 0.77–0.86
  under chroma.

- **Blind head-to-head A/B audition.** Three seeds rated (45 tracks per space, one rater,
  verdicts match=3 / close=2 / off=1 / wrong=0):
  | Seed | 2DFTM avg | Legacy avg | Δ |
  |---|---|---|---|
  | Jean-Yves Thibaudet (classical piano) | **2.53** | 0.87 | **+1.66** |
  | Green-House (ambient new-age) | 1.67 | 1.53 | +0.13 |
  | Minor Threat (hardcore punk) | 1.29 | **2.13** | **−0.84** |
  | **Overall** | **1.84** | **1.51** | **+0.33** |

  The Minor Threat counter was deliberately sought: power chords (root+fifth, no third)
  have minimal harmonic texture, so the key acts as an accidental genre proxy for legacy.
  The net result still favors 2DFTM ~2:1, and harmony is weighted at 0.30 of the blend.

- **Blend grows from 86 → 162 dimensions.** New layout: rhythm 9 + timbre 57 + harmony 96,
  all via `sqrt(w) * L2(tower)` per tower. The artifact's `tower_dims=[9,57,96]` is now
  the authoritative source for axis slicing (exposed on `ArtifactBundle.tower_dims`).

- **`ArtifactBundle.tower_dims` added.** The per-tower blend split (rhythm/timbre/harmony)
  is now loaded from the artifact and exposed on the bundle. `worker._resolve_tower_pca_dims`
  prefers it over width-based inference, which went wrong for the non-default 162-dim blend
  (inferred `(40,81,41)` vs true `(9,57,96)`). Affected path: GUI track-replacement pace/sound
  divergence scoring.

- **Segment harmony is global.** Start/mid/end harmony all use the full-track 2DFTM
  (only whole-track features were extracted). Rhythm and timbre segments remain
  position-specific.

### New scripts

| Script | Role |
|---|---|
| `scripts/extract_harmony_2dftm_sidecar.py` | Full-library 2DFTM extraction (~17h, resumable, 0 failures) |
| `scripts/fold_2dftm_into_artifact.py` | Surgical harmony-tower replacement with backup + atomic write |
| `scripts/sonic_audition_build.py --head-to-head` | Blinded legacy-vs-2DFTM A/B manifest builder |

Full investigation write-up (probe methodology, gate framework, weight sweep, key-invariance
proof, beat-sync comparison, A/B results): `docs/SONIC_PHASE2_HARMONY_FINDINGS.md`.

### Tests

1432 unit tests pass. New: `test_worker_tower_pca_dims.py` (resolution priority for the
authoritative tower split), `test_artifact_tower_weighted_load.py` extended (tower_dims
load + None fallback). Updated: `test_sonic_audition_build.py` (4-space assertion).

---

## v4.2.0 - Upstream Transition Alignment and Edge Repair Fallback

**Release Date:** 2026-05-21
**Branch:** `codex-artist-mode-genre-conflict`
**Focus:** Make the beam, builder stats, reporter, and repair fallback use one final-edge transition metric

### Highlights

- **Shared final-edge transition metric.** Added `src/playlist/transition_metrics.py` so beam `trans_score_in_beam`, builder `edge_scores`, reporter `T`, and edge repair evaluate edges through the same context and formula.
- **T-mismatch is now a regression.** The selected-edge audit still reports mismatches, but they now indicate stale audit data or missing-data fallback rather than an expected beam-vs-reporter formulation gap.
- **Beam hard gates use final `T`.** `transition_floor`, `min_edge_objective`, and the centered-cos catastrophic gate now operate on the shared edge dict.
- **Opt-in edge repair fallback.** Added default-off `pier_bridge.edge_repair` config. Repair protects seeds and piers, rejects duplicate/disallowed/title-artifact candidates, and accepts swaps only when both adjacent edges clear the floor and improve worst adjacent `T` by at least `0.05`.

### Tests

- Added shared metric parity coverage for beam-vs-reporter scoring and catastrophic centered-cos detection.
- Added focused edge-repair tests for clean no-ops, interior swaps, source-before-pier swaps, and refusal rules.

## v4.1.0 - Candidate Pool Fixes, Edge Diagnostics, and Transition-Score Alignment

**Release Date:** 2026-05-21
**Branch:** `codex-artist-mode-genre-conflict`
**Focus:** Restore narrow-style playlist generation, add per-edge diagnostics, and align beam transition scoring with the perceived-quality metric

### Highlights

- **Generation restored for narrow-style artist playlists.** A previously-introduced raw genre-conflict hard gate (`min_confidence=0.50`) was rejecting ~50% of legitimate candidates against the 764-dim raw artifact vocabulary with identity affinity; downstream One Each fallbacks could not relax the gate, so segments starved (Tiger Trap / 50-track playlists at `very_low` artist presence failed at every floor). Hard gate is now off by default; the soft penalty (`strength=0.20`) still demotes off-axis tracks.
- **Per-edge audit diagnostic.** New `emit_selected_edge_audit: true` flag emits a "Selected-edge audit" log block with one row per final-playlist edge showing `T`, `T_centered_cos`, `S`, `G`, `bridge_score`, the beam's own `trans_score_in_beam`, `progress_t/jump`, `local_sonic_raw_cos`, applied penalties, `title_flags`, and a `⚠` prefix for below-floor edges.
- **Beam vs. final-T mismatch detector.** Cross-checks every emitted edge against the beam's internal score and warns when the beam optimistically scored an edge that the reporter then judged below the transition floor.
- **Transition-score weight alignment.** Default `transition_weights` changed from `rhythm 0.50 / timbre 0.25 / harmony 0.15` (rhythm-dominant) to `rhythm 0.20 / timbre 0.50 / harmony 0.30` to match `tower_weights`. Brings beam transition scoring into the same feature balance as the rest of the pipeline, eliminating large beam-vs-reporter score divergences on timbre-dominated mismatches. Measured impact on a representative seeded playlist: `mean_T` 0.828 → 0.898, `p10_T` 0.567 → 0.709, `min_T` 0.366 → 0.459.
- **Segment pool no longer hard-collapsed to one track per artist.** The segment-pool builder previously kept only the top-harmonic-mean track per artist before the beam ran, biasing every segment toward mid-projection tracks. New `collapse_segment_pool_by_artist: false` (default in `config.yaml`) lets the beam see multiple tracks per artist across the projection range; the beam still enforces one-per-segment artist diversity via `used_artists`. Critical for long narrow-style segments.
- **Cluster + neighbor pool sizes increased.** `per_cluster_candidate_pool_size` 800 → 2000 and `genre_neighbor_pool_size` 500 → 1500 in `config.yaml`. Provides more sonically-diverse bridging candidates for artist-style playlists.

### New opt-in knobs (default off, backward compatible)

All four can be enabled in `playlists.ds_pipeline.pier_bridge:` and are documented in `docs/PLAYLIST_ORDERING_TUNING.md`.

| Knob | Purpose |
|---|---|
| `emit_selected_edge_audit` | Per-edge audit table + T-mismatch detector (now treated as a regression check in v4.2) |
| `title_artifact_penalty` | Demote demo/live/medley/remix/instrumental/take/outtake/alternate titles |
| `local_sonic_edge_penalty_mode: scaled` + `local_sonic_edge_penalty_scale` | Replaces the decorative legacy local-sonic penalty math (max ≈0.03) with a scale-based formula that can produce meaningful 0.05-0.30 demotions |
| `min_edge_objective: min_edge` | Beam selection prefers paths whose worst edge is highest (lexicographic by `min trans_score_in_beam`, ties by total score) |

### Title-quality detection

New pure-function module `src/playlist/title_quality.py` with `detect_title_artifacts(title) -> Set[str]` recognising `live`, `demo`, `medley`, `remix`, `instrumental`, `remaster`, `version`, `take`, `mono`, `stereo`, `edit`, `outtake`, `alternate`. Word-boundary matching to avoid false positives (`demolish` ≠ `demo`); `mono`/`stereo` are gated to parenthetical/bracketed contexts to avoid demoting tracks like "Stereo Hearts".

### Tests

925 unit tests pass. New focused tests cover: `test_selected_edge_audit`, `test_title_quality`, `test_title_artifact_penalty`, `test_local_sonic_scaled_mode`, `test_min_edge_objective`, `test_beam_vs_final_t_diagnostic`.

---

## v4.0.0 - Native GUI Overhaul and CLI Parity

**Release Date:** 2026-05-12
**Version:** 4.0.0
**Branch:** `release/v4.0`
**Focus:** Native PySide6 GUI overhaul, CLI parity, shared request validation, Analyze Library UX, and GUI reliability

### Highlights

- Restored GUI/CLI parity for Artist, Genre, Seeds, and History generation flows.
- Added first-class `strict`, `narrow`, `dynamic`, `discover`, and `off` matching controls for genre and sonic axes.
- Rebuilt generation controls with responsive grouped cards and compact mode-specific panels.
- Added shared request models and inline validation so GUI and CLI dispatch the same generation shape.
- Improved Analyze Library job UX with summary readouts, job details, stage results, and controlled logging.
- Modernized results table, export footer, logs, jobs panel, dialogs, diagnostics banner, and Advanced Settings dark-theme styling.
- Disabled conflicting tool actions while worker jobs are busy.
- Removed the deprecated active-config DJ pooling flat key in favor of nested `dj_bridging.pooling.strategy`.
- Expanded GUI, worker, request-model, and Analyze Library regression coverage.

---

## v3.5.0 - Quality-of-Life Improvements

**Release Date:** 2026-04-29
**Version:** 3.5.0
**Branch:** `release/v3.5`
**Focus:** Faster scans, resumable GUI jobs, cache-backed genre updates, and diagnostics

### Highlights

- Added cancellation/checkpoint infrastructure for long-running GUI worker operations.
- Added job details UI with progress, checkpoint, error, and performance diagnostics.
- Added persistent genre lookup caching for MusicBrainz/Discogs metadata refreshes.
- Improved library scan throughput with batch-oriented processing.
- Improved artist-style clustering with collaboration-aware matching and duration-aware medoid selection.
- Added verbose worker logging and performance event plumbing for easier debugging.
- Updated genre normalization imports in tests to the unified genre module path.

---

# Playlist Generator v3.4 - DJ Bridge Mode & Multi-Seed Playlists

**Release Date:** 2026-01-10
**Version:** 3.4.0
**Branch:** `release/v3.4` (60 commits from dj-ordering)
**Focus:** Multi-seed playlist generation with genre-aware bridging

---

## 🎧 What's New: DJ Bridge Mode

Version 3.4 introduces **DJ Bridge Mode**, a revolutionary approach to multi-seed playlist generation that creates smooth transitions between 2-10 seed tracks using genre-aware routing and intelligent candidate pooling.

### The Problem We Solved

**Before v3.4:**
- Single-seed playlists only: Start with one artist, expand outward
- Multi-artist playlists had no control over genre evolution
- Bridging between stylistically distant artists was hit-or-miss
- No way to plan genre arcs (e.g., shoegaze → dream pop → indie rock)

**With v3.4:**
- **Multi-seed support**: 2-10 seed tracks as anchors ("piers")
- **Genre-aware routing**: Plans optimal genre paths between seeds
- **Union pooling**: Combines local, toward, and genre candidates
- **Beam search**: Explores multiple paths to find optimal bridges
- **Artist diversity**: Enforces constraints within and across segments

### Real-World Example

```
Seeds: Slowdive, Beach House, Deerhunter, Helvetia
Mode: DJ Bridge (dj_union pooling)
Result: 30-track playlist with 3 segments

Segment 1 (Slowdive → Beach House): 10 tracks
  Genre arc: shoegaze → dream pop → psychedelic

Segment 2 (Beach House → Deerhunter): 10 tracks
  Genre arc: dream pop → indie rock → noise rock

Segment 3 (Deerhunter → Helvetia): 10 tracks
  Genre arc: noise rock → indie rock → lo-fi
```

---

## 🏗️ DJ Bridge Architecture

### 1. Seed Ordering

**What it does:** Arranges N seed tracks in optimal order to minimize total bridging distance.

**How it works:**
- Evaluates all permutations of seed order
- Scores each ordering by sonic similarity between consecutive pairs
- Selects ordering with highest total bridgeability

**Why it matters:** Starting with Radiohead → Aphex Twin → Boards of Canada is much easier to bridge than Radiohead → Boards of Canada → Aphex Twin. Optimal ordering reduces jarring transitions.

---

### 2. Union Pooling Strategy (Phase 1)

**The Innovation:** Instead of one candidate pool, DJ Bridge Mode combines three specialized pools:

#### **S1: Local Pool**
- Top-K neighbors of current pier
- Provides familiar, safe transitions
- Example: k_local=200 → 100 near pier A + 100 near pier B

#### **S2: Toward Pool**
- Top-K candidates moving toward destination pier per step
- Ensures progress toward bridge endpoint
- Example: k_toward=80 per step × 30 steps = 2,400 candidates

#### **S3: Genre Pool**
- Top-K candidates matching genre waypoint targets per step
- Introduces discovery while staying on-genre
- Example: k_genre=40 per step × 30 steps = 1,200 candidates

**Deduplication:** Union of S1 + S2 + S3, capped at k_union_max (default: 900)

**Result:** Balanced pool that's safe (local), progressive (toward), and adventurous (genre).

---

### 3. Genre Waypoint Planning

**Two Modes:**

#### **Onehot Mode (Legacy)**
- Finds shortest path in genre similarity graph
- Selects single-label waypoints
- **Problem:** Collapses to hub genres ("indie rock"), loses nuance

#### **Vector Mode (Phase 2 - NEW)**
- Direct interpolation between anchor genre vectors
- Preserves full multi-genre signatures
- Formula: `g_step = (1-t) * vA + t * vB` where t = step fraction
- **Benefit:** Shoegaze → dream pop → psychedelic (not "indie rock" × 30)

---

### 4. Phase 2: IDF Weighting & Coverage Bonus

#### **IDF (Inverse Document Frequency) Weighting**

**The Problem:** Common genres dominate scoring.
- "indie rock" appears in 40% of library → weight = 1.0
- "shoegaze" appears in 2% of library → weight = 1.0
- Result: System treats rare and common genres equally

**The Solution:** IDF down-weights common genres, emphasizes rare genres.
- Formula: `idf = log((N+1)/(df+1))^power`
- "indie rock" (40% of library) → idf = 0.2
- "shoegaze" (2% of library) → idf = 0.9
- Result: Rare genre signatures preserved

**Configuration:**
```yaml
dj_genre_use_idf: true
dj_genre_idf_power: 0.5          # Exponential scaling
dj_genre_idf_norm: l2            # Normalization method
```

#### **Coverage Bonus**

**The Problem:** Waypoint scoring alone insufficient to prefer genre-aligned candidates.

**The Solution:** Reward candidates matching anchor's top-K genres with schedule decay.
- Tracks top-8 genres per anchor
- Bonus weight: 0.15 (additive to score)
- Schedule decay: `wA = (1-s)^power, wB = s^power`
- Early steps favor pier A genres, late steps favor pier B genres

**Configuration:**
```yaml
dj_genre_use_coverage: true
dj_genre_coverage_weight: 0.15
dj_genre_coverage_top_k: 8
```

**Impact:** Genre pool candidates now competitive with local/toward pools.

---

## 🎯 Beam Search with Waypoint Guidance

### How It Works

1. **Initialize beam** at pier A with width=40 (configurable)
2. **For each step:**
   - Expand each beam state to all valid candidates
   - Score candidates by:
     - **Bridge score** (harmonic mean of sim_A and sim_B)
     - **Transition score** (end-to-start segment similarity)
     - **Waypoint bonus** (genre target similarity, optional)
     - **Coverage bonus** (anchor genre matching, optional)
   - Keep top-K states (beam pruning)
3. **Select path** with highest total score reaching pier B

### Waypoint Scoring

**Centered Mode (Default):**
- Computes baseline similarity (median/mean of all candidates)
- Calculates centered delta: `Δ = sim(cand, target) - baseline`
- Applies tanh squashing: `bonus = waypoint_cap * tanh(Δ / waypoint_cap)`
- Allows negative deltas (penalizes off-genre candidates)

**Configuration:**
```yaml
dj_waypoint_enabled: true
dj_waypoint_weight: 0.25         # Additive weight
dj_waypoint_cap: 0.10            # Maximum bonus/penalty
dj_waypoint_delta_mode: centered # Allow negative deltas
```

---

## 👥 Artist Identity Resolution (Enhanced in v3.4)

### The Problem (Fixed in v3.4)

**Featured artists were invisible:**
- "Charli XCX feat. MØ" → system only saw "charli xcx"
- "Bob Brookmeyer & Bill Evans" → system only saw "bob brookmeyer"
- Min-gap constraints couldn't block collaborators

**Root cause:** Identity resolution received pre-normalized strings with collaboration markers stripped.

### The Solution (v3.4)

**Use raw artist strings** that preserve collaboration markers:
- "Charli XCX feat. MØ" → `{"charli xcx", "mø"}`
- "Bob Brookmeyer & Bill Evans" → `{"bob brookmeyer", "bill evans"}`
- "Mount Eerie with Julie Doiron & Fred Squire" → `{"mount eerie", "julie doiron", "fred squire"}`

**Supports all collaboration types:**
- feat., featuring, ft., with, &, and, x, +, comma-separated

**Configuration:**
```yaml
constraints:
  artist_identity:
    enabled: true
    strip_trailing_ensemble_terms: true  # "Bill Evans Trio" → "bill evans"
```

### The Impact

- ✅ **True artist diversity** in collaboration-heavy libraries
- ✅ **Jazz playlists** no longer repeat featured soloists
- ✅ **Electronic playlists** block all remix artists and collaborators
- ✅ **Hip-hop playlists** respect featured artist constraints
- ✅ **Ensemble normalization** still works (Trio, Quartet, Orchestra, etc.)

---

## 📊 Comprehensive Diagnostics

### Always-On Pool Visibility

Every segment logs pool composition:
```
DJ union pool: S1_local=200 S2_toward=720 S3_genre=0 | overlaps: L∩T=156 L∩G=0 T∩G=0
```

Shows:
- Raw pool sizes before deduplication
- Overlap counts (how many tracks in multiple pools)
- Dedup efficiency

### Per-Track Membership

Each chosen track logs which pools it came from:
```
[Track 0] idx=24932 pools=L+T
[Track 1] idx=18843 pools=T
[Track 2] idx=31204 pools=L+T+G
```

Legend: L=local, T=toward, G=genre, B=baseline

### Waypoint Saturation Metrics

Measures if waypoint scoring is effective:
```
Waypoint saturation: sim0=0.673 delta(mean=-0.0087 p50=0.0011 p90=0.0626) near_cap=20.5% at_cap=15.5%
```

Shows:
- **sim0**: Baseline similarity (centered mode)
- **delta distribution**: Mean, p50, p90 of centered deltas
- **near_cap**: Fraction of candidates with |delta| > 0.8*cap
- **at_cap**: Fraction of candidates with |delta| > 0.9*cap

**Good:** <30% near_cap means waypoint scoring has room to differentiate
**Bad:** >50% at_cap means cap too low, all candidates bunching

---

## ⚡ Performance Optimizations

### 1. Transition Score Caching

**Moved computation outside per-step loop:**
- Before: 30 steps × 1,000 candidates = 30,000 operations
- After: 1,000 candidates = 1,000 operations
- **Speedup: 30×** for this component

### 2. Genre Pool Caching

**Cache genre pool selections per step:**
- Reused across beam states
- Avoids redundant similarity computations
- Especially impactful for wide beams (width=60+)

---

## 🛠️ Code Quality Improvements

### IDF Consistency in Waypoint Scoring

**Problem:** Waypoint similarity used non-IDF matrix even when IDF enabled.

**Fix:** Use IDF-weighted matrix (`X_genre_norm_idf`) when available, matching target construction.

**Impact:** Waypoint deltas now accurate when IDF is enabled.

---

### Genre Vector Source Alignment

**Problem:** When `dj_genre_vector_source=raw`, targets used raw matrix but scoring used smoothed matrix (vector space mismatch).

**Fix:** Scoring matrix (`X_genre_for_sim`) now matches target construction mode:
- Raw mode: Use raw matrix + IDF (if enabled) + normalize
- Smoothed mode: Use smoothed matrix (X_genre_norm_idf or X_genre_norm)

**Impact:** Targets and scoring use same space, eliminating distorted rankings.

---

### Raw Genre Vector Normalization

**Problem:** Raw genre weights can exceed 1.0 (track/album weights are 1.2), causing coverage >1.

**Fix:** Normalize raw vectors before computing weighted coverage.

**Impact:** Coverage values stay in 0-1 range, bonuses don't exceed configured weights.

---

### Progress Arc Override Support

**Problem:** Overrides that set `progress_arc.enabled: false` were ignored if base config had it enabled.

**Fix:** Apply override regardless of enabled state (fully bidirectional).

**Impact:** Can now disable progress_arc via overrides without editing base config.

---

## 📚 Documentation Overhaul

### Professional Structure

- Created **`docs/README.md`** as documentation hub with navigation guide
- Added **"I want to..."** quick links for common tasks
- Archived **20+ diagnostic files** to `docs/archive/diagnostics_2026-01/` (git-ignored)
- Created **clean TODO.md** focused on future roadmap (not historical issues)

### Archive Policy

Diagnostic reports, A/B tests, and design docs archived after development cycles complete. Retained locally for reference but git-ignored to keep repo clean.

---

## 🔧 Configuration Reference

### DJ Bridge Mode Settings

```yaml
pier_bridge:
  # Pooling strategy
  pooling:
    strategy: dj_union           # Use union pooling
    k_local: 200                 # Local pool size
    k_toward: 80                 # Toward pool per step
    k_genre: 0                   # Genre pool per step (opt-in)
    k_union_max: 900             # Max after dedup

  # Genre waypoint planning
  dj_ladder_target_mode: vector  # Direct interpolation (Phase 2)
  dj_genre_vector_source: smoothed  # Use smoothed matrix

  # IDF weighting (Phase 2)
  dj_genre_use_idf: true
  dj_genre_idf_power: 0.5
  dj_genre_idf_norm: l2

  # Coverage bonus (Phase 2)
  dj_genre_use_coverage: true
  dj_genre_coverage_weight: 0.15
  dj_genre_coverage_top_k: 8

  # Waypoint scoring
  dj_waypoint_enabled: true
  dj_waypoint_weight: 0.25
  dj_waypoint_cap: 0.10
  dj_waypoint_delta_mode: centered

  # Artist diversity
  disallow_pier_artists_in_interiors: true
  disallow_seed_artist_in_interiors: false

constraints:
  artist_identity:
    enabled: true
    strip_trailing_ensemble_terms: true
```

---

## 📈 Benefits Summary

| Feature | Before | After | Impact |
|---------|--------|-------|--------|
| **Multi-seed playlists** | Not supported | 2-10 seeds | Revolutionary workflow |
| **Genre bridging** | Accidental | Planned arcs | Smooth evolution |
| **Pool diversity** | Single source | 3 sources (L+T+G) | Balanced safe/adventurous |
| **Genre nuance** | Hub collapse | Multi-genre preservation | "shoegaze" not "indie rock" |
| **IDF weighting** | Equal weights | Rare genre emphasis | Signature preservation |
| **Coverage bonus** | Waypoint only | +Top-K matching | Genre pool competitive |
| **Artist diversity** | Name-based | Identity-based | Blocks collaborators |
| **Diagnostics** | Limited | Comprehensive | Full visibility |
| **Performance** | Baseline | 30× cache speedup | Faster generation |
| **Documentation** | Scattered | Professional hub | Easy navigation |

---

## 🚀 How to Use DJ Bridge Mode

### Basic Multi-Seed Playlist

```bash
# CLI
python main_app.py \
  --seed-list "Slowdive,Beach House,Deerhunter,Helvetia" \
  --tracks 30 \
  --output my_playlist.m3u

# GUI
1. Select "Seed List Mode" from dropdown
2. Enter comma-separated seeds: "Slowdive, Beach House, Deerhunter"
3. Set target tracks: 30
4. Click "Generate Playlist"
```

### Configuration Tips

**For tight genre cohesion:**
```yaml
pooling:
  k_genre: 40           # Enable genre pool

dj_waypoint_weight: 0.30  # Strong waypoint guidance
```

**For adventurous discovery:**
```yaml
pooling:
  k_toward: 120         # More toward candidates
  k_genre: 0            # Disable genre pool (pure sonic)

dj_waypoint_enabled: false
```

**For collaboration-heavy libraries (jazz, electronic):**
```yaml
constraints:
  artist_identity:
    enabled: true       # Must enable for featured artist blocking
```

---

## 🧪 Testing

All test suites passing:
- **16 DJ/pier-bridge tests** - Core algorithm validation
- **34 artist identity tests** - Collaboration splitting, ensemble normalization
- **3 segment pool builder tests** - Union pooling, deduplication
- **28 pool diagnostics tests** - Overlap metrics, provenance tracking
- **5 DJ ladder planner tests** - Genre path planning, vector mode

**Total: 86+ tests** with zero regressions.

---

## 🎓 Technical Deep-Dive

### Why Union Pooling?

**Baseline pooling (before v3.4):**
- Top-K neighbors of current position
- Problem: Gets stuck in local optima
- Problem: Can't discover genre-aligned alternatives

**Union pooling (v3.4):**
- Local pool: Safety net (always have familiar options)
- Toward pool: Progress guarantee (always moving toward goal)
- Genre pool: Discovery engine (finds genre-aligned alternatives)
- Result: Balanced exploration-exploitation

### Why Vector Mode > Onehot Mode?

**Onehot mode:**
- Shortest-path label selection
- Picks single genre per waypoint
- Multi-genre tracks lose information
- Example: "indie rock" × 30 steps (hub genre collapse)

**Vector mode:**
- Direct multi-genre interpolation
- Preserves full genre signatures
- No information loss
- Example: shoegaze=0.38 + dream pop=0.33 + psychedelic=0.25 (nuanced)

### Why IDF + Coverage?

**IDF:** Makes rare genres "louder" in scoring
- Without IDF: "indie rock" dominates (appears everywhere)
- With IDF: "shoegaze" gets fair representation (rare but important)

**Coverage:** Rewards matching anchor top-K genres
- Without coverage: Waypoint scoring alone is weak signal
- With coverage: Genre pool candidates become competitive
- Schedule decay: Early steps favor pier A genres, late steps favor pier B genres

---

## 🔮 Future Roadmap (v3.5)

- **Genre path visualization** in audit reports
- **User-specified genre waypoints** (force route through specific genres)
- **A/B comparison mode** (baseline vs dj_union side-by-side)
- **Preset configurations** for common use cases (tight cohesion, discovery, etc.)
- **Pool parameter auto-tuning** based on saturation metrics

---

## 📦 Upgrade Notes

**No breaking changes.** All DJ Bridge features are additive and backward-compatible.

**To enable DJ Bridge Mode:**
1. Set `pooling.strategy: dj_union` in config
2. Enable artist identity: `constraints.artist_identity.enabled: true`
3. Use seed list mode with 2+ seeds

**Upgrading to v3.4:**
- All existing v3.3 playlists continue to work without changes
- Single-seed playlists unaffected (use traditional mode)
- DJ Bridge Mode only activates with 2+ seeds
- Enable artist identity for collaboration-aware diversity

---

## 🙏 Acknowledgments

DJ Bridge Mode represents 59 commits and comprehensive architectural evolution:
- **Phase 1:** Union pooling foundation, seed ordering, segment allocation
- **Phase 2:** Vector mode, IDF weighting, coverage bonus
- **Phase 3:** Quality fixes, diagnostics, documentation

Special thanks to the code review process for identifying critical bugs in artist identity resolution and genre vector alignment.

---

**Version:** 3.4.0
**Branch:** `release/v3.4`
**Commits:** 60 (from dj-ordering)
**Release Type:** Major Feature Release
**Tag:** v3.4.0
**Documentation:** `docs/DJ_BRIDGE_ARCHITECTURE.md` (54KB comprehensive guide)

For complete technical details, see `DJ_BRIDGE_ARCHITECTURE.md` and commit history.
