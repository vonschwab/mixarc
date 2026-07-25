import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useInfiniteSearch } from "../lib/useInfiniteSearch";
import { useLocalStorage } from "../lib/useLocalStorage";
import type { FlowDial, GenerateRequestBody, Mode, PaceDial, RangeDial } from "../lib/types";
import { StylePopover } from "./StylePopover";
import { GenreAutocomplete } from "./GenreAutocomplete";
import { btnPrimary } from "../lib/ui";

// ── Constants ────────────────────────────────────────────────────────────────

const DIVERSITY_GAMMAS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.08];
const DIVERSITY_LABELS = ["very low", "low", "normal", "high", "very high", "one each"];

// ── Shared style helpers ──────────────────────────────────────────────────────

const SEL = "bg-well border border-border rounded text-xs text-muted py-[3px] pl-[5px] pr-[18px] appearance-none cursor-pointer disabled:opacity-30 disabled:cursor-default"
  + " [background-image:url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%235b6470'/%3E%3C/svg%3E\")] [background-repeat:no-repeat] [background-position:right_5px_center]";

function Lbl({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <span title={title} className="text-2xs uppercase tracking-[.08em] text-faint font-medium whitespace-nowrap select-none">
      {children}
    </span>
  );
}

// A bordered cell within a toolbar row
function Cell({ children, grow, push, className = "" }: {
  children: React.ReactNode;
  grow?: boolean;
  push?: boolean;
  className?: string;
}) {
  return (
    <div className={[
      "flex items-center gap-[6px] px-3 py-[6px]",
      "border-r border-hairline last:border-r-0",
      grow ? "flex-1 min-w-0" : "",
      push ? "ml-auto border-l border-r-0 !border-r-0" : "",
      className,
    ].join(" ")}>
      {children}
    </div>
  );
}

function Segmented({ value, options, onChange, offDefault }: {
  value: string; options: { v: string; label: string; tip: string }[];
  onChange: (v: string) => void; offDefault: boolean;
}) {
  return (
    <div className="flex rounded border border-border overflow-hidden">
      {options.map((o) => (
        <button key={o.v} type="button" title={o.tip} onClick={() => onChange(o.v)}
          className={[
            "text-xs px-2 py-[3px] whitespace-nowrap",
            o.v === value ? "bg-border text-text" : "bg-well text-faint hover:text-muted",
          ].join(" ")}>
          {o.label}{o.v === value && offDefault ? " •" : ""}
        </button>
      ))}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export function GenerateControls({
  mode,
  onModeChange,
  seedTrackIds,
  seedDisplays,
  onSubmit,
  busy,
  initialValues,
}: {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  seedTrackIds: string[];
  // Display strings ("Title - Artist") parallel to seedTrackIds. The backend
  // requires seed_tracks (the fuzzy-match fallback) alongside seed_track_ids
  // (the exact-match optimization); sending IDs alone fails generation.
  seedDisplays: string[];
  onSubmit: (body: GenerateRequestBody) => void;
  busy: boolean;
  initialValues?: Partial<GenerateRequestBody>;
}) {
  const [seed, setSeed] = useLocalStorage("pg_seed_text", "");
  const [tracks, setTracks] = useLocalStorage("pg_tracks", 30);
  const DIAL_DEFAULTS = { range: "open", flow: "normal", pace: "natural" } as const;
  // Forward-migrate dial values saved before the 2026-07-06 Flow/Pace rename so a
  // stale localStorage entry can't silently no-op or leave a detent un-highlighted.
  const DIAL_ALIASES: Record<string, string> = {
    drift: "smooth", balanced: "normal", journey: "normal", steady: "locked_in",
  };
  const [dialsRaw, setDials] = useLocalStorage<Record<string, string>>("pg_dials", { ...DIAL_DEFAULTS });
  const mig = (v: string | undefined, def: string) => (v ? DIAL_ALIASES[v] ?? v : def);
  const dials = {
    range: mig(dialsRaw.range, DIAL_DEFAULTS.range),
    flow: mig(dialsRaw.flow, DIAL_DEFAULTS.flow),
    pace: mig(dialsRaw.pace, DIAL_DEFAULTS.pace),
  };
  const dialsOffDefault =
    dials.range !== DIAL_DEFAULTS.range || dials.flow !== DIAL_DEFAULTS.flow || dials.pace !== DIAL_DEFAULTS.pace;

  // Row 3 — freshness + spacing
  const [recencyEnabled, setRecencyEnabled] = useLocalStorage("pg_recency_enabled", true);
  const [recencyDays, setRecencyDays] = useLocalStorage("pg_recency_days", 14);
  const [recencyPlays, setRecencyPlays] = useLocalStorage("pg_recency_plays", 1);
  const [excludeRecentSeeds, setExcludeRecentSeeds] = useLocalStorage("pg_exclude_recent_seeds", false);
  const [instrumental, setInstrumental] = useLocalStorage("pg_instrumental", false);
  const [artistSpacing, setArtistSpacing] = useLocalStorage("pg_artist_spacing", "normal");
  const [diversityLevel, setDiversityLevel] = useLocalStorage("pg_diversity_level", 2);

  // Artist-mode Row 1 extras
  const [artistPresence, setArtistPresence] = useLocalStorage("pg_artist_presence", "medium");
  const [artistVariety, setArtistVariety] = useLocalStorage("pg_artist_variety", "balanced");
  const [includeCollabs, setIncludeCollabs] = useLocalStorage("pg_include_collabs", false);
  const [popularSeedsMode, setPopularSeedsMode] = useLocalStorage<"off" | "on" | "fire">("pg_popular_seeds_mode", "off");
  const [popularityMode, setPopularityMode] = useLocalStorage<"off" | "on" | "oops">("pg_popularity_mode", "off");
  const [seedEpoch, setSeedEpoch] = useState(0);

  // On a narrow *container* (<@md = 448px) Rows 2–3 + artist extras collapse
  // behind this toggle. At >=448px they are always shown regardless of state.
  const [showMore, setShowMore] = useState(false);

  // Autocomplete (artist mode) — bounded-page infinite scroll
  const artistSearch = useInfiniteSearch<string>({ fetchPage: api.autocomplete, pageSize: 30 });
  // Initialize from the persisted seed so a remount (e.g. mobile tab switch
  // unmounting this component) treats the stored name as already-selected and
  // does NOT reopen the autocomplete dropdown.
  const selectedRef = useRef<string | null>(seed);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // Tag steering (artist mode): the artist's published genres as selectable chips.
  const [artistTags, setArtistTags] = useState<
    { name: string; release_count: number; confidence: number }[]
  >([]);
  const [steeringTags, setSteeringTags] = useState<string[]>([]);
  const [tagsFetched, setTagsFetched] = useState(false);
  // Bump to force a genre (re)fetch even when `seed` is unchanged — e.g. clicking the
  // autocomplete item whose full name you already typed (so setSeed is a no-op and the
  // fetch effect would never re-run), or opening the Style popover to refresh.
  const [tagEpoch, setTagEpoch] = useState(0);

  // Reset the chips whenever the artist / mode changes.
  useEffect(() => {
    setSteeringTags([]);
    setArtistTags([]);
    setTagsFetched(false);
  }, [seed, mode]);

  // Fetch the confirmed artist's published genres. Re-runs on selection AND on tagEpoch
  // bumps (re-selecting the same name, or Style opened). Always resolves `tagsFetched` so
  // a transient fetch error can't leave the chips stuck on "Loading…".
  useEffect(() => {
    if (mode !== "artist" || !seed.trim() || seed !== selectedRef.current) return;
    let cancelled = false;
    setTagsFetched(false);
    api.artistGenres(seed)
      .then((r) => { if (!cancelled) { setArtistTags(r.genres); setTagsFetched(true); } })
      .catch(() => { if (!cancelled) setTagsFetched(true); });
    return () => { cancelled = true; };
  }, [seed, mode, tagEpoch]);

  function toggleSteeringTag(name: string) {
    setSteeringTags((prev) =>
      prev.includes(name) ? prev.filter((t) => t !== name)
        : prev.length >= 3 ? prev
        : [...prev, name],
    );
  }

  // Apply initialValues (re-run prefill) once per change
  useEffect(() => {
    if (!initialValues) return;
    if (initialValues.range_dial || initialValues.flow_dial || initialValues.pace_dial) {
      setDials({
        range: initialValues.range_dial ?? dials.range,
        flow: initialValues.flow_dial ?? dials.flow,
        pace: initialValues.pace_dial ?? dials.pace,
      });
    }
    if (typeof initialValues.tracks === "number") setTracks(initialValues.tracks);
    if (typeof initialValues.artist === "string") { selectedRef.current = initialValues.artist; setSeed(initialValues.artist); setTagEpoch((e) => e + 1); }
    else if (typeof initialValues.genre === "string") setSeed(initialValues.genre);
    if (initialValues.artist_spacing) setArtistSpacing(initialValues.artist_spacing);
    if (initialValues.artist_presence) setArtistPresence(initialValues.artist_presence);
    if (initialValues.artist_variety) setArtistVariety(initialValues.artist_variety);
    if (typeof initialValues.include_collaborations === "boolean") setIncludeCollabs(initialValues.include_collaborations);
    if (typeof initialValues.recency_enabled === "boolean") setRecencyEnabled(initialValues.recency_enabled);
    if (typeof initialValues.recency_days === "number") setRecencyDays(initialValues.recency_days);
    if (typeof initialValues.recency_plays_threshold === "number") setRecencyPlays(initialValues.recency_plays_threshold);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialValues]);

  // Close autocomplete on outside click
  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        artistSearch.reset();
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Drive the artist autocomplete from the seed input. Skip one cycle after a
  // selection so picking a name doesn't immediately re-open the dropdown.
  useEffect(() => {
    if (mode !== "artist") { artistSearch.reset(); selectedRef.current = null; return; }
    if (seed === selectedRef.current) return; // just selected this name — don't reopen the dropdown
    artistSearch.setQuery(seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, mode]);

  // Shared visual style for the artist-name input and the genre autocomplete input.
  const inputClassName = "w-full bg-well border border-border rounded text-xs text-text px-2.5 py-[3px]";

  function submit(epoch: number = seedEpoch) {
    const body: GenerateRequestBody = {
      mode,
      tracks,
      artist: mode === "artist" ? seed : undefined,
      genre: mode === "genre" ? seed : undefined,
      seed_tracks: mode === "seeds" ? seedDisplays : undefined,
      seed_track_ids: mode === "seeds" ? seedTrackIds : undefined,
      range_dial: dials.range as RangeDial,
      flow_dial: dials.flow as FlowDial,
      pace_dial: dials.pace as PaceDial,
      recency_enabled: recencyEnabled,
      recency_days: recencyDays,
      recency_plays_threshold: recencyPlays,
      exclude_seed_tracks_from_recency: excludeRecentSeeds,
      instrumental: instrumental,
      artist_spacing: artistSpacing,
      diversity_gamma: DIVERSITY_GAMMAS[diversityLevel],
      artist_diversity_mode: diversityLevel === DIVERSITY_GAMMAS.length - 1 ? "one_per_artist" : "weighted",
      artist_presence: artistPresence,
      artist_variety: artistVariety,
      include_collaborations: includeCollabs,
      popular_seeds_mode: popularityMode === "oops" ? "fire" : popularSeedsMode,
      popularity_mode: popularityMode,
      seed_epoch: epoch,
      steering_tags: mode === "artist" && steeringTags.length ? steeringTags : undefined,
    };
    onSubmit(body);
  }

  return (
    <div className="@container border border-border rounded-none border-x-0 border-t-0 overflow-hidden">

      {/* ── ROW 1: mode-specific ──────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center bg-panel border-b border-hairline">

        {/* Mode selector */}
        <Cell>
          <select
            value={mode}
            onChange={(e) => { onModeChange(e.target.value as Mode); setSeed(""); artistSearch.reset(); }}
            className={SEL}
            title="Generation mode"
          >
            <option value="artist">ARTIST</option>
            <option value="seeds">SEEDS</option>
            <option value="genre">GENRE</option>
            <option value="history">HISTORY</option>
          </select>
        </Cell>

        {/* Text input: artist mode — free-text with its own name autocomplete */}
        {mode === "artist" && (
          <Cell className="flex-1 min-w-[220px]">
            <div ref={dropdownRef} className="relative w-full">
              <input
                data-testid="seed-input"
                value={seed}
                onChange={(e) => { selectedRef.current = null; setSeed(e.target.value); }}
                onKeyDown={(e) => e.key === "Enter" && submit()}
                placeholder="Artist name…"
                className={inputClassName}
              />
              {artistSearch.items.length > 0 && (
                <ul
                  ref={listRef}
                  onScroll={() => {
                    const el = listRef.current;
                    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 48) artistSearch.loadMore();
                  }}
                  className="absolute z-10 mt-1 w-full bg-panel border border-border rounded shadow-xl max-h-48 overflow-auto"
                >
                  {artistSearch.items.map((s) => (
                    <li
                      key={s}
                      onClick={() => { selectedRef.current = s; setSeed(s); artistSearch.reset(); setTagEpoch((e) => e + 1); }}
                      className="px-2.5 py-1.5 text-xs text-text hover:bg-rowsel cursor-pointer"
                    >
                      {s}
                    </li>
                  ))}
                  {(artistSearch.loading || artistSearch.hasMore) && (
                    <li className="px-2.5 py-1.5 text-2xs text-faint">
                      {artistSearch.loading ? "Loading…" : "Scroll for more"}
                    </li>
                  )}
                </ul>
              )}
            </div>
          </Cell>
        )}

        {/* Text input: genre mode — alias-aware canonical genre autocomplete
            (picking a suggestion guarantees a real canonical name; see
            GenreAutocomplete.tsx and /api/genres/search). */}
        {mode === "genre" && (
          <Cell className="flex-1 min-w-[220px]">
            <GenreAutocomplete
              data-testid="seed-input"
              value={seed}
              onChange={setSeed}
              onPick={(name) => setSeed(name)}
              placeholder="Genre…"
              className={inputClassName}
            />
          </Cell>
        )}

        {/* Seeds mode: show count badge */}
        {mode === "seeds" && seedTrackIds.length > 0 && (
          <Cell grow>
            <span className="text-2xs text-faint">
              {seedTrackIds.length} seed{seedTrackIds.length !== 1 ? "s" : ""} — manage below ↓
            </span>
          </Cell>
        )}

        {/* Tracks */}
        <Cell>
          <Lbl title="Number of tracks to generate.">tracks</Lbl>
          <input
            type="number"
            inputMode="numeric"
            min={1}
            max={200}
            value={tracks}
            onChange={(e) => setTracks(Number(e.target.value))}
            className="w-12 bg-well border border-border rounded text-xs text-text px-2 py-[3px] text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
        </Cell>

        {/* Artist-mode extras */}
        {mode === "artist" && (
          <>
            <Cell>
              <Lbl title="Target share of the playlist from the seed artist. Very Low ≈ 5%, Low ≈ 10%, Medium ≈ 12%, High ≈ 20%, Very High ≈ 33%.">
                artist presence
              </Lbl>
              <select
                value={artistPresence}
                onChange={(e) => setArtistPresence(e.target.value)}
                className={SEL}
                title="Target share of the playlist from the seed artist. Very Low ≈ 5%, Low ≈ 10%, Medium ≈ 12%, High ≈ 20%, Very High ≈ 33%."
              >
                <option value="very_low">very low</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
                <option value="very_high">very high</option>
              </select>
            </Cell>
            <Cell>
              <StylePopover
                artistVariety={artistVariety}
                onVarietyChange={setArtistVariety}
                artistTags={artistTags}
                steeringTags={steeringTags}
                onToggleTag={toggleSteeringTag}
                tagsFetched={tagsFetched}
                onOpen={() => setTagEpoch((e) => e + 1)}
              />
            </Cell>
            <Cell>
              <label
                className="flex items-center gap-1.5 cursor-pointer select-none"
                title="Include collaboration tracks (e.g. 'Miles Davis & John Coltrane', 'Bill Evans Trio') in the seed pool alongside solo tracks."
              >
                <input
                  type="checkbox"
                  checked={includeCollabs}
                  onChange={(e) => setIncludeCollabs(e.target.checked)}
                  className="accent-accent cursor-pointer"
                />
                <Lbl>include collaborations</Lbl>
              </label>
            </Cell>
            <Cell>
              <Lbl title="Seed piers from this artist's most popular tracks. Off = cluster-medoid selection; On = popularity-weighted medoids; 🔥 Pure Hits = top-N most popular tracks only. Forced 🔥 when Bangers = Oops.">
                popular seeds
              </Lbl>
              <select
                value={popularityMode === "oops" ? "fire" : popularSeedsMode}
                onChange={(e) => setPopularSeedsMode(e.target.value as "off" | "on" | "fire")}
                disabled={popularityMode === "oops"}
                className={SEL}
                title="Seed piers from this artist's most popular tracks. Off = cluster-medoid selection; On = popularity-weighted medoids; 🔥 Pure Hits = top-N most popular tracks only. Forced 🔥 when Bangers = Oops."
              >
                <option value="off">Off</option>
                <option value="on">On</option>
                <option value="fire">🔥</option>
              </select>
            </Cell>
          </>
        )}

        {/* Generate */}
        <Cell push>
          <button
            onClick={() => submit()}
            disabled={busy}
            className={`${btnPrimary} font-bold whitespace-nowrap`}
          >
            {busy ? "Generating…" : "▸ Generate"}
          </button>
        </Cell>
        {mode === "artist" && (
          <Cell>
            <button onClick={() => { setSeedEpoch((e) => e + 1); submit(seedEpoch + 1); }}
              disabled={busy}
              className="border border-accent text-accent text-xs px-3 py-1 pointer-coarse:min-h-11 rounded disabled:opacity-50 whitespace-nowrap"
              title="Re-roll: same settings, fresh seed tracks.">
              ↻ New Seeds
            </button>
          </Cell>
        )}
        <Cell>
          <button
            type="button"
            data-testid="more-controls"
            aria-expanded={showMore}
            aria-controls="advanced-controls-region"
            onClick={() => setShowMore((v) => !v)}
            className="@md:hidden border border-border text-muted text-xs px-3 py-[4px] rounded whitespace-nowrap"
          >
            {showMore ? "Less controls ▴" : "More controls ▾"}
          </button>
        </Cell>
      </div>

      {/* ── Advanced controls: Rows 2–3 (collapse below @md container width) ── */}
      <div id="advanced-controls-region" data-testid="advanced-controls" className={showMore ? "" : "@max-md:hidden"}>

      {/* ── ROW 2: dials + matching ──────────────────────────────────────── */}
      <div className="flex flex-wrap items-center bg-panel2 border-b border-hairline">
        <Cell>
          <Lbl title="How far from home the music can come from — what's considered for the playlist.">range</Lbl>
          <Segmented value={dials.range} offDefault={dials.range !== DIAL_DEFAULTS.range}
            onChange={(v) => setDials({ ...dials, range: v })}
            options={[
              { v: "home", label: "Home", tip: "Only the seed's home territory — sound and genre." },
              { v: "close", label: "Close", tip: "Close neighbors of the seed's sound and genre." },
              { v: "open", label: "Open", tip: "Balanced reach (default)." },
              { v: "wander", label: "Wander", tip: "Range further afield in sound and genre." },
            ]} />
        </Cell>
        <Cell>
          <Lbl title="How smoothly one track flows into the next.">flow</Lbl>
          <Segmented value={dials.flow} offDefault={dials.flow !== DIAL_DEFAULTS.flow}
            onChange={(v) => setDials({ ...dials, flow: v })}
            options={[
              { v: "normal", label: "Normal", tip: "The usual mix of smooth joins and momentum (default)." },
              { v: "smooth", label: "Smooth", tip: "The gentlest possible track-to-track joins." },
            ]} />
        </Cell>
        <Cell>
          <Lbl title="Tempo discipline.">pace</Lbl>
          <Segmented value={dials.pace} offDefault={dials.pace !== DIAL_DEFAULTS.pace}
            onChange={(v) => setDials({ ...dials, pace: v })}
            options={[
              { v: "locked_in", label: "Locked-In", tip: "Hold the seed's tempo and groove tightly." },
              { v: "natural", label: "Natural", tip: "Gentle tempo anchoring (default)." },
              { v: "free", label: "Free", tip: "No tempo constraint." },
            ]} />
        </Cell>
        {dialsOffDefault && (
          <Cell>
            <button type="button" title="Reset dials to defaults"
              onClick={() => setDials({ ...DIAL_DEFAULTS })}
              className="text-xs text-faint hover:text-muted">↺</button>
          </Cell>
        )}
        <Cell>
          <Lbl title="Oops, All Bangers — bias the bridge tracks toward each artist's most popular (Last.fm) songs. Off = today; On = lean popular; Oops, All Bangers = library greatest-hits.">bangers</Lbl>
          <select value={popularityMode} onChange={(e) => setPopularityMode(e.target.value as "off" | "on" | "oops")} className={SEL}
            title="Bias the bridge tracks toward each artist's most popular (Last.fm) songs.">
            <option value="off">Off</option>
            <option value="on">On</option>
            <option value="oops">Oops, All Bangers</option>
          </select>
        </Cell>
      </div>

      {/* ── ROW 3: freshness + spacing ──────────────────────────────────────── */}
      <div className="flex flex-wrap items-center bg-bg">
        <Cell>
          <label
            className="flex items-center gap-1.5 cursor-pointer select-none"
            title="Exclude tracks played recently. Uncheck to ignore playback history entirely."
          >
            <input
              type="checkbox"
              checked={recencyEnabled}
              onChange={(e) => setRecencyEnabled(e.target.checked)}
              className="accent-accent cursor-pointer"
            />
            <Lbl>freshness</Lbl>
          </label>
        </Cell>
        <Cell>
          <Lbl title="How far back to look for recently-played tracks.">within</Lbl>
          <select value={recencyDays} onChange={(e) => setRecencyDays(Number(e.target.value))} disabled={!recencyEnabled} className={SEL}
            title="How far back to look for recently-played tracks.">
            {[7, 14, 30, 60, 90].map((d) => <option key={d} value={d}>{d}d</option>)}
          </select>
        </Cell>
        <Cell>
          <Lbl title="Minimum play count to consider a track recently played.">played</Lbl>
          <select value={recencyPlays} onChange={(e) => setRecencyPlays(Number(e.target.value))} disabled={!recencyEnabled} className={SEL}
            title="Minimum play count to consider a track recently played.">
            {[1, 2, 3].map((n) => <option key={n} value={n}>{n}+</option>)}
          </select>
          <Lbl>times</Lbl>
        </Cell>
        <Cell>
          <label
            className="flex items-center gap-1.5 cursor-pointer select-none"
            title="Also apply the recency filter to seed candidates, not just bridge tracks."
          >
            <input
              type="checkbox"
              checked={excludeRecentSeeds}
              onChange={(e) => setExcludeRecentSeeds(e.target.checked)}
              disabled={!recencyEnabled}
              className="accent-accent cursor-pointer disabled:opacity-30"
            />
            <Lbl>skip recent seeds</Lbl>
          </label>
        </Cell>
        <Cell>
          <label
            className="flex items-center gap-1.5 cursor-pointer select-none"
            title="Demote vocal-classified tracks (spoken-word / poetry) from bridges. Soft, never a hard filter. Note: heavily-processed vocals (vocoder/talkbox) may read as instrumental and slip through."
          >
            <input
              type="checkbox"
              checked={instrumental}
              onChange={(e) => setInstrumental(e.target.checked)}
              className="accent-accent cursor-pointer"
            />
            <Lbl>instrumental</Lbl>
          </label>
        </Cell>
        <Cell>
          <Lbl title="Minimum tracks between the same artist appearing again. Loose = 3, Normal = 6, Strong = 9, Very Strong = 12.">
            artist gap
          </Lbl>
          <select value={artistSpacing} onChange={(e) => setArtistSpacing(e.target.value)} className={SEL}
            title="Minimum tracks between the same artist appearing again. Loose = 3, Normal = 6, Strong = 9, Very Strong = 12.">
            <option value="loose">loose</option>
            <option value="normal">normal</option>
            <option value="strong">strong</option>
            <option value="very_strong">very strong</option>
          </select>
        </Cell>
        <Cell>
          <Lbl title="Scoring bonus for picking tracks from artists not yet in the playlist. Higher = more variety across artists. 'One each' hard-caps each non-seed artist to one track.">
            artist diversity
          </Lbl>
          <select value={diversityLevel} onChange={(e) => setDiversityLevel(Number(e.target.value))} className={SEL}
            title="Scoring bonus for picking tracks from artists not yet in the playlist. Higher = more variety across artists. 'One each' hard-caps each non-seed artist to one track.">
            {DIVERSITY_LABELS.map((label, i) => (
              <option key={i} value={i}>{label}</option>
            ))}
          </select>
        </Cell>
      </div>
      </div>

    </div>
  );
}
