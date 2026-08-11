import { useCallback, useEffect, useRef, useState } from "react";
import { friendlyError } from "../lib/errors";
import { api } from "../lib/api";
import type { BlacklistEntry, BlacklistFetchResponse, LiveAlbumEntry } from "../lib/types";

const DOT: Record<string, string> = {
  artist: "var(--color-danger)",
  album: "var(--color-warn)",
  track: "var(--color-chipText)",
};

// Live albums are a *semi*-blacklist (studio duplicates get demoted, not
// dropped) — its own accent keeps it visually distinct from the DOT map above.
const LIVE_DOT = "var(--color-accent)";

export function BlacklistPanel() {
  const [data, setData] = useState<BlacklistFetchResponse | null>(null);
  const [live, setLive] = useState<LiveAlbumEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const timer = useRef<number | undefined>(undefined);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const [blacklist, liveAlbums] = await Promise.all([api.getBlacklist(), api.liveAlbums()]);
      setData(blacklist);
      setLive(liveAlbums.entries);
      setError(null);
    } catch (e) {
      setError(friendlyError(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (q.length < 2) { setSuggestions([]); return; }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      const page = await api.autocomplete(q).catch(() => ({ items: [] as string[], has_more: false }));
      setSuggestions(page.items);
    }, 180);
  }, [q]);

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) setSuggestions([]);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  async function addArtist() {
    const name = q.trim();
    if (!name) return;
    setBusy(true);
    try {
      await api.blacklistArtist(name);
      setQ("");
      setSuggestions([]);
      await refresh();
    } catch (e) {
      setError(friendlyError(e));
      setBusy(false);
    }
  }

  async function remove(entry: BlacklistEntry) {
    setBusy(true);
    try {
      if (entry.scope === "artist") {
        await api.blacklist({ scope: "artist", value: entry.artist ?? "", enabled: false });
      } else if (entry.scope === "album") {
        await api.blacklist({ scope: "album", value: entry.album ?? "", artist: entry.artist ?? "", enabled: false });
      } else {
        await api.blacklist({ track_ids: [entry.track_id ?? ""], enabled: false });
      }
      await refresh();
    } catch (e) {
      setError(friendlyError(e));
      setBusy(false);
    }
  }

  async function unmarkLive(entry: LiveAlbumEntry) {
    setBusy(true);
    try {
      await api.setLiveAlbum({ ...entry, enabled: false });
      await refresh();
    } catch (e) {
      setError(friendlyError(e));
      setBusy(false);
    }
  }

  const section = (title: string, entries: BlacklistEntry[]) => (
    <div>
      <div className="flex justify-between text-2xs uppercase tracking-[.08em] text-faint mt-2 mb-1">
        <span>{title}</span><span className="text-faint">{entries.length}</span>
      </div>
      {entries.map((e, i) => (
        <div key={`${e.scope}-${i}`} className="flex items-center gap-1.5 py-0.5 border-b border-hairline">
          <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: DOT[e.scope] }} />
          <span className="text-2xs text-muted flex-1 truncate">{e.display_name}</span>
          <button onClick={() => remove(e)} className="text-faint hover:text-danger text-sm leading-none">×</button>
        </div>
      ))}
    </div>
  );

  // Live albums are a *semi*-blacklist — studio duplicates get demoted, not
  // dropped — so this section gets its own heading/accent, separated from the
  // hard-blacklist sections above by a divider, instead of blending into `section()`.
  const liveSection = (title: string, entries: LiveAlbumEntry[]) => (
    <div className="mt-3 pt-2 border-t border-hairline">
      <div className="flex justify-between text-2xs uppercase tracking-[.08em] text-accent mt-2 mb-1">
        <span>{title}</span><span className="text-faint">{entries.length}</span>
      </div>
      {entries.length === 0 ? (
        <div className="text-faint text-2xs">No albums marked live yet. Use the track table context menu.</div>
      ) : (
        entries.map((e, i) => (
          <div key={`live-${i}`} className="flex items-center gap-1.5 py-0.5 border-b border-hairline">
            <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: LIVE_DOT }} />
            <span className="text-2xs text-muted flex-1 truncate">{e.artist} — {e.album}</span>
            <button onClick={() => unmarkLive(e)} className="text-faint hover:text-accent text-sm leading-none">×</button>
          </div>
        ))
      )}
    </div>
  );

  const empty = data && data.total === 0;

  return (
    <div className="h-full overflow-y-auto p-3 text-xs" data-testid="blacklist-panel">
      <div ref={dropdownRef} className="relative flex gap-1 mb-2">
        <input
          data-testid="blacklist-search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addArtist()}
          placeholder="Artist to blacklist…"
          className="flex-1 bg-well border border-border rounded text-2xs text-text px-2 py-1"
        />
        <button data-testid="blacklist-add" onClick={addArtist} disabled={busy}
          className="text-2xs bg-accent/10 text-accent rounded px-2 disabled:opacity-50">+ Add</button>
        {suggestions.length > 0 && (
          <ul className="absolute z-20 top-8 left-0 right-0 bg-panel border border-border rounded shadow-xl max-h-40 overflow-auto">
            {suggestions.map((s) => (
              <li key={s} onClick={() => { setQ(s); setSuggestions([]); }}
                className="px-2 py-1 text-2xs text-text hover:bg-rowsel cursor-pointer">{s}</li>
            ))}
          </ul>
        )}
      </div>

      {error && <div className="text-danger text-2xs mb-2">{error}</div>}
      {busy && <div className="text-faint text-2xs mb-1">Refreshing…</div>}

      {empty ? (
        <div className="text-faint text-2xs">Nothing blacklisted yet. Use the track table context menu or search above.</div>
      ) : data ? (
        <>
          {section("Artists", data.artists)}
          {section("Albums", data.albums)}
          {section("Tracks", data.tracks)}
        </>
      ) : null}

      {data && liveSection("Live albums", live)}
    </div>
  );
}