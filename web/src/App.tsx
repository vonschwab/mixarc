import { useCallback, useEffect, useRef, useState } from "react";
import { friendlyError } from "./lib/errors";
import { Shell } from "./components/Shell";
import { AdvancedPanel } from "./components/AdvancedPanel";
import { GenerateControls } from "./components/GenerateControls";
import { SeedTrackSection } from "./components/SeedTrackSection";
import { TrackTable } from "./components/TrackTable";
import { QualityStats } from "./components/QualityStats";
import { LogPanel } from "./components/LogPanel";
import { JobsPanel } from "./components/JobsPanel";
import { ToolsPanel } from "./components/ToolsPanel";
import { PlayerProvider } from "./contexts/PlayerContext";
import { MiniPlayer } from "./components/MiniPlayer";
import { api } from "./lib/api";
import { useWorkerEvents } from "./lib/ws";
import { useJobReconcile } from "./lib/useJobReconcile";
import { useLocalStorage } from "./lib/useLocalStorage";
import { useMediaQuery } from "./lib/useMediaQuery";
import type { CandidateOut, GenerateRequestBody, JobOut, Mode, PlaylistOut, SeedTrack, SetupStatus, TrackOut, WsEvent } from "./lib/types";
import { TrackContextMenu, type MenuTarget } from "./components/TrackContextMenu";
import { ReplaceDialog } from "./components/ReplaceDialog";
import { EditGenresDialog } from "./components/EditGenresDialog";
import { ExportPlexDialog } from "./components/ExportPlexDialog";
import { ExportM3U8Dialog } from "./components/ExportM3U8Dialog";
import { DegradedNotice } from "./components/DegradedNotice";
import { RelaxationNotice } from "./components/RelaxationNotice";
import { GeneratingIndicator } from "./components/GeneratingIndicator";
import { defaultPlaylistName } from "./lib/playlistName";
import SetupPage from "./components/SetupPage";

// The last generated playlist, persisted so it survives a page reload.
const PLAYLIST_KEY = "pg_current_playlist";
function loadStoredPlaylist(): PlaylistOut | null {
  try {
    const raw = localStorage.getItem(PLAYLIST_KEY);
    return raw ? (JSON.parse(raw) as PlaylistOut) : null;
  } catch {
    return null;
  }
}

export default function App() {
  // Setup gate (Task 5): fetched once on mount, alongside the rest of the
  // app's fetch-on-mount state. `needs_setup` renders SetupPage instead of
  // the generator tree; `needs_analyze`/`ready` fall through unchanged.
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [setupLoading, setSetupLoading] = useState(true);
  useEffect(() => {
    api.getSetupStatus()
      .then((s: SetupStatus) => setSetupStatus(s))
      // A rejected fetch or non-ok response leaves setupStatus null, which the
      // render below treats as "not needs_setup" — the app renders normally
      // rather than blocking behind a gate the server can't answer. Never
      // worse than pre-gate behavior.
      .catch(() => setSetupStatus(null))
      .finally(() => setSetupLoading(false));
  }, []);
  // I2: the wizard's Analyze step calls this ONLY on a successful terminal
  // analyze job (never mid-wizard, never on Review's config write) — a
  // re-fetch right after config write would flip state to needs_analyze,
  // which still isn't "needs_setup" but would be wrong regardless; the only
  // safe trigger is "the wizard is fully done." This re-fetch is what lets
  // needs_setup -> ready happen without the user manually reloading MixArc.
  const refetchSetupStatus = useCallback(() => {
    api.getSetupStatus().then((s: SetupStatus) => setSetupStatus(s)).catch(() => {});
  }, []);

  const [mode, setMode] = useLocalStorage<Mode>("pg_mode", "artist");
  const [seedTracks, setSeedTracks] = useLocalStorage<SeedTrack[]>("pg_seed_tracks", []);
  const addSeed = useCallback((t: SeedTrack) => {
    setSeedTracks(seedTracks.some((s) => s.track_id === t.track_id) ? seedTracks : [...seedTracks, t]);
  }, [seedTracks, setSeedTracks]);
  const removeSeed = useCallback((id: string) => {
    setSeedTracks(seedTracks.filter((t) => t.track_id !== id));
  }, [seedTracks, setSeedTracks]);
  const clearSeeds = useCallback(() => setSeedTracks([]), [setSeedTracks]);

  const [tab, setTab] = useState<"generate" | "tools">("generate");
  const [busy, setBusy] = useState(false);
  const [rerunValues, setRerunValues] = useState<GenerateRequestBody | null>(null);
  const [playlist, setPlaylist] = useState<PlaylistOut | null>(loadStoredPlaylist);
  // Persist the current playlist so it survives a reload; clear on reset.
  useEffect(() => {
    try {
      if (playlist) localStorage.setItem(PLAYLIST_KEY, JSON.stringify(playlist));
      else localStorage.removeItem(PLAYLIST_KEY);
    } catch {}
  }, [playlist]);
  // On the phone, the payoff owns the viewport (discipline S5): when a
  // playlist arrives, the generate controls collapse into a summary bar.
  // Same breakpoint as the Shell's desktop/mobile swap.
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const desktopRef = useRef(isDesktop);
  desktopRef.current = isDesktop;
  // Start collapsed when a playlist was restored, so a reloaded phone shows the
  // playlist (not the controls) straight away; desktop ignores collapse.
  const [controlsOpen, setControlsOpen] = useState(() => playlist == null);
  const showPlaylist = useCallback((pl: PlaylistOut) => {
    setPlaylist(pl);
    if (!desktopRef.current) setControlsOpen(false);
  }, []);
  const controlsCollapsed = !isDesktop && !controlsOpen && playlist != null;
  const [logs, setLogs] = useState<string[]>([]);
  const [jobs, setJobs] = useState<JobOut[]>(() => {
    try {
      const raw = localStorage.getItem("pg_jobs_history");
      if (!raw) return [];
      return (JSON.parse(raw) as Array<Omit<JobOut, "playlist">>).map((j) => ({ ...j, playlist: null }));
    } catch { return []; }
  });
  const [error, setError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuTarget, setMenuTarget] = useState<MenuTarget | null>(null);
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [blacklisted, setBlacklisted] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string>("");
  // The generation awaiting its `done` event. Distinct from `jobId`, which the
  // Replace dialog keeps pointing at the LAST completed job indefinitely.
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);
  const [replaceOpen, setReplaceOpen] = useState(false);
  const [replacePos, setReplacePos] = useState(0);
  const [editGenresOpen, setEditGenresOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<{ artist: string; album: string; genres: string[] }>({ artist: "", album: "", genres: [] });
  const [plexOpen, setPlexOpen] = useState(false);
  const [m3u8Open, setM3u8Open] = useState(false);

  // Persist slim job history (no playlist payload) across server restarts.
  useEffect(() => {
    try {
      const slim = jobs.map(({ playlist: _, ...j }) => j);
      localStorage.setItem("pg_jobs_history", JSON.stringify(slim.slice(0, 30)));
    } catch {}
  }, [jobs]);

  // Merge server jobs into local history; server wins for the same job_id.
  const refreshJobs = useCallback(() => {
    api.jobs().then((serverJobs) => {
      setJobs((prev) => {
        const byId = new Map(prev.map((j) => [j.job_id, j]));
        for (const sj of serverJobs) byId.set(sj.job_id, sj);
        return [...byId.values()]
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
          .slice(0, 30);
      });
    }).catch(() => {});
  }, []);

  const handleCancel = useCallback(async (j: JobOut) => {
    try { await api.cancelJob(j.job_id); refreshJobs(); }
    catch (e) { setError(friendlyError(e)); }
  }, [refreshJobs]);

  // Clear finished (and dead-"running") jobs. The server keeps only a genuinely
  // in-flight job; we then adopt its post-clear list as the truth so client-only
  // stale entries (a "running" job localStorage kept across a worker death, which
  // the server no longer tracks) are dropped too — a plain status filter would
  // leave those zombies behind.
  const handleClearJobs = useCallback(async () => {
    try { await api.clearJobs(); }
    catch (e) { setError(friendlyError(e)); return; }
    try { setJobs(await api.jobs()); }
    catch { setJobs((prev) => prev.filter((j) => j.status === "running" || j.status === "pending")); }
  }, []);

  const handleRerun = useCallback((params: GenerateRequestBody) => {
    setMode((params.mode as Mode) ?? "artist");
    setRerunValues({ ...params });
  }, [setMode]);

  const openMenu = useCallback((track: TrackOut, index: number, x: number, y: number) => {
    const last = (playlist?.tracks.length ?? 0) - 1;
    setMenuTarget({ track, index, isPier: index === 0 || index === last });
    setMenuPos({ x, y });
    setMenuOpen(true);
  }, [playlist]);

  const markBlacklisted = useCallback((ids: string[]) => {
    setBlacklisted((prev) => { const n = new Set(prev); ids.forEach((i) => n.add(i)); return n; });
  }, []);

  const handleBlacklistTrack = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ track_ids: [t.track.rating_key ?? ""], enabled: true });
      markBlacklisted([t.track.rating_key ?? ""]);
    } catch (e) { setError(friendlyError(e)); }
  }, [markBlacklisted]);

  const handleBlacklistAlbum = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ scope: "album", value: t.track.album, artist: t.track.artist, enabled: true });
    } catch (e) { setError(friendlyError(e)); }
  }, []);

  const handleBlacklistArtist = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.blacklist({ scope: "artist", value: t.track.artist, enabled: true });
    } catch (e) { setError(friendlyError(e)); }
  }, []);

  const handleMarkAlbumLive = useCallback(async (t: MenuTarget) => {
    setMenuOpen(false);
    try {
      await api.setLiveAlbum({ artist: t.track.artist, album: t.track.album, enabled: true });
    } catch (e) { setError(friendlyError(e)); }
  }, []);

  const handleReplace = useCallback((t: MenuTarget) => {
    setMenuOpen(false);
    setReplacePos(t.index);
    setReplaceOpen(true);
  }, []);

  const applyReplacement = useCallback((position: number, cand: CandidateOut) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = [...pl.tracks];
      const old = tracks[position];
      tracks[position] = {
        ...old,
        rating_key: cand.track_id,
        // file_path is the identity the Plex/M3U exporters resolve on — without it
        // the export keeps the OLD track's path and the replacement is lost.
        file_path: cand.file_path,
        duration_ms: cand.duration_ms,
        title: cand.title,
        artist: cand.artist,
        album: cand.album,
        genres: cand.genres,
        sonic_similarity: cand.fit_score,
      };
      return { ...pl, tracks };
    });
  }, []);

  const handleEditGenres = useCallback((t: MenuTarget) => {
    setMenuOpen(false);
    setEditTarget({ artist: t.track.artist, album: t.track.album, genres: t.track.genres });
    setEditGenresOpen(true);
  }, []);

  const applyGenreEdit = useCallback((artist: string, album: string, genres: string[]) => {
    setPlaylist((pl) => {
      if (!pl) return pl;
      const tracks = pl.tracks.map((tr) =>
        (tr.album === album && tr.artist === artist) ? { ...tr, genres } : tr
      );
      return { ...pl, tracks };
    });
  }, []);

  useWorkerEvents(useCallback((e: WsEvent) => {
    if (e.type === "log") setLogs((l) => [...l, `${e["level"] ?? "INFO"}: ${e["msg"] ?? ""}`].slice(-500));
    if (e.type === "error") setError(friendlyError(e["message"]));
    if (e.type === "done") {
      setBusy(false);
      const jid = e.job_id;
      if (jid) {
        setPendingJobId((p) => (p === jid ? null : p));
        api.job(jid).then((j) => { if (j.playlist) showPlaylist(j.playlist); }).catch(() => {});
      }
      refreshJobs();
    }
  }, [refreshJobs]));

  // Backstop for a dead/backgrounded WebSocket (iOS Safari kills the socket
  // when the tab leaves the foreground): while a generation is pending, poll
  // the job registry and finish from its authoritative state. Idempotent with
  // the WS `done` path — whichever fires first clears pendingJobId.
  const finishGenerate = useCallback((jid: string) => {
    setPendingJobId(null);
    setBusy(false);
    api.job(jid).then((j) => {
      if (j.playlist) showPlaylist(j.playlist);
      else if (j.status === "failed") setError(j.error ?? "generation failed");
    }).catch(() => {});
    refreshJobs();
  }, [refreshJobs]);
  useJobReconcile(pendingJobId, finishGenerate);

  async function submit(body: GenerateRequestBody) {
    setError(null); setBusy(true); setLogs([]); setPlaylist(null);
    try {
      const { job_id } = await api.generate(body);
      setJobs((prev) => {
        const pending: JobOut = {
          job_id,
          status: "pending",
          stage: "queued",
          created_at: Date.now() / 1000,
          request_params: body as unknown as Record<string, unknown>,
          playlist: null,
        };
        const byId = new Map(prev.map((j) => [j.job_id, j]));
        byId.set(job_id, pending);
        return [...byId.values()]
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
          .slice(0, 30);
      });
      setJobId(job_id);
      setPendingJobId(job_id);
      refreshJobs();
    } catch (err) {
      setError(friendlyError(err)); setBusy(false);
    }
  }

  if (setupLoading) {
    return (
      <div className="h-dvh w-full bg-bg">
        <GeneratingIndicator label="Checking setup…" />
      </div>
    );
  }
  if (setupStatus?.state === "needs_setup") {
    return <SetupPage status={setupStatus} onSetupComplete={refetchSetupStatus} />;
  }

  return (
    <PlayerProvider>
      <Shell
        topBar={
          <>
            <div className="font-bold text-sm"><span className="text-accent">◆</span> MixArc</div>
            <div className="flex items-center gap-1 ml-4">
              {(["generate", "tools"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={[
                    "px-3 py-1 pointer-coarse:min-h-11 text-xs rounded transition-colors capitalize",
                    tab === t
                      ? "bg-border text-text"
                      : "text-faint hover:text-muted",
                  ].join(" ")}
                >
                  {t}
                </button>
              ))}
            </div>
            {error && <div className="text-danger text-xs ml-auto">{error}</div>}
          </>
        }
        jobs={<JobsPanel jobs={jobs} onSelect={(j) => (j.playlist ? showPlaylist(j.playlist) : setPlaylist(null))} onCancel={handleCancel} onRerun={handleRerun} onClear={handleClearJobs} />}
        center={
          tab === "tools" ? (
            <ToolsPanel externalBusy={busy} refreshJobs={refreshJobs} />
          ) : (
            <div className="h-full flex flex-col overflow-hidden">
              {controlsCollapsed && (
                <div
                  data-testid="controls-summary"
                  className="flex items-center justify-between gap-2 px-4 py-2 bg-panel border-b border-border"
                >
                  <div className="text-xs text-muted truncate">
                    <span className="text-text font-medium">{playlist?.name ?? "Playlist"}</span>
                    {playlist ? ` — ${playlist.track_count} tracks` : ""}
                  </div>
                  <button
                    data-testid="controls-summary-open"
                    onClick={() => setControlsOpen(true)}
                    className="border border-accent text-accent text-xs px-3 py-1 pointer-coarse:min-h-11 rounded whitespace-nowrap"
                  >
                    New playlist
                  </button>
                </div>
              )}
              {/* Hidden, not unmounted: the form state must survive the collapse. */}
              <div className={controlsCollapsed ? "hidden" : "contents"}>
                <GenerateControls
                  mode={mode}
                  onModeChange={setMode}
                  seedTrackIds={seedTracks.map((t) => t.track_id)}
                  seedDisplays={seedTracks.map((t) => `${t.title} - ${t.artist}`)}
                  onSubmit={submit}
                  busy={busy}
                  initialValues={rerunValues ?? undefined}
                />
                {mode === "seeds" && (
                  <SeedTrackSection
                    tracks={seedTracks}
                    onAdd={addSeed}
                    onRemove={removeSeed}
                    onClear={clearSeeds}
                  />
                )}
              </div>
              {busy ? (
                <div className="flex-1 overflow-auto">
                  <GeneratingIndicator status={logs[logs.length - 1]} />
                </div>
              ) : (
                <>
                  <QualityStats
                    metrics={playlist?.metrics}
                    receipt={playlist?.receipt}
                    count={playlist?.track_count ?? 0}
                    tracks={playlist?.tracks ?? []}
                    onExportM3U8={() => setM3u8Open(true)}
                    onExportPlex={() => setPlexOpen(true)}
                  />
                  <DegradedNotice warnings={playlist?.warnings ?? []} />
                  <RelaxationNotice relaxations={playlist?.relaxations ?? []} />
                  <div className="flex-1 overflow-auto">
                    <TrackTable
                      tracks={playlist?.tracks ?? []}
                      blacklisted={blacklisted}
                      onContextAction={openMenu}
                    />
                  </div>
                </>
              )}
            </div>
          )
        }
        right={<AdvancedPanel playlist={playlist} />}
        logs={<LogPanel lines={logs} />}
      />
      <MiniPlayer />
      <TrackContextMenu
        open={menuOpen}
        onOpenChange={setMenuOpen}
        target={menuTarget}
        pos={menuPos}
        onReplace={handleReplace}
        onBlacklistTrack={handleBlacklistTrack}
        onBlacklistAlbum={handleBlacklistAlbum}
        onBlacklistArtist={handleBlacklistArtist}
        onMarkAlbumLive={handleMarkAlbumLive}
        onEditGenres={handleEditGenres}
      />
      <ReplaceDialog
        open={replaceOpen}
        onOpenChange={setReplaceOpen}
        jobId={jobId}
        position={replacePos}
        prevTitle={playlist?.tracks[replacePos - 1]?.title ?? ""}
        nextTitle={playlist?.tracks[replacePos + 1]?.title ?? ""}
        onConfirm={applyReplacement}
      />
      <EditGenresDialog
        open={editGenresOpen}
        onOpenChange={setEditGenresOpen}
        artist={editTarget.artist}
        album={editTarget.album}
        initialGenres={editTarget.genres}
        onSaved={applyGenreEdit}
      />
      <ExportPlexDialog
        open={plexOpen}
        onOpenChange={setPlexOpen}
        tracks={playlist?.tracks ?? []}
        defaultName={defaultPlaylistName(playlist)}
      />
      <ExportM3U8Dialog
        open={m3u8Open}
        onOpenChange={setM3u8Open}
        tracks={playlist?.tracks ?? []}
        defaultName={defaultPlaylistName(playlist)}
      />
    </PlayerProvider>
  );
}