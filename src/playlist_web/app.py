# src/playlist_web/app.py
from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config_loader import resolve_database_path
from src.mixarc.paths import resolve_home
from src.playlist_gui.policy import derive_runtime_config, resolve_dial_axes
from src.playlist_gui.ui_state import UIStateModel

from .audio import stream_audio
from .jobs import JobRegistry
from .plex_export import PlexNotConfigured, run_plex_export
from .schemas import (
    AnalyzeToolRequest,
    ArtistLinksSaveRequest,
    BlacklistArtistRequest,
    BlacklistFetchResponse,
    BlacklistRequest,
    EditGenresRequest,
    EnrichToolRequest,
    EscalationDecisionRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsRequest,
    ReplaceSuggestionsResponse,
    SetupConfigRequest,
    SetupTestRequest,
    TaxonomyAdjudicateRequest,
    TaxonomyDecisionRequest,
    TaxonomyValidateRequest,
    TrackGenresRequest,
)
from .setup_state import SetupState, derive_setup_state
from .worker_bridge import (
    BridgeBusy,
    WorkerBridge,
    WorkerCommandError,
    WorkerTimeout,
    WorkerUnavailable,
)
from .ws import WsHub

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_CMD = [sys.executable, "-m", "src.playlist_gui.worker"]
DEFAULT_CONFIG = str(ROOT / "config.yaml")

logger = logging.getLogger(__name__)


def resolve_static_dir(root: Optional[Path] = None, packaged: Optional[Path] = None) -> Path:
    """Locate the built front-end bundle.

    Prefers the repo's `web/dist` (dev checkout); falls back to the
    packaged `static_dist/` copied alongside this module by
    `scripts/build_wheel.py` (wheel install, where there is no `web/`
    source tree at all).
    """
    if root is None:
        root = ROOT
    repo_dist = root / "web" / "dist"
    if repo_dist.exists():
        return repo_dist
    if packaged is None:
        packaged = Path(__file__).resolve().parent / "static_dist"
    return packaged

# Import-time fallbacks only — create_app() rebinds these two module globals
# from the config's library.database_path via resolve_database_path()
# (2026-07-16 fix). A satellite clone's data/metadata.db is a 0-byte stub;
# without the rebind every endpoint below silently read the stub instead of
# the canonical DB named in config.yaml. Endpoints close over these globals,
# so rebinding them in create_app() is the smallest correct diff. Routed
# through resolve_database_path(None) (repo-root default), not a ROOT-joined
# path literal — the latter is exactly what test_no_relative_db_literal.py
# bans in this file.
DB_PATH = Path(resolve_database_path(None))
SIDECAR_DB_PATH = DB_PATH.parent / "ai_genre_enrichment.db"


def _resolve_seed_artist_keys(track_ids: list[str]) -> list[str]:
    """Resolve seed track IDs to artist keys for policy evaluation (read-only).

    Policy gates DJ bridging on >= 2 unique seed artists; without this lookup
    it conservatively force-disables DJ bridging on every web request.
    """
    if not track_ids:
        return []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" for _ in track_ids)
            rows = conn.execute(
                "SELECT track_id, COALESCE(NULLIF(artist_key, ''), artist) "
                f"FROM tracks WHERE track_id IN ({placeholders})",
                [str(t) for t in track_ids],
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        logger.warning(
            "Seed artist resolution failed; policy will keep DJ bridging disabled",
            exc_info=True,
        )
        return []
    by_id = {str(r[0]): str(r[1] or "") for r in rows}
    return [by_id.get(str(t), "") for t in track_ids]


def _genres_for_tracks(
    conn: sqlite3.Connection, track_ids: list[str], per_track: int = 5
) -> dict[str, list[str]]:
    """Genre lookup for a page of tracks. Published authority first
    (`release_effective_genres` via authority.py, graph-canonically ordered),
    then `track_effective_genres` (priority), then a `track_genres` fallback for
    tracks still without any. Per-track authority lookup is fine for a page;
    the effective/raw fallbacks stay batched (audit [P#1])."""
    if not track_ids:
        return {}
    from src.genre.authority import display_genre_names_for_track
    from src.genre.granularity import order_genres_for_display

    out: dict[str, list[str]] = {}
    for tid in track_ids:
        names = order_genres_for_display(display_genre_names_for_track(conn, str(tid)))
        out[tid] = names[:per_track]
    missing = [tid for tid, gl in out.items() if not gl]
    if missing:
        ph = ",".join("?" for _ in missing)
        for tid, genre in conn.execute(
            f"SELECT track_id, genre FROM track_effective_genres "
            f"WHERE track_id IN ({ph}) ORDER BY priority",
            tuple(missing),
        ):
            if len(out[tid]) < per_track:
                out[tid].append(genre)
    still_missing = [tid for tid in missing if not out[tid]]
    if still_missing:
        ph2 = ",".join("?" for _ in still_missing)
        for tid, genre in conn.execute(
            f"SELECT track_id, genre FROM track_genres "
            f"WHERE track_id IN ({ph2}) ORDER BY weight DESC",
            tuple(still_missing),
        ):
            if len(out[tid]) < per_track:
                out[tid].append(genre)
    return out


def _resolve_db_paths(config_path: str, anchor: Optional[Path] = None) -> tuple[Path, Path]:
    """DB_PATH / SIDECAR_DB_PATH for this app instance, from ``config_path``.

    Reads ``library.database_path`` the same way the CLI + worker do
    (``resolve_database_path`` — the 2026-07-07 multi-site DB-path fix), so a
    satellite clone's absolute config points endpoints at the real canonical
    DB instead of the clone's 0-byte stub. A bare ``yaml.safe_load`` (not the
    full ``Config`` class) keeps this cheap and side-effect-free — ``Config``
    additionally validates unrelated required fields and publishes artifact
    settings process-wide, neither of which this path resolution needs.

    ``anchor`` resolves a relative ``database_path`` against a MixArc home
    (wheel-install data dir) instead of the repo root — passed through from
    ``create_app``'s resolved ``home.anchor_dir`` (MixArc SP-1). Omitted, it
    falls back to the repo root exactly as before.

    There is no config key for the enrichment sidecar DB (checked
    config.example.yaml and the sidecar readers — none resolve it from
    config); it is derived as a sibling of the resolved metadata.db.
    """
    cfg: dict = {}
    try:
        cfg_file = Path(config_path)
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Failed to load config %s for DB path resolution; falling back "
            "to repo-root defaults: %s", config_path, exc,
        )
    db_path = Path(resolve_database_path(cfg, anchor=anchor))
    sidecar_path = db_path.parent / "ai_genre_enrichment.db"
    return db_path, sidecar_path


def create_app(
    worker_cmd: Optional[list[str]] = None,
    config_path: str | None = None,
    seed_artist_resolver: Optional[Callable[[list[str]], list[str]]] = None,
) -> FastAPI:
    # Config-less boot (MixArc SP-1): resolve_home() finds the right
    # config.yaml (cli arg > $MIXARC_HOME > repo checkout > platformdirs) and
    # never raises when nothing exists yet — a missing config becomes
    # SetupState.NEEDS_SETUP, not a crash. `home.source == "repo"` for a
    # normal repo checkout with a real config.yaml keeps this identical to
    # the old DEFAULT_CONFIG-default behavior.
    home = resolve_home(cli_config=config_path)
    config_path = str(home.config_path)

    global DB_PATH, SIDECAR_DB_PATH
    DB_PATH, SIDECAR_DB_PATH = _resolve_db_paths(config_path, anchor=home.anchor_dir)

    registry = JobRegistry()
    hub = WsHub()

    async def on_event(event: dict) -> None:
        registry.apply_event(event)
        await hub.broadcast(event)

    bridge = WorkerBridge(worker_cmd or DEFAULT_WORKER_CMD, on_event=on_event)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Worker guard (MixArc SP-1): spawning the worker subprocess against
        # a config-less / not-yet-analyzed home is pointless (every command
        # would just fail once the app is used) — skip it and let
        # /api/setup/status drive the frontend's setup flow instead. Warn,
        # never raise: startup must still succeed so the setup page can load.
        status = derive_setup_state(home)
        if status.state != SetupState.NEEDS_SETUP:
            await bridge.start()
        else:
            logger.warning(
                "setup incomplete (%s) — worker not started (will start after setup)",
                status.detail,
            )
        yield
        await bridge.stop()

    app = FastAPI(title="MixArc", lifespan=lifespan)
    app.state.bridge = bridge
    app.state.registry = registry

    # Worker stall/death must degrade to a clean 5xx with a message, never a bare
    # 500 traceback. Covers every bridge.command/submit caller in one place
    # (WorkerTimeout is a WorkerUnavailable subclass; register the specific one
    # first so it maps to 504 rather than 503).
    @app.exception_handler(WorkerTimeout)
    async def _on_worker_timeout(request: Request, exc: WorkerTimeout) -> JSONResponse:
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    @app.exception_handler(WorkerUnavailable)
    async def _on_worker_unavailable(request: Request, exc: WorkerUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "worker_running": bridge.running}

    @app.get("/api/setup/status")
    def setup_status() -> dict:
        from dataclasses import asdict

        status = derive_setup_state(home)
        base = status.to_dict()
        # I1: `checks` (incl. a full metadata.db PRAGMA integrity_check) is
        # only ever rendered by the wizard's Environment step, which only
        # mounts for a needs_setup user. Every other state (needs_analyze,
        # ready) pays nothing for it — this endpoint is fetched once on
        # every app load, so an already-configured user must not carry the
        # full doctor-style health sweep on their steady-state hot path.
        if status.state == SetupState.NEEDS_SETUP:
            from src.setup.checks import run_all_checks
            base["checks"] = [asdict(c) for c in run_all_checks(home)]
        else:
            base["checks"] = []
        return base

    @app.post("/api/setup/test/{service}")
    def setup_test(service: str, body: SetupTestRequest) -> dict:
        from dataclasses import asdict

        from src.setup.services import test_connection

        return asdict(test_connection(service, body.config))

    @app.post("/api/setup/config")
    def setup_config(body: SetupConfigRequest) -> dict:
        from src.setup.config_writer import ConfigExistsError, write_config

        draft = body.model_dump(exclude_none=True)
        reconfigure = bool(draft.pop("reconfigure", False))
        try:
            path = write_config(home, draft, reconfigure=reconfigure)
        except ConfigExistsError as exc:
            raise HTTPException(status_code=409, detail=f"config.yaml already exists at {exc}")
        except ValueError as exc:
            # C1: write_config's own required-field guard (e.g. an
            # empty/whitespace music_directory reaching this route via a
            # rail-jump that skipped the Music step) — authoritative even
            # for a non-wizard caller of this endpoint.
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "config_path": path}

    @app.get("/api/setup/browse")
    def setup_browse(path: Optional[str] = Query(default=None)) -> dict:
        from src.setup.browse import list_directory

        try:
            return list_directory(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=f"Not a directory: {exc}")

    @app.post("/api/generate")
    async def generate(body: GenerateRequestBody) -> dict:
        axes = resolve_dial_axes(body.range_dial, body.flow_dial, body.pace_dial)
        req = body.to_request(axes)
        err = req.validation_error()
        if err:
            raise HTTPException(status_code=422, detail=err)
        job_id = registry.create(request_params=body.model_dump())
        ui = UIStateModel(
            mode=body.mode,  # type: ignore[arg-type]
            cohesion_mode=axes["cohesion_mode"],  # type: ignore[arg-type]
            genre_mode=axes["genre_mode"],  # type: ignore[arg-type]
            sonic_mode=axes["sonic_mode"],  # type: ignore[arg-type]
            pace_mode=axes["pace_mode"],  # type: ignore[arg-type]
            track_count=body.tracks,
            seed_track_ids=list(body.seed_track_ids),
            steering_tags=list(body.steering_tags),
            recency_enabled=body.recency_enabled,
            recency_days=body.recency_days,
            recency_plays_threshold=body.recency_plays_threshold,
            instrumental=body.instrumental,
            exclude_live=body.exclude_live if body.exclude_live is not None else True,
            artist_spacing=body.artist_spacing,  # type: ignore[arg-type]
            diversity_gamma=body.diversity_gamma,
            artist_diversity_mode=body.artist_diversity_mode,  # type: ignore[arg-type]
            artist_presence=body.artist_presence,  # type: ignore[arg-type]
            artist_variety=body.artist_variety,  # type: ignore[arg-type]
            popularity_mode=body.popularity_mode,
        )
        # Resolve seed artists so the policy can evaluate DJ-bridging
        # eligibility (it conservatively disables when keys are missing).
        seed_artist_keys: Optional[list[str]] = None
        if body.mode == "seeds" and body.seed_track_ids:
            resolver = seed_artist_resolver or _resolve_seed_artist_keys
            seed_artist_keys = [k for k in resolver(list(body.seed_track_ids)) if k] or None
        policy = derive_runtime_config(ui, seed_artist_keys=seed_artist_keys)
        for note in policy.notes:
            logger.info("Policy: %s", note)
        overrides = policy.overrides
        try:
            await bridge.submit({
                "cmd": "generate_playlist",
                "job_id": job_id,
                "base_config_path": config_path,
                "overrides": overrides,
                "args": req.to_worker_args(),
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is already running.")
        return {"job_id": job_id}

    @app.get("/api/jobs")
    async def jobs() -> list[JobOut]:
        return registry.recent()

    @app.delete("/api/jobs")
    async def clear_jobs() -> dict:
        # Preserve a genuinely in-flight job (bridge busy) so its still-arriving
        # worker events aren't orphaned. When the bridge is idle, any job still
        # marked "running" is a zombie whose worker died before emitting `done` —
        # clear it too. Keying on bridge.busy (not the status string) is what
        # lets Clear sweep dead "running" jobs.
        return {"cleared": registry.clear(keep_running=bridge.busy)}

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str) -> JobOut:
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/jobs/{job_id}/logs")
    async def job_logs(job_id: str) -> dict:
        return {"logs": registry.logs(job_id)}

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "running":
            raise HTTPException(status_code=409, detail="Job is not running")
        await bridge.cancel()
        return {"ok": True}

    @app.post("/api/tools/analyze")
    async def tools_analyze(body: AnalyzeToolRequest) -> dict:
        job_id = registry.create(request_params=body.model_dump())
        try:
            await bridge.submit({
                "cmd": "analyze_library",
                "job_id": job_id,
                "base_config_path": config_path,
                "overrides": {},
                "stages": body.stages or None,
                "force": body.force,
                "dry_run": body.dry_run,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.post("/api/tools/enrich")
    async def tools_enrich(body: EnrichToolRequest) -> dict:
        job_id = registry.create(request_params=body.model_dump())
        try:
            await bridge.submit({
                "cmd": "enrich_genres",
                "job_id": job_id,
                "scope": body.scope,
                "artist": body.artist,
                "album": body.album,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.get("/api/review/queue")
    async def review_queue(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            return await bridge.command({
                "cmd": "get_escalation_queue", "search": search,
                "limit": limit, "offset": offset}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/api/review/completed")
    async def review_completed(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            return await bridge.command({
                "cmd": "get_escalation_completed", "search": search,
                "limit": limit, "offset": offset}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/review/decision")
    async def review_decision(body: EscalationDecisionRequest) -> dict:
        if not body.album_id.strip() or not body.decision.strip():
            raise HTTPException(status_code=422, detail="album_id and decision are required")
        try:
            result = await bridge.command({
                "cmd": "apply_escalation_decision", "album_id": body.album_id,
                "decision": body.decision, "genres": body.genres}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.post("/api/review/publish")
    async def review_publish() -> dict:
        job_id = registry.create(request_params={"tool": "publish_decided"})
        try:
            await bridge.submit({"cmd": "publish_decided", "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    # ── Taxonomy term adjudication (vocabulary-level review) ──────────────────
    @app.get("/api/taxonomy/queue")
    async def taxonomy_queue(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            return await bridge.command({
                "cmd": "get_taxonomy_queue", "search": search,
                "limit": limit, "offset": offset}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/api/taxonomy/completed")
    async def taxonomy_completed(search: str = "", limit: int = 50, offset: int = 0) -> dict:
        try:
            return await bridge.command({
                "cmd": "get_taxonomy_completed", "search": search,
                "limit": limit, "offset": offset}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/taxonomy/adjudicate")
    async def taxonomy_adjudicate(body: TaxonomyAdjudicateRequest) -> dict:
        # Tracked job: the Claude call is slow (often > the 60s untracked cap) and
        # must run off the reader thread. The client polls /api/jobs/{id} for the
        # verdict in tool_result.
        if not body.term.strip():
            raise HTTPException(status_code=422, detail="term is required")
        job_id = registry.create(
            request_params={"tool": "adjudicate_taxonomy_term", "term": body.term})
        try:
            await bridge.submit({"cmd": "adjudicate_taxonomy_term",
                                 "term": body.term, "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    @app.post("/api/taxonomy/decision")
    async def taxonomy_decision(body: TaxonomyDecisionRequest) -> dict:
        if not body.term.strip() or not body.verdict.strip():
            raise HTTPException(status_code=422, detail="term and verdict are required")
        try:
            result = await bridge.command({
                "cmd": "record_taxonomy_decision", "term": body.term,
                "raw_term": body.raw_term, "verdict": body.verdict,
                "proposal": body.proposal, "claude": body.claude,
                "human_edited": body.human_edited}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.post("/api/taxonomy/validate")
    async def taxonomy_validate(body: TaxonomyValidateRequest) -> dict:
        # Untracked: validate_proposal is a fast read-only check against the
        # live taxonomy; the ADD wizard blocks Stage until errors == [].
        try:
            result = await bridge.command({
                "cmd": "validate_taxonomy_proposal",
                "proposal": body.proposal}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {"ok": True, **result}

    @app.post("/api/taxonomy/apply")
    async def taxonomy_apply() -> dict:
        job_id = registry.create(request_params={"tool": "apply_taxonomy_decisions"})
        try:
            await bridge.submit({"cmd": "apply_taxonomy_decisions", "job_id": job_id})
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is already running.")
        return {"job_id": job_id}

    # ── Artist-alias links (Artist Links panel) ────────────────────────────
    @app.get("/api/artists/links")
    async def artist_links_list() -> dict:
        try:
            return await bridge.command({"cmd": "list_artist_links"}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/artists/links/save")
    async def artist_links_save(body: ArtistLinksSaveRequest) -> dict:
        groups = [g.model_dump() for g in body.groups]
        try:
            result = await bridge.command({"cmd": "save_artist_links", "groups": groups}, untracked=True)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="Worker is busy — try again when the current job finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.get("/api/artists/search")
    async def artists_search(q: str = "", limit: int = Query(20, ge=1, le=100)) -> dict:
        """Distinct-library-artist typeahead for the Artist Links panel."""
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": []}
        from src.metadata_client import search_artists
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                return {"items": search_artists(conn, q, limit)}
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("artists_search query failed against DB %s: %s", DB_PATH, exc)
            return {"items": []}

    @app.get("/api/tracks/search")
    async def track_search(
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(25, ge=1, le=200),
    ) -> dict:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": [], "has_more": False}
        pattern = f"%{q.lower()}%"
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    """
                    SELECT t.track_id, t.title, t.artist, t.album, t.duration_ms, t.file_path
                    FROM tracks t
                    WHERE lower(t.title) LIKE ? OR lower(t.artist) LIKE ? OR lower(t.album) LIKE ?
                    ORDER BY t.artist, t.title
                    LIMIT ? OFFSET ?
                    """,
                    (pattern, pattern, pattern, limit + 1, offset),
                ).fetchall()
                has_more = len(rows) > limit
                rows = rows[:limit]
                genres_by_id = _genres_for_tracks(conn, [r[0] for r in rows])
                items = [
                    {
                        "track_id": r[0],
                        "title": r[1] or "Unknown",
                        "artist": r[2] or "Unknown",
                        "album": r[3] or "",
                        "duration_ms": r[4] or 0,
                        "file_path": r[5] or "",
                        "genres": genres_by_id.get(r[0], []),
                    }
                    for r in rows
                ]
                return {"items": items, "has_more": has_more}
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("track_search query failed against DB %s: %s", DB_PATH, exc)
            return {"items": [], "has_more": False}

    @app.post("/api/tracks/genres")
    async def track_genres(body: TrackGenresRequest) -> dict[str, list[str]]:
        """Display genres for staged seed tracks: enriched -> metadata fallback,
        canonicalized through the taxonomy, ordered most-specific first.

        Called when the staged seed set changes (NOT per keystroke). Unknown
        track ids are omitted from the response.
        """
        ids = [str(t) for t in body.track_ids if str(t).strip()]
        if not ids or not DB_PATH.exists():
            return {}
        from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver
        from src.genre.authority import display_genre_names_for_track
        from src.genre.granularity import order_genres_for_display

        resolver = EnrichedGenreResolver(SIDECAR_DB_PATH)
        out: dict[str, list[str]] = {}
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                placeholders = ",".join("?" for _ in ids)
                rows = conn.execute(
                    f"SELECT track_id, artist, album FROM tracks WHERE track_id IN ({placeholders})",
                    ids,
                ).fetchall()
                for track_id, artist, album in rows:
                    tid = str(track_id)
                    # Published authority first (release_effective_genres); the
                    # sidecar signature is the older bandcamp-era layer, kept as
                    # fallback for enriched-but-unpublished releases.
                    raw = display_genre_names_for_track(conn, tid)
                    if not raw:
                        raw = resolver.get_enriched_genres(artist=artist or "", album=album) or []
                    if not raw:
                        raw = [g[0] for g in conn.execute(
                            "SELECT genre FROM track_effective_genres WHERE track_id = ? ORDER BY priority",
                            (tid,),
                        ).fetchall()]
                    if not raw:
                        raw = [g[0] for g in conn.execute(
                            "SELECT genre FROM track_genres WHERE track_id = ? ORDER BY weight DESC",
                            (tid,),
                        ).fetchall()]
                    out[tid] = order_genres_for_display(raw)
            finally:
                conn.close()
        except sqlite3.Error:
            logger.warning("track_genres lookup failed", exc_info=True)
            return {}
        return out

    @app.get("/api/genres/search")
    async def genres_search(q: str = "", limit: int = Query(20, ge=1, le=100)) -> dict:
        """Autocomplete over the canonical taxonomy vocabulary (active genres)."""
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": []}
        from src.genre.authority import canonical_genre_search
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                return {"items": [
                    {"genre_id": gid, "name": name}
                    for gid, name in canonical_genre_search(conn, q, limit)
                ]}
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("genres_search query failed against DB %s: %s", DB_PATH, exc)
            return {"items": []}

    @app.get("/api/genres/for_album")
    async def genres_for_album(artist: str = "", album: str = "") -> dict:
        """Current authoritative genres for a release (seeds the edit dialog)."""
        if not artist.strip() or not album.strip() or not DB_PATH.exists():
            return {"genres": []}
        from src.genre.authority import display_genre_names_for_album
        from src.genre.genre_edit import album_id_for_release
        from src.genre.granularity import order_genres_for_display
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                album_id = album_id_for_release(conn, artist, album)
                if not album_id:
                    return {"genres": []}
                names = display_genre_names_for_album(conn, album_id)
                return {"genres": order_genres_for_display(names)}
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("genres_for_album query failed against DB %s: %s", DB_PATH, exc)
            return {"genres": []}

    @app.get("/api/genres/for_artist")
    async def genres_for_artist(artist: str = "") -> dict:
        """Published observed-leaf genres across an artist's releases (steering chips)."""
        if not artist.strip() or not DB_PATH.exists():
            return {"genres": []}
        from src.genre.authority import resolved_genres_for_artist
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                tags = resolved_genres_for_artist(conn, artist)
                return {"genres": [
                    {"name": t.name, "release_count": t.release_count,
                     "confidence": round(t.max_confidence, 3)}
                    for t in tags[:12]
                ]}
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("genres_for_artist query failed against DB %s: %s", DB_PATH, exc)
            return {"genres": []}

    @app.get("/api/autocomplete")
    async def autocomplete(
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(30, ge=1, le=200),
    ) -> dict:
        q = q.strip()
        if not q or not DB_PATH.exists():
            return {"items": [], "has_more": False}
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    # Read the actual library (tracks), not the separate `artists` table:
                    # that table lags the scanner (689 scanned artists were missing from
                    # it, e.g. "REX"), so newly-scanned artists never autocompleted.
                    # Match the query as a SUBSTRING (so "radio" finds "The Radio Dept."),
                    # not just a leading prefix. Relevance tiers: name starts with the
                    # query (0) > query starts a word, e.g. after "The " (1) > query is a
                    # mid-word substring (2); alphabetical within a tier. GROUP BY
                    # LOWER(artist) collapses case variants; MIN() is a stable casing.
                    "SELECT MIN(artist) AS name FROM tracks "
                    "WHERE artist LIKE '%' || :q || '%' "
                    "  AND artist IS NOT NULL AND TRIM(artist) <> '' "
                    "GROUP BY LOWER(artist) "
                    "ORDER BY CASE "
                    "    WHEN name LIKE :q || '%' THEN 0 "
                    "    WHEN name LIKE '% ' || :q || '%' THEN 1 "
                    "    ELSE 2 END, "
                    "  name "
                    "LIMIT :lim OFFSET :off",
                    {"q": q, "lim": limit + 1, "off": offset},
                ).fetchall()
            finally:
                conn.close()
            has_more = len(rows) > limit
            return {"items": [r[0] for r in rows[:limit]], "has_more": has_more}
        except Exception as exc:
            logger.warning("autocomplete query failed against DB %s: %s", DB_PATH, exc)
            return {"items": [], "has_more": False}

    @app.get("/api/audio/{track_id}")
    async def audio(track_id: str, request: Request):
        return stream_audio(track_id, DB_PATH, request)

    @app.post("/api/replace_suggestions")
    async def replace_suggestions(body: ReplaceSuggestionsRequest) -> ReplaceSuggestionsResponse:
        try:
            result = await bridge.command({
                "cmd": "find_replacement_suggestions",
                "position": body.position,
                "top_k": body.top_k,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ReplaceSuggestionsResponse.from_worker_candidates(
            position=result.get("position", body.position),
            raw=result.get("candidates", []),
        )

    @app.post("/api/blacklist")
    async def blacklist(body: BlacklistRequest) -> dict:
        if body.scope:
            if body.scope not in ("album", "artist"):
                raise HTTPException(status_code=422, detail="scope must be 'album' or 'artist'")
            if body.scope == "album" and not body.artist:
                raise HTTPException(status_code=422, detail="album scope requires 'artist'")
            cmd = {
                "cmd": "blacklist_scope_set",
                "base_config_path": config_path,
                "overrides": {},
                "scope": body.scope,
                "value": body.value,
                "artist": body.artist,
                "enabled": body.enabled,
            }
        else:
            if not body.track_ids:
                raise HTTPException(status_code=422, detail="track_ids required when no scope given")
            cmd = {
                "cmd": "blacklist_set",
                "base_config_path": config_path,
                "overrides": {},
                "track_ids": body.track_ids,
                "value": body.enabled,
            }
        try:
            result = await bridge.command(cmd)
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.get("/api/blacklist")
    async def get_blacklist() -> BlacklistFetchResponse:
        try:
            result = await bridge.command({
                "cmd": "blacklist_fetch_scopes",
                "base_config_path": config_path,
                "overrides": {},
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return BlacklistFetchResponse.from_worker(result)

    @app.post("/api/blacklist/artist")
    async def blacklist_artist(body: BlacklistArtistRequest) -> dict:
        if not body.artist.strip():
            raise HTTPException(status_code=422, detail="artist is required")
        try:
            result = await bridge.command({
                "cmd": "blacklist_scope_set",
                "base_config_path": config_path,
                "overrides": {},
                "scope": "artist",
                "value": body.artist,
                "artist": body.artist,
                "enabled": True,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.post("/api/edit_genres")
    async def edit_genres(body: EditGenresRequest) -> dict:
        if not body.artist.strip() or not body.album.strip():
            raise HTTPException(status_code=422, detail="artist and album are required")
        try:
            result = await bridge.command({
                "cmd": "edit_genres",
                "base_config_path": config_path,
                "artist": body.artist,
                "album": body.album,
                "genres": body.genres,
                "base_genres": body.base_genres,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}

    @app.post("/api/refresh_genre_artifact")
    async def refresh_genre_artifact() -> dict:
        """Re-bake the genre matrices in the artifact so generation sees edits."""
        job_id = registry.create(request_params={"tool": "refresh_genre_artifact"})
        try:
            await bridge.submit({
                "cmd": "refresh_genre_artifact",
                "job_id": job_id,
                "base_config_path": config_path,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A job is in progress — try again when it finishes.")
        return {"job_id": job_id}

    @app.post("/api/export/plex")
    async def export_plex(body: PlexExportRequest) -> dict:
        try:
            key = run_plex_export(body.title, body.tracks, config_path)
        except PlexNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface any export failure to the client
            raise HTTPException(status_code=502, detail=f"Plex export failed: {exc}")
        return {"ok": True, "playlist_key": key}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive; client may send pings
        except WebSocketDisconnect:
            hub.disconnect(ws)

    static_dir = resolve_static_dir()
    if static_dir.exists():
        # Mount the whole dist root (html=True serves index.html at "/") so
        # root-level files — icons, manifest.webmanifest, /fonts — are served
        # too; the old /assets-only mount silently 404'd all of them. Mounted
        # last, so /api and /ws routes registered above keep precedence.
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="dist")

    return app
