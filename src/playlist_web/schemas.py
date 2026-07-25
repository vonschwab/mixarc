"""Pydantic schemas for web API request/response serialization."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.playlist.request_models import GeneratePlaylistRequest


class GenerateRequestBody(BaseModel):
    """API request body for playlist generation."""

    mode: str = "artist"
    tracks: int = 30
    artist: Optional[str] = None
    genre: Optional[str] = None
    seed_tracks: list[str] = Field(default_factory=list)
    seed_track_ids: list[str] = Field(default_factory=list)
    steering_tags: list[str] = Field(default_factory=list)
    # GUI dials (spec 2026-07-04). Translated to engine axes exactly once in
    # the generate route via policy.resolve_dial_axes.
    range_dial: Optional[str] = None
    flow_dial: Optional[str] = None
    pace_dial: Optional[str] = None
    include_collaborations: bool = False
    exclude_seed_tracks_from_recency: bool = False
    # Policy fields — translated into config overrides via UIStateModel + derive_runtime_config
    recency_enabled: bool = True
    instrumental: bool = False
    recency_days: int = 14
    recency_plays_threshold: int = 1
    artist_spacing: str = "normal"
    diversity_gamma: float = 0.04
    artist_diversity_mode: str = "weighted"
    artist_presence: str = "medium"
    artist_variety: str = "balanced"
    popular_seeds_mode: str = "off"
    popularity_mode: str = "off"  # Oops All Bangers: off / on / oops
    seed_epoch: int = 0

    def to_request(self, axes: dict) -> GeneratePlaylistRequest:
        """Convert to internal GeneratePlaylistRequest. `axes` is the
        policy-resolved dial translation (resolve_dial_axes output)."""
        return GeneratePlaylistRequest(
            mode=self.mode,
            tracks=self.tracks,
            artist=self.artist,
            genre=self.genre,
            seed_tracks=list(self.seed_tracks),
            seed_track_ids=list(self.seed_track_ids),
            genre_mode=axes.get("genre_mode"),
            sonic_mode=axes.get("sonic_mode"),
            pace_mode=axes.get("pace_mode"),
            include_collaborations=self.include_collaborations,
            exclude_seed_tracks_from_recency=self.exclude_seed_tracks_from_recency,
            popular_seeds_mode=self.popular_seeds_mode,
            popularity_mode=self.popularity_mode,
            seed_epoch=self.seed_epoch,
        )


class TrackOut(BaseModel):
    """Response model for a single track in a playlist."""

    position: int
    rating_key: Optional[str] = None
    artist: str = "Unknown"
    title: str = "Unknown"
    album: str = ""
    duration_ms: int = 0
    file_path: str = ""
    sonic_similarity: Optional[float] = None
    genre_similarity: Optional[float] = None
    transition_score: Optional[float] = None
    popularity_rank: Optional[int] = None  # Oops All Bangers: Last.fm rank (1-based), or null
    genres: list[str] = Field(default_factory=list)


class MetricsOut(BaseModel):
    """Response model for playlist quality metrics."""

    mean_transition: Optional[float] = None
    min_transition: Optional[float] = None
    p10_transition: Optional[float] = None
    p90_transition: Optional[float] = None
    distinct_artists: Optional[int] = None


class PlaylistOut(BaseModel):
    """Response model for a generated playlist."""

    name: str = "Generated Playlist"
    track_count: int = 0
    tracks: list[TrackOut] = Field(default_factory=list)
    metrics: MetricsOut = Field(default_factory=MetricsOut)
    relaxations: list[dict] = Field(default_factory=list)
    receipt: Optional[dict] = None
    # Hard-invariant breaches (artist gap, per-artist cap). The playlist is still
    # returned; `degraded` tells the UI to say so rather than presenting it as clean.
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False

    @classmethod
    def from_worker(cls, raw: dict[str, Any]) -> "PlaylistOut":
        """Parse a worker result dict into a PlaylistOut response."""
        # Extract tracks, handling missing fields gracefully
        tracks_raw = raw.get("tracks", [])
        tracks = [
            TrackOut(
                position=t.get("position", 0),
                rating_key=t.get("rating_key"),
                artist=t.get("artist", "Unknown"),
                title=t.get("title", "Unknown"),
                album=t.get("album", ""),
                duration_ms=t.get("duration_ms", 0),
                file_path=t.get("file_path", ""),
                sonic_similarity=t.get("sonic_similarity"),
                genre_similarity=t.get("genre_similarity"),
                transition_score=t.get("transition_score"),
                popularity_rank=t.get("popularity_rank"),
                genres=t.get("genres", []),
            )
            for t in tracks_raw
        ]

        # Extract metrics
        metrics_raw = raw.get("metrics") or {}
        metrics = MetricsOut(
            mean_transition=metrics_raw.get("mean_transition"),
            min_transition=metrics_raw.get("min_transition"),
            p10_transition=metrics_raw.get("p10_transition"),
            p90_transition=metrics_raw.get("p90_transition"),
            distinct_artists=metrics_raw.get("distinct_artists"),
        )

        relaxations = [
            e for e in (raw.get("relaxations") or [])
            if isinstance(e, dict)
        ]

        return cls(
            name=raw.get("name", "Generated Playlist"),
            track_count=raw.get("track_count", len(tracks_raw)),
            tracks=tracks,
            metrics=metrics,
            relaxations=relaxations,
            receipt=raw.get("receipt"),
            warnings=[str(w) for w in (raw.get("warnings") or [])],
            degraded=bool(raw.get("degraded", False)),
        )


class JobOut(BaseModel):
    """Response model for a job status query or completion."""

    job_id: str
    status: str
    stage: str = ""
    error: Optional[str] = None
    playlist: Optional[PlaylistOut] = None
    tool_result: Optional[dict] = None
    created_at: Optional[float] = None
    request_params: Optional[dict] = None


class CandidateOut(BaseModel):
    """A replacement candidate track."""

    track_id: str
    title: str = "Unknown"
    artist: str = "Unknown"
    album: str = ""
    genres: list[str] = Field(default_factory=list)
    fit_score: float = 0.0
    # file_path is the identity the Plex/M3U exporters resolve on; it must reach the
    # GUI so a replacement stamps the new track's path onto the playlist. Without it
    # the export resolves the OLD track's path and the replacement is lost.
    file_path: str = ""
    duration_ms: int = 0


class ReplaceSuggestionsRequest(BaseModel):
    # job_id is unused by the worker (it reads _LAST_GENERATION_CACHE) but kept
    # for client correlation and future multi-job support.
    job_id: str = ""
    position: int
    top_k: int = 10


class ReplaceSuggestionsResponse(BaseModel):
    position: int
    candidates: list[CandidateOut] = Field(default_factory=list)

    @classmethod
    def from_worker_candidates(cls, position: int, raw: list[dict]) -> "ReplaceSuggestionsResponse":
        cands = [
            CandidateOut(
                track_id=str(c.get("rating_key") or c.get("track_id") or ""),
                title=c.get("title", "Unknown"),
                artist=c.get("artist", "Unknown"),
                album=c.get("album", ""),
                genres=list(c.get("genres", []) or []),
                fit_score=float(c.get("mean_t", 0.0) or 0.0),
                file_path=str(c.get("file_path", "") or ""),
                duration_ms=int(c.get("duration_ms", 0) or 0),
            )
            for c in raw
        ]
        return cls(position=position, candidates=cands)


class BlacklistRequest(BaseModel):
    """Blacklist request body.

    Mutually exclusive modes:
    - Track mode: provide track_ids (non-empty list), leave scope empty.
    - Album scope: scope="album", value=album_title, artist=artist_name.
    - Artist scope: scope="artist", value=artist_name.

    Route-level validation enforces these constraints.
    """

    track_ids: list[str] = Field(default_factory=list)
    scope: Optional[str] = None          # "album" | "artist"
    value: str = ""                      # album title (album scope) or artist name (artist scope)
    artist: str = ""                     # required for album scope
    enabled: bool = True


class EditGenresRequest(BaseModel):
    artist: str
    album: str
    genres: list[str] = Field(default_factory=list)
    # The genres the GUI displayed when the edit dialog opened (the graph
    # authority). The worker diffs `genres` against this to compute the
    # add/remove override — see handle_edit_genres.
    base_genres: list[str] = Field(default_factory=list)


class PlexExportRequest(BaseModel):
    title: str
    tracks: list[dict] = Field(default_factory=list)


class BlacklistEntryOut(BaseModel):
    scope: str                              # "artist" | "album" | "track"
    display_name: str
    track_id: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None


class BlacklistFetchResponse(BaseModel):
    artists: list[BlacklistEntryOut] = Field(default_factory=list)
    albums: list[BlacklistEntryOut] = Field(default_factory=list)
    tracks: list[BlacklistEntryOut] = Field(default_factory=list)
    total: int = 0

    @classmethod
    def from_worker(cls, raw: dict) -> "BlacklistFetchResponse":
        artists = [
            BlacklistEntryOut(scope="artist", display_name=a.get("artist_name", ""),
                              artist=a.get("artist_name", ""))
            for a in raw.get("artists", [])
        ]
        albums = [
            BlacklistEntryOut(scope="album", display_name=al.get("album_name", ""),
                              artist=al.get("artist_name", ""), album=al.get("album_name", ""))
            for al in raw.get("albums", [])
        ]
        tracks = [
            BlacklistEntryOut(scope="track",
                              display_name=t.get("title", "") or t.get("track_id", ""),
                              track_id=t.get("track_id", ""), artist=t.get("artist", ""),
                              album=t.get("album", ""))
            for t in raw.get("tracks", [])
        ]
        return cls(artists=artists, albums=albums, tracks=tracks,
                   total=len(artists) + len(albums) + len(tracks))


class BlacklistArtistRequest(BaseModel):
    artist: str


class AnalyzeToolRequest(BaseModel):
    stages: list[str] = Field(default_factory=list)
    force: bool = False
    dry_run: bool = False


class EnrichToolRequest(BaseModel):
    scope: str = "all_unenriched"
    artist: Optional[str] = None
    album: Optional[str] = None


class ReviewDecisionRequest(BaseModel):
    release_key: str
    term: str
    decision: str  # accept | reject | revert — validated by the worker


class EscalationDecisionRequest(BaseModel):
    album_id: str
    decision: str  # accept | edit | reject | revert
    genres: list[str] | None = None


class TaxonomyAdjudicateRequest(BaseModel):
    term: str


class TaxonomyDecisionRequest(BaseModel):
    term: str
    raw_term: str = ""
    verdict: str  # add | alias | reject | revert — validated by the worker
    proposal: dict | None = None  # GrowthProposal asdict (add/alias) or {reject_reason, rationale}
    claude: dict | None = None    # Claude's original verdict, for audit
    human_edited: bool = False


class TaxonomyValidateRequest(BaseModel):
    """Structural pre-flight for a manual ADD-wizard proposal (no staging)."""

    proposal: dict


class ArtistLinkGroup(BaseModel):
    type: str  # "alias" | "sibling" — validated by the worker
    members: list[str]


class ArtistLinksSaveRequest(BaseModel):
    groups: list[ArtistLinkGroup]


class TrackGenresRequest(BaseModel):
    """Batch lookup of display genres for staged seed tracks."""

    track_ids: list[str] = Field(default_factory=list)


class SetupTestRequest(BaseModel):
    """Body for POST /api/setup/test/{service} — the Services step's Test button."""

    config: dict = Field(default_factory=dict)


class SetupConfigRequest(BaseModel):
    """Body for POST /api/setup/config — the wizard's final-step config write.

    Field shape matches `src.setup.config_writer.write_config`'s expected
    draft dict: flat `ai_genre_provider`, nested `lastfm`/`discogs`/`plex`.
    """

    music_directory: str
    lastfm: Optional[dict] = None
    discogs: Optional[dict] = None
    plex: Optional[dict] = None
    ai_genre_provider: Optional[str] = None
    reconfigure: bool = False
