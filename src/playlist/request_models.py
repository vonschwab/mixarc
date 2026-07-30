"""Shared playlist request models for CLI, GUI, and worker boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


GenerateMode = Literal["artist", "genre", "seeds", "history"]
ModeValue = Literal["strict", "narrow", "dynamic", "discover", "off"]
PaceModeValue = Literal["strict", "narrow", "dynamic"]
LibraryOperation = Literal[
    "analyze_library",
    "scan_library",
    "update_genres",
    "update_sonic",
    "build_artifacts",
]
AnalyzeLibraryStage = Literal[
    "scan",
    "genres",
    "discogs",
    "lastfm",
    "sonic",
    "muq",
    "adjudicate",
    "apply",
    "enrich",
    "publish",
    "genre-sim",
    "artifacts",
    "energy",
    "popularity",
    "genre-embedding",
    "verify",
]

# Canonical default stage order — the single source of truth shared by the CLI
# (scripts/analyze_library.STAGE_ORDER_DEFAULT imports this) and the GUI/web
# worker. The album-grain Claude (Sonnet) adjudicator (`adjudicate` + `apply`)
# is the genre-production path; the legacy tag-grain `enrich` stage is opt-in
# only (still a valid stage for `--stages enrich`, but never in the default run).
# Keep these in sync — a past divergence silently left the GUI on `enrich`.
ANALYZE_LIBRARY_STAGE_ORDER: tuple[AnalyzeLibraryStage, ...] = (
    "scan",
    "genres",
    "discogs",
    "lastfm",
    "sonic",
    "muq",
    "adjudicate",
    "apply",
    "publish",
    "genre-sim",
    "artifacts",
    "energy",
    "popularity",
    "genre-embedding",
    "verify",
)


def _clean_text(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_list(value: Optional[list[Any]]) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _clean_text(item))]


def _clean_stages(value: Optional[list[Any]]) -> list[AnalyzeLibraryStage]:
    requested = _clean_list(value)
    allowed = set(ANALYZE_LIBRARY_STAGE_ORDER)
    return [
        stage
        for stage in ANALYZE_LIBRARY_STAGE_ORDER
        if stage in requested and stage in allowed
    ]


@dataclass
class GeneratePlaylistRequest:
    """Typed generation request shared by CLI, GUI, and worker protocol."""

    mode: GenerateMode = "artist"
    tracks: int = 30
    artist: Optional[str] = None
    artists: list[str] = field(default_factory=list)
    """2+ names activate the multi-artist blend; `artist` stays the primary name."""
    genre: Optional[str] = None
    track: Optional[str] = None
    seed_tracks: list[str] = field(default_factory=list)
    seed_track_ids: list[str] = field(default_factory=list)
    anchor_seed_ids: list[str] = field(default_factory=list)
    genre_mode: Optional[ModeValue] = None
    sonic_mode: Optional[ModeValue] = None
    pace_mode: Optional[PaceModeValue] = None
    include_collaborations: bool = False
    exclude_seed_tracks_from_recency: bool = False
    artist_only: bool = False
    popular_seeds_mode: str = "off"
    popularity_mode: str = "off"  # Oops All Bangers: off / on / oops
    seed_epoch: int = 0

    @classmethod
    def from_ui_state(
        cls,
        ui_state: Any,
        *,
        seed_tracks: Optional[list[str]] = None,
        seed_track_ids: Optional[list[str]] = None,
    ) -> "GeneratePlaylistRequest":
        return cls(
            mode=ui_state.mode,
            tracks=ui_state.track_count,
            artist=ui_state.primary_artist() if ui_state.mode == "artist" else None,
            artists=_clean_list(list(getattr(ui_state, "artist_queries", []) or [])),
            genre=ui_state.genre_query if ui_state.mode == "genre" else None,
            seed_tracks=_clean_list(seed_tracks or []),
            seed_track_ids=_clean_list(seed_track_ids if seed_track_ids is not None else ui_state.seed_track_ids),
            genre_mode=ui_state.genre_mode,
            sonic_mode=ui_state.sonic_mode,
            pace_mode=getattr(ui_state, "pace_mode", "dynamic"),
            include_collaborations=ui_state.include_collaborations,
            exclude_seed_tracks_from_recency=bool(
                getattr(ui_state, "exclude_seed_tracks_from_recency", False)
            ),
        )

    @classmethod
    def from_worker_args(cls, args: dict[str, Any]) -> "GeneratePlaylistRequest":
        mode = args.get("mode", "artist")
        if mode not in ("artist", "genre", "seeds", "history"):
            mode = "artist"
        try:
            tracks = int(args.get("tracks", 30))
        except (TypeError, ValueError):
            tracks = 30
        return cls(
            mode=mode,
            tracks=tracks,
            artist=_clean_text(args.get("artist")),
            artists=_clean_list(args.get("artists")),
            genre=_clean_text(args.get("genre")),
            track=_clean_text(args.get("track")),
            seed_tracks=_clean_list(args.get("seed_tracks")),
            seed_track_ids=_clean_list(args.get("seed_track_ids")),
            anchor_seed_ids=_clean_list(args.get("anchor_seed_ids")),
            genre_mode=_clean_text(args.get("genre_mode")),
            sonic_mode=_clean_text(args.get("sonic_mode")),
            pace_mode=_clean_text(args.get("pace_mode")),
            include_collaborations=bool(args.get("include_collaborations", False)),
            exclude_seed_tracks_from_recency=bool(args.get("exclude_seed_tracks_from_recency", False)),
            artist_only=bool(args.get("artist_only", False)),
            popular_seeds_mode=str(args.get("popular_seeds_mode") or "off"),
            popularity_mode=str(args.get("popularity_mode") or "off"),
            seed_epoch=int(args.get("seed_epoch", 0)),
        )

    @classmethod
    def from_cli_args(
        cls,
        args: Any,
        *,
        genre_mode: Optional[ModeValue] = None,
        sonic_mode: Optional[ModeValue] = None,
        pace_mode: Optional[PaceModeValue] = None,
    ) -> "GeneratePlaylistRequest":
        artist = _clean_text(getattr(args, "artist", None))
        genre = _clean_text(getattr(args, "genre", None))
        if artist:
            mode: GenerateMode = "artist"
        elif genre:
            mode = "genre"
        else:
            mode = "history"

        try:
            tracks = int(getattr(args, "tracks", 30))
        except (TypeError, ValueError):
            tracks = 30

        anchor_seed_ids_raw = getattr(args, "anchor_seed_ids", None)
        if isinstance(anchor_seed_ids_raw, str):
            anchor_seed_ids = _clean_list(anchor_seed_ids_raw.split(","))
        else:
            anchor_seed_ids = _clean_list(anchor_seed_ids_raw)

        return cls(
            mode=mode,
            tracks=tracks,
            artist=artist if mode == "artist" else None,
            genre=genre if mode == "genre" else None,
            track=_clean_text(getattr(args, "track", None)),
            anchor_seed_ids=anchor_seed_ids,
            genre_mode=genre_mode,
            sonic_mode=sonic_mode,
            pace_mode=pace_mode,
            artist_only=bool(getattr(args, "artist_only", False)),
        )

    def to_worker_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {"mode": self.mode, "tracks": int(self.tracks)}
        optional_values = {
            "artist": _clean_text(self.artist),
            "genre": _clean_text(self.genre),
            "track": _clean_text(self.track),
            "genre_mode": _clean_text(self.genre_mode),
            "sonic_mode": _clean_text(self.sonic_mode),
            "pace_mode": _clean_text(self.pace_mode),
        }
        for key, value in optional_values.items():
            if value:
                args[key] = value

        seed_tracks = _clean_list(self.seed_tracks)
        seed_track_ids = _clean_list(self.seed_track_ids)
        anchor_seed_ids = _clean_list(self.anchor_seed_ids)
        if seed_tracks:
            args["seed_tracks"] = seed_tracks
        if seed_track_ids:
            args["seed_track_ids"] = seed_track_ids
        if anchor_seed_ids:
            args["anchor_seed_ids"] = anchor_seed_ids
        artists = _clean_list(self.artists)
        if artists:
            args["artists"] = artists
        if self.include_collaborations:
            args["include_collaborations"] = True
        if self.exclude_seed_tracks_from_recency:
            args["exclude_seed_tracks_from_recency"] = True
        if self.artist_only:
            args["artist_only"] = True
        if self.popular_seeds_mode and self.popular_seeds_mode != "off":
            args["popular_seeds_mode"] = str(self.popular_seeds_mode)
        if self.popularity_mode and self.popularity_mode != "off":
            args["popularity_mode"] = str(self.popularity_mode)
        if self.seed_epoch:
            args["seed_epoch"] = int(self.seed_epoch)
        return args

    def validation_error(self) -> Optional[str]:
        if self.mode == "artist" and not (_clean_text(self.artist) or _clean_list(self.artists)):
            return "Enter an artist before generating."
        if self.mode == "genre" and not _clean_text(self.genre):
            return "Enter a genre before generating."
        if self.mode == "seeds" and not (_clean_list(self.seed_tracks) or _clean_list(self.seed_track_ids)):
            return "Add at least one seed track before generating."
        return None


@dataclass
class LibraryOperationRequest:
    """Typed request for a single library maintenance operation."""

    operation: Any
    config_path: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_worker_command(self) -> dict[str, Any]:
        operation_value = getattr(self.operation, "value", self.operation)
        return {
            "cmd": str(operation_value),
            "base_config_path": self.config_path,
            "overrides": dict(self.overrides or {}),
        }


@dataclass
class LibraryPipelineRequest:
    """Typed request for the existing Analyze Library workflow."""

    config_path: str
    overrides: dict[str, Any] = field(default_factory=dict)
    stages: list[AnalyzeLibraryStage] = field(
        default_factory=lambda: list(ANALYZE_LIBRARY_STAGE_ORDER)
    )
    force: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        cleaned = _clean_stages(list(self.stages or []))
        self.stages = cleaned or list(ANALYZE_LIBRARY_STAGE_ORDER)

    def operations(self) -> list[LibraryOperationRequest]:
        return [
            LibraryOperationRequest(
                operation="analyze_library",
                config_path=self.config_path,
                overrides=dict(self.overrides or {}),
            )
        ]

    def to_worker_command(self) -> dict[str, Any]:
        command = LibraryOperationRequest(
            operation="analyze_library",
            config_path=self.config_path,
            overrides=dict(self.overrides or {}),
        ).to_worker_command()
        if list(self.stages) != list(ANALYZE_LIBRARY_STAGE_ORDER):
            command["stages"] = list(self.stages)
        if self.force:
            command["force"] = True
        if self.dry_run:
            command["dry_run"] = True
        return command
