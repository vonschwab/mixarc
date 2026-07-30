"""
Configuration Loader - Manages YAML configuration and environment variables
"""
import yaml
import os
from pathlib import Path
from typing import Any

# Single source of truth for the ``playlists.min_track_duration_seconds``
# fallback used when the key is absent from config.yaml. Before this constant
# existed, every reader hardcoded its own literal and they had quietly drifted
# apart (47 here, 47 in playlist_generator.py's bridge-side gate, 46 in genre
# mode and the pier-duration-gate fix, 90 in playlist_gui/worker.py's
# MergedConfig shim) -- inert only because config.yaml and config.example.yaml
# both pin the key to 46 explicitly, so a config missing the key would have
# silently made the bridge floor and the pier floor (or the GUI worker path
# and the CLI path) disagree by a second or more. 46 is the value that
# matches shipped reality (config.example.yaml), so it is the one kept.
# Coordinator review, 2026-07-30 (duration-gate-report.md).
DEFAULT_MIN_TRACK_DURATION_SECONDS = 46


class Config:
    """Configuration manager for MixArc"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._apply_mode_presets()  # Resolve genre_mode/sonic_mode to settings
        self._validate_config()
        self._publish_artifact_settings()

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _apply_mode_presets(self):
        """
        Apply genre_mode and sonic_mode presets if specified in config.

        Mode settings (genre_mode/sonic_mode) take precedence over manual
        weight/threshold settings. If both are present, mode wins.
        """
        playlists_cfg = self.config.get('playlists', {})
        if not playlists_cfg:
            return
        try:
            from src.playlist.mode_presets import apply_mode_presets
        except ImportError:
            return

        apply_mode_presets(playlists_cfg)

    def _validate_config(self):
        """Validate required configuration fields"""
        # Note: openai/api_key was removed when the legacy OpenAI wiring was
        # dropped (Tier-0.10b). It is no longer a required section.
        required_fields = [
            ('library', 'database_path'),
        ]

        for section, field in required_fields:
            if section not in self.config:
                raise ValueError(f"Missing configuration section: {section}")
            if field not in self.config[section]:
                raise ValueError(f"Missing configuration field: {section}.{field}")

            # Check if value is placeholder or empty
            value = self.config[section][field]
            if not value or str(value).startswith('YOUR_'):
                raise ValueError(f"Please set {section}.{field} in {self.config_path}")

    def _publish_artifact_settings(self):
        """Publish artifacts.* config to the artifact loader (process-wide).

        ``artifacts.sonic_variant_override`` must win over the variant the
        artifact declares; if the override names a variant the artifact lacks,
        the artifact load raises (configured-knob-must-act rule).
        """
        from src.features.artifacts import set_sonic_variant_override

        set_sonic_variant_override(self.sonic_variant_override)

    @property
    def sonic_variant_override(self):
        """artifacts.sonic_variant_override — A/B switch for the sonic space.

        When set (e.g. 'mert' or 'tower_weighted'), it wins over the
        artifact-declared X_sonic_variant. Default absent (None): the artifact
        decides.
        """
        return (self.config.get('artifacts') or {}).get('sonic_variant_override')

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default
        return self.config[section].get(key, default)

    @property
    def library_database_path(self) -> str:
        """Get library database path"""
        return self.config['library']['database_path']

    @property
    def library_music_directory(self) -> str:
        """Get music directory path"""
        return self.config['library'].get('music_directory', 'E:\\MUSIC')

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key (env var takes precedence over config.yaml)"""
        if 'openai' not in self.config or 'api_key' not in self.config.get('openai', {}):
            return ''
        return os.getenv('OPENAI_API_KEY') or self.config['openai']['api_key']

    @property
    def ai_genre_provider(self) -> str:
        """LLM provider for the genre enrichment pipeline ('claude_code' or 'openai')."""
        return (self.config.get('ai_genre') or {}).get('provider', 'claude_code')

    @property
    def ai_genre_claude_model(self) -> str:
        """Claude model alias for the claude_code provider (e.g. 'haiku', 'sonnet')."""
        return (self.config.get('ai_genre') or {}).get('claude_model', 'haiku')

    @property
    def lastfm_api_key(self) -> str:
        """Get Last.FM API key (with environment variable override)"""
        if 'lastfm' not in self.config or 'api_key' not in self.config.get('lastfm', {}):
            return ''
        return os.getenv('LASTFM_API_KEY') or self.config['lastfm']['api_key']

    @property
    def lastfm_username(self) -> str:
        """Get Last.FM username (with environment variable override)"""
        if 'lastfm' not in self.config or 'username' not in self.config.get('lastfm', {}):
            return ''
        return os.getenv('LASTFM_USERNAME') or self.config['lastfm']['username']

    @property
    def lastfm_history_days(self) -> int:
        """Get Last.FM history days"""
        return self.config.get('lastfm', {}).get('history_days', 90)

    @property
    def lastfm_recheck_miss_days(self) -> int:
        """Days to skip a Last.fm album that returned no tags before retrying.

        The lastfm stage records a 'miss' for releases Last.fm has no tags for
        and skips them on reruns until this many days have passed (then it
        retries once, since tags accrue over time). 0 (or negative) disables the
        cache and re-fetches every miss every run (the legacy behavior).
        """
        return int(self.config.get('lastfm', {}).get('recheck_miss_days', 30))

    def get_ds_artifact_path(self) -> str:
        """Get DS pipeline artifact path."""
        return self.config.get('playlists', {}).get('ds_pipeline', {}).get(
            'artifact_path',
            'data/artifacts/beat3tower_32k/data_matrices_step1.npz',
        )

    def get_ds_genre_source(self) -> str:
        """Get artifact-build genre source. Legacy is the backward-compatible default."""
        return self._get_ds_pipeline("genre_source", default="legacy")

    def get_ds_genre_similarity_source(self) -> str:
        """Which generator produces the genre-similarity smoothing matrix at
        artifact-build time. cooccurrence (legacy Jaccard) is the default;
        graph derives it from the layered genre taxonomy."""
        raw = str(
            self._get_ds_pipeline("genre_similarity", "source", default="cooccurrence")
            or "cooccurrence"
        ).strip().lower()
        if raw not in {"cooccurrence", "graph"}:
            return "cooccurrence"
        return raw

    def get_ds_genre_graph_source(self) -> str:
        """Get runtime genre graph source. Legacy is the backward-compatible default."""
        raw = self._get_ds_pipeline("genre_graph", "source", default=None)
        if raw is None:
            raw = self.get_ds_genre_source()
            if raw == "enriched":
                raw = "legacy"
        source = str(raw or "legacy").strip().lower()
        if source not in {"legacy", "layered_shadow", "layered"}:
            return "legacy"
        return source

    @property
    def min_duration_minutes(self) -> int:
        """Get minimum playlist duration in minutes"""
        return self.config.get('playlists', {}).get('min_duration_minutes', 90)

    @property
    def min_track_duration_seconds(self) -> int:
        """Get minimum track duration in seconds (filter out short tracks)"""
        return self.config.get('playlists', {}).get(
            'min_track_duration_seconds', DEFAULT_MIN_TRACK_DURATION_SECONDS
        )

    @property
    def max_track_duration_seconds(self) -> int:
        """Get maximum track duration in seconds (filter out overly long tracks)"""
        return self.config.get('playlists', {}).get('max_track_duration_seconds', 720)

    @property
    def recently_played_filter_enabled(self) -> bool:
        """Check if recently played filtering is enabled"""
        return self.config.get('playlists', {}).get('recently_played_filter', {}).get('enabled', True)

    @property
    def recently_played_lookback_days(self) -> int:
        """Get recently played filter lookback window in days (0 = all history)"""
        return self.config.get('playlists', {}).get('recently_played_filter', {}).get('lookback_days', 0)

    @property
    def recently_played_min_playcount(self) -> int:
        """Get minimum playcount threshold for recently played filter"""
        return self.config.get('playlists', {}).get('recently_played_filter', {}).get('min_playcount_threshold', 0)

    @property
    def max_tracks_per_artist(self) -> int:
        """Get maximum tracks per artist in a playlist"""
        return self.config.get('playlists', {}).get('max_tracks_per_artist', 3)

    @property
    def artist_window_size(self) -> int:
        """Get window size for artist frequency limiting"""
        return self.config.get('playlists', {}).get('artist_window_size', 8)

    @property
    def max_artist_per_window(self) -> int:
        """Get maximum times an artist can appear in the window"""
        return self.config.get('playlists', {}).get('max_artist_per_window', 1)

    @property
    def min_seed_artist_ratio(self) -> float:
        """Get minimum ratio of seed artist tracks (e.g., 0.125 = 1/8)"""
        return self.config.get('playlists', {}).get('min_seed_artist_ratio', 0.125)

    # Dynamic mode settings
    @property
    def dynamic_sonic_ratio(self) -> float:
        """Get percentage of tracks from sonic similarity in dynamic mode"""
        return self.config.get('playlists', {}).get('dynamic_mode', {}).get('sonic_ratio', 0.6)

    @property
    def dynamic_genre_ratio(self) -> float:
        """Get percentage of tracks from genre matching in dynamic mode"""
        return self.config.get('playlists', {}).get('dynamic_mode', {}).get('genre_ratio', 0.4)

    # Similarity settings
    @property
    def similarity_min_threshold(self) -> float:
        """Get minimum similarity threshold for pairing (0.0-1.0)"""
        return self.config.get('playlists', {}).get('similarity', {}).get('min_threshold', 0.5)

    @property
    def similarity_artist_direct_match(self) -> float:
        """Get similarity score for direct artist match"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_direct_match', 0.9)

    @property
    def similarity_artist_shared_base(self) -> float:
        """Get base similarity score for shared similar artists"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_shared_base', 0.4)

    @property
    def similarity_artist_shared_increment(self) -> float:
        """Get score increment per common similar artist"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_shared_increment', 0.05)

    @property
    def similarity_artist_shared_max(self) -> float:
        """Get maximum similarity score for shared similar artists"""
        return self.config.get('playlists', {}).get('similarity', {}).get('artist_shared_max', 0.7)

    # Limits
    @property
    def limit_similar_tracks(self) -> int:
        """Get number of similar tracks to fetch per seed"""
        return self.config.get('playlists', {}).get('limits', {}).get('similar_tracks', 50)

    @property
    def limit_similar_artists(self) -> int:
        """Get number of similar artists to fetch"""
        return self.config.get('playlists', {}).get('limits', {}).get('similar_artists', 30)

    @property
    def limit_extension_base(self) -> int:
        """Get base number of tracks when extending playlist"""
        return self.config.get('playlists', {}).get('limits', {}).get('extension_base', 10)

    @property
    def limit_extension_increment(self) -> int:
        """Get additional tracks per extension attempt"""
        return self.config.get('playlists', {}).get('limits', {}).get('extension_increment', 20)

    # Instrumentation / artifact emission settings
    @property
    def emit_run_artifacts(self) -> bool:
        """Check if run artifact emission is enabled"""
        return self.config.get('playlists', {}).get('instrumentation', {}).get('emit_run_artifacts', False)

    @property
    def run_artifact_dir(self) -> str:
        """Get directory for run artifacts"""
        return self.config.get('playlists', {}).get('instrumentation', {}).get('artifact_dir', 'diagnostics/run_artifacts')

    # Title deduplication settings
    @property
    def title_dedupe_enabled(self) -> bool:
        """Check if title deduplication is enabled"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('enabled', True)

    @property
    def title_dedupe_threshold(self) -> int:
        """Get fuzzy match threshold for title deduplication (0-100)"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('threshold', 92)

    @property
    def title_dedupe_mode(self) -> str:
        """Get title deduplication mode ('strict' or 'loose')"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('mode', 'loose')

    @property
    def title_dedupe_short_title_min_len(self) -> int:
        """Get minimum title length for fuzzy matching (shorter titles require exact match)"""
        return self.config.get('playlists', {}).get('dedupe', {}).get('title', {}).get('short_title_min_len', 6)

    # ─────────────────────────────────────────────────────────────────────────
    # DS Pipeline Tuning Parameters
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ds_pipeline(self, *keys, default=None):
        """Helper to access nested ds_pipeline config values."""
        val = self.config.get('playlists', {}).get('ds_pipeline', {})
        for key in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(key, default)
            if val is default:
                return default
        return val

    @property
    def ds_embedding_sonic_components(self) -> int:
        """PCA dimensions for sonic features in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'sonic_components', default=32)

    @property
    def ds_embedding_genre_components(self) -> int:
        """PCA dimensions for genre features in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'genre_components', default=32)

    @property
    def ds_embedding_sonic_weight(self) -> float:
        """Weight for sonic similarity in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'sonic_weight', default=0.60)

    @property
    def ds_embedding_genre_weight(self) -> float:
        """Weight for genre similarity in hybrid embedding."""
        return self._get_ds_pipeline('embedding', 'genre_weight', default=0.40)

    @property
    def ds_candidate_similarity_floor(self) -> float:
        """Minimum sonic similarity for candidate pool."""
        return self._get_ds_pipeline('candidate_pool', 'similarity_floor', default=0.20)

    @property
    def ds_candidate_max_pool_size(self) -> int:
        """Maximum candidates before construction phase."""
        return self._get_ds_pipeline('candidate_pool', 'max_pool_size', default=1200)

    @property
    def ds_candidate_max_artist_fraction(self) -> float:
        """Max fraction of playlist from one artist."""
        return self._get_ds_pipeline('candidate_pool', 'max_artist_fraction', default=0.125)

    @property
    def ds_scoring_alpha(self) -> float:
        """Seed similarity weight in scoring."""
        return self._get_ds_pipeline('scoring', 'alpha', default=0.55)

    @property
    def ds_scoring_beta(self) -> float:
        """Transition similarity weight in scoring."""
        return self._get_ds_pipeline('scoring', 'beta', default=0.55)

    @property
    def ds_scoring_gamma(self) -> float:
        """Artist diversity bonus in scoring."""
        return self._get_ds_pipeline('scoring', 'gamma', default=0.04)

    @property
    def ds_scoring_alpha_schedule(self) -> str:
        """Alpha schedule: 'constant' or 'arc'."""
        return self._get_ds_pipeline('scoring', 'alpha_schedule', default='arc')

    @property
    def ds_scoring_alpha_start(self) -> float:
        """Alpha at playlist start (if arc schedule)."""
        return self._get_ds_pipeline('scoring', 'alpha_start', default=0.65)

    @property
    def ds_scoring_alpha_mid(self) -> float:
        """Alpha at playlist midpoint (if arc schedule)."""
        return self._get_ds_pipeline('scoring', 'alpha_mid', default=0.45)

    @property
    def ds_scoring_alpha_end(self) -> float:
        """Alpha at playlist end (if arc schedule)."""
        return self._get_ds_pipeline('scoring', 'alpha_end', default=0.60)

    @property
    def ds_scoring_arc_midpoint(self) -> float:
        """Where midpoint falls in playlist (0.0-1.0)."""
        return self._get_ds_pipeline('scoring', 'arc_midpoint', default=0.55)

    @property
    def ds_constraints_min_gap(self) -> int:
        """Minimum positions between same artist."""
        return self._get_ds_pipeline('constraints', 'min_gap', default=6)

    @property
    def ds_constraints_hard_floor(self) -> bool:
        """Reject (True) vs penalize (False) below floor."""
        return self._get_ds_pipeline('constraints', 'hard_floor', default=True)

    @property
    def ds_constraints_transition_floor(self) -> float:
        """Minimum acceptable transition similarity."""
        return self._get_ds_pipeline('constraints', 'transition_floor', default=0.20)

    @property
    def ds_constraints_center_transitions(self) -> bool:
        """Whether to center X_start/X_end matrices."""
        return self._get_ds_pipeline('constraints', 'center_transitions', default=True)

    def get_ds_tuning_dict(self) -> dict:
        """Return all DS tuning parameters as a dict for pipeline consumption."""
        return {
            'embedding': {
                'sonic_components': self.ds_embedding_sonic_components,
                'genre_components': self.ds_embedding_genre_components,
                'sonic_weight': self.ds_embedding_sonic_weight,
                'genre_weight': self.ds_embedding_genre_weight,
            },
            'candidate_pool': {
                'similarity_floor': self.ds_candidate_similarity_floor,
                'max_pool_size': self.ds_candidate_max_pool_size,
                'max_artist_fraction': self.ds_candidate_max_artist_fraction,
            },
            'genre_graph': {
                'source': self.get_ds_genre_graph_source(),
            },
            'genre_similarity': {
                'source': self.get_ds_genre_similarity_source(),
            },
            'genre_source': self.get_ds_genre_source(),
            'scoring': {
                'alpha': self.ds_scoring_alpha,
                'beta': self.ds_scoring_beta,
                'gamma': self.ds_scoring_gamma,
                'alpha_schedule': self.ds_scoring_alpha_schedule,
                'alpha_start': self.ds_scoring_alpha_start,
                'alpha_mid': self.ds_scoring_alpha_mid,
                'alpha_end': self.ds_scoring_alpha_end,
                'arc_midpoint': self.ds_scoring_arc_midpoint,
            },
            'constraints': {
                'min_gap': self.ds_constraints_min_gap,
                'hard_floor': self.ds_constraints_hard_floor,
                'transition_floor': self.ds_constraints_transition_floor,
                'center_transitions': self.ds_constraints_center_transitions,
            },
        }

    def __repr__(self) -> str:
        """String representation (hides sensitive data)"""
        return f"Config(database={self.library_database_path}, lastfm_user={self.lastfm_username})"


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_REL = "data/metadata.db"


def resolve_database_path(config: "Config | dict | None", anchor: "Path | None" = None) -> str:
    """Absolute metadata.db path from config — the single source of truth.

    Accepts a Config object or a plain dict (the two shapes at call sites).
    Reads library.database_path; an absolute value is used as-is, a relative
    value is resolved against the REPO ROOT (not the process cwd), and a
    missing/blank value falls back to <repo-root>/data/metadata.db. Always
    returns an absolute path string — never a bare relative path. This makes
    the DB location independent of where the process was launched, which is
    what lets a satellite clone (cwd != data root) read the real canonical DB.

    `anchor` optionally overrides the base a relative path is resolved
    against (e.g. a MixarcHome.anchor_dir for a wheel install); when omitted,
    relative paths resolve against _REPO_ROOT exactly as before.
    """
    if isinstance(config, dict):
        raw = (config.get("library") or {}).get("database_path")
    elif config is None:
        raw = None
    else:
        # Config, MergedConfig, or any object exposing get(section, key, default)
        try:
            raw = config.get("library", "database_path", default=None)
        except (TypeError, AttributeError):
            raw = None
    raw = (str(raw).strip() if raw else "") or _DEFAULT_DB_REL
    p = Path(raw)
    if not p.is_absolute():
        base = Path(anchor) if anchor is not None else _REPO_ROOT
        p = base / p
    return str(p.resolve())


# Example usage
if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    try:
        config = Config()
        logger.info(f"Configuration loaded successfully: {config}")
        logger.info(f"Database path: {config.library_database_path}")
        logger.info(f"Music directory: {config.library_music_directory}")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
