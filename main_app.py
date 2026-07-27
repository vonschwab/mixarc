# -*- coding: utf-8 -*-
"""
MixArc - Main Application
Automatically generates playlists using beat3tower sonic analysis and genre metadata
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import sys
import argparse

from src.logging_utils import configure_logging, add_logging_args, resolve_log_level, playlist_log_file
from src.console_output import (
    header, section, blank, bullet,
    PlaylistReport, BatchReport, print_startup_banner
)
from src.config_loader import Config, resolve_database_path
from src.local_library_client import LocalLibraryClient
from src.playlist_generator import PlaylistGenerator
from src.lastfm_client import LastFMClient
from src.track_matcher import TrackMatcher
from src.m3u_exporter import M3UExporter
from src.metadata_client import MetadataClient
from src.plex_exporter import PlexExporter
from src.playlist.request_models import GeneratePlaylistRequest

logger = logging.getLogger(__name__)


class PlaylistApp:
    """Main application orchestrator"""

    def __init__(self, config_path: str = "config.yaml", cohesion_mode_override: Optional[str] = None):
        # Load configuration
        self.config = Config(config_path)
        self.cohesion_mode_override = cohesion_mode_override

        # Get logger (logging should already be configured by main())
        self.logger = logging.getLogger(__name__)

        # Initialize library client
        self.logger.info("Initializing MixArc")
        self.library = LocalLibraryClient(db_path=resolve_database_path(self.config))

        # Initialize Last.FM client (history-only). Only create if credentials exist.
        self.lastfm = None
        if self.config.lastfm_api_key and self.config.lastfm_username:
            self.lastfm = LastFMClient(
                api_key=self.config.lastfm_api_key,
                username=self.config.lastfm_username,
                db_path=resolve_database_path(self.config),
            )
        else:
            self.logger.info("Last.FM credentials missing; skipping Last.FM client (history will be local-only).")

        # Use the same metadata DB path as the library client to ensure normalized columns are present
        self.matcher = TrackMatcher(
            self.library,
            library_id=None,
            db_path=resolve_database_path(self.config),
        )
        # Cosmetic export label only (M3U #EXTVARIANT tag + filename suffix) — SP-B
        # retired the sonic-space transform selector (resolve_sonic_variant); this
        # no longer selects a transform, it just surfaces the configured name.
        sonic_cfg = self.config.get('playlists', 'sonic', default={}) or {}
        self.sonic_variant = str(sonic_cfg.get("sim_variant") or "raw").strip().lower()

        # Initialize metadata database client (optional, for enhanced genre matching)
        self.metadata = None
        try:
            db_path = resolve_database_path(self.config)
            self.metadata = MetadataClient(db_path)
            self.logger.info(f"Metadata database connected: {db_path}")
        except Exception as e:
            self.logger.warning(f"Metadata database not available: {e}")
            self.logger.info("Dynamic mode will use local metadata as fallback")

        self.generator = PlaylistGenerator(
            self.library,
            self.config,
            lastfm_client=self.lastfm,
            track_matcher=self.matcher,
            metadata_client=self.metadata
        )

        # Initialize M3U exporter (always on with E:\PLAYLISTS fallback)
        export_path = self.config.get('playlists', 'm3u_export_path', default="E:\\PLAYLISTS")
        export_enabled = self.config.get('playlists', 'export_m3u', default=True)
        if not export_enabled:
            self.logger.warning("M3U export disabled in config; overriding to ensure playlists are written to disk.")

        self.m3u_exporter = None
        if export_path:
            try:
                self.m3u_exporter = M3UExporter(export_path)
                self.logger.info(f"M3U export enabled (always on): {export_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize M3U exporter at {export_path}: {e}")

        # Initialize Plex exporter (optional)
        plex_enabled = bool(self.config.get('plex', 'enabled', default=False))
        plex_base_url = self.config.get('plex', 'base_url', default=None)
        plex_token = os.getenv("PLEX_TOKEN") or self.config.get('plex', 'token', default=None)
        plex_section = self.config.get('plex', 'music_section', default=None)
        plex_verify_ssl = bool(self.config.get('plex', 'verify_ssl', default=True))
        plex_replace = bool(self.config.get('plex', 'replace_existing', default=True))
        plex_path_map = self.config.get('plex', 'path_map', default=None)

        self.plex_exporter = None
        if plex_enabled:
            if plex_base_url and plex_token:
                try:
                    self.plex_exporter = PlexExporter(
                        plex_base_url,
                        plex_token,
                        music_section=plex_section,
                        verify_ssl=plex_verify_ssl,
                        replace_existing=plex_replace,
                        path_map=plex_path_map,
                    )
                    self.logger.info("Plex export enabled")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Plex exporter: {e}")
            else:
                self.logger.warning("Plex export enabled but base_url/token not configured")

    def cleanup_old_playlists(self) -> int:
        """
        Delete old auto-generated playlists

        Returns:
            Number of playlists deleted
        """
        self.logger.info("Cleaning up old playlists")

        name_prefix = self.config.get('playlists', 'name_prefix', default='Auto:')
        max_age_days = self.config.get('playlists', 'max_age_days', default=14)

        # Get all auto-generated playlists
        playlists = self.library.get_playlists(name_prefix=name_prefix)

        if not playlists:
            self.logger.info("No auto-generated playlists found")
            return 0

        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=max_age_days)
        cutoff_timestamp = int(cutoff.timestamp())

        deleted_count = 0
        for playlist in playlists:
            # Check if playlist is too old
            created_at = playlist.get('created_at', 0)

            if created_at < cutoff_timestamp:
                playlist_id = playlist['id']
                playlist_title = playlist['title']

                try:
                    self.library.delete_playlist(playlist_id)
                    self.logger.info(f"Deleted old playlist: {playlist_title}")
                    deleted_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete playlist {playlist_title}: {e}")

        self.logger.info(f"Cleanup complete: {deleted_count} playlists deleted")
        return deleted_count

    def generate_playlists(self, dry_run: bool = False, dynamic: bool = False) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Main workflow: Generate new playlists

        Args:
            dry_run: If True, skip creating playlists and exporting M3U
            dynamic: If True, enable dynamic mode (mix sonic + genre-based discovery)

        Returns:
            Tuple of (created playlist metadata, raw playlist data for reporting)
        """
        self.logger.info("Starting playlist generation workflow")

        # Step 1: Clean up old playlists (skip in dry-run mode)
        if not dry_run:
            self.cleanup_old_playlists()
        else:
            self.logger.debug("Skipping cleanup (dry run mode)")

        # Step 2: Generate playlists with tracks and metadata
        playlist_count = self.config.get('playlists', 'count', default=3)
        playlists = self.generator.create_playlist_batch(
            playlist_count,
            dynamic=dynamic,
            cohesion_mode_override=self.cohesion_mode_override,
        )

        if not playlists:
            self.logger.warning("No playlists generated - insufficient data")
            return [], []

        # Step 3: Create playlists (skip in dry-run mode)
        name_prefix = self.config.get('playlists', 'name_prefix', default='Auto:')
        created_playlists = []

        for i, playlist_data in enumerate(playlists, 1):
            title = playlist_data['title']
            tracks = playlist_data['tracks']
            full_title = f"{name_prefix} {title}"
            total_duration_ms = sum(t.get('duration', 0) for t in tracks)
            duration_minutes = total_duration_ms / 1000 / 60

            if dry_run:
                self.logger.debug(f"Would create: {full_title} ({len(tracks)} tracks)")
                created_playlists.append({
                    'title': full_title,
                    'id': 'dry-run',
                    'track_count': len(tracks),
                    'duration_min': duration_minutes
                })
            else:
                # Export to M3U
                m3u_exported = False
                if self.m3u_exporter:
                    try:
                        m3u_path = self.m3u_exporter.export_playlist(full_title, tracks, self.library, sonic_variant=self.sonic_variant)
                        if m3u_path:
                            m3u_exported = True
                            self.logger.debug(f"Exported M3U: {m3u_path}")
                    except Exception as e:
                        self.logger.error(f"M3U export failed for '{full_title}': {e}")

                # Export to Plex
                plex_exported = False
                if self.plex_exporter:
                    try:
                        plex_key = self.plex_exporter.export_playlist(full_title, tracks)
                        if plex_key:
                            plex_exported = True
                            self.logger.debug(f"Exported to Plex: {full_title}")
                    except Exception as e:
                        self.logger.error(f"Plex export failed for '{full_title}': {e}")

                if m3u_exported or plex_exported:
                    created_playlists.append({
                        'title': full_title,
                        'id': f'm3u_{i}',
                        'track_count': len(tracks),
                        'duration_min': duration_minutes
                    })

        self.logger.info(f"Generation complete: {len(created_playlists)}/{len(playlists)} playlists")
        return created_playlists, playlists

    def run(self, dry_run: bool = False, dynamic: bool = False):
        """Run the application with beautiful batch report output."""
        try:
            playlist_count = self.config.get('playlists', 'count', default=3)
            batch_report = BatchReport(playlist_count)

            # Generate playlists
            created_playlists, raw_playlists = self.generate_playlists(dry_run=dry_run, dynamic=dynamic)

            if created_playlists:
                # Add each playlist to the batch report
                for created in created_playlists:
                    batch_report.add_created(
                        created['title'],
                        created['track_count'],
                        created.get('duration_min', 0)
                    )

                # Print the beautiful summary
                batch_report.print_summary()

            else:
                header("NO PLAYLISTS CREATED", "")
                section("POSSIBLE REASONS")
                bullet("Insufficient listening history")
                bullet("No matching tracks in library")
                bullet("Check logs for details")
                blank()

        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            header("ERROR", "Application failed")
            section("DETAILS")
            bullet(str(e))
            bullet("Check the log file for more details")
            blank()
            sys.exit(1)

    def run_single_artist(
        self,
        artist_name: str,
        track_count: int = 30,
        track_title: Optional[str] = None,
        dry_run: bool = False,
        dynamic: bool = False,
        verbose: bool = False,
        artist_only: bool = False,
        anchor_seed_ids: Optional[List[str]] = None,
    ):
        """
        Generate a single playlist for a specific artist.
        Uses PlaylistReport for beautiful, organized output.
        """
        try:
            self._configure_verbose_logging(verbose)

            # Create the report object to track everything
            playlist_title = f"Auto: {artist_name}"
            report = PlaylistReport(playlist_title, artist_name)
            report.dry_run = dry_run
            report.mode = self.cohesion_mode_override or "dynamic"

            # Log to file but keep console clean
            self.logger.info(f"Generating playlist for: {artist_name} (dry_run={dry_run})")

            playlist_data = self._generate_single_artist_playlist(
                artist_name,
                track_count,
                track_title,
                dynamic,
                dry_run,
                verbose,
                artist_only,
                anchor_seed_ids,
            )

            if not playlist_data:
                header("PLAYLIST GENERATION FAILED", f"Artist: {artist_name}")
                section("POSSIBLE REASONS")
                bullet("Artist not found in your library")
                bullet("Artist has too few tracks")
                bullet("No similar tracks available")
                blank()
                return

            # Populate the report
            report.set_tracks(playlist_data['tracks'])

            # Extract edge scores if available
            ds_report = playlist_data.get('ds_report', {})
            if ds_report:
                metrics = ds_report.get('metrics', {})
                if metrics:
                    report.set_edge_scores({
                        'Transition (T)': metrics.get('T_mean', 0),
                        'Sonic (S)': metrics.get('S_mean', 0),
                        'Genre (G)': metrics.get('G_mean', 0),
                    })

            # Export if not dry run
            if not dry_run:
                if self.m3u_exporter:
                    m3u_path = self.m3u_exporter.export_playlist(
                        playlist_title,
                        playlist_data['tracks'],
                        self.library,
                        sonic_variant=self.sonic_variant
                    )
                    if m3u_path:
                        report.export_path = str(m3u_path)
                        self.logger.info(f"Exported to M3U: {m3u_path}")

                if self.plex_exporter:
                    try:
                        plex_key = self.plex_exporter.export_playlist(playlist_title, playlist_data['tracks'])
                        if plex_key:
                            report.plex_exported = True
                            self.logger.info(f"Exported to Plex: {playlist_title}")
                    except Exception as e:
                        self.logger.error(f"Plex export failed: {e}")

            # Print the beautiful report
            report.print_full(show_all_tracks=verbose)

        except Exception as e:
            self.logger.error(f"Error creating playlist for {artist_name}: {e}", exc_info=True)
            header("ERROR", "Playlist generation failed")
            section("DETAILS")
            bullet(str(e))
            bullet("Check the log file for more details")
            blank()
            sys.exit(1)

    def run_single_genre(self, genre_name: str, track_count: int = 30, dry_run: bool = False, dynamic: bool = False, verbose: bool = False):
        """
        Generate a single playlist for a specific genre.
        Uses PlaylistReport for beautiful, organized output.
        """
        try:
            self._configure_verbose_logging(verbose)

            # Create the report object to track everything
            playlist_title = f"Auto: {genre_name.title()}"
            report = PlaylistReport(playlist_title, artist_name=None)
            report.dry_run = dry_run
            report.mode = self.cohesion_mode_override or "dynamic"

            # Log to file but keep console clean
            self.logger.info(f"Generating playlist for genre: {genre_name} (dry_run={dry_run})")

            playlist_data = self._generate_single_genre_playlist(
                genre_name,
                track_count,
                dynamic,
                dry_run,
                verbose,
            )

            if not playlist_data:
                header("PLAYLIST GENERATION FAILED", f"Genre: {genre_name}")
                section("POSSIBLE REASONS")
                bullet("Genre not found in your library")
                bullet("Genre has too few tracks (need at least 4)")
                bullet("Check suggestions above for similar genres")
                blank()
                return

            # Populate the report
            report.set_tracks(playlist_data['tracks'])

            # Extract edge scores if available
            ds_report = playlist_data.get('ds_report', {})
            if ds_report:
                metrics = ds_report.get('metrics', {})
                if metrics:
                    report.set_edge_scores({
                        'Transition (T)': metrics.get('T_mean', 0),
                        'Sonic (S)': metrics.get('S_mean', 0),
                        'Genre (G)': metrics.get('G_mean', 0),
                    })

            # Export if not dry run
            if not dry_run:
                if self.m3u_exporter:
                    m3u_path = self.m3u_exporter.export_playlist(
                        playlist_title,
                        playlist_data['tracks'],
                        self.library,
                        sonic_variant=self.sonic_variant
                    )
                    if m3u_path:
                        report.export_path = str(m3u_path)
                        self.logger.info(f"Exported to M3U: {m3u_path}")

                if self.plex_exporter:
                    try:
                        plex_key = self.plex_exporter.export_playlist(playlist_title, playlist_data['tracks'])
                        if plex_key:
                            report.plex_exported = True
                            self.logger.info(f"Exported to Plex: {playlist_title}")
                    except Exception as e:
                        self.logger.error(f"Plex export failed: {e}")

            # Print the beautiful report
            report.print_full(show_all_tracks=verbose)

        except Exception as e:
            self.logger.error(f"Error creating playlist for genre {genre_name}: {e}", exc_info=True)
            header("ERROR", "Playlist generation failed")
            section("DETAILS")
            bullet(str(e))
            bullet("Check the log file for more details")
            blank()
            sys.exit(1)

    def _configure_verbose_logging(self, verbose: bool) -> None:
        """Enable verbose logging for generator/similarity when requested."""
        if not verbose:
            return
        logging.getLogger('src.playlist_generator').setLevel(logging.DEBUG)
        logging.getLogger('src.similarity_calculator').setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers + self.logger.handlers:
            handler.setLevel(logging.DEBUG)
        self.logger.info("Verbose logging enabled")

    def _generate_single_artist_playlist(
        self,
        artist_name: str,
        track_count: int,
        track_title: Optional[str],
        dynamic: bool,
        dry_run: bool,
        verbose: bool,
        artist_only: bool,
        anchor_seed_ids: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        """Generate playlist data for a single artist."""
        return self.generator.create_playlist_for_artist(
            artist_name,
            track_count,
            track_title=track_title,
            dynamic=dynamic,
            dry_run=dry_run,
            verbose=verbose,
            cohesion_mode_override=self.cohesion_mode_override,
            artist_only=artist_only,
            anchor_seed_ids=anchor_seed_ids,
        )

    def _generate_single_genre_playlist(
        self,
        genre_name: str,
        track_count: int,
        dynamic: bool,
        dry_run: bool,
        verbose: bool,
    ) -> Optional[Dict[str, Any]]:
        """Generate playlist data for a single genre."""
        return self.generator.create_playlist_for_genre(
            genre_name,
            track_count,
            dynamic=dynamic,
            dry_run=dry_run,
            verbose=verbose,
            cohesion_mode_override=self.cohesion_mode_override,
        )


def _playlist_log_settings(config: Config) -> Tuple[bool, str, int]:
    """Resolve logging.playlist_logs.* from config, with the documented
    defaults when the block is absent (see
    docs/superpowers/specs/2026-07-02-per-playlist-logging-design.md).
    """
    cfg = config.get('logging', 'playlist_logs', default={}) or {}
    enabled = bool(cfg.get('enabled', True))
    directory = cfg.get('dir', 'logs/playlists')
    level_name = cfg.get('level', 'DEBUG')
    level = getattr(logging, str(level_name).upper(), logging.DEBUG)
    return enabled, directory, level


def main():
    """Entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate playlists based on listening history, a specific artist, or genre"
    )

    # Create mutually exclusive group for seed mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--artist",
        type=str,
        help="Generate a single playlist for a specific artist (e.g., --artist \"Radiohead\")"
    )
    mode_group.add_argument(
        "--genre",
        type=str,
        help='Generate a single playlist for a specific genre (e.g., --genre "new age")'
    )
    parser.add_argument(
        "--track",
        type=str,
        help="Optional: specify a seed track title for the artist (e.g., --track \"Life On Mars\")"
    )
    parser.add_argument(
        "--anchor-seed-ids",
        type=str,
        help="Comma-separated list of rating_key IDs to fix pier seeds (artist mode only).",
    )
    parser.add_argument(
        "--tracks",
        type=int,
        default=30,
        help="Number of tracks in the playlist (default: 30)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview playlists without creating them or exporting M3U files"
    )
    parser.add_argument(
        "--artist-only",
        action="store_true",
        help="Restrict DS pipeline to the requested artist only (no discovery)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging: show detailed DS transition metrics and constraint enforcement"
    )
    parser.add_argument(
        "--cohesion-mode",
        choices=["strict", "narrow", "dynamic", "discover"],
        help="Select cohesion mode (default from config playlists.cohesion_mode)",
    )

    # Genre and sonic similarity mode presets (simpler alternative to manual tuning)
    parser.add_argument(
        "--genre-mode",
        choices=["strict", "narrow", "dynamic", "discover", "off"],
        help="Genre similarity mode: strict (0.80 weight), narrow (0.65), dynamic (0.50), discover (0.35), off (sonic-only)",
    )
    parser.add_argument(
        "--sonic-mode",
        choices=["strict", "narrow", "dynamic", "discover", "off"],
        help="Sonic similarity mode: strict (0.85 weight), narrow (0.70), dynamic (0.50), discover (0.35), off (genre-only)",
    )
    parser.add_argument(
        "--pace-mode",
        choices=["strict", "narrow", "dynamic", "off"],
        help="Pace/rhythm mode: strict / narrow / dynamic (default) / off (no gate)",
    )
    parser.add_argument(
        "--mode",
        choices=["balanced", "tight", "exploratory", "sonic_only", "genre_only", "varied_sound", "sonic_thread"],
        help="Quick preset: balanced (default), tight (strict+strict), exploratory (discover+discover), sonic_only, genre_only, etc.",
    )

    parser.add_argument(
        "--audit-run",
        action="store_true",
        help="Write a detailed pier-bridge markdown audit report for this run (see playlists.ds_pipeline.pier_bridge.audit_run).",
    )
    parser.add_argument(
        "--audit-run-dir",
        type=str,
        help="Override run audit output directory (default: docs/run_audits).",
    )
    # --pb-backoff removed 2026-07: it set infeasible_handling.enabled, one of
    # the keys retired with the legacy per-segment relaxation ladder (corridor
    # Phase 1 Task 8). The corridor widening ladder (corridor_widen_*) is the
    # sole segment-level recovery mechanism and has no runtime flag.
    parser.add_argument(
        "--pb-experiment-bridge-scoring",
        action="store_true",
        help="Enable experimental pier-bridge segment scoring (dry-run/audit only).",
    )
    parser.add_argument(
        "--pb-experiment-bridge-min-weight",
        type=float,
        help="Experimental bridge score weight for min(sim_a, sim_b).",
    )
    parser.add_argument(
        "--pb-experiment-bridge-balance-weight",
        type=float,
        help="Experimental bridge score weight for balance between endpoints.",
    )
    parser.add_argument(
        "--pb-experiment-progress-arc",
        action="store_true",
        help="Enable progress-arc beam scoring for this run (sets pier_bridge.progress_arc).",
    )
    parser.add_argument(
        "--pb-experiment-progress-weight",
        type=float,
        help="Experimental progress score weight for beam search.",
    )
    parser.add_argument(
        "--pb-experiment-progress-shape",
        choices=["linear", "arc"],
        help="Experimental progress target shape for beam search.",
    )
    # Add standard logging arguments
    add_logging_args(parser)
    args = parser.parse_args()

    # Configure logging (once, before anything else)
    log_level = resolve_log_level(args)
    if args.verbose:
        log_level = 'DEBUG'
    # Default log file if not specified
    log_file = getattr(args, 'log_file', None) or 'playlist_generator.log'
    configure_logging(
        level=log_level,
        log_file=log_file,
        show_run_id=getattr(args, "show_run_id", False),
    )

    print_startup_banner()

    # Check for config file
    import os
    if not os.path.exists("config.yaml"):
        header("CONFIGURATION ERROR", "")
        section("MISSING FILE")
        bullet("config.yaml not found")
        blank()
        bullet("Please create config.yaml with your settings")
        bullet("See config.example.yaml for reference")
        blank()
        sys.exit(1)

    # Run application
    try:
        app = PlaylistApp(
            cohesion_mode_override=getattr(args, 'cohesion_mode', None),
        )

        # Apply genre/sonic mode presets if provided (CLI overrides config)
        from src.playlist.mode_presets import apply_mode_presets, resolve_quick_preset

        genre_mode = None
        sonic_mode = None

        # Check for quick preset first
        if getattr(args, "mode", None):
            genre_mode, sonic_mode = resolve_quick_preset(args.mode)
            logger.info(f"Using quick preset '{args.mode}': genre={genre_mode}, sonic={sonic_mode}")

        # Individual mode args override quick preset
        if getattr(args, "genre_mode", None):
            genre_mode = args.genre_mode
        if getattr(args, "sonic_mode", None):
            sonic_mode = args.sonic_mode
        pace_mode = getattr(args, "pace_mode", None)

        playlists_cfg = app.generator.config.config.setdefault('playlists', {})
        if genre_mode:
            playlists_cfg['genre_mode'] = genre_mode
        if sonic_mode:
            playlists_cfg['sonic_mode'] = sonic_mode
        if pace_mode:
            playlists_cfg['pace_mode'] = pace_mode
        if genre_mode or sonic_mode:
            apply_mode_presets(playlists_cfg)

        if (
            getattr(args, "pb_experiment_bridge_scoring", False)
            or getattr(args, "pb_experiment_bridge_min_weight", None) is not None
            or getattr(args, "pb_experiment_bridge_balance_weight", None) is not None
        ):
            ds_cfg = playlists_cfg.setdefault("ds_pipeline", {})
            pb_cfg = ds_cfg.setdefault("pier_bridge", {})
            experiments_cfg = pb_cfg.setdefault("experiments", {})
            bridge_cfg = experiments_cfg.setdefault("bridge_scoring", {})
            bridge_cfg["enabled"] = True
            if getattr(args, "pb_experiment_bridge_min_weight", None) is not None:
                bridge_cfg["min_weight"] = float(args.pb_experiment_bridge_min_weight)
            if getattr(args, "pb_experiment_bridge_balance_weight", None) is not None:
                bridge_cfg["balance_weight"] = float(args.pb_experiment_bridge_balance_weight)
        if (
            getattr(args, "pb_experiment_progress_arc", False)
            or getattr(args, "pb_experiment_progress_weight", None) is not None
            or getattr(args, "pb_experiment_progress_shape", None) is not None
        ):
            ds_cfg = playlists_cfg.setdefault("ds_pipeline", {})
            pb_cfg = ds_cfg.setdefault("pier_bridge", {})
            progress_cfg = pb_cfg.setdefault("progress_arc", {})
            progress_cfg["enabled"] = True
            if getattr(args, "pb_experiment_progress_weight", None) is not None:
                progress_cfg["weight"] = float(args.pb_experiment_progress_weight)
            if getattr(args, "pb_experiment_progress_shape", None) is not None:
                progress_cfg["shape"] = str(args.pb_experiment_progress_shape)

        # Runtime flags (do not change defaults unless enabled)
        if getattr(args, "audit_run", False):
            app.generator._audit_run_enabled = True
        if getattr(args, "audit_run_dir", None):
            app.generator._audit_run_dir = str(getattr(args, "audit_run_dir"))
        dynamic_flag = getattr(args, "cohesion_mode", None) == "dynamic"
        if args.dry_run:
            section("DRY RUN MODE")
            bullet("No playlists will be created or exported")
            blank()
        generation_request = GeneratePlaylistRequest.from_cli_args(
            args,
            genre_mode=genre_mode,
            sonic_mode=sonic_mode,
            pace_mode=pace_mode,
        )

        # Per-playlist DEBUG log file for this CLI run (independent of the
        # explicit --log-file / configure_logging file, which keeps working
        # unchanged). One CLI invocation = one generation = one log.
        log_enabled, log_dir, log_level = _playlist_log_settings(app.config)

        if generation_request.mode == "artist" and generation_request.artist:
            # Single artist mode
            with playlist_log_file(
                generation_request.artist, None,
                enabled=log_enabled, dir=log_dir, level=log_level,
            ):
                app.run_single_artist(
                    generation_request.artist,
                    generation_request.tracks,
                    track_title=generation_request.track,
                    dry_run=getattr(args, 'dry_run', False),
                    dynamic=dynamic_flag,
                    verbose=getattr(args, 'verbose', False),
                    artist_only=generation_request.artist_only,
                    anchor_seed_ids=generation_request.anchor_seed_ids or None,
                )
        elif generation_request.mode == "genre" and generation_request.genre:
            # Single genre mode
            with playlist_log_file(
                generation_request.genre, None,
                enabled=log_enabled, dir=log_dir, level=log_level,
            ):
                app.run_single_genre(
                    generation_request.genre,
                    generation_request.tracks,
                    dry_run=getattr(args, 'dry_run', False),
                    dynamic=dynamic_flag,
                    verbose=getattr(args, 'verbose', False),
                )
        else:
            # Normal mode - generate multiple playlists from history
            with playlist_log_file(
                "history", None,
                enabled=log_enabled, dir=log_dir, level=log_level,
            ):
                app.run(dry_run=getattr(args, 'dry_run', False), dynamic=dynamic_flag)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease check your config.yaml file.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
