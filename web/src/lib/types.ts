export type Mode = "artist" | "genre" | "seeds" | "history";

export type RangeDial = "home" | "close" | "open" | "wander";
export type FlowDial = "normal" | "smooth";
export type PaceDial = "locked_in" | "natural" | "free";

// GET /api/setup/status shape (Task 4/5). Shared here (rather than kept local
// to SetupPage) so lib/api.ts can wrap the endpoint like every other call.
export interface SetupStatus {
  state: "needs_setup" | "needs_analyze" | "ready";
  config_path: string;
  config_exists: boolean;
  music_directory: string | null;
  db_path: string | null;
  track_count: number | null;
  detail: string;
  checks?: CheckResult[];
}

// Result of a single setup-wizard check (env/DB/service probe).
export interface CheckResult {
  id: string;
  status: "pass" | "warn" | "fail";
  summary: string;
  fix_hint: string | null;
}

// GET /api/setup/browse — one entry in a directory listing.
export interface BrowseEntry {
  name: string;
  path: string;
  audio_count: number;
}

// GET /api/setup/browse response.
export interface BrowseResponse {
  path: string;
  parent: string | null;
  entries: BrowseEntry[];
  is_music_dir: boolean;
}

// POST /api/setup/config body — the wizard's accumulated draft.
export interface SetupConfigDraft {
  music_directory: string;
  lastfm?: { api_key: string; username: string };
  discogs?: { token: string };
  plex?: Record<string, unknown>;
  ai_genre_provider?: string;
  reconfigure?: boolean;
}

export interface GenerateRequestBody {
  mode: Mode;
  tracks: number;
  artist?: string;
  genre?: string;
  seed_tracks?: string[];
  seed_track_ids?: string[];
  range_dial?: RangeDial;
  flow_dial?: FlowDial;
  pace_dial?: PaceDial;
  include_collaborations?: boolean;
  popular_seeds_mode?: "off" | "on" | "fire";
  popularity_mode?: "off" | "on" | "oops";
  seed_epoch?: number;
  exclude_seed_tracks_from_recency?: boolean;
  recency_enabled?: boolean;
  instrumental?: boolean;
  recency_days?: number;
  recency_plays_threshold?: number;
  artist_spacing?: string;
  diversity_gamma?: number;
  artist_diversity_mode?: string;
  artist_presence?: string;
  artist_variety?: string;
  steering_tags?: string[];
}

export interface SeedTrack {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genres: string[];
  duration_ms: number;
  file_path: string;
}

export interface TrackOut {
  position: number;
  rating_key?: string;
  artist: string;
  title: string;
  album: string;
  duration_ms: number;
  file_path: string;
  sonic_similarity?: number | null;
  genre_similarity?: number | null;
  transition_score?: number | null;
  popularity_rank?: number | null;
  genres: string[];
}

export interface MetricsOut {
  mean_transition?: number | null;
  min_transition?: number | null;
  p10_transition?: number | null;
  p90_transition?: number | null;
  distinct_artists?: number | null;
}

export interface RelaxationEntry {
  segment_index?: number;
  bridge: string;
  relaxed: string[];
  severity: string;
}

export interface Receipt {
  range: { pool: number | null; considered: number | null };
  flow: { worst: number | null; mean: number | null };
  pace: { bpm_mean: number | null; bpm_std: number | null; n: number | null; total: number | null };
  notes: string[];
}

export interface PlaylistOut {
  name: string;
  track_count: number;
  tracks: TrackOut[];
  metrics: MetricsOut;
  relaxations?: RelaxationEntry[];
  receipt?: Receipt | null;
  /** Hard-invariant breaches (artist gap, per-artist cap). Distinct from
   *  `relaxations`: a relaxation is the generator bending a guideline on
   *  purpose; `degraded` means a stated guarantee was broken. */
  warnings?: string[];
  degraded?: boolean;
}

export interface JobOut {
  job_id: string;
  status: "pending" | "running" | "success" | "failed" | "cancelled";
  stage: string;
  error?: string | null;
  playlist?: PlaylistOut | null;
  tool_result?: Record<string, unknown> | null;
  created_at?: number | null;
  request_params?: Record<string, unknown> | null;
}

export interface WsEvent {
  type: "log" | "progress" | "result" | "error" | "done";
  job_id?: string;
  [k: string]: unknown;
}

export interface CandidateOut {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genres: string[];
  fit_score: number;
  // Identity the exporters resolve on — must be stamped onto the replaced track,
  // or Plex/M3U export keeps the old track's file_path.
  file_path: string;
  duration_ms: number;
}

export interface ReplaceSuggestionsResponse {
  position: number;
  candidates: CandidateOut[];
}

export interface BlacklistRequest {
  track_ids?: string[];
  scope?: "album" | "artist";
  value?: string;
  artist?: string;
  enabled?: boolean;
}

export interface EditGenresRequest {
  artist: string;
  album: string;
  genres: string[];
  // What the dialog displayed when opened (graph authority). The backend
  // diffs `genres` against this to compute the add/remove override.
  base_genres: string[];
}

export interface CanonicalGenre {
  genre_id: string;
  name: string;
}

export interface EditGenresResponse {
  ok: boolean;
  resolved: string[];
  unknown: string[];
  added: string[];
  removed: string[];
  no_change: boolean;
}

export interface PlexExportRequest {
  title: string;
  tracks: TrackOut[];
}

export interface BlacklistEntry {
  scope: "artist" | "album" | "track";
  display_name: string;
  track_id?: string | null;
  artist?: string | null;
  album?: string | null;
}

export interface BlacklistFetchResponse {
  artists: BlacklistEntry[];
  albums: BlacklistEntry[];
  tracks: BlacklistEntry[];
  total: number;
}

export interface AnalyzeToolRequest {
  stages?: string[];
  force?: boolean;
  dry_run?: boolean;
}

export interface EnrichToolRequest {
  scope: "all_unenriched" | "artist" | "release";
  artist?: string;
  album?: string;
}

export interface ProposedGenre {
  term: string;
  confidence: number | null;
}

export interface EscalationOut {
  album_id: string;
  artist: string;
  album: string;
  prior_observed_leaf: string[];
  proposed_genres: ProposedGenre[];
  escalate_reason: string;
  dropped_file_tags: string[];
  status: "pending" | "accepted" | "edited" | "rejected";
  decision_genres: string[] | null;
}

export interface EscalationQueueResponse {
  escalations: EscalationOut[];
  pending_albums: number;
  decided_albums: number;
}

export interface EscalationDecisionRequest {
  album_id: string;
  decision: "accept" | "edit" | "reject" | "revert";
  genres?: string[];
}

// ── Taxonomy term adjudication (vocabulary-level review) ──────────────────────
export interface TaxonomyParentEdge {
  target: string;
  edge_type: string;
  weight: number;
  confidence: number;
}

export interface TaxonomyProposal {
  name?: string;
  kind?: string;
  status?: string;
  specificity_score?: number;
  parent_edges?: TaxonomyParentEdge[];
  similar_to?: string[];
  alias_variants?: string[];
  term_kind_confirm?: string;
  rationale?: string;
  facet_type?: string | null;
  canonical_target?: string | null;
  reject_reason?: string; // present on reject verdicts
}

export interface TaxonomyStagedDecision {
  verdict: "add" | "alias" | "reject";
  status?: string; // pending | applied
}

export interface TaxonomyQueueItem {
  term: string;
  raw_term: string;
  album_frequency: number;
  cooccurring_tags: string[];
  examples: string[];
  variants: string[];
  source: string;
  decision: TaxonomyStagedDecision | null;
}

export interface TaxonomyQueueResponse {
  terms: TaxonomyQueueItem[];
  untriaged_terms: number;
  decided_terms: number;
}

export interface TaxonomyVerdict {
  ok: boolean;
  verdict: "add" | "alias" | "reject";
  term: string;
  proposal: TaxonomyProposal;
}

export interface TaxonomyDecisionRequest {
  term: string;
  raw_term?: string;
  verdict: "add" | "alias" | "reject" | "revert";
  proposal?: TaxonomyProposal | null;
  claude?: TaxonomyProposal | null;
  human_edited?: boolean;
}

export interface Page<T> {
  items: T[];
  has_more: boolean;
}

export interface ArtistLinkGroup {
  type: "alias" | "sibling";
  members: string[];
}
export interface ArtistLinksListResponse {
  groups: ArtistLinkGroup[];
}
export interface ArtistLinksSaveRequest {
  groups: ArtistLinkGroup[];
}
export interface ArtistSearchResponse {
  items: string[];
}
