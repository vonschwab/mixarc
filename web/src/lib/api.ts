import type {
  AnalyzeToolRequest,
  ArtistLinksListResponse,
  ArtistLinksSaveRequest,
  ArtistSearchResponse,
  BlacklistFetchResponse,
  BlacklistRequest,
  BrowseResponse,
  CanonicalGenre,
  CheckResult,
  EditGenresRequest,
  EditGenresResponse,
  EnrichToolRequest,
  GenerateRequestBody,
  JobOut,
  LiveAlbumsResponse,
  Page,
  PlexExportRequest,
  EscalationDecisionRequest,
  EscalationQueueResponse,
  ReplaceSuggestionsResponse,
  SeedTrack,
  SetupConfigDraft,
  SetupStatus,
  TaxonomyDecisionRequest,
  TaxonomyProposal,
  TaxonomyQueueResponse,
} from "./types";

// Thrown by jsonOrThrow on a non-ok response. `status` lets callers branch on
// the HTTP status code (e.g. 409 conflict) instead of substring-matching the
// message text, which couples them to the server's exact wording.
export type ApiError = Error & { status?: number };

async function jsonOrThrow(resp: Response) {
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    const err: ApiError = new Error(body.detail || `HTTP ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

export const api = {
  async getSetupStatus(): Promise<SetupStatus> {
    return jsonOrThrow(await fetch("/api/setup/status"));
  },
  async generate(body: GenerateRequestBody): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }));
  },
  async job(id: string): Promise<JobOut> {
    return jsonOrThrow(await fetch(`/api/jobs/${id}`));
  },
  async jobs(): Promise<JobOut[]> {
    return jsonOrThrow(await fetch("/api/jobs"));
  },
  async clearJobs(): Promise<{ cleared: number }> {
    return jsonOrThrow(await fetch("/api/jobs", { method: "DELETE" }));
  },
  async autocomplete(q: string, offset = 0, limit = 30): Promise<Page<string>> {
    const params = new URLSearchParams({ q, offset: String(offset), limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/autocomplete?${params}`));
  },
  async searchTracks(q: string, offset = 0, limit = 25): Promise<Page<SeedTrack>> {
    const params = new URLSearchParams({ q, offset: String(offset), limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/tracks/search?${params}`));
  },
  async trackGenres(trackIds: string[]): Promise<Record<string, string[]>> {
    return jsonOrThrow(await fetch("/api/tracks/genres", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_ids: trackIds }),
    }));
  },
  async replaceSuggestions(jobId: string, position: number, topK = 10): Promise<ReplaceSuggestionsResponse> {
    return jsonOrThrow(await fetch("/api/replace_suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, position, top_k: topK }),
    }));
  },
  async blacklist(req: BlacklistRequest): Promise<{ ok: boolean; updated?: number }> {
    return jsonOrThrow(await fetch("/api/blacklist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async getBlacklist(): Promise<BlacklistFetchResponse> {
    return jsonOrThrow(await fetch("/api/blacklist"));
  },
  async liveAlbums(): Promise<LiveAlbumsResponse> {
    return jsonOrThrow(await fetch("/api/live-albums"));
  },
  async setLiveAlbum(req: { artist: string; album: string; enabled: boolean }): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch("/api/live-albums", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async blacklistArtist(artist: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch("/api/blacklist/artist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ artist }),
    }));
  },
  async cancelJob(jobId: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }));
  },
  async editGenres(req: EditGenresRequest): Promise<EditGenresResponse> {
    return jsonOrThrow(await fetch("/api/edit_genres", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async genresSearch(q: string, limit = 20): Promise<{ items: CanonicalGenre[] }> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/genres/search?${params}`));
  },
  async albumGenres(artist: string, album: string): Promise<{ genres: string[] }> {
    const params = new URLSearchParams({ artist, album });
    return jsonOrThrow(await fetch(`/api/genres/for_album?${params}`));
  },
  async artistGenres(artist: string): Promise<{ genres: { name: string; release_count: number; confidence: number }[] }> {
    const params = new URLSearchParams({ artist });
    return jsonOrThrow(await fetch(`/api/genres/for_artist?${params}`));
  },
  async refreshGenreArtifact(): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/refresh_genre_artifact", { method: "POST" }));
  },
  async exportPlex(req: PlexExportRequest): Promise<{ ok: boolean; playlist_key: string }> {
    return jsonOrThrow(await fetch("/api/export/plex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async analyzeLibrary(req: AnalyzeToolRequest): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/tools/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async enrich(req: EnrichToolRequest): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/tools/enrich", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async reviewQueue(search = "", limit = 50, offset = 0): Promise<EscalationQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/review/queue?${params}`));
  },
  async reviewCompleted(search = "", limit = 50, offset = 0): Promise<EscalationQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/review/completed?${params}`));
  },
  async reviewDecision(req: EscalationDecisionRequest): Promise<{ ok: boolean; status: string }> {
    return jsonOrThrow(await fetch("/api/review/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async reviewPublish(): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/review/publish", { method: "POST" }));
  },
  async taxonomyQueue(search = "", limit = 50, offset = 0): Promise<TaxonomyQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/taxonomy/queue?${params}`));
  },
  async taxonomyCompleted(search = "", limit = 50, offset = 0): Promise<TaxonomyQueueResponse> {
    const params = new URLSearchParams({ search, limit: String(limit), offset: String(offset) });
    return jsonOrThrow(await fetch(`/api/taxonomy/completed?${params}`));
  },
  async taxonomyAdjudicate(term: string): Promise<{ job_id: string }> {
    // Tracked job — poll api.job(job_id) for the verdict in tool_result.
    return jsonOrThrow(await fetch("/api/taxonomy/adjudicate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term }),
    }));
  },
  async taxonomyDecision(req: TaxonomyDecisionRequest): Promise<{ ok: boolean; status: string }> {
    return jsonOrThrow(await fetch("/api/taxonomy/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async taxonomyValidate(proposal: TaxonomyProposal): Promise<{ ok: boolean; errors: string[] }> {
    return jsonOrThrow(await fetch("/api/taxonomy/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proposal }),
    }));
  },
  async taxonomyApply(): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/taxonomy/apply", { method: "POST" }));
  },
  async artistLinksList(): Promise<ArtistLinksListResponse> {
    return jsonOrThrow(await fetch("/api/artists/links"));
  },
  async artistLinksSave(req: ArtistLinksSaveRequest): Promise<{ ok: boolean; count: number }> {
    return jsonOrThrow(await fetch("/api/artists/links/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async artistsSearch(q: string, limit = 20): Promise<ArtistSearchResponse> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return jsonOrThrow(await fetch(`/api/artists/search?${params}`));
  },
  async browseDir(path?: string): Promise<BrowseResponse> {
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    return jsonOrThrow(await fetch(`/api/setup/browse${q}`));
  },
  async testService(service: string, config: object): Promise<CheckResult> {
    return jsonOrThrow(await fetch(`/api/setup/test/${service}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    }));
  },
  async writeConfig(draft: SetupConfigDraft): Promise<{ ok: boolean; config_path: string }> {
    return jsonOrThrow(await fetch(`/api/setup/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(draft),
    }));
  },
};
