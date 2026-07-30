import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { GenerateControls } from "./GenerateControls";
import { api } from "../lib/api";

afterEach(() => { cleanup(); localStorage.clear(); vi.restoreAllMocks(); });

function renderControls(props: Partial<Parameters<typeof GenerateControls>[0]> = {}) {
  return render(
    <GenerateControls
      mode="artist"
      onModeChange={() => {}}
      seedTrackIds={[]}
      seedDisplays={[]}
      onSubmit={() => {}}
      busy={false}
      {...props}
    />,
  );
}

describe("GenerateControls disclosure", () => {
  it("renders a collapsed advanced region with a More controls toggle", () => {
    renderControls();
    const toggle = screen.getByTestId("more-controls");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // Collapsed: advanced region carries the container-query hide class.
    expect(screen.getByTestId("advanced-controls").className).toContain("@max-md:hidden");
  });

  it("expands and collapses when the toggle is clicked", () => {
    renderControls();
    const toggle = screen.getByTestId("more-controls");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByTestId("advanced-controls").className).not.toContain("@max-md:hidden");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByTestId("advanced-controls").className).toContain("@max-md:hidden");
  });
});

describe("GenerateControls genre mode", () => {
  it("shows genre autocomplete suggestions in genre mode and picking one fills the input with the canonical name", async () => {
    vi.spyOn(api, "genresSearch").mockResolvedValue({
      items: [{ genre_id: "shoegaze", name: "shoegaze" }],
    });
    renderControls({ mode: "genre" });
    const input = screen.getByPlaceholderText("Genre…") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "shoe" } });
    await waitFor(() => expect(api.genresSearch).toHaveBeenCalled());
    const suggestion = await screen.findByText("shoegaze");
    fireEvent.click(suggestion);
    // The picked canonical name is what ultimately feeds GenerateRequestBody.genre —
    // asserting only that the suggestion rendered would pass even if onPick were wired
    // to a no-op.
    expect(input.value).toBe("shoegaze");
  });

  it("hides the artist Style popover in genre mode", () => {
    renderControls({ mode: "genre" });
    expect(screen.queryByText(/style/i)).toBeNull();
  });

  it("Enter submits when no genre suggestions are open", async () => {
    vi.spyOn(api, "genresSearch").mockResolvedValue({ items: [] });
    const onSubmit = vi.fn();
    renderControls({ mode: "genre", onSubmit });
    const input = screen.getByPlaceholderText("Genre…");
    fireEvent.change(input, { target: { value: "shoegaze" } });
    await waitFor(() => expect(api.genresSearch).toHaveBeenCalled());
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("Enter picks the top suggestion instead of submitting when suggestions are open", async () => {
    vi.spyOn(api, "genresSearch").mockResolvedValue({
      items: [{ genre_id: "shoegaze", name: "shoegaze" }],
    });
    const onSubmit = vi.fn();
    renderControls({ mode: "genre", onSubmit });
    const input = screen.getByPlaceholderText("Genre…") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "shoe" } });
    await screen.findByText("shoegaze"); // suggestion dropdown open
    fireEvent.keyDown(input, { key: "Enter" });
    expect(input.value).toBe("shoegaze");
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe("GenerateControls New Seeds button", () => {
  it("renders the New Seeds button in artist mode", () => {
    renderControls({ mode: "artist" });
    const button = screen.queryByTitle("Re-roll: same settings, fresh seed tracks.");
    expect(button).not.toBeNull();
    expect(button?.textContent).toContain("↻ New Seeds");
  });

  it("renders the New Seeds button in genre mode", () => {
    renderControls({ mode: "genre" });
    const button = screen.queryByTitle("Re-roll: same settings, fresh seed tracks.");
    expect(button).not.toBeNull();
    expect(button?.textContent).toContain("↻ New Seeds");
  });

  it("does not render the New Seeds button in seeds mode", () => {
    renderControls({ mode: "seeds" });
    const button = screen.queryByTitle("Re-roll: same settings, fresh seed tracks.");
    expect(button).toBeNull();
  });

  it("clicking the New Seeds button in genre mode submits with incremented seed_epoch", () => {
    const onSubmit = vi.fn();
    renderControls({ mode: "genre", onSubmit });

    // First click should submit with epoch 1
    const button = screen.queryByTitle("Re-roll: same settings, fresh seed tracks.");
    expect(button).not.toBeNull();
    fireEvent.click(button!);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const firstCall = onSubmit.mock.calls[0][0];
    expect(firstCall.seed_epoch).toBe(1);

    // Second click should submit with epoch 2
    fireEvent.click(button!);
    expect(onSubmit).toHaveBeenCalledTimes(2);
    const secondCall = onSubmit.mock.calls[1][0];
    expect(secondCall.seed_epoch).toBe(2);
  });
});

describe("GenerateControls multi-artist blend", () => {
  it("sends the artists list when a second artist chip is added", () => {
    const onSubmit = vi.fn();
    renderControls({ mode: "artist", onSubmit });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i }));
    fireEvent.change(screen.getByPlaceholderText(/second artist/i), { target: { value: "Harold Budd" } });
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    const payload = onSubmit.mock.calls[0][0];
    expect(payload.artist).toBe("Brian Eno");
    expect(payload.artists).toEqual(["Brian Eno", "Harold Budd"]);
  });

  it("omits artists entirely for a single artist (wire payload unchanged)", () => {
    const onSubmit = vi.fn();
    renderControls({ mode: "artist", onSubmit });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    const payload = onSubmit.mock.calls[0][0];
    // JSON.stringify (api.generate) drops undefined-valued keys, so this
    // matches the pre-feature wire payload byte-for-byte.
    expect(payload.artists).toBeUndefined();
  });

  it("removing an extra artist chip drops it from the artists list", () => {
    const onSubmit = vi.fn();
    renderControls({ mode: "artist", onSubmit });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i }));
    fireEvent.change(screen.getByPlaceholderText(/second artist/i), { target: { value: "Harold Budd" } });
    fireEvent.click(screen.getByRole("button", { name: /remove artist 2/i }));
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    const payload = onSubmit.mock.calls[0][0];
    expect(payload.artists).toBeUndefined();
  });

  it("blank extra-artist chips are stripped before sending", () => {
    const onSubmit = vi.fn();
    renderControls({ mode: "artist", onSubmit });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i }));
    fireEvent.change(screen.getByPlaceholderText(/second artist/i), { target: { value: "   " } });
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    const payload = onSubmit.mock.calls[0][0];
    expect(payload.artists).toBeUndefined();
  });

  it("shows autocomplete suggestions on an artist chip and picking one fills the chip and reaches the payload", async () => {
    // Scoped to a "har" prefix so the primary artist input's own autocomplete
    // (query "Brian Eno", same mocked endpoint) doesn't also surface "Harold
    // Budd" and create an ambiguous duplicate match for screen.findByText.
    vi.spyOn(api, "autocomplete").mockImplementation(async (q: string) => ({
      items: q.toLowerCase().startsWith("har") ? ["Harold Budd"] : [],
      has_more: false,
    }));
    const onSubmit = vi.fn();
    renderControls({ mode: "artist", onSubmit });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i }));
    const chipInput = screen.getByPlaceholderText(/second artist/i) as HTMLInputElement;
    fireEvent.focus(chipInput);
    fireEvent.change(chipInput, { target: { value: "Har" } });
    await waitFor(() => expect(api.autocomplete).toHaveBeenCalled());

    const suggestion = await screen.findByText("Harold Budd");
    fireEvent.click(suggestion);
    expect(chipInput.value).toBe("Harold Budd");
    // Picking a chip suggestion must not immediately reopen the dropdown.
    expect(screen.queryByText("Harold Budd", { selector: "li" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /generate/i }));
    const payload = onSubmit.mock.calls[0][0];
    expect(payload.artists).toEqual(["Brian Eno", "Harold Budd"]);
  });

  it("refocusing an already-picked chip does not reopen its dropdown or refire the search", async () => {
    // Scoped to a "har" prefix so the primary artist input's own autocomplete
    // can't also surface "Harold Budd" and create an ambiguous duplicate match.
    const autocompleteMock = vi.spyOn(api, "autocomplete").mockImplementation(async (q: string) => ({
      items: q.toLowerCase().startsWith("har") ? ["Harold Budd"] : [],
      has_more: false,
    }));
    renderControls({ mode: "artist" });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i })); // chip 0 ("Second artist…")
    fireEvent.click(screen.getByRole("button", { name: /add artist/i })); // chip 1 ("Another artist…")
    const chip0 = screen.getByPlaceholderText(/second artist/i) as HTMLInputElement;
    const chip1 = screen.getByPlaceholderText(/another artist/i) as HTMLInputElement;

    // Pick a suggestion into chip 0.
    fireEvent.focus(chip0);
    fireEvent.change(chip0, { target: { value: "Har" } });
    const suggestion = await screen.findByText("Harold Budd");
    fireEvent.click(suggestion);
    expect(chip0.value).toBe("Harold Budd");
    const callsAfterPick = autocompleteMock.mock.calls.length;

    // Focus away to chip 1, then refocus chip 0 without editing it.
    fireEvent.focus(chip1);
    fireEvent.focus(chip0);

    // Give the hook's debounce window a chance to fire if it were going to.
    await new Promise((resolve) => setTimeout(resolve, 250));

    expect(screen.queryByText("Harold Budd", { selector: "li" })).toBeNull();
    expect(autocompleteMock.mock.calls.length).toBe(callsAfterPick);
  });

  it("removing an earlier chip re-targets the active search to the same logical chip, not a stale index", async () => {
    const autocompleteMock = vi.spyOn(api, "autocomplete").mockImplementation(async (q: string) => ({
      items: q ? [`${q} Result`] : [],
      has_more: false,
    }));
    renderControls({ mode: "artist" });

    fireEvent.change(screen.getByPlaceholderText("Artist name…"), { target: { value: "Brian Eno" } });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i })); // chip 0
    fireEvent.click(screen.getByRole("button", { name: /add artist/i })); // chip 1
    const chip0 = screen.getByPlaceholderText(/second artist/i) as HTMLInputElement;
    const chip1 = screen.getByPlaceholderText(/another artist/i) as HTMLInputElement;
    expect(chip0.value).toBe(""); // chip 0 stays untouched — it's what gets removed below

    // Focus chip 1 (index 1) and start typing — it owns the shared search.
    fireEvent.focus(chip1);
    fireEvent.change(chip1, { target: { value: "Budd" } });
    await screen.findByText("Budd Result");

    // Remove chip 0 (index 0). Chip 1's content ("Budd") is now at index 0;
    // a stale `activeChip === 1` would re-read the now-blank index 1 (or an
    // out-of-range index) and silently retarget/drop the search instead of
    // following the surviving chip.
    fireEvent.click(screen.getByRole("button", { name: /remove artist 2/i }));

    // Exactly one chip remains, and it still holds what was typed — the
    // removal did not clear or misdirect the in-progress chip.
    const remainingInputs = screen.getAllByPlaceholderText(/second artist|another artist/i) as HTMLInputElement[];
    expect(remainingInputs).toHaveLength(1);
    expect(remainingInputs[0].value).toBe("Budd");

    // The shared search must still be live for THIS (now re-indexed) chip —
    // not stuck pointing at the removed or an out-of-range index. Typing
    // further and getting a matching result proves it re-targeted correctly
    // rather than silently going stale.
    fireEvent.change(remainingInputs[0], { target: { value: "Buddy" } });
    await screen.findByText("Buddy Result");
    expect(autocompleteMock.mock.calls.some(([q]) => q === "Buddy")).toBe(true);
  });

  it("clears extra artist chips when leaving artist mode", () => {
    const { rerender } = renderControls({ mode: "artist" });
    fireEvent.click(screen.getByRole("button", { name: /add artist/i }));
    expect(screen.queryByPlaceholderText(/second artist/i)).not.toBeNull();

    rerender(
      <GenerateControls
        mode="genre"
        onModeChange={() => {}}
        seedTrackIds={[]}
        seedDisplays={[]}
        onSubmit={() => {}}
        busy={false}
      />,
    );
    rerender(
      <GenerateControls
        mode="artist"
        onModeChange={() => {}}
        seedTrackIds={[]}
        seedDisplays={[]}
        onSubmit={() => {}}
        busy={false}
      />,
    );

    expect(screen.queryByPlaceholderText(/second artist/i)).toBeNull();
  });
});
