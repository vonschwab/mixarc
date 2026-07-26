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
