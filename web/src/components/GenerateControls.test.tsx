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
  it("shows genre autocomplete suggestions in genre mode", async () => {
    vi.spyOn(api, "genresSearch").mockResolvedValue({
      items: [{ genre_id: "shoegaze", name: "shoegaze" }],
    });
    renderControls({ mode: "genre" });
    fireEvent.change(screen.getByPlaceholderText("Genre…"), { target: { value: "shoe" } });
    await waitFor(() => expect(api.genresSearch).toHaveBeenCalled());
    expect(await screen.findByText("shoegaze")).toBeTruthy();
  });

  it("hides the artist Style popover in genre mode", () => {
    renderControls({ mode: "genre" });
    expect(screen.queryByText(/style/i)).toBeNull();
  });
});
