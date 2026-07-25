import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { DegradedNotice } from "./DegradedNotice";

afterEach(() => cleanup());

describe("DegradedNotice", () => {
  it("renders nothing when the playlist is clean", () => {
    const { container } = render(<DegradedNotice warnings={[]} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when warnings is undefined", () => {
    const { container } = render(<DegradedNotice warnings={undefined} />);
    expect(container.innerHTML).toBe("");
  });

  it("names the broken guarantee verbatim so it is actionable", () => {
    render(
      <DegradedNotice
        warnings={[
          "artist_gap_violation: artist=augustus pablo positions=[26, 27] configured_min_gap=6",
        ]}
      />,
    );
    // The raw validator string is shown as-is: the artist and the positions are
    // what make it fixable, so they must survive to the UI unparaphrased.
    expect(
      screen.getByText(
        "artist_gap_violation: artist=augustus pablo positions=[26, 27] configured_min_gap=6",
      ),
    ).toBeTruthy();
  });

  it("lists every warning, not just the first", () => {
    render(
      <DegradedNotice
        warnings={["artist_gap_violation: pablo", "artist_cap_violation: tubby"]}
      />,
    );
    expect(screen.getByText("artist_gap_violation: pablo")).toBeTruthy();
    expect(screen.getByText("artist_cap_violation: tubby")).toBeTruthy();
  });

  it("is not dismissible — a broken guarantee must not be hideable", () => {
    render(<DegradedNotice warnings={["artist_gap_violation: pablo"]} />);
    expect(screen.queryByLabelText("Dismiss")).toBeNull();
  });

  it("is announced to assistive tech", () => {
    render(<DegradedNotice warnings={["artist_gap_violation: pablo"]} />);
    expect(screen.getByRole("alert")).toBeTruthy();
  });
});
