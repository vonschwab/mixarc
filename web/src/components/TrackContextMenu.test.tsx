import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { TrackContextMenu, type MenuTarget } from "./TrackContextMenu";
import type { TrackOut } from "../lib/types";

afterEach(() => cleanup());

const track: TrackOut = {
  position: 1,
  rating_key: "t1",
  artist: "Acetone",
  title: "Sundown",
  album: "York Blvd",
  duration_ms: 1000,
  file_path: "/x.mp3",
  genres: [],
};
const target: MenuTarget = { track, index: 1, isPier: false };
const noop = () => {};

describe("TrackContextMenu copy", () => {
  it("labels the single-track action in plain words (no 'Track(s)')", () => {
    render(
      <TrackContextMenu
        open
        onOpenChange={noop}
        target={target}
        pos={{ x: 0, y: 0 }}
        onReplace={noop}
        onBlacklistTrack={noop}
        onBlacklistAlbum={noop}
        onBlacklistArtist={noop}
        onMarkAlbumLive={noop}
        onEditGenres={noop}
      />,
    );
    // S3: computed copy, never "1 Track(s)".
    expect(screen.getByText("Blacklist this track")).toBeTruthy();
    expect(screen.queryByText(/Track\(s\)/)).toBeNull();
  });
});
