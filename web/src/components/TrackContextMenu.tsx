import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import type { TrackOut } from "../lib/types";

export interface MenuTarget {
  track: TrackOut;
  index: number;
  isPier: boolean;
}

export interface TrackContextMenuProps {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  target: MenuTarget | null;
  pos: { x: number; y: number };
  onReplace: (t: MenuTarget) => void;
  onBlacklistTrack: (t: MenuTarget) => void;
  onBlacklistAlbum: (t: MenuTarget) => void;
  onBlacklistArtist: (t: MenuTarget) => void;
  onMarkAlbumLive: (t: MenuTarget) => void;
  onEditGenres: (t: MenuTarget) => void;
}

const item =
  "px-3 py-1.5 text-xs text-text hover:bg-border rounded cursor-pointer outline-none data-[disabled]:opacity-40 data-[disabled]:cursor-default";

export function TrackContextMenu(props: TrackContextMenuProps) {
  const t = props.target;
  return (
    <DropdownMenu.Root open={props.open} onOpenChange={props.onOpenChange}>
      <DropdownMenu.Trigger asChild>
        <span
          aria-hidden
          style={{ position: "fixed", left: props.pos.x, top: props.pos.y, width: 0, height: 0 }}
        />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="start"
          className="z-50 min-w-[200px] bg-panel border border-border rounded-md shadow-2xl p-1"
        >
          {t && (
            <>
              <DropdownMenu.Item className={item} disabled={t.isPier} onSelect={() => props.onReplace(t)}>
                Replace this track…
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onBlacklistTrack(t)}>
                Blacklist this track
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onBlacklistAlbum(t)}>
                Blacklist Album: {t.track.album}
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onBlacklistArtist(t)}>
                Blacklist Artist: {t.track.artist}
              </DropdownMenu.Item>
              <DropdownMenu.Item className={item} onSelect={() => props.onMarkAlbumLive(t)}>
                Mark album as live
              </DropdownMenu.Item>
              <DropdownMenu.Separator className="h-px bg-border my-1" />
              <DropdownMenu.Item className={item} onSelect={() => props.onEditGenres(t)}>
                Edit genres for album: {t.track.album}
              </DropdownMenu.Item>
            </>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
