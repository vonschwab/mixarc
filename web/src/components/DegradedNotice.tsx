interface DegradedNoticeProps {
  warnings?: string[];
}

/**
 * A generated playlist broke a stated hard guarantee (artist spacing, per-artist
 * cap) but is still usable, so it is shown with the breach named.
 *
 * Deliberately distinct from `RelaxationNotice`: a relaxation is the generator
 * bending a guideline on purpose to stay feasible — the system working. This is
 * the system failing a guarantee it claims to enforce, so it uses the danger
 * token and is NOT dismissible. Hiding it would recreate the silent failure this
 * exists to end.
 */
export function DegradedNotice({ warnings }: DegradedNoticeProps) {
  if (!warnings || warnings.length === 0) return null;

  return (
    <div
      role="alert"
      data-testid="degraded-notice"
      className="flex items-start gap-2 px-3 py-2 border-b border-danger/40 bg-danger/10 text-danger text-xs"
    >
      <span className="mt-0.5 shrink-0" aria-hidden="true">⚠</span>
      <div className="flex-1">
        <div className="font-semibold">
          Playlist degraded — a guarantee was not met
        </div>
        {warnings.map((w, i) => (
          <div key={i} className="mt-0.5 font-mono text-2xs break-words">
            {w}
          </div>
        ))}
      </div>
    </div>
  );
}
