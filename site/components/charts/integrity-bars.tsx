const W = 620;
const H = 232;
const L = 210;
const R = 190;

/**
 * What the same detector scores under three different evaluations.
 *
 * This is the most persuasive thing on the page and it was pure prose. The bars
 * make the argument in one look: evaluated in-sample the model would claim
 * near-perfection, the honest out-of-fold number is far lower, and when the
 * labels are shuffled it collapses to exactly chance — which is the proof that
 * nothing is leaking.
 *
 * The shuffled bar is the load-bearing one. A pipeline with a leak would still
 * score above chance on noise.
 */
const rows = [
  {
    label: "In-sample",
    sub: "scored on the data it fit",
    v: 0.999,
    tone: "var(--color-high)",
    note: "what a careless report would claim",
  },
  {
    label: "Out-of-fold",
    sub: "held-out patients",
    v: 0.56,
    tone: "var(--color-accent)",
    note: "the honest number on this fixture",
  },
  {
    label: "Labels shuffled",
    sub: "the leakage tripwire",
    v: 0.497,
    tone: "var(--color-dim)",
    note: "chance — so nothing is leaking",
  },
];

export function IntegrityBars() {
  const lo = 0.45;
  const hi = 1.0;
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const rowH = 58;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={
          "The same detector under three evaluations: in-sample AUROC 0.999, out-of-fold 0.560, and with labels shuffled 0.497 — chance, which demonstrates the pipeline does not leak."
        }
        className="block w-full"
        style={{ overflow: "visible" }}
      >
        {[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((v) => (
          <g key={v}>
            <line x1={x(v)} y1={16} x2={x(v)} y2={H - 26} stroke="var(--color-line-soft)" />
            <text x={x(v)} y={10} textAnchor="middle" className="fill-faint font-mono" fontSize={9}>
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* chance */}
        <line
          x1={x(0.5)}
          y1={16}
          x2={x(0.5)}
          y2={H - 26}
          stroke="var(--color-faint)"
          strokeDasharray="3 3"
        />
        <text x={x(0.5)} y={H - 12} textAnchor="middle" className="fill-faint font-mono" fontSize={9}>
          chance
        </text>

        {rows.map((r, i) => {
          const y = 40 + i * rowH;
          return (
            <g key={r.label}>
              <text x={0} y={y + 2} className="fill-ink" fontSize={13} fontWeight={500}>
                {r.label}
              </text>
              <text x={0} y={y + 17} className="fill-faint font-mono" fontSize={9.5}>
                {r.sub}
              </text>

              <rect
                x={x(lo)}
                y={y - 11}
                width={Math.max(1, x(r.v) - x(lo))}
                height={22}
                fill={r.tone}
                fillOpacity={0.75}
                rx={2}
              />

              {/* Value and note both sit AFTER the bar. Putting the note inside
                  the bar clipped it to nothing on the short ones — the shuffled
                  bar is barely wider than the word it was meant to contain. */}
              <text
                x={x(r.v) + 9}
                y={y + 5}
                className="fill-ink font-mono"
                fontSize={12.5}
                fontWeight={600}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {r.v.toFixed(3)}
              </text>
              <text
                x={x(r.v) + 9}
                y={y + 19}
                className="fill-faint font-mono"
                fontSize={9}
              >
                {r.note}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
