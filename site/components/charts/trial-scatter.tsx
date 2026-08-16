import { ChartFrame, Annotation } from "./chart-frame";
import { scaleX, scaleY } from "@/lib/scales";
import { f4, f3 } from "@/lib/format";
import { selfImprove } from "@/lib/results";

const M = { l: 52, r: 20, t: 16, b: 38 };
const W = 620;
const H = 340;

/**
 * Sixteen self-improvement trials, plotted as effect size against significance.
 *
 * This replaces a table, and it earns the swap: a table of sixteen rows makes
 * you do the comparison in your head, whereas the picture shows in one glance
 * that the acceptance region — bottom-right, where a result must be BOTH large
 * enough and significant enough — is empty.
 *
 * Trial #5 is the one that matters. It sits far to the right on effect size, so
 * a naive loop reading only ΔAUROC would have banked it, and it is nowhere near
 * the significance line. The rejection is the contribution.
 */
export function TrialScatter() {
  const trials = selfImprove.trials;
  const deltas = trials.map((t) => t.delta);
  const dMin = Math.min(-0.025, Math.floor(Math.min(...deltas) * 100) / 100);
  const dMax = Math.max(0.02, Math.ceil(Math.max(...deltas) * 100) / 100);

  const sx = scaleX([dMin, dMax], W, M);
  // p ascending downward: significant (small p) at the TOP, so "good" is up-right
  const sy = scaleY([0, 1], H, M);

  const ALPHA = 0.0125; // Bonferroni-corrected, 4 challengers at 0.05
  const MIN_EFFECT = 0.005;

  const top = trials[0];

  return (
    <ChartFrame
      label={`Sixteen self-improvement trials plotted by effect size and p-value. None fell in the acceptance region, which requires both a delta above ${MIN_EFFECT} AUROC and a p-value below ${ALPHA}. The largest apparent gain, trial ${top.id}, scored ${f4(top.delta)} AUROC at p ${f3(top.p)}.`}
      xDomain={[dMin, dMax]}
      yDomain={[0, 1]}
      width={W}
      height={H}
      margins={M}
      grid={false}
      xLabel="Δ AUROC vs champion"
      yLabel="p-value"
      table={
        <table>
          <caption>Self-improvement trials</caption>
          <thead>
            <tr>
              <th scope="col">Trial</th>
              <th scope="col">Δ AUROC</th>
              <th scope="col">p</th>
              <th scope="col">Accepted</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t) => (
              <tr key={t.id}>
                <td>{t.id}</td>
                <td>{f4(t.delta)}</td>
                <td>{f3(t.p)}</td>
                <td>{t.accepted ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      {({ h }) => {
        const accX = sx(MIN_EFFECT);
        const accY = sy(ALPHA);
        return (
          <>
            {/* THE ACCEPTANCE REGION — large enough AND significant enough.
                Drawn in the clinical "good" hue because it encodes a real
                verdict, and left conspicuously empty. */}
            <rect
              x={accX}
              y={M.t}
              width={Math.max(0, W - M.r - accX)}
              height={Math.max(0, accY - M.t)}
              fill="var(--color-low)"
              opacity={0.1}
            />
            <rect
              x={accX}
              y={M.t}
              width={Math.max(0, W - M.r - accX)}
              height={Math.max(0, accY - M.t)}
              fill="none"
              stroke="var(--color-low)"
              strokeOpacity={0.45}
              strokeDasharray="3 3"
            />
            <text
              x={accX + 10}
              y={M.t + 18}
              className="fill-low-ink font-mono"
              fontSize={10}
              fontWeight={600}
            >
              ACCEPT
            </text>
            <text x={accX + 10} y={M.t + 32} className="fill-dim font-mono" fontSize={9}>
              Δ &gt; {MIN_EFFECT} and p &lt; {ALPHA}
            </text>
            <text
              x={accX + 10}
              y={M.t + 50}
              className="fill-faint font-mono"
              fontSize={9}
              fontStyle="italic"
            >
              — empty, across all 16 —
            </text>

            {/* threshold lines */}
            <line
              x1={accX}
              y1={M.t}
              x2={accX}
              y2={h - M.b}
              stroke="var(--color-line-bright)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <line
              x1={M.l}
              y1={accY}
              x2={W - M.r}
              y2={accY}
              stroke="var(--color-line-bright)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {/* zero-effect line */}
            <line
              x1={sx(0)}
              y1={M.t}
              x2={sx(0)}
              y2={h - M.b}
              stroke="var(--color-line)"
              strokeWidth={1}
            />

            {trials.map((t) => {
              const isTop = t.id === top.id;
              return (
                <g key={t.id}>
                  <circle
                    cx={sx(t.delta)}
                    cy={sy(t.p)}
                    r={isTop ? 6 : 4.5}
                    fill={isTop ? "var(--color-med)" : "var(--color-dim)"}
                    fillOpacity={isTop ? 0.9 : 0.55}
                    stroke={isTop ? "var(--color-med)" : "var(--color-line-bright)"}
                    strokeWidth={1.2}
                  />
                </g>
              );
            })}

            <Annotation
              x={sx(top.delta)}
              y={sy(top.p)}
              dx={-18}
              dy={-30}
              anchor="end"
              title={`trial #${top.id} · ${f4(top.delta)}`}
              detail={`p = ${f3(top.p)} — fold noise`}
              color="var(--color-med)"
            />
          </>
        );
      }}
    </ChartFrame>
  );
}
