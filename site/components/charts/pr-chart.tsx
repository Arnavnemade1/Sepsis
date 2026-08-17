import { ChartFrame, AreaGradient } from "./chart-frame";
import { DrawPath, DrawArea } from "@/components/motion/draw-path";
import { toPath, zip, scaleX, scaleY } from "@/lib/scales";
import { f3, pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

const M = { l: 50, r: 20, t: 18, b: 40 };
const W = 620;
const H = 440;

/**
 * Precision-recall, read as "how many of your alerts are real".
 *
 * At ~1.4% prevalence precision is low everywhere, and the curve collapses
 * almost immediately — that cliff IS the finding, not a rendering artefact. So
 * rather than hide it, this annotates the one point a reader can act on: at
 * half of all septic patients caught, what fraction of the alerts firing are
 * genuine? The prevalence floor is drawn as a filled band, because the distance
 * between the curve and that band is the entire value the model adds.
 */
export function PrChart({
  stacked,
  gbm,
  label,
  id,
}: {
  stacked: ModelResult;
  gbm: ModelResult;
  label: string;
  id: string;
}) {
  const prev = stacked.prevalence;

  // Scale to the 98th percentile of precision, NOT the max: a PR curve touches
  // 1.0 at its single highest-confidence point, which is one patient-hour and
  // carries no information. Scaling to it squashed the informative band into
  // the bottom tenth of the plot.
  const sorted = [...stacked.pr.precision].sort((a, b) => a - b);
  const p98 = sorted[Math.floor(sorted.length * 0.98)] ?? 0.1;
  const yMax = Math.max(0.06, Math.ceil(p98 * 1.4 * 100) / 100);

  const sx = scaleX([0, 1], W, M);
  const sy = scaleY([0, yMax], H, M);

  const pts = zip(stacked.pr.recall, stacked.pr.precision);
  const curve = toPath(pts, sx, sy);
  const area = `${curve} L${sx(1).toFixed(2)} ${sy(0).toFixed(2)} L${sx(0).toFixed(2)} ${sy(0).toFixed(2)} Z`;

  // precision at 50% recall — "half the septic patients caught"
  let half = pts[0];
  for (const p of pts) {
    if (p[0] >= 0.5) {
      half = p;
      break;
    }
  }
  const lift = half[1] / prev;

  return (
    <ChartFrame
      label={`Precision-recall, ${label}. AUPRC ${f3(
        stacked.auprc,
      )} against a prevalence floor of ${pct(prev, 2)}. At 50% recall, precision is ${pct(
        half[1],
        1,
      )} — about ${lift.toFixed(1)} times better than chance.`}
      xDomain={[0, 1]}
      yDomain={[0, yMax]}
      width={W}
      height={H}
      margins={M}
      xLabel="recall — share of septic patients caught"
      yLabel="precision"
      defs={<AreaGradient id={`pr-${id}`} color="var(--color-accent)" />}
      table={
        <table>
          <caption>Precision-recall summary — {label}</caption>
          <tbody>
            <tr>
              <th scope="row">Production (stacked) AUPRC</th>
              <td>{f3(stacked.auprc)}</td>
            </tr>
            <tr>
              <th scope="row">Baseline (GBM) AUPRC</th>
              <td>{f3(gbm.auprc)}</td>
            </tr>
            <tr>
              <th scope="row">Prevalence floor (a random classifier)</th>
              <td>{pct(prev, 2)}</td>
            </tr>
            <tr>
              <th scope="row">Precision at 50% recall</th>
              <td>{pct(half[1], 2)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      {({ h }) => (
        <>
          {/* what a coin flip already achieves */}
          <rect
            x={M.l}
            y={sy(prev)}
            width={W - M.l - M.r}
            height={Math.max(0, h - M.b - sy(prev))}
            fill="var(--color-nodata)"
            opacity={0.2}
          />
          <line
            x1={M.l}
            y1={sy(prev)}
            x2={W - M.r}
            y2={sy(prev)}
            stroke="var(--color-faint)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />
          <text
            x={W - M.r - 4}
            y={sy(prev) - 7}
            textAnchor="end"
            className="fill-faint font-mono"
            fontSize={9}
          >
            chance = prevalence {pct(prev, 2)}
          </text>

          <DrawArea d={area} fill={`url(#pr-${id})`} />
          <DrawPath
            d={toPath(zip(gbm.pr.recall, gbm.pr.precision), sx, sy)}
            stroke="var(--color-dim)"
            width={1.5}
            delay={0.05}
          />
          <DrawPath d={curve} stroke="var(--color-accent)" width={2.4} delay={0.15} />

          {/* the point a reader can act on */}
          <line
            x1={sx(half[0])}
            y1={sy(half[1])}
            x2={sx(half[0])}
            y2={h - M.b}
            stroke="var(--color-line-bright)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <circle
            cx={sx(half[0])}
            cy={sy(half[1])}
            r={4.5}
            fill="var(--color-bg)"
            stroke="var(--color-accent)"
            strokeWidth={2}
          />

          <g transform={`translate(${sx(0.30)} ${M.t + 6})`}>
            <rect
              x={0}
              y={0}
              width={W - M.r - sx(0.30)}
              height={58}
              fill="var(--color-bg)"
              fillOpacity={0.6}
              stroke="var(--color-line)"
              rx={4}
            />
            <text x={11} y={18} className="fill-dim font-mono" fontSize={9.5}>
              catch half of all septic patients, and…
            </text>
            <text x={11} y={38} className="fill-ink font-mono" fontSize={13} fontWeight={600}>
              {pct(half[1], 1)}
            </text>
            <text x={56} y={38} className="fill-muted font-mono" fontSize={10}>
              of alerts are real — {lift.toFixed(1)}× chance
            </text>
          </g>
        </>
      )}
    </ChartFrame>
  );
}
