import { ChartFrame, AreaGradient, SeriesLabel } from "./chart-frame";
import { DrawPath, DrawArea } from "@/components/motion/draw-path";
import { toPath, zip, scaleX, scaleY } from "@/lib/scales";
import { f3, pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

const M = { l: 46, r: 18, t: 14, b: 34 };
const W = 460;
const H = 320;

/**
 * Precision-recall against the prevalence floor.
 *
 * This is the chart that keeps an AUROC of 0.72 honest. Sepsis occupies ~1.4% of
 * patient-hours, so a random classifier scores that as precision — and the gap
 * between the curve and that floor is the whole of the model's real value. The
 * floor is drawn as a filled band rather than a dashed line so the comparison is
 * a visible distance, not an arithmetic exercise.
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

  // Scale to the 98th percentile of precision, NOT the max.
  //
  // A PR curve reaches precision 1.0 at its very first point — the single
  // highest-confidence prediction, which is one patient-hour and carries no
  // information. Scaling to that squashed the entire informative band (roughly
  // 0.01–0.10 on a 1.4% prevalence problem) into the bottom tenth of the plot
  // and made the chart look broken. The spike is still drawn; it just runs off
  // the top, which is the honest way to show an uninformative extremum.
  const sorted = [...stacked.pr.precision].sort((a, b) => a - b);
  const p98 = sorted[Math.floor(sorted.length * 0.98)] ?? 0.1;
  const yMax = Math.max(0.06, Math.ceil(p98 * 1.35 * 100) / 100);

  const sx = scaleX([0, 1], W, M);
  const sy = scaleY([0, yMax], H, M);

  const pts = zip(stacked.pr.recall, stacked.pr.precision);
  const curve = toPath(pts, sx, sy);
  const area = `${curve} L${sx(1).toFixed(2)} ${sy(0).toFixed(2)} L${sx(0).toFixed(2)} ${sy(0).toFixed(2)} Z`;

  const at = (p: readonly (readonly [number, number])[], f: number) =>
    p[Math.max(0, Math.min(p.length - 1, Math.round(p.length * f)))];
  const lp = at(pts, 0.3);

  return (
    <ChartFrame
      label={`Precision-recall, ${label}. Production AUPRC ${f3(
        stacked.auprc,
      )} against a prevalence floor of ${pct(prev, 2)}.`}
      xDomain={[0, 1]}
      yDomain={[0, yMax]}
      width={W}
      height={H}
      margins={M}
      xLabel="recall"
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
          </tbody>
        </table>
      }
    >
      {({ h }) => (
        <>
          {/* the floor a coin-flip would reach, as a filled band */}
          <rect
            x={M.l}
            y={sy(prev)}
            width={W - M.l - M.r}
            height={Math.max(0, h - M.b - sy(prev))}
            fill="var(--color-nodata)"
            opacity={0.16}
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
            x={W - M.r}
            y={sy(prev) - 6}
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
            width={1.4}
            delay={0.05}
          />
          <DrawPath d={curve} stroke="var(--color-accent)" width={2.2} delay={0.15} />

          <SeriesLabel
            x={sx(lp[0]) + 8}
            y={sy(lp[1]) - 8}
            text={`AUPRC ${f3(stacked.auprc)}`}
            color="var(--color-accent-ink)"
          />
        </>
      )}
    </ChartFrame>
  );
}
