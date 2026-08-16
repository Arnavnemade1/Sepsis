import { ChartFrame, AreaGradient, SeriesLabel } from "./chart-frame";
import { DrawPath, DrawArea } from "@/components/motion/draw-path";
import { toPath, zip, scaleX, scaleY } from "@/lib/scales";
import { f3, ci, pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

const M = { l: 46, r: 20, t: 18, b: 40 };
const W = 560;
const H = 400;

/**
 * ROC, with the region between the curve and the chance diagonal shaded.
 *
 * The earlier version filled the whole area under the curve, which is wrong on
 * two counts: half of that area is the triangle a coin flip gets for free, and
 * shading it produced a heavy wedge that read as noise. The crescent between
 * curve and diagonal is the part the model actually earned — literally
 * (AUROC − 0.5) — so it is the only part worth ink.
 *
 * The operating points live in a corner readout rather than on leader lines.
 * Leaders across a plot this dense always end up crossing the other series.
 */
export function RocChart({
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
  const sx = scaleX([0, 1], W, M);
  const sy = scaleY([0, 1], H, M);

  const pts = zip(stacked.roc.fpr, stacked.roc.tpr);
  const curve = toPath(pts, sx, sy);
  // close along the diagonal → the crescent of lift over chance
  const lift = `${curve} L${sx(0).toFixed(2)} ${sy(0).toFixed(2)} Z`;

  const s90 = stacked.sensAtSpec["0.90"];
  const s95 = stacked.sensAtSpec["0.95"];

  const at = (p: readonly (readonly [number, number])[], f: number) =>
    p[Math.max(0, Math.min(p.length - 1, Math.round(p.length * f)))];
  const lp = at(pts, 0.5);
  const gp = at(zip(gbm.roc.fpr, gbm.roc.tpr), 0.82);

  return (
    <ChartFrame
      label={`Receiver operating characteristic, ${label}. Production model AUROC ${f3(
        stacked.auroc,
      )}, 95% confidence interval ${ci(stacked.aurocCi)}. Baseline AUROC ${f3(
        gbm.auroc,
      )}. At 90% specificity the production model catches ${pct(s90.sensitivity, 0)} of septic patients.`}
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      width={W}
      height={H}
      margins={M}
      xLabel="false positive rate"
      yLabel="true positive rate"
      defs={<AreaGradient id={`roc-${id}`} color="var(--color-accent)" />}
      table={
        <table>
          <caption>ROC summary — {label}</caption>
          <tbody>
            <tr>
              <th scope="row">Production (stacked) AUROC</th>
              <td>
                {f3(stacked.auroc)} (95% CI {ci(stacked.aurocCi)})
              </td>
            </tr>
            <tr>
              <th scope="row">Baseline (GBM) AUROC</th>
              <td>{f3(gbm.auroc)}</td>
            </tr>
            <tr>
              <th scope="row">Sensitivity at specificity ≥ 0.90</th>
              <td>{pct(s90.sensitivity, 1)}</td>
            </tr>
            <tr>
              <th scope="row">Sensitivity at specificity ≥ 0.95</th>
              <td>{pct(s95.sensitivity, 1)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      {() => (
        <>
          <line
            x1={sx(0)}
            y1={sy(0)}
            x2={sx(1)}
            y2={sy(1)}
            stroke="var(--color-faint)"
            strokeWidth={1.2}
            strokeDasharray="4 4"
          />

          <DrawArea d={lift} fill={`url(#roc-${id})`} />

          <DrawPath
            d={toPath(zip(gbm.roc.fpr, gbm.roc.tpr), sx, sy)}
            stroke="var(--color-dim)"
            width={1.5}
            delay={0.05}
          />
          <DrawPath d={curve} stroke="var(--color-accent)" width={2.4} delay={0.15} />

          <SeriesLabel
            x={sx(lp[0]) + 8}
            y={sy(lp[1]) - 10}
            text={`stacked · ${f3(stacked.auroc)}`}
            color="var(--color-accent-ink)"
          />
          <SeriesLabel
            x={sx(gp[0]) - 8}
            y={sy(gp[1]) + 16}
            text={`gbm · ${f3(gbm.auroc)}`}
            color="var(--color-dim)"
            anchor="end"
          />

          {/* the clinically meaningful operating points, ringed on the curve */}
          {[
            { fpr: 0.1, s: s90.sensitivity },
            { fpr: 0.05, s: s95.sensitivity },
          ].map((o) => (
            <circle
              key={o.fpr}
              cx={sx(o.fpr)}
              cy={sy(o.s)}
              r={4.5}
              fill="var(--color-bg)"
              stroke="var(--color-accent)"
              strokeWidth={2}
            />
          ))}

          {/* corner readout — no leaders to cross the other series */}
          <g transform={`translate(${sx(0.40)} ${sy(0.34)})`}>
            <rect
              x={0}
              y={-18}
              width={W - M.r - sx(0.40)}
              height={64}
              fill="var(--color-bg)"
              fillOpacity={0.6}
              stroke="var(--color-line)"
              rx={4}
            />
            <text x={11} y={-4} className="fill-dim font-mono" fontSize={9.5}>
              at a false-alarm rate a ward could tolerate
            </text>
            <text x={11} y={17} className="fill-ink font-mono" fontSize={13} fontWeight={600}>
              {pct(s90.sensitivity, 0)}
            </text>
            <text x={52} y={17} className="fill-muted font-mono" fontSize={10}>
              caught · 90% specificity
            </text>
            <text x={11} y={35} className="fill-ink font-mono" fontSize={13} fontWeight={600}>
              {pct(s95.sensitivity, 0)}
            </text>
            <text x={52} y={35} className="fill-muted font-mono" fontSize={10}>
              caught · 95% specificity
            </text>
          </g>

          <text
            x={sx(0.97)}
            y={sy(0.92)}
            textAnchor="end"
            className="fill-faint font-mono"
            fontSize={9}
          >
            chance
          </text>
        </>
      )}
    </ChartFrame>
  );
}
