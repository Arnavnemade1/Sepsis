import { ChartFrame, RefLine, AreaGradient, SeriesLabel, Annotation } from "./chart-frame";
import { DrawPath, DrawArea } from "@/components/motion/draw-path";
import { toPath, zip, scaleX, scaleY } from "@/lib/scales";
import { f3, ci, pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

const M = { l: 42, r: 18, t: 14, b: 34 };
const W = 460;
const H = 320;

/**
 * ROC with the area under the production curve shaded.
 *
 * The shading is the point: AUROC literally *is* that area, so showing it makes
 * the headline number legible as a picture rather than a claim you have to take
 * on trust. The baseline is a thin neutral line beneath it, and both curves are
 * labelled on themselves — no legend to cross-reference.
 *
 * The sens@spec≥0.90 point is annotated, because that is the operating regime a
 * clinician would actually accept, and it is far less flattering than the AUROC.
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
  const area = `${curve} L${sx(1).toFixed(2)} ${sy(0).toFixed(2)} L${sx(0).toFixed(2)} ${sy(0).toFixed(2)} Z`;

  // marker at ~90% specificity, i.e. FPR 0.10
  const sens90 = stacked.sensAtSpec["0.90"].sensitivity;

  // put the direct labels around 55% along each curve
  const at = (p: readonly (readonly [number, number])[], f: number) =>
    p[Math.max(0, Math.min(p.length - 1, Math.round(p.length * f)))];
  const lp = at(pts, 0.55);
  const gp = at(zip(gbm.roc.fpr, gbm.roc.tpr), 0.78);

  return (
    <ChartFrame
      label={`Receiver operating characteristic, ${label}. Production model AUROC ${f3(
        stacked.auroc,
      )}, 95% confidence interval ${ci(stacked.aurocCi)}. Baseline AUROC ${f3(gbm.auroc)}.`}
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
              <td>
                {f3(gbm.auroc)} (95% CI {ci(gbm.aurocCi)})
              </td>
            </tr>
            <tr>
              <th scope="row">Sensitivity at specificity ≥ 0.90</th>
              <td>{pct(sens90, 1)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      {() => (
        <>
          <RefLine d={`M${sx(0)} ${sy(0)} L${sx(1)} ${sy(1)}`} />
          <DrawArea d={area} fill={`url(#roc-${id})`} />

          <DrawPath
            d={toPath(zip(gbm.roc.fpr, gbm.roc.tpr), sx, sy)}
            stroke="var(--color-dim)"
            width={1.4}
            delay={0.05}
          />
          <DrawPath d={curve} stroke="var(--color-accent)" width={2.2} delay={0.15} />

          <SeriesLabel
            x={sx(lp[0]) + 6}
            y={sy(lp[1]) - 8}
            text={`stacked · ${f3(stacked.auroc)}`}
            color="var(--color-accent-ink)"
          />
          <SeriesLabel
            x={sx(gp[0]) + 6}
            y={sy(gp[1]) + 14}
            text={`gbm · ${f3(gbm.auroc)}`}
            color="var(--color-dim)"
          />

          <Annotation
            x={sx(0.1)}
            y={sy(sens90)}
            dx={22}
            dy={44}
            title={`${pct(sens90, 0)} caught`}
            detail="at 90% specificity"
            color="var(--color-accent)"
          />

          <text
            x={sx(0.97)}
            y={sy(0.06)}
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
