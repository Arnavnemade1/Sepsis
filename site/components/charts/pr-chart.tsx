import { ChartFrame, RefLine } from "./chart-frame";
import { DrawPath } from "@/components/motion/draw-path";
import { toPath, zip } from "@/lib/scales";
import { f3, pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

/**
 * Precision-recall, with the prevalence line — the baseline a random classifier
 * would achieve. On a 1.4-2.2% prevalence problem this is the chart that keeps
 * an AUROC of 0.72 honest: precision is low because sepsis is rare, and hiding
 * that behind ROC alone would be the flattering choice.
 */
export function PrChart({
  stacked,
  gbm,
  label,
}: {
  stacked: ModelResult;
  gbm: ModelResult;
  label: string;
}) {
  const prev = stacked.prevalence;
  const yMax = Math.max(0.1, Math.ceil(Math.max(...stacked.pr.precision) * 20) / 20);

  return (
    <ChartFrame
      label={`Precision-recall, ${label}. Production AUPRC ${f3(
        stacked.auprc,
      )} against a prevalence baseline of ${pct(prev, 1)}.`}
      xDomain={[0, 1]}
      yDomain={[0, yMax]}
      xLabel="recall"
      yLabel="precision"
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
              <th scope="row">Prevalence (a random classifier)</th>
              <td>{pct(prev, 2)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      {({ sx, sy }) => (
        <>
          <RefLine
            d={`M${sx(0)} ${sy(prev)} L${sx(1)} ${sy(prev)}`}
            label={`prevalence ${pct(prev, 1)}`}
            x={sx(1)}
            y={sy(prev) - 5}
          />
          <DrawPath
            d={toPath(zip(gbm.pr.recall, gbm.pr.precision), sx, sy)}
            stroke="var(--color-dim)"
            width={1.6}
            delay={0.05}
          />
          <DrawPath
            d={toPath(zip(stacked.pr.recall, stacked.pr.precision), sx, sy)}
            stroke="var(--color-accent)"
            width={2}
            delay={0.15}
          />
        </>
      )}
    </ChartFrame>
  );
}
