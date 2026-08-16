import { ChartFrame, RefLine } from "./chart-frame";
import { DrawPath } from "@/components/motion/draw-path";
import { toPath, zip } from "@/lib/scales";
import { f3, ci } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

/**
 * ROC, both models, one direction. Production is the accent; the baseline is
 * neutral. No clinical hue here — a model comparison is not a patient state, and
 * Rule 1 reserves saturated colour for things that are.
 */
export function RocChart({
  stacked,
  gbm,
  label,
}: {
  stacked: ModelResult;
  gbm: ModelResult;
  label: string;
}) {
  return (
    <ChartFrame
      label={`Receiver operating characteristic, ${label}. Production model AUROC ${f3(
        stacked.auroc,
      )}, 95% confidence interval ${ci(stacked.aurocCi)}. Baseline AUROC ${f3(gbm.auroc)}.`}
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      xLabel="false positive rate"
      yLabel="true positive rate"
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
          </tbody>
        </table>
      }
    >
      {({ sx, sy }) => (
        <>
          <RefLine d={`M${sx(0)} ${sy(0)} L${sx(1)} ${sy(1)}`} />
          <DrawPath
            d={toPath(zip(gbm.roc.fpr, gbm.roc.tpr), sx, sy)}
            stroke="var(--color-dim)"
            width={1.6}
            delay={0.05}
          />
          <DrawPath
            d={toPath(zip(stacked.roc.fpr, stacked.roc.tpr), sx, sy)}
            stroke="var(--color-accent)"
            width={2}
            delay={0.15}
          />
        </>
      )}
    </ChartFrame>
  );
}
