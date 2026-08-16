import { ChartFrame } from "./chart-frame";
import { DrawPath } from "@/components/motion/draw-path";
import { toPath, zip } from "@/lib/scales";
import type { ModelResult } from "@/lib/results";

/**
 * Decision-curve analysis: net clinical benefit against the two strategies that
 * need no model at all — treat everybody, and treat nobody. A model only earns
 * its place in the band of threshold probabilities where its curve sits above
 * both. This is a harsher and more useful test than AUROC.
 */
export function DecisionCurve({
  model,
  label,
}: {
  model: ModelResult;
  label: string;
}) {
  const dc = model.decisionCurve;
  const all = [...dc.model, ...dc.treatAll, ...dc.treatNone];
  const lo = Math.min(...all, 0);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.12 || 0.01;
  const xMax = Math.max(...dc.thresholds);

  return (
    <ChartFrame
      label={`Decision curve analysis, ${label}: net benefit of the model against treat-all and treat-none strategies across threshold probabilities.`}
      xDomain={[0, xMax]}
      yDomain={[lo - pad, hi + pad]}
      xLabel="threshold probability"
      yLabel="net benefit"
      table={
        <table>
          <caption>Decision curve — {label}</caption>
          <thead>
            <tr>
              <th scope="col">Threshold</th>
              <th scope="col">Model</th>
              <th scope="col">Treat all</th>
              <th scope="col">Treat none</th>
            </tr>
          </thead>
          <tbody>
            {dc.thresholds.map((t, i) => (
              <tr key={i}>
                <td>{t.toFixed(4)}</td>
                <td>{dc.model[i]?.toFixed(4)}</td>
                <td>{dc.treatAll[i]?.toFixed(4)}</td>
                <td>{dc.treatNone[i]?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      {({ sx, sy }) => (
        <>
          <DrawPath
            d={toPath(zip(dc.thresholds, dc.treatAll), sx, sy)}
            stroke="var(--color-faint)"
            width={1.4}
            dash="4 4"
          />
          <DrawPath
            d={toPath(zip(dc.thresholds, dc.treatNone), sx, sy)}
            stroke="var(--color-faint)"
            width={1.4}
            dash="2 3"
            delay={0.05}
          />
          <DrawPath
            d={toPath(zip(dc.thresholds, dc.model), sx, sy)}
            stroke="var(--color-accent)"
            width={2}
            delay={0.12}
          />
        </>
      )}
    </ChartFrame>
  );
}
