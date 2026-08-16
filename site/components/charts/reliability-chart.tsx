import { ChartFrame, RefLine } from "./chart-frame";
import { DrawPath, DrawDots } from "@/components/motion/draw-path";
import { toPath, zip } from "@/lib/scales";
import { f4 } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

/**
 * Calibration. The diagonal is perfect calibration: a bin the model called 4%
 * should contain 4% septic hours. Landing on it is what lets a clinician read
 * the number as a probability rather than a ranking.
 *
 * Honest note the page prints alongside: the cross-hospital run has only a
 * couple of reliability bins because predicted probabilities on a ~1.4%
 * prevalence problem cluster hard at the low end. Few bins is a real limitation
 * of this plot, not a sign of a suspiciously perfect fit.
 */
export function ReliabilityChart({
  model,
  label,
}: {
  model: ModelResult;
  label: string;
}) {
  const pts = zip(model.reliability.meanPred, model.reliability.obsFreq);
  const hi = Math.max(0.02, ...model.reliability.meanPred, ...model.reliability.obsFreq);
  const max = Math.ceil(hi * 100) / 100;

  return (
    <ChartFrame
      label={`Calibration reliability diagram, ${label}. Expected calibration error ${f4(
        model.ece,
      )}.`}
      xDomain={[0, max]}
      yDomain={[0, max]}
      xLabel="predicted probability"
      yLabel="observed frequency"
      table={
        <table>
          <caption>Calibration bins — {label}</caption>
          <thead>
            <tr>
              <th scope="col">Mean predicted</th>
              <th scope="col">Observed frequency</th>
              <th scope="col">Hours in bin</th>
            </tr>
          </thead>
          <tbody>
            {model.reliability.meanPred.map((p, i) => (
              <tr key={i}>
                <td>{f4(p)}</td>
                <td>{f4(model.reliability.obsFreq[i])}</td>
                <td>{model.reliability.count[i]?.toLocaleString("en-US")}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">Expected calibration error</th>
              <td colSpan={2}>{f4(model.ece)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      {({ sx, sy }) => {
        const screen = pts.map(([x, y]) => [sx(x), sy(y)] as const);
        return (
          <>
            <RefLine
              d={`M${sx(0)} ${sy(0)} L${sx(max)} ${sy(max)}`}
              label="perfect"
              x={sx(max)}
              y={sy(max) - 6}
            />
            <DrawPath
              d={toPath(pts, sx, sy)}
              stroke="var(--color-accent)"
              width={2}
              delay={0.1}
            />
            <DrawDots pts={screen} />
          </>
        );
      }}
    </ChartFrame>
  );
}
