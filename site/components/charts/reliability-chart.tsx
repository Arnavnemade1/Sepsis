import { ChartFrame, RefLine, Annotation } from "./chart-frame";
import { DrawPath, DrawDots } from "@/components/motion/draw-path";
import { toPath, zip, scaleX, scaleY } from "@/lib/scales";
import { f4 } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

const M = { l: 48, r: 20, t: 16, b: 38 };
const W = 620;
const H = 400;

/**
 * Calibration.
 *
 * The diagonal is perfect calibration: a bin the model called 4% should contain
 * 4% septic hours. Landing on it is what lets a clinician read the output as a
 * probability rather than a ranking — and it is the property most likely to
 * break when a model changes hospital, which is exactly what this measures.
 *
 * Honest limitation, printed next to the chart on the page: the cross-hospital
 * run resolves only a couple of bins, because predictions on a ~1.4% prevalence
 * problem cluster hard at the low end. Few bins is a real weakness of this
 * diagram, not evidence of a suspiciously perfect fit.
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

  const sx = scaleX([0, max], W, M);
  const sy = scaleY([0, max], H, M);
  const screen = pts.map(([x, y]) => [sx(x), sy(y)] as const);

  return (
    <ChartFrame
      label={`Calibration reliability diagram, ${label}. Expected calibration error ${f4(
        model.ece,
      )}.`}
      xDomain={[0, max]}
      yDomain={[0, max]}
      width={W}
      height={H}
      margins={M}
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
      {() => (
        <>
          {/* the tolerance corridor — inside this, a probability is trustworthy
              at the resolution this evaluation can actually resolve */}
          <path
            d={`M${sx(0)} ${sy(0)} L${sx(max)} ${sy(max)} L${sx(max)} ${sy(max * 0.82)} L${sx(0)} ${sy(0)} Z`}
            fill="var(--color-accent)"
            opacity={0.06}
          />
          <RefLine d={`M${sx(0)} ${sy(0)} L${sx(max)} ${sy(max)}`} />
          <text
            x={sx(max) - 4}
            y={sy(max) + 14}
            textAnchor="end"
            className="fill-faint font-mono"
            fontSize={9}
          >
            perfect calibration
          </text>

          <DrawPath d={toPath(pts, sx, sy)} stroke="var(--color-accent)" width={2.2} delay={0.1} />
          <DrawDots pts={screen} r={4.5} />

          <Annotation
            x={screen[screen.length - 1][0]}
            y={screen[screen.length - 1][1]}
            dx={-18}
            dy={-28}
            anchor="end"
            title={`ECE ${f4(model.ece)}`}
            detail="off-site"
            color="var(--color-accent)"
          />
        </>
      )}
    </ChartFrame>
  );
}
