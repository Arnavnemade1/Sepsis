import { ChartFrame } from "./chart-frame";
import { DrawPath } from "@/components/motion/draw-path";
import { toPath, zip } from "@/lib/scales";
import type { ModelResult } from "@/lib/results";

/** Shared with every other figure on the page, so all axis type renders at the
 *  same size. Relying on ChartFrame's default made this one scale 1.70x while
 *  its neighbours sat at 1.26x. */
const W = 620;
const H = 380;
const M = { l: 52, r: 20, t: 16, b: 38 };

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
  const xMax = Math.max(...dc.thresholds);

  // Scale to the MODEL, not to every series.
  //
  // "Treat all" dives to about −0.23 at high thresholds — expected, and
  // uninformative, because treating every patient gets ruinous once the
  // threshold implies you should treat almost nobody. Including it in the
  // domain squashed the model's entire range (−0.002 to +0.009 here) into the
  // top 4% of the plot, which is why this chart read as a flat line.
  //
  // Zero stays in view because it is the decision boundary that matters: below
  // it, the model is worse than treating nobody.
  const hi = Math.max(...dc.model, 0);
  const lo = Math.min(...dc.model, 0);
  const span = hi - lo || 0.01;
  const yMax = hi + span * 0.22;
  const yMin = lo - span * 0.35;

  // Where treat-all leaves the frame, so the chart can say so rather than
  // letting the clamped line look like a plateau along the axis.
  const exitIdx = dc.treatAll.findIndex((v) => v < yMin);
  const exitAt = exitIdx > 0 ? dc.thresholds[exitIdx] : null;

  return (
    <ChartFrame
      label={`Decision curve analysis, ${label}: net benefit of the model against treat-all and treat-none strategies across threshold probabilities.`}
      xDomain={[0, xMax]}
      yDomain={[yMin, yMax]}
      width={W}
      height={H}
      margins={M}
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
          {/* zero is the decision boundary: below it the model is worse than
              doing nothing, so it gets a real rule rather than a gridline */}
          <line
            x1={M.l}
            y1={sy(0)}
            x2={W - M.r}
            y2={sy(0)}
            stroke="var(--color-line-bright)"
            strokeWidth={1}
          />
          <text
            x={W - M.r - 4}
            y={sy(0) - 6}
            textAnchor="end"
            className="fill-faint font-mono"
            fontSize={9}
          >
            treat none — no benefit either way
          </text>

          <DrawPath
            d={toPath(zip(dc.thresholds, dc.treatAll), sx, sy)}
            stroke="var(--color-faint)"
            width={1.4}
            dash="4 4"
          />
          {exitAt !== null && (
            <text
              x={sx(exitAt) + 6}
              y={sy(yMin) - 8}
              className="fill-faint font-mono"
              fontSize={9}
            >
              ↓ treat all falls off scale
            </text>
          )}
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
