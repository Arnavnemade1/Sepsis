import { ChartFrame } from "./chart-frame";
import { DrawPath } from "@/components/motion/draw-path";
import { toPath, zip } from "@/lib/scales";
import { pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

/**
 * The chart that decides whether this is a product or a paper.
 *
 * Detection rate and false-alarm rate are plotted against the same threshold
 * axis, so you can read the trade the operating point actually buys. The
 * clinical hues ARE warranted here: these two curves are the alarm a bedside
 * nurse would or would not receive, which is exactly the "clinical state"
 * Rule 1 reserves colour for.
 */
export function AlarmBurden({
  model,
  label,
  shippedThreshold,
}: {
  model: ModelResult;
  label: string;
  shippedThreshold?: number;
}) {
  const ab = model.alarmBurden;
  const xMax = Math.max(...ab.thresholds);

  return (
    <ChartFrame
      label={`Alarm burden, ${label}: detection rate and false-alarm rate across alarm thresholds. At the shipped operating point, detection is ${pct(
        model.earlyWarning.detectionRate,
      )} and the false-alarm rate is ${pct(model.earlyWarning.falseAlarmRate)}.`}
      xDomain={[0, xMax]}
      yDomain={[0, 1]}
      xLabel="alarm threshold"
      yLabel="rate"
      table={
        <table>
          <caption>Alarm burden — {label}</caption>
          <thead>
            <tr>
              <th scope="col">Threshold</th>
              <th scope="col">Detection rate</th>
              <th scope="col">False-alarm rate</th>
              <th scope="col">False alerts per 100 non-septic patient-hours</th>
            </tr>
          </thead>
          <tbody>
            {ab.thresholds.map((t, i) => (
              <tr key={i}>
                <td>{t.toFixed(4)}</td>
                <td>{pct(ab.detectionRate[i], 1)}</td>
                <td>{pct(ab.falseAlarmRate[i], 1)}</td>
                <td>{ab.falseAlertsPer100h[i]?.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      {({ sx, sy, h }) => (
        <>
          {shippedThreshold !== undefined && shippedThreshold <= xMax && (
            <>
              <line
                x1={sx(shippedThreshold)}
                y1={sy(1)}
                x2={sx(shippedThreshold)}
                y2={h - 34}
                stroke="var(--color-line-bright)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <text
                x={sx(shippedThreshold) + 4}
                y={sy(1) + 9}
                className="fill-dim font-mono"
                fontSize={9}
              >
                shipped
              </text>
            </>
          )}
          <DrawPath
            d={toPath(zip(ab.thresholds, ab.detectionRate), sx, sy)}
            stroke="var(--color-low)"
            width={2}
            delay={0.1}
          />
          <DrawPath
            d={toPath(zip(ab.thresholds, ab.falseAlarmRate), sx, sy)}
            stroke="var(--color-high)"
            width={2}
            delay={0.2}
          />
        </>
      )}
    </ChartFrame>
  );
}
