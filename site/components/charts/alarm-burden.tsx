import { ChartFrame, SeriesLabel, Annotation } from "./chart-frame";
import { DrawPath, DrawArea } from "@/components/motion/draw-path";
import { toPath, zip, scaleX, scaleY } from "@/lib/scales";
import { pct } from "@/lib/format";
import type { ModelResult } from "@/lib/results";

const M = { l: 44, r: 20, t: 16, b: 38 };
const W = 460;
const H = 320;

/**
 * The chart that decides whether this is a product or a paper.
 *
 * Detection and false-alarm rate share a threshold axis, and the band BETWEEN
 * them is the thing worth looking at: it is the margin by which catching a
 * septic patient beats waking a nurse for a healthy one. Where the band is thin,
 * the detector is not buying much.
 *
 * Clinical hues are warranted here — these two curves are literally the alarm a
 * nurse does or does not receive, which is the "clinical state" Rule 1 reserves
 * saturated colour for.
 */
export function AlarmBurden({
  model,
  label,
  shippedThreshold,
  id,
}: {
  model: ModelResult;
  label: string;
  shippedThreshold?: number;
  id: string;
}) {
  const ab = model.alarmBurden;
  const xMax = Math.max(...ab.thresholds);
  const sx = scaleX([0, xMax], W, M);
  const sy = scaleY([0, 1], H, M);

  const det = zip(ab.thresholds, ab.detectionRate);
  const far = zip(ab.thresholds, ab.falseAlarmRate);

  // the band between the two curves
  const band = `${toPath(det, sx, sy)} L${[...far]
    .reverse()
    .map(([x, y], i) => `${i ? "L" : ""}${sx(x).toFixed(2)} ${sy(y).toFixed(2)}`)
    .join(" ")} Z`;

  const at = (p: readonly (readonly [number, number])[], f: number) =>
    p[Math.max(0, Math.min(p.length - 1, Math.round(p.length * f)))];
  const dl = at(det, 0.42);
  const fl = at(far, 0.42);

  return (
    <ChartFrame
      label={`Alarm burden, ${label}: detection rate and false-alarm rate across alarm thresholds. At the shipped operating point, detection is ${pct(
        model.earlyWarning.detectionRate,
      )} and the false-alarm rate is ${pct(model.earlyWarning.falseAlarmRate)}.`}
      xDomain={[0, xMax]}
      yDomain={[0, 1]}
      width={W}
      height={H}
      margins={M}
      xLabel="alarm threshold"
      yLabel="rate"
      defs={
        <linearGradient id={`band-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--color-low)" stopOpacity={0.22} />
          <stop offset="100%" stopColor="var(--color-low)" stopOpacity={0.04} />
        </linearGradient>
      }
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
      {({ h }) => (
        <>
          <DrawArea d={band} fill={`url(#band-${id})`} />

          {shippedThreshold !== undefined && shippedThreshold <= xMax && (
            <>
              <line
                x1={sx(shippedThreshold)}
                y1={M.t}
                x2={sx(shippedThreshold)}
                y2={h - M.b}
                stroke="var(--color-line-bright)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <text
                x={sx(shippedThreshold) + 5}
                y={M.t + 10}
                className="fill-dim font-mono"
                fontSize={9}
              >
                shipped
              </text>
            </>
          )}

          <DrawPath d={toPath(det, sx, sy)} stroke="var(--color-low)" width={2.2} delay={0.1} />
          <DrawPath d={toPath(far, sx, sy)} stroke="var(--color-high)" width={2.2} delay={0.2} />

          <SeriesLabel
            x={sx(dl[0]) + 6}
            y={sy(dl[1]) - 8}
            text="caught"
            color="var(--color-low-ink)"
          />
          <SeriesLabel
            x={sx(fl[0]) + 6}
            y={sy(fl[1]) + 14}
            text="false alarms"
            color="var(--color-high-ink)"
          />

          {shippedThreshold !== undefined && (
            <Annotation
              x={sx(shippedThreshold)}
              y={sy(model.earlyWarning.falseAlarmRate)}
              dx={16}
              dy={-24}
              title={pct(model.earlyWarning.falseAlarmRate, 0)}
              detail="of healthy patients alarmed"
              color="var(--color-high)"
            />
          )}
        </>
      )}
    </ChartFrame>
  );
}
