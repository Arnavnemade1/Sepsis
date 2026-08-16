import { ChartFrame, Annotation } from "./chart-frame";
import { scaleX, scaleY, toPath } from "@/lib/scales";
import { pct, signed } from "@/lib/format";
import { operatingPoint } from "@/lib/results";

const M = { l: 48, r: 22, t: 16, b: 38 };
const W = 620;
const H = 340;

/**
 * The operating-point finding, as a picture.
 *
 * The held-out rows in artifacts/operating_point.json are three samples of one
 * curve: utility as a function of where you put the alarm threshold. Shown as a
 * table they look like three options. Shown as a curve they show the actual
 * finding — the shipped hand-set threshold sat far out on the tail, where almost
 * nothing crossed it, and simply moving it left recovered most of the available
 * utility without touching the model.
 *
 * The AUROC is identical at every point on this axis. That is the whole story:
 * everything here was won in the decision, not the score.
 */
export function ThresholdSweep() {
  const rows = operatingPoint.heldOut.rows;
  const shipped = rows.find((r) => r.threshold === 0.12)!;
  const derived = rows.find((r) => r.threshold === operatingPoint.high)!;
  const optimal = rows.find((r) => r.threshold === 0.025)!;

  const xs = [0, 0.025, 0.05, 0.0725, 0.1, 0.12, 0.15];
  const sx = scaleX([0, 0.16], W, M);
  const sy = scaleY([0, 0.42], H, M);

  // Interpolate a smooth utility curve through the three measured points plus
  // the known endpoints, so the shape is readable without implying we measured
  // every threshold.
  const known: [number, number][] = [
    [0.0, 0.05],
    [0.025, optimal.utility],
    [0.0725, derived.utility],
    [0.12, shipped.utility],
    [0.15, 0.031],
  ];
  const curve = toPath(known, sx, sy);

  return (
    <ChartFrame
      label={`Held-out utility against alarm threshold. The shipped hand-set threshold of 0.12 returned ${signed(
        shipped.utility,
        3,
      )}; the out-of-fold derived threshold of ${operatingPoint.high} returned ${signed(
        derived.utility,
        3,
      )} — a gain gathered entirely by moving the decision boundary, at an unchanged AUROC of ${operatingPoint.heldOut.auroc}.`}
      xDomain={[0, 0.16]}
      yDomain={[0, 0.42]}
      width={W}
      height={H}
      margins={M}
      xTicks={xs}
      xLabel="alarm threshold"
      yLabel="held-out utility"
      table={
        <table>
          <caption>Held-out utility by operating point</caption>
          <thead>
            <tr>
              <th scope="col">Operating point</th>
              <th scope="col">Threshold</th>
              <th scope="col">Utility</th>
              <th scope="col">Detection</th>
              <th scope="col">False alarms</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.threshold}>
                <td>{r.label}</td>
                <td>{r.threshold}</td>
                <td>{signed(r.utility, 4)}</td>
                <td>{pct(r.detection, 1)}</td>
                <td>{pct(r.far, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      {({ h }) => (
        <>
          {/* the tail the shipped threshold was sitting in */}
          <rect
            x={sx(operatingPoint.probP95)}
            y={M.t}
            width={Math.max(0, W - M.r - sx(operatingPoint.probP95))}
            height={h - M.b - M.t}
            fill="var(--color-high)"
            opacity={0.07}
          />
          <text
            x={W - M.r - 6}
            y={M.t + 14}
            textAnchor="end"
            className="fill-high-ink font-mono"
            fontSize={9}
          >
            above the 95th percentile of the model&apos;s own output
          </text>

          <path
            d={curve}
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth={2.2}
            strokeLinejoin="round"
          />

          {/* measured points */}
          {[
            { r: optimal, c: "var(--color-dim)" },
            { r: derived, c: "var(--color-low)" },
            { r: shipped, c: "var(--color-high)" },
          ].map(({ r, c }) => (
            <circle
              key={r.threshold}
              cx={sx(r.threshold)}
              cy={sy(r.utility)}
              r={5}
              fill={c}
              stroke="var(--color-bg)"
              strokeWidth={1.5}
            />
          ))}

          {/* the gain, as a vertical distance you can actually see */}
          <line
            x1={sx(shipped.threshold)}
            y1={sy(shipped.utility)}
            x2={sx(shipped.threshold)}
            y2={sy(derived.utility)}
            stroke="var(--color-low)"
            strokeWidth={1.5}
            strokeDasharray="3 3"
          />
          <line
            x1={sx(derived.threshold)}
            y1={sy(derived.utility)}
            x2={sx(shipped.threshold)}
            y2={sy(derived.utility)}
            stroke="var(--color-low)"
            strokeWidth={1.5}
            strokeDasharray="3 3"
          />
          {/* State the absolute gain AND the relative one. "+4.1" alone reads
              as either 4.1 points or 4.1 percent, and it is neither. */}
          <text
            x={sx(shipped.threshold) + 8}
            y={(sy(shipped.utility) + sy(derived.utility)) / 2 - 4}
            className="fill-low-ink font-mono"
            fontSize={11}
            fontWeight={600}
          >
            {signed(derived.utility - shipped.utility, 3)} utility
          </text>
          <text
            x={sx(shipped.threshold) + 8}
            y={(sy(shipped.utility) + sy(derived.utility)) / 2 + 9}
            className="fill-dim font-mono"
            fontSize={9}
          >
            +{Math.round(((derived.utility - shipped.utility) / shipped.utility) * 100)}% relative
          </text>

          <Annotation
            x={sx(derived.threshold)}
            y={sy(derived.utility)}
            dx={-14}
            dy={-34}
            anchor="end"
            title={`derived ${operatingPoint.high}`}
            detail={`${pct(derived.detection, 0)} detection`}
            color="var(--color-low)"
          />
          <Annotation
            x={sx(shipped.threshold)}
            y={sy(shipped.utility)}
            dx={10}
            dy={34}
            title="shipped guess 0.12"
            detail={`${pct(shipped.detection, 0)} detection`}
            color="var(--color-high)"
          />
        </>
      )}
    </ChartFrame>
  );
}
