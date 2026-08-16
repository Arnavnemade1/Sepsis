import { headline } from "@/lib/results";
import { pct } from "@/lib/format";

const COLS = 20;
const ROWS = 5;
const R = 5.2;
const GAP = 15;
const PAD = 2;

/**
 * A hundred patients, one dot each.
 *
 * This is the visual the page was missing. AUROC 0.716 is a statement about
 * ranking that almost nobody can convert into an expectation; "of a hundred
 * septic patients, fifty-three were alarmed and forty-seven were missed" is the
 * same evaluation stated in the only unit that matters at a bedside.
 *
 * It also refuses to flatter: the missed patients occupy just as much of the
 * picture as the caught ones, and the companion panel showing false alarms is
 * deliberately the same size.
 */
function Grid({
  filled,
  fillColor,
  restColor,
  ariaLabel,
}: {
  filled: number;
  fillColor: string;
  restColor: string;
  ariaLabel: string;
}) {
  const W = COLS * GAP + PAD * 2;
  const H = ROWS * GAP + PAD * 2;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={ariaLabel}
      className="block w-full"
    >
      {Array.from({ length: COLS * ROWS }).map((_, i) => {
        const row = Math.floor(i / COLS);
        const col = i % COLS;
        const on = i < filled;
        return (
          <circle
            key={i}
            cx={PAD + col * GAP + GAP / 2}
            cy={PAD + row * GAP + GAP / 2}
            r={R}
            fill={on ? fillColor : restColor}
            fillOpacity={on ? 0.95 : 0.22}
          />
        );
      })}
    </svg>
  );
}

export function OutcomeWaffle() {
  const detected = Math.round(headline.production.detectionRate * 100);
  const missed = 100 - detected;
  const falseAlarmed = Math.round(headline.production.falseAlarmRate * 100);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* the win, and its limit, in the same picture */}
      <div className="rounded-lg border border-line bg-s1/62 p-6 backdrop-blur-xl">
        <p className="eyebrow">Of 100 patients who developed sepsis</p>
        <div className="mt-5 flex items-baseline gap-3">
          <span className="num font-mono text-[40px] leading-none text-low-ink">
            {detected}
          </span>
          <span className="text-[14px] text-muted">
            were alarmed before onset —{" "}
            <span className="text-ink">{missed} were not</span>
          </span>
        </div>
        <div className="mt-6">
          <Grid
            filled={detected}
            fillColor="var(--color-low)"
            restColor="var(--color-nodata)"
            ariaLabel={`${detected} of 100 septic patients were alarmed before onset; ${missed} were missed.`}
          />
        </div>
        <p className="mt-5 text-[13px] leading-relaxed text-dim text-pretty">
          Median warning for the ones it caught:{" "}
          <span className="num text-ink">
            {headline.production.medianLeadHours.toFixed(1)} hours
          </span>{" "}
          before Sepsis-3 onset.
        </p>
      </div>

      {/* the cost, at the same scale, deliberately */}
      <div className="rounded-lg border border-line bg-s1/62 p-6 backdrop-blur-xl">
        <p className="eyebrow">Of 100 patients who did not</p>
        <div className="mt-5 flex items-baseline gap-3">
          <span className="num font-mono text-[40px] leading-none text-high-ink">
            {falseAlarmed}
          </span>
          <span className="text-[14px] text-muted">
            still triggered at least one alarm
          </span>
        </div>
        <div className="mt-6">
          <Grid
            filled={falseAlarmed}
            fillColor="var(--color-high)"
            restColor="var(--color-nodata)"
            ariaLabel={`${falseAlarmed} of 100 non-septic patients triggered at least one false alarm.`}
          />
        </div>
        <p className="mt-5 text-[13px] leading-relaxed text-dim text-pretty">
          This is the honest bottleneck, and it is not fixable by policy — four
          families of alarm rule were searched and none beat a plain threshold.
        </p>
      </div>
    </div>
  );
}

export { pct };
