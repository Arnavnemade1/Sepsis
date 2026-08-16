import { qhdc } from "@/lib/results";
import { f3 } from "@/lib/format";

const W = 620;
const ROW = 56;

/**
 * The QHDC ablation as bars with error whiskers.
 *
 * A table of mean ± SD hides the finding; bars show it instantly — the quantum
 * model lands on top of the classical HDC model, and both sit well below plain
 * gradient-boosted trees. The whiskers matter as much as the bars: the two
 * losing models overlap each other completely, which is the honest reading.
 */
export function AblationBars() {
  const entries = Object.entries(qhdc.ablation);
  const H = entries.length * ROW + 46;
  const lo = 0.5;
  const hi = 0.75;
  const L = 168;
  const R = 96;
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Ablation: ${entries
          .map(([n, m]) => `${n} AUROC ${f3(m.auroc.mean)} ± ${f3(m.auroc.sd)}`)
          .join("; ")}`}
        className="block w-full"
        style={{ overflow: "visible" }}
      >
        {[0.5, 0.55, 0.6, 0.65, 0.7, 0.75].map((v) => (
          <g key={v}>
            <line x1={x(v)} y1={26} x2={x(v)} y2={H - 20} stroke="var(--color-line-soft)" />
            <text x={x(v)} y={18} textAnchor="middle" className="fill-faint font-mono" fontSize={9.5}>
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {/* chance */}
        <line
          x1={x(0.5)}
          y1={26}
          x2={x(0.5)}
          y2={H - 20}
          stroke="var(--color-faint)"
          strokeDasharray="3 3"
        />

        {entries.map(([name, m], i) => {
          const y = 44 + i * ROW;
          const best = name.startsWith("GBM");
          const c = best ? "var(--color-accent)" : "var(--color-dim)";
          return (
            <g key={name}>
              <text
                x={0}
                y={y + 4}
                className={best ? "fill-ink" : "fill-muted"}
                fontSize={12}
                fontWeight={best ? 600 : 400}
              >
                {name}
              </text>

              <rect
                x={x(lo)}
                y={y - 9}
                width={Math.max(0, x(m.auroc.mean) - x(lo))}
                height={18}
                fill={c}
                fillOpacity={best ? 0.85 : 0.4}
                rx={2}
              />

              {/* ± one standard deviation across folds */}
              <line
                x1={x(m.auroc.mean - m.auroc.sd)}
                y1={y}
                x2={x(m.auroc.mean + m.auroc.sd)}
                y2={y}
                stroke="var(--color-ink)"
                strokeOpacity={0.65}
                strokeWidth={1.4}
              />
              {[m.auroc.mean - m.auroc.sd, m.auroc.mean + m.auroc.sd].map((v, k) => (
                <line
                  key={k}
                  x1={x(v)}
                  y1={y - 5}
                  x2={x(v)}
                  y2={y + 5}
                  stroke="var(--color-ink)"
                  strokeOpacity={0.65}
                  strokeWidth={1.4}
                />
              ))}

              <text
                x={W - R + 10}
                y={y + 4}
                className="fill-ink font-mono"
                fontSize={12}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {f3(m.auroc.mean)}
              </text>
              <text
                x={W - R + 58}
                y={y + 4}
                className="fill-faint font-mono"
                fontSize={10}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                ±{f3(m.auroc.sd)}
              </text>
            </g>
          );
        })}

        <text x={x(0.5) + 5} y={H - 6} className="fill-faint font-mono" fontSize={9}>
          chance
        </text>
      </svg>
    </figure>
  );
}
