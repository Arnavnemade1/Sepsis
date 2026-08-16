import { generalization } from "@/lib/results";
import { f3 } from "@/lib/format";

const W = 620;
const H = 260;

/**
 * The generalization gap as a slope, not a sentence.
 *
 * Two points and the line between them. The drop is the entire message, and a
 * slope chart is the only form where the magnitude of a drop is the literal
 * visual magnitude of the thing you are looking at.
 */
export function GapSlope() {
  const lo = 0.6;
  const hi = 0.8;
  const y = (v: number) => 40 + (1 - (v - lo) / (hi - lo)) * (H - 90);
  const xa = 150;
  const xb = W - 190;

  const a = generalization.internalAuroc;
  const b = generalization.externalAuroc;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Within-hospital AUROC ${f3(a)} falls to ${f3(
          b,
        )} across hospitals — a generalization gap of ${f3(generalization.gap)} for the same model on the same features.`}
        className="block w-full"
        style={{ overflow: "visible" }}
      >
        {[0.6, 0.65, 0.7, 0.75, 0.8].map((v) => (
          <g key={v}>
            <line
              x1={xa - 40}
              y1={y(v)}
              x2={xb + 40}
              y2={y(v)}
              stroke="var(--color-line-soft)"
            />
            <text
              x={xa - 50}
              y={y(v) + 3}
              textAnchor="end"
              className="fill-faint font-mono"
              fontSize={9.5}
            >
              {v.toFixed(2)}
            </text>
          </g>
        ))}

        {/* the drop, shaded */}
        <path
          d={`M${xa} ${y(a)} L${xb} ${y(b)} L${xb} ${y(lo)} L${xa} ${y(lo)} Z`}
          fill="var(--color-high)"
          opacity={0.08}
        />
        <line
          x1={xa}
          y1={y(a)}
          x2={xb}
          y2={y(b)}
          stroke="var(--color-high)"
          strokeWidth={2}
        />

        {/* endpoints */}
        <circle cx={xa} cy={y(a)} r={6} fill="var(--color-dim)" stroke="var(--color-bg)" strokeWidth={2} />
        <circle cx={xb} cy={y(b)} r={6} fill="var(--color-ink)" stroke="var(--color-bg)" strokeWidth={2} />

        <text x={xa} y={y(a) - 20} textAnchor="middle" className="fill-ink font-mono" fontSize={22} fontWeight={500}>
          {f3(a)}
        </text>
        <text x={xa} y={y(a) - 6} textAnchor="middle" className="fill-dim font-mono" fontSize={10}>
          within hospital
        </text>
        <text x={xa} y={H - 22} textAnchor="middle" className="fill-faint font-mono" fontSize={9}>
          held-out split, same site
        </text>
        <text x={xa} y={H - 10} textAnchor="middle" className="fill-faint font-mono" fontSize={9}>
          40,335 patients
        </text>

        <text x={xb} y={y(b) - 20} textAnchor="middle" className="fill-ink font-mono" fontSize={22} fontWeight={500}>
          {f3(b)}
        </text>
        <text x={xb} y={y(b) - 6} textAnchor="middle" className="fill-dim font-mono" fontSize={10}>
          across hospitals
        </text>
        <text x={xb} y={H - 22} textAnchor="middle" className="fill-faint font-mono" fontSize={9}>
          trained on one, tested on the other
        </text>
        <text x={xb} y={H - 10} textAnchor="middle" className="fill-faint font-mono" fontSize={9}>
          6,000 + 6,000 patients
        </text>

        {/* the gap itself */}
        <line
          x1={xb + 34}
          y1={y(a)}
          x2={xb + 34}
          y2={y(b)}
          stroke="var(--color-high)"
          strokeWidth={1.5}
        />
        <line x1={xb + 28} y1={y(a)} x2={xb + 40} y2={y(a)} stroke="var(--color-high)" strokeWidth={1.5} />
        <line x1={xb + 28} y1={y(b)} x2={xb + 40} y2={y(b)} stroke="var(--color-high)" strokeWidth={1.5} />
        <text
          x={xb + 48}
          y={(y(a) + y(b)) / 2 + 1}
          className="fill-high-ink font-mono"
          fontSize={15}
          fontWeight={600}
        >
          −{f3(generalization.gap)}
        </text>
        <text x={xb + 48} y={(y(a) + y(b)) / 2 + 15} className="fill-dim font-mono" fontSize={9}>
          what a new
        </text>
        <text x={xb + 48} y={(y(a) + y(b)) / 2 + 26} className="fill-dim font-mono" fontSize={9}>
          hospital costs
        </text>
      </svg>
    </figure>
  );
}
