/**
 * Hyperdimensional computing, shown as it is ACTUALLY used.
 *
 * The temptation with HDC is to present it as the thing that makes the prediction,
 * because the algebra is striking and 10,000 dimensions sound impressive. That would
 * be false here. The shipped detector's roster is hgb+hgb2+et+lr+xgb@utility with
 * use_hdc=False, and in this project's own ablation classical HDC scores AUROC 0.594
 * against gradient boosting's 0.690.
 *
 * What HDC earns its place doing is different and more interesting: it is an
 * associative memory over precedent patient-hours, and it is the substrate of the
 * cross-modal resonance gate that removes 91% of artifact false alarms. It is also,
 * measurably, most of where the runtime goes. All three facts are on the panel.
 *
 * The bit strips are deterministic (seeded LCG at module scope), so the server and
 * client renders agree and no hydration warning appears.
 */

const D = 64; // strip length on screen; the real vectors are 10,000-D

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function bits(seed: number, n = D): number[] {
  const r = lcg(seed);
  return Array.from({ length: n }, () => (r() < 0.5 ? -1 : 1));
}

const bind = (a: number[], b: number[]) => a.map((v, i) => v * b[i]);
const bundle = (vs: number[][]) =>
  vs[0].map((_, i) => (vs.reduce((s, v) => s + v[i], 0) >= 0 ? 1 : -1));

const CHANNEL = bits(11);
const LEVEL = bits(29);
const BOUND = bind(CHANNEL, LEVEL);
const OTHER = [bind(bits(41), bits(53)), bind(bits(67), bits(71))];
const BUNDLED = bundle([BOUND, ...OTHER]);

function Strip({ v, tone = "ink" }: { v: number[]; tone?: "ink" | "dim" | "accent" }) {
  const fill =
    tone === "accent" ? "fill-accent" : tone === "dim" ? "fill-dim" : "fill-muted";
  return (
    <svg
      viewBox={`0 0 ${D} 8`}
      preserveAspectRatio="none"
      className="block h-4 w-full"
      role="img"
      aria-label="bipolar hypervector segment"
    >
      <rect x={0} y={0} width={D} height={8} className="fill-s2" />
      {v.map((b, i) =>
        b > 0 ? <rect key={i} x={i} y={0} width={1} height={8} className={fill} /> : null,
      )}
    </svg>
  );
}

function Step({
  n,
  op,
  title,
  body,
  children,
}: {
  n: string;
  op: string;
  title: string;
  body: string;
  children?: React.ReactNode;
}) {
  return (
    <li className="grid grid-cols-[auto_1fr] gap-x-4 border-b border-line-soft py-4 last:border-b-0">
      <span className="num pt-0.5 font-mono text-[11px] text-faint">{n}</span>
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-2.5">
          <h4 className="text-[13.5px] font-semibold text-ink">{title}</h4>
          <code className="rounded-sm border border-line bg-s1 px-1.5 py-0.5 font-mono text-[11px] text-accent-ink">
            {op}
          </code>
        </div>
        <p className="mt-1 text-[12.5px] leading-[1.55] text-dim">{body}</p>
        {children ? <div className="mt-2.5 space-y-1">{children}</div> : null}
      </div>
    </li>
  );
}

export interface HdcFacts {
  ablation: { name: string; auroc: number; aurocSd: number; utility: number }[];
  shippedRoster: string[];
  runtime: { beds: number; phases: { phase: string; share: number; msPerTick: number }[] };
}

export function HdcPipeline({ facts }: { facts: HdcFacts }) {
  const hdcShare = facts.runtime.phases
    .filter((p) => p.phase === "hdc_memory" || p.phase === "resonance_gate")
    .reduce((s, p) => s + p.share, 0);
  const best = facts.ablation.reduce((a, b) => (b.auroc > a.auroc ? b : a));
  const hdcRow = facts.ablation.find((a) => /HDC \(classical\)/.test(a.name));

  return (
    <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      {/* ── the algebra, step by step ── */}
      <div className="min-w-0">
        <h3 className="font-display text-[15px] font-semibold tracking-[-0.01em] text-ink">
          The algebra, as it runs on every tick
        </h3>
        <p className="mt-1.5 text-[13px] leading-[1.55] text-dim">
          Three operations over 10,000-dimensional bipolar vectors. Each strip below is a
          64-bit window onto one real vector — lit cells are <code className="font-mono text-[11.5px] text-muted">+1</code>,
          dark are <code className="font-mono text-[11.5px] text-muted">−1</code>.
        </p>

        <ol className="mt-4 border-t border-line">
          <Step
            n="01"
            op="encode"
            title="A reading becomes a vector"
            body="Every channel gets a fixed random id vector, and every value a level vector that morphs continuously from the channel's floor to its ceiling — so numeric closeness becomes geometric closeness."
          >
            <Strip v={CHANNEL} tone="dim" />
            <Strip v={LEVEL} tone="dim" />
          </Step>

          <Step
            n="02"
            op="bind  ⊗"
            title="Channel and value are fused"
            body="Elementwise multiply. The result is near-orthogonal to both inputs — it says 'this channel held this value' as a single object that neither factor can be mistaken for."
          >
            <Strip v={BOUND} tone="accent" />
          </Step>

          <Step
            n="03"
            op="bundle  ⊕"
            title="A patient-hour becomes one object"
            body="Sum the bound pairs and take the sign. Every channel a patient had this hour collapses into one vector of fixed width — the same size whether six channels arrived or sixty."
          >
            <Strip v={BUNDLED} />
          </Step>

          <Step
            n="04"
            op="project"
            title="Projected onto the sepsis axis"
            body="One direction is learned from real precedent: the septic-minus-healthy centroid. Each acquisition pathway — camera, wearable, labs — is projected onto it separately, giving one calibrated score per pathway on a shared scale."
          />

          <Step
            n="05"
            op="consensus"
            title="Agreement, not magnitude, decides"
            body="A patient deteriorating moves every independent pathway together. A sensor glitching moves one. The gate takes the median and holds an uncorroborated pathway out until a second confirms it — which is where the 91% artifact false-alarm removal comes from."
          />
        </ol>
      </div>

      {/* ── the honest scoreboard ── */}
      <div className="min-w-0">
        <h3 className="font-display text-[15px] font-semibold tracking-[-0.01em] text-ink">
          What it is, and is not, doing
        </h3>

        <div className="mt-4 rounded-md border border-med/30 bg-med/5 p-4">
          <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-med-ink">
            not the detector
          </p>
          <p className="mt-2 text-[13px] leading-[1.6] text-muted">
            The shipped roster is{" "}
            <code className="font-mono text-[12px] text-ink">
              {facts.shippedRoster.join("+")}
            </code>{" "}
            with <code className="font-mono text-[12px] text-ink">use_hdc=False</code>. On
            this project&rsquo;s own ablation, classical HDC scores{" "}
            <span className="num font-semibold text-ink">{hdcRow?.auroc.toFixed(3)}</span>{" "}
            AUROC against gradient boosting&rsquo;s{" "}
            <span className="num font-semibold text-ink">{best.auroc.toFixed(3)}</span>. It
            earns its place as an associative memory and a robustness gate — not as the
            thing that makes the prediction.
          </p>
        </div>

        <table className="mt-5 w-full border-collapse text-[13px]">
          <caption className="sr-only">
            Discrimination ablation: gradient boosting versus classical and quantum HDC
          </caption>
          <thead>
            <tr className="border-b border-line text-left">
              <th className="py-2 pr-3 font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-faint">
                model
              </th>
              <th className="py-2 pr-3 text-right font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-faint">
                AUROC
              </th>
              <th className="py-2 text-right font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-faint">
                utility
              </th>
            </tr>
          </thead>
          <tbody>
            {facts.ablation.map((a) => (
              <tr key={a.name} className="border-b border-line-soft">
                <td className="py-2 pr-3 text-muted">{a.name}</td>
                <td className="num py-2 pr-3 text-right text-ink">
                  {a.auroc.toFixed(3)}
                  <span className="ml-1 text-faint">±{a.aurocSd.toFixed(3)}</span>
                </td>
                <td className="num py-2 text-right text-muted">
                  {a.utility >= 0 ? "+" : ""}
                  {a.utility.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="mt-6 rounded-md border border-line bg-s1 p-4">
          <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-faint">
            where the tick actually goes · {facts.runtime.beds} beds
          </p>
          <ul className="mt-3 space-y-2">
            {facts.runtime.phases.slice(0, 5).map((p) => {
              const isHdc = p.phase === "hdc_memory" || p.phase === "resonance_gate";
              return (
                <li key={p.phase} className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-2">
                  <span
                    className={`font-mono text-[11.5px] ${isHdc ? "text-accent-ink" : "text-dim"}`}
                  >
                    {p.phase}
                  </span>
                  <span className="h-1.5 overflow-hidden rounded-sm bg-s3">
                    <span
                      className={`block h-full rounded-sm ${isHdc ? "bg-accent" : "bg-dim"}`}
                      style={{ width: `${Math.round(p.share * 100)}%` }}
                    />
                  </span>
                  <span className="num text-right font-mono text-[11.5px] text-muted">
                    {(p.share * 100).toFixed(1)}%
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="mt-3 text-[12.5px] leading-[1.55] text-dim">
            HDC work is{" "}
            <span className="num font-semibold text-accent-ink">
              {(hdcShare * 100).toFixed(0)}%
            </span>{" "}
            of the runtime budget — the single most expensive thing in the loop, for a
            component deliberately kept out of the detector. That trade is stated rather
            than hidden: it buys artifact rejection and case retrieval, not accuracy.
          </p>
        </div>
      </div>
    </div>
  );
}
