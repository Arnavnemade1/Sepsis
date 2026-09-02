import { AbsoluteFill, useCurrentFrame } from "remotion";

import { DURATION, SCENE, SEAM_FRAMES, W, H } from "./constants";
import { Mono, Hairline, Card, CountUp, Waveform, Stamp, ramp, noise, colors } from "./primitives";

const { surface, line, text, state, stateInk, accent, font } = colors;

/**
 * Real numbers, passed in from data/results.json so the video cannot drift out
 * of sync with the evaluation it describes.
 *
 * Declared as a `type`, not an `interface`, deliberately: Remotion's
 * <Composition> constrains props to `Record<string, unknown>`, and only a type
 * alias gets the implicit index signature that satisfies it. As an interface
 * this fails to compile with a confusing "missing the following properties"
 * error pointing at Root.tsx rather than here.
 */
export type ExplainerProps = {
  auroc: number;
  ece: number;
  leadHours: number;
  threshold: number;
};

/* ── signal generators — deterministic, so every render matches ──────────── */

function vitalsPath(seed: number, n = 90, amp = 0.28, drift = 0) {
  return Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    const base = 0.5 + Math.sin(t * Math.PI * 6 + seed) * amp * (1 - t * drift);
    const jitter = (noise(i + seed * 100) - 0.5) * 0.06;
    return [t, Math.min(0.95, Math.max(0.05, base + jitter))] as const;
  });
}

/** A risk trajectory that stays flat, then climbs — the shape the detector is
 *  actually looking for. */
function riskPath(n = 90) {
  return Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    const y = 0.92 - 0.62 / (1 + Math.exp(-(t - 0.62) * 14));
    return [t, y + (noise(i * 3.1) - 0.5) * 0.02] as const;
  });
}

/* ── the composition ─────────────────────────────────────────────────────── */

export function Explainer({ auroc, ece, leadHours, threshold }: ExplainerProps) {
  const frame = useCurrentFrame();

  // The seam: over the last SEAM_FRAMES everything decays back to the state at
  // frame 0, so `loop` has nothing visible to cut. This multiplier is applied to
  // every scene's opacity.
  const seam = ramp(frame, [DURATION - SEAM_FRAMES, DURATION], [1, 0]);

  const scene = (k: keyof typeof SCENE) => {
    const { from, len } = SCENE[k];
    const local = frame - from;
    const inOp = ramp(frame, [from, from + 12]);
    const outOp = ramp(frame, [from + len - 12, from + len], [1, 0]);
    return { local, active: frame >= from && frame < from + len, opacity: inOp * outOp * seam };
  };

  const acquire = scene("acquire");
  const feature = scene("feature");
  const calibrate = scene("calibrate");
  const verify = scene("verify");
  const deliver = scene("deliver");

  return (
    <AbsoluteFill
      style={{
        background: surface.bg,
        fontFamily: font.sans,
        padding: 56,
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* persistent chrome */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, opacity: seam }}>
        <div
          style={{
            width: 6,
            height: 6,
            borderRadius: 3,
            background: state.low,
          }}
        />
        <Mono size={11} color={text.dim} style={{ textTransform: "uppercase", letterSpacing: "0.16em" }}>
          AegisOS · one patient · live
        </Mono>
        <div style={{ flex: 1 }} />
        <Mono size={11} color={text.faint}>
          {String(Math.floor(frame / 30)).padStart(2, "0")}:
          {String(frame % 30).padStart(2, "0")}
        </Mono>
      </div>
      <Hairline style={{ marginTop: 14, opacity: seam }} />

      <div style={{ position: "relative", flex: 1, marginTop: 40 }}>
        {/* ── 1 · ACQUIRE ─────────────────────────────────────────────── */}
        <SceneBox opacity={acquire.opacity}>
          <SceneTitle n="01" title="Acquire" sub="wearable · 1 Hz · six-hour window" />
          <div style={{ display: "flex", flexDirection: "column", gap: 26, marginTop: 34 }}>
            {[
              { label: "HR", seed: 0.2, amp: 0.3 },
              { label: "SpO₂", seed: 1.7, amp: 0.16 },
              { label: "RESP", seed: 3.4, amp: 0.24 },
            ].map((s, i) => (
              <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 20 }}>
                <Mono size={11} color={text.faint} style={{ width: 46 }}>
                  {s.label}
                </Mono>
                <Waveform
                  points={vitalsPath(s.seed, 90, s.amp)}
                  width={880}
                  height={62}
                  progress={ramp(frame, [10 + i * 14, 78 + i * 14])}
                  stroke={text.dim}
                />
              </div>
            ))}
          </div>
        </SceneBox>

        {/* ── 2 · FEATURE ─────────────────────────────────────────────── */}
        <SceneBox opacity={feature.opacity}>
          <SceneTitle n="02" title="Construct" sub="94 features · identical at train and serve time" />
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(12, 1fr)",
              gap: 8,
              marginTop: 40,
              width: 880,
            }}
          >
            {Array.from({ length: 48 }).map((_, i) => {
              const on = ramp(frame, [SCENE.feature.from + 8 + i * 1.1, SCENE.feature.from + 22 + i * 1.1]);
              const h = 14 + noise(i * 7.7) * 34;
              return (
                <div
                  key={i}
                  style={{
                    height: 52,
                    display: "flex",
                    alignItems: "flex-end",
                  }}
                >
                  <div
                    style={{
                      width: "100%",
                      height: h * on,
                      background: line.lineBright,
                      borderRadius: 1,
                    }}
                  />
                </div>
              );
            })}
          </div>
          <Mono size={11} color={text.faint} style={{ marginTop: 22, display: "block" }}>
            no labs · no chart-derived shortcuts · no train/serve skew
          </Mono>
        </SceneBox>

        {/* ── 3 · CALIBRATE — the scene that earns the video ───────────── */}
        <SceneBox opacity={calibrate.opacity}>
          <SceneTitle
            n="03"
            title="Calibrate"
            sub="a probability, not a ranking"
          />
          <div style={{ display: "flex", gap: 44, marginTop: 34, alignItems: "flex-start" }}>
            {/* risk trajectory crossing the threshold */}
            <div style={{ position: "relative" }}>
              <Waveform
                points={riskPath()}
                width={520}
                height={190}
                progress={ramp(frame, [SCENE.calibrate.from + 10, SCENE.calibrate.from + 88])}
                stroke={accent.accent}
                strokeWidth={2}
              />
              {/* alarm threshold */}
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: 190 * 0.62,
                  width: 520,
                  borderTop: `1px dashed ${line.lineBright}`,
                  opacity: ramp(frame, [SCENE.calibrate.from + 30, SCENE.calibrate.from + 46]),
                }}
              />
              <Mono
                size={10}
                color={text.faint}
                style={{
                  position: "absolute",
                  left: 0,
                  top: 190 * 0.62 + 6,
                  opacity: ramp(frame, [SCENE.calibrate.from + 30, SCENE.calibrate.from + 46]),
                }}
              >
                alarm threshold {threshold}
              </Mono>
            </div>

            {/* reliability diagram — observed frequency landing on the diagonal */}
            <div>
              <Mono size={10} color={text.faint}>
                calibration
              </Mono>
              <svg width={190} height={190} style={{ display: "block", marginTop: 8 }}>
                <rect x={0} y={0} width={190} height={190} fill="none" stroke={line.line} />
                <line x1={0} y1={190} x2={190} y2={0} stroke={text.faint} strokeDasharray="4 4" />
                {[0.18, 0.35, 0.52, 0.7, 0.86].map((p, i) => {
                  const appear = ramp(frame, [
                    SCENE.calibrate.from + 44 + i * 9,
                    SCENE.calibrate.from + 58 + i * 9,
                  ]);
                  const off = (noise(i * 5.5) - 0.5) * 0.05;
                  return (
                    <circle
                      key={i}
                      cx={p * 190}
                      cy={190 - (p + off) * 190}
                      r={4 * appear}
                      fill={accent.accent}
                      stroke={surface.s1}
                      strokeWidth={1.5}
                    />
                  );
                })}
              </svg>
            </div>
          </div>

          <div style={{ display: "flex", gap: 40, marginTop: 26 }}>
            <Readout
              label="cross-hospital AUROC"
              node={
                <CountUp
                  frame={frame}
                  from={0}
                  to={auroc}
                  start={SCENE.calibrate.from + 20}
                  end={SCENE.calibrate.from + 70}
                  decimals={3}
                  size={26}
                  color={accent.accentInk}
                />
              }
            />
            <Readout
              label="calibration error"
              node={
                <CountUp
                  frame={frame}
                  from={0}
                  to={ece}
                  start={SCENE.calibrate.from + 40}
                  end={SCENE.calibrate.from + 90}
                  decimals={4}
                  size={26}
                  color={accent.accentInk}
                />
              }
            />
            {/* Was "median lead time". Retracted 2026-08-30 — it does not beat a
                random-score null at matched detection (docs/LEAD_TIME_NULL.md). */}
            <Readout
              label="calibration error"
              node={
                <CountUp
                  frame={frame}
                  from={0}
                  to={ece}
                  start={SCENE.calibrate.from + 55}
                  end={SCENE.calibrate.from + 105}
                  decimals={4}
                  size={26}
                  color={text.ink}
                />
              }
            />
          </div>
          <Mono size={11} color={text.faint} style={{ marginTop: 18, display: "block" }}>
            ECE {ece.toFixed(4)} — the number means what it says, at a hospital it was not trained on
          </Mono>
        </SceneBox>

        {/* ── 4 · VERIFY ──────────────────────────────────────────────── */}
        <SceneBox opacity={verify.opacity}>
          <SceneTitle n="04" title="Verify" sub="every protocol proved before it is shown" />
          <div style={{ display: "flex", gap: 22, marginTop: 34 }}>
            <Card style={{ padding: 20, width: 430 }}>
              <Mono size={10} color={text.faint} style={{ textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Candidate A
              </Mono>
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 9 }}>
                {[
                  "antibiotic := pip/tazo",
                  "fluid_rate := 4.2 mL/kg/hr",
                  "vasopressor := 0.08 mcg/kg/min",
                ].map((l, i) => (
                  <div
                    key={l}
                    style={{ opacity: ramp(frame, [SCENE.verify.from + 8 + i * 7, SCENE.verify.from + 20 + i * 7]) }}
                  >
                    <Mono size={12} color={text.muted}>
                      {l}
                    </Mono>
                  </div>
                ))}
              </div>
              <Hairline style={{ margin: "16px 0" }} />
              <Stamp
                kind="certified"
                label="certified"
                opacity={ramp(frame, [SCENE.verify.from + 38, SCENE.verify.from + 50])}
              />
            </Card>

            <Card
              style={{ padding: 20, width: 430 }}
              borderColor={
                ramp(frame, [SCENE.verify.from + 84, SCENE.verify.from + 96]) > 0.5
                  ? state.high
                  : line.line
              }
            >
              <Mono size={10} color={text.faint} style={{ textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Candidate B
              </Mono>
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 9 }}>
                {[
                  "antibiotic := pip/tazo",
                  "fluid_rate := 6.5 mL/kg/hr",
                  "patient has CHF",
                ].map((l, i) => (
                  <div
                    key={l}
                    style={{ opacity: ramp(frame, [SCENE.verify.from + 56 + i * 7, SCENE.verify.from + 68 + i * 7]) }}
                  >
                    <Mono size={12} color={i === 2 ? text.dim : text.muted}>
                      {l}
                    </Mono>
                  </div>
                ))}
              </div>
              <Hairline style={{ margin: "16px 0" }} />
              <Stamp
                kind="refused"
                label="refused"
                opacity={ramp(frame, [SCENE.verify.from + 84, SCENE.verify.from + 96])}
              />
              <div
                style={{
                  marginTop: 10,
                  opacity: ramp(frame, [SCENE.verify.from + 96, SCENE.verify.from + 108]),
                }}
              >
                <Mono size={11} color={state.high}>
                  CHF ⇒ fluid_rate ≤ 5.0
                </Mono>
              </div>
            </Card>
          </div>
          <Mono size={11} color={text.faint} style={{ marginTop: 22, display: "block" }}>
            the refusal is the feature — an uncertified order is never emitted
          </Mono>
        </SceneBox>

        {/* ── 5 · DELIVER ─────────────────────────────────────────────── */}
        <SceneBox opacity={deliver.opacity}>
          <SceneTitle n="05" title="Deliver" sub="with the proof attached" />
          <div style={{ marginTop: 40, width: 880 }}>
            <div style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden" }}>
              {[
                { c: state.low, w: 52 },
                { c: state.med, w: 26 },
                { c: state.high, w: 22 },
              ].map((b, i) => (
                <div
                  key={i}
                  style={{
                    width: `${b.w}%`,
                    background: b.c,
                    opacity: ramp(frame, [SCENE.deliver.from + 8 + i * 8, SCENE.deliver.from + 24 + i * 8]),
                  }}
                />
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
              <Mono size={10} color={text.faint}>
                low
              </Mono>
              <Mono size={10} color={text.faint}>
                elevated
              </Mono>
              <Mono size={10} color={text.faint}>
                high
              </Mono>
            </div>

            <div style={{ display: "flex", gap: 40, marginTop: 34 }}>
              {/* Retracted 2026-08-30: this used to read "alarm raised 37.8 h before
                  onset". Lead time does not beat a random-score null at matched
                  detection rate, so it is not a claim we can make. */}
              <Readout
                label="alarm raised"
                node={
                  <Mono size={26} color={stateInk.med} style={{ fontWeight: 500 }}>
                    before onset
                  </Mono>
                }
              />
              <Readout
                label="before onset"
                node={
                  <Mono size={26} color={text.ink} style={{ fontWeight: 500 }}>
                    median
                  </Mono>
                }
              />
            </div>
          </div>
        </SceneBox>
      </div>

      {/* footer provenance — always on, so no frame is unattributed */}
      <div style={{ opacity: seam }}>
        <Hairline style={{ marginBottom: 14 }} />
        <Mono size={10} color={text.faint}>
          PhysioNet/CinC 2019 · trained on one hospital, measured on the other · not a medical device
        </Mono>
      </div>
    </AbsoluteFill>
  );
}

/* ── layout helpers ──────────────────────────────────────────────────────── */

function SceneBox({ opacity, children }: { opacity: number; children: React.ReactNode }) {
  if (opacity <= 0.001) return null;
  return (
    <div style={{ position: "absolute", inset: 0, opacity }}>{children}</div>
  );
}

function SceneTitle({ n, title, sub }: { n: string; title: string; sub: string }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
      <Mono size={11} color={text.faint}>
        {n}
      </Mono>
      <span
        style={{
          fontFamily: font.sans,
          fontSize: 30,
          fontWeight: 600,
          letterSpacing: "-0.02em",
          color: text.ink,
        }}
      >
        {title}
      </span>
      <Mono size={11} color={text.dim}>
        {sub}
      </Mono>
    </div>
  );
}

function Readout({ label, node }: { label: string; node: React.ReactNode }) {
  return (
    <div>
      <div>{node}</div>
      <Mono size={10} color={text.faint} style={{ display: "block", marginTop: 8 }}>
        {label}
      </Mono>
    </div>
  );
}

export const explainerDefaults = { W, H };
