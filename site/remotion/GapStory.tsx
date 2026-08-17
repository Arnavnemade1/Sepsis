import { AbsoluteFill, useCurrentFrame } from "remotion";

import { Mono, Hairline, ramp, colors } from "./primitives";

const { surface, line, text, state, accent, font } = colors;

export const GAP_FPS = 30;
export const GAP_W = 1280;
export const GAP_H = 560;
export const GAP_DURATION = 420; // 14 s

export type GapStoryProps = {
  internal: number;
  external: number;
  gap: number;
};

/* ── geometry ───────────────────────────────────────────────────────────────
   Laid out once, up here, because the previous version positioned things
   ad hoc and they collided: the verdict line sat on top of the 0.6 gridline,
   the gap bracket floated off at the right edge unattached to anything, and
   once the model travelled there was nothing left at Hospital A — so the "gap"
   had only one visible end.                                                   */
const PAD = 48;
const X_A = 330;
const X_B = 950;
const WARD_TOP = 92;
const WARD_H = 62;
const TRACK_Y = 232; // the model rides this line between wards
const AXIS_TOP = 300; // 0.80
const AXIS_H = 140; // to 0.60
const AXIS_BOTTOM = AXIS_TOP + AXIS_H;
const LO = 0.6;
const HI = 0.8;
const barY = (v: number) => AXIS_TOP + (1 - (v - LO) / (HI - LO)) * AXIS_H;

/**
 * "It worked here. Then we moved it."
 *
 * The single most important idea on the research page, told as a fourteen-second
 * loop: a model is built and measured at one hospital, then carried to another,
 * and the number falls. Static charts state the gap; this shows the move that
 * causes it, which is the part people remember.
 *
 * Hospital A's reading stays on screen as a dimmed bar after the model leaves,
 * so the gap bracket has two real ends to span rather than pointing at empty
 * space.
 *
 * Everything is inline-styled from lib/tokens.ts — Remotion's bundler cannot see
 * globals.css, so `var(--color-*)` would resolve to nothing here.
 */
export function GapStory({ internal, external, gap }: GapStoryProps) {
  const frame = useCurrentFrame();

  const buildIn = ramp(frame, [6, 26]);
  const measureA = ramp(frame, [40, 78]);
  const wardB = ramp(frame, [104, 134]);
  const travel = ramp(frame, [120, 190]);
  const measureB = ramp(frame, [196, 250]);
  const gapReveal = ramp(frame, [262, 292]);
  const verdict = ramp(frame, [300, 330]);
  const seam = ramp(frame, [GAP_DURATION - 30, GAP_DURATION], [1, 0]);

  const modelX = X_A + (X_B - X_A) * travel;

  /** The live reading: climbs at A, then falls once the model is at B. */
  const shown = measureB > 0 ? internal - (internal - external) * measureB : internal * measureA;

  /** A's result stays visible after the model departs. */
  const aLocked = frame > 96;

  return (
    <AbsoluteFill
      style={{ background: surface.bg, fontFamily: font.sans, padding: PAD, opacity: seam }}
    >
      <Mono size={11} color={text.dim} style={{ textTransform: "uppercase", letterSpacing: "0.16em" }}>
        The generalization gap
      </Mono>
      <Hairline style={{ marginTop: 12 }} />

      {/* ── the two wards ─────────────────────────────────────────────── */}
      {[
        { x: X_A, name: "Hospital A", sub: "BIDMC · trained here", on: buildIn },
        { x: X_B, name: "Hospital B", sub: "Emory · never seen", on: wardB },
      ].map((h) => (
        <div
          key={h.name}
          style={{
            position: "absolute",
            left: h.x - 110,
            top: WARD_TOP,
            width: 220,
            textAlign: "center",
            opacity: h.on,
          }}
        >
          <div
            style={{
              height: WARD_H,
              border: `1px solid ${line.line}`,
              borderRadius: 6,
              background: surface.s1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
            }}
          >
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                style={{
                  width: 18,
                  height: 26,
                  border: `1px solid ${line.lineBright}`,
                  borderRadius: 2,
                  background: surface.s2,
                }}
              />
            ))}
          </div>
          <div style={{ marginTop: 9 }}>
            <Mono size={13} color={text.ink} style={{ fontWeight: 600 }}>
              {h.name}
            </Mono>
          </div>
          <div style={{ marginTop: 2 }}>
            <Mono size={10} color={text.faint}>
              {h.sub}
            </Mono>
          </div>
        </div>
      ))}

      {/* ── travel track + the model riding it ────────────────────────── */}
      <div
        style={{
          position: "absolute",
          left: X_A,
          top: TRACK_Y,
          width: X_B - X_A,
          height: 1,
          background: line.lineSoft,
          opacity: wardB,
        }}
      />
      <div style={{ position: "absolute", left: modelX - 46, top: TRACK_Y - 15, opacity: buildIn }}>
        <div
          style={{
            width: 92,
            padding: "7px 0",
            textAlign: "center",
            border: `1px solid ${travel > 0.02 && travel < 0.98 ? accent.accent : line.lineBright}`,
            borderRadius: 5,
            background: surface.s2,
          }}
        >
          <Mono size={10} color={accent.accentInk} style={{ letterSpacing: "0.1em" }}>
            MODEL
          </Mono>
        </div>
      </div>

      {/* ── axis ──────────────────────────────────────────────────────── */}
      <div style={{ position: "absolute", left: PAD, top: AXIS_TOP - 26 }}>
        <Mono size={10} color={text.faint}>
          AUROC on the hospital it is standing in
        </Mono>
      </div>
      {[0.8, 0.7, 0.6].map((v) => (
        <div key={v} style={{ position: "absolute", left: PAD, top: barY(v), width: GAP_W - PAD * 2 }}>
          <div style={{ height: 1, background: line.lineSoft }} />
          <div style={{ position: "absolute", left: -34, top: -7 }}>
            <Mono size={9} color={text.faint}>
              {v.toFixed(2)}
            </Mono>
          </div>
        </div>
      ))}

      {/* ── Hospital A's locked-in reading ────────────────────────────── */}
      {aLocked && (
        <>
          <div
            style={{
              position: "absolute",
              left: X_A - 3,
              top: barY(internal),
              width: 6,
              height: AXIS_BOTTOM - barY(internal),
              background: text.dim,
              borderRadius: 1,
            }}
          />
          <div style={{ position: "absolute", left: X_A + 16, top: barY(internal) - 13 }}>
            <Mono size={22} color={text.muted} style={{ fontWeight: 500 }}>
              {internal.toFixed(3)}
            </Mono>
          </div>
        </>
      )}

      {/* ── the live reading, travelling with the model ───────────────── */}
      {measureA > 0 && !(aLocked && travel > 0.98 && measureB === 0) && (
        <>
          <div
            style={{
              position: "absolute",
              left: modelX - 3,
              top: barY(shown),
              width: 6,
              height: AXIS_BOTTOM - barY(shown),
              background: measureB > 0.5 ? state.high : accent.accent,
              borderRadius: 1,
            }}
          />
          <div style={{ position: "absolute", left: modelX + 16, top: barY(shown) - 13 }}>
            <Mono size={22} color={text.ink} style={{ fontWeight: 500 }}>
              {shown.toFixed(3)}
            </Mono>
          </div>
        </>
      )}

      {/* ── the gap, bracketed between the two readings ───────────────── */}
      {gapReveal > 0 && (
        <div style={{ opacity: gapReveal }}>
          {/* carry A's level across so the drop is a visible distance */}
          <div
            style={{
              position: "absolute",
              left: X_A,
              top: barY(internal),
              width: X_B - X_A + 96,
              height: 1,
              background: line.lineBright,
            }}
          />
          <div
            style={{
              position: "absolute",
              left: X_B + 96,
              top: barY(internal),
              width: 2,
              height: barY(external) - barY(internal),
              background: state.high,
            }}
          />
          {[barY(internal), barY(external)].map((y, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: X_B + 90,
                top: y,
                width: 14,
                height: 2,
                background: state.high,
              }}
            />
          ))}
          <div
            style={{
              position: "absolute",
              left: X_B + 118,
              top: (barY(internal) + barY(external)) / 2 - 16,
            }}
          >
            <Mono size={22} color={state.high} style={{ fontWeight: 600 }}>
              −{gap.toFixed(3)}
            </Mono>
            <div style={{ marginTop: 2 }}>
              <Mono size={9} color={text.dim}>
                what a new hospital costs
              </Mono>
            </div>
          </div>
        </div>
      )}

      {/* ── verdict, clear of the axis ────────────────────────────────── */}
      <div style={{ position: "absolute", left: PAD, top: AXIS_BOTTOM + 48, opacity: verdict }}>
        <Mono size={15} color={text.ink}>
          Same model. Same features. A different hospital.
        </Mono>
        <div style={{ marginTop: 7 }}>
          <Mono size={11} color={text.dim}>
            {external.toFixed(3)} is the number this project reports — not {internal.toFixed(3)}.
          </Mono>
        </div>
      </div>
    </AbsoluteFill>
  );
}
