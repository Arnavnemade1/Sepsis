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

/**
 * "It worked here. Then we moved it."
 *
 * The single most important idea on the research page, told as a fourteen-second
 * loop: a model is built and measured at one hospital, then physically carried
 * to another, and the number falls. Static charts state the gap; this shows the
 * move that causes it, which is the part people actually remember.
 *
 * Everything is inline-styled from lib/tokens.ts — Remotion's bundler cannot see
 * globals.css, so `var(--color-*)` would resolve to nothing here.
 */
export function GapStory({ internal, external, gap }: GapStoryProps) {
  const frame = useCurrentFrame();

  // scene clock
  const buildIn = ramp(frame, [6, 26]);
  const measureA = ramp(frame, [40, 78]); // needle climbs to `internal`
  const travel = ramp(frame, [120, 190]); // the model slides to hospital B
  const measureB = ramp(frame, [196, 250]); // needle falls to `external`
  const gapReveal = ramp(frame, [262, 292]);
  const verdict = ramp(frame, [300, 330]);
  const seam = ramp(frame, [GAP_DURATION - 30, GAP_DURATION], [1, 0]);

  const xA = 250;
  const xB = 1030;
  const modelX = xA + (xB - xA) * travel;

  // the reading: climbs at A, then falls at B
  const shown = measureB > 0 ? internal - (internal - external) * measureB : internal * measureA;

  const barTop = 300;
  const barH = 150;
  const lo = 0.6;
  const hi = 0.8;
  const barY = (v: number) => barTop + (1 - (v - lo) / (hi - lo)) * barH;

  return (
    <AbsoluteFill
      style={{
        background: surface.bg,
        fontFamily: font.sans,
        padding: 48,
        opacity: seam,
      }}
    >
      <Mono size={11} color={text.dim} style={{ textTransform: "uppercase", letterSpacing: "0.16em" }}>
        The generalization gap
      </Mono>
      <Hairline style={{ marginTop: 12 }} />

      {/* the two hospitals */}
      {[
        { x: xA, name: "Hospital A", sub: "BIDMC · trained here", on: buildIn },
        { x: xB, name: "Hospital B", sub: "Emory · never seen", on: ramp(frame, [110, 140]) },
      ].map((h) => (
        <div
          key={h.name}
          style={{
            position: "absolute",
            left: h.x - 110,
            top: 96,
            width: 220,
            textAlign: "center",
            opacity: h.on,
          }}
        >
          <div
            style={{
              height: 64,
              border: `1px solid ${line.line}`,
              borderRadius: 6,
              background: surface.s1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
            }}
          >
            {/* a crude ward: five beds */}
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
          <div style={{ marginTop: 10 }}>
            <Mono size={13} color={text.ink} style={{ fontWeight: 600 }}>
              {h.name}
            </Mono>
          </div>
          <div style={{ marginTop: 3 }}>
            <Mono size={10} color={text.faint}>
              {h.sub}
            </Mono>
          </div>
        </div>
      ))}

      {/* the model itself, travelling */}
      <div
        style={{
          position: "absolute",
          left: modelX - 46,
          top: 222,
          opacity: buildIn,
        }}
      >
        <div
          style={{
            width: 92,
            padding: "8px 0",
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

      {/* travel track */}
      <div
        style={{
          position: "absolute",
          left: xA,
          top: 248,
          width: xB - xA,
          height: 1,
          background: line.lineSoft,
          opacity: ramp(frame, [110, 130]),
        }}
      />

      {/* the reading */}
      <div style={{ position: "absolute", left: 48, top: barTop - 46 }}>
        <Mono size={10} color={text.faint}>
          AUROC on the hospital it is standing in
        </Mono>
      </div>

      {/* axis */}
      {[0.6, 0.7, 0.8].map((v) => (
        <div key={v} style={{ position: "absolute", left: 48, top: barY(v), width: GAP_W - 96 }}>
          <div style={{ height: 1, background: line.lineSoft }} />
          <div style={{ position: "absolute", left: -4, top: -14 }}>
            <Mono size={9} color={text.faint}>
              {v.toFixed(1)}
            </Mono>
          </div>
        </div>
      ))}

      {/* the measurement bar — climbs at A, falls at B */}
      {measureA > 0 && (
        <>
          <div
            style={{
              position: "absolute",
              left: modelX - 3,
              top: barY(shown),
              width: 6,
              height: barTop + barH - barY(shown),
              background: measureB > 0.5 ? state.high : accent.accent,
              opacity: 0.9,
              borderRadius: 1,
            }}
          />
          <div
            style={{
              position: "absolute",
              left: modelX + 16,
              top: barY(shown) - 14,
            }}
          >
            <Mono size={26} color={text.ink} style={{ fontWeight: 500 }}>
              {shown.toFixed(3)}
            </Mono>
          </div>
        </>
      )}

      {/* the gap, once both readings exist */}
      {gapReveal > 0 && (
        <div style={{ opacity: gapReveal }}>
          <div
            style={{
              position: "absolute",
              left: 48,
              top: barY(internal),
              width: GAP_W - 96,
              height: 1,
              background: line.lineBright,
            }}
          />
          <div
            style={{
              position: "absolute",
              left: xB + 110,
              top: barY(internal),
              width: 2,
              height: barY(external) - barY(internal),
              background: state.high,
            }}
          />
          <div style={{ position: "absolute", left: xB + 124, top: (barY(internal) + barY(external)) / 2 - 12 }}>
            <Mono size={22} color={state.high} style={{ fontWeight: 600 }}>
              −{gap.toFixed(3)}
            </Mono>
          </div>
        </div>
      )}

      {/* verdict */}
      <div style={{ position: "absolute", left: 48, bottom: 40, opacity: verdict }}>
        <Mono size={15} color={text.ink}>
          Same model. Same features. A different hospital.
        </Mono>
        <div style={{ marginTop: 6 }}>
          <Mono size={11} color={text.dim}>
            {external.toFixed(3)} is the number this project reports — not {internal.toFixed(3)}.
          </Mono>
        </div>
      </div>
    </AbsoluteFill>
  );
}
