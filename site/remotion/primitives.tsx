import { interpolate, Easing } from "remotion";
import type { CSSProperties, ReactNode } from "react";

import { surface, line, text, state, stateInk, accent, font } from "@/lib/tokens";

/**
 * Composition primitives.
 *
 * INLINE STYLES ONLY — no Tailwind classes anywhere under remotion/. Remotion's
 * bundler is separate from Next's and has no access to app/globals.css, so a
 * `var(--color-high)` here would resolve against a :root that does not exist and
 * silently paint nothing. Feeding styles from lib/tokens.ts instead makes the
 * <Player> output and the rendered MP4 pixel-identical with zero extra config,
 * and `npm run check:tokens` proves those tokens match the website's.
 */

/** The instrument easing curve, --ease. Nothing here bounces. */
export const ease = Easing.bezier(0.2, 0.7, 0.2, 1);

/** Clamped interpolation — the form used everywhere in this composition. */
export function ramp(
  frame: number,
  input: readonly [number, number],
  output: readonly [number, number] = [0, 1],
) {
  return interpolate(frame, input as [number, number], output as [number, number], {
    easing: ease,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

export const colors = { surface, line, text, state, stateInk, accent, font };

/** Monospace label at instrument sizes, with tabular figures. */
export function Mono({
  children,
  size = 13,
  color = text.muted,
  style,
}: {
  children: ReactNode;
  size?: number;
  color?: string;
  style?: CSSProperties;
}) {
  return (
    <span
      style={{
        fontFamily: font.mono,
        fontSize: size,
        letterSpacing: "0.04em",
        color,
        fontVariantNumeric: "tabular-nums slashed-zero",
        ...style,
      }}
    >
      {children}
    </span>
  );
}

/** A 1px rule. Rule 2: structure is hairlines. */
export function Hairline({
  width,
  color = line.line,
  style,
}: {
  width?: number | string;
  color?: string;
  style?: CSSProperties;
}) {
  return <div style={{ height: 1, width: width ?? "100%", background: color, ...style }} />;
}

/** A flat hairline-bordered panel. */
export function Card({
  children,
  style,
  borderColor = line.line,
}: {
  children: ReactNode;
  style?: CSSProperties;
  borderColor?: string;
}) {
  return (
    <div
      style={{
        border: `1px solid ${borderColor}`,
        borderRadius: 8,
        background: surface.s1,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** Frame-deterministic numeric readout. Remotion renders by frame, so this
 *  cannot use Motion — the value must be a pure function of `frame`. */
export function CountUp({
  frame,
  from,
  to,
  start,
  end,
  decimals = 3,
  size = 13,
  color = text.ink,
  suffix = "",
}: {
  frame: number;
  from: number;
  to: number;
  start: number;
  end: number;
  decimals?: number;
  size?: number;
  color?: string;
  suffix?: string;
}) {
  const v = ramp(frame, [start, end], [from, to]);
  return (
    <Mono size={size} color={color} style={{ fontWeight: 500 }}>
      {v.toFixed(decimals)}
      {suffix}
    </Mono>
  );
}

/**
 * A waveform that draws itself left to right via strokeDashoffset.
 * `progress` 0→1 is supplied by the scene so the seam can reverse it.
 */
export function Waveform({
  points,
  width,
  height,
  progress,
  stroke = text.dim,
  strokeWidth = 1.5,
}: {
  points: readonly (readonly [number, number])[];
  width: number;
  height: number;
  progress: number;
  stroke?: string;
  strokeWidth?: number;
}) {
  const d = points
    .map(([x, y], i) => `${i ? "L" : "M"}${(x * width).toFixed(2)} ${(y * height).toFixed(2)}`)
    .join(" ");
  // Generous dash length so a single dash covers the whole path at any scale.
  const LEN = 4000;
  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <path
        d={d}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
        strokeDasharray={LEN}
        strokeDashoffset={LEN * (1 - progress)}
      />
    </svg>
  );
}

/** The verdict stamp: certified or refused. The only saturated colour in the
 *  whole composition, because it is the only thing that encodes a real state. */
export function Stamp({
  kind,
  label,
  opacity,
  scale = 1,
}: {
  kind: "certified" | "refused";
  label: string;
  opacity: number;
  scale?: number;
}) {
  const c = kind === "certified" ? state.low : state.high;
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        border: `1px solid ${c}`,
        borderRadius: 4,
        padding: "6px 12px",
        opacity,
        transform: `scale(${scale})`,
        transformOrigin: "left center",
      }}
    >
      <Mono size={13} color={c} style={{ fontWeight: 600 }}>
        {kind === "certified" ? "⊢" : "⊬"}
      </Mono>
      <Mono size={12} color={c} style={{ textTransform: "uppercase", letterSpacing: "0.1em" }}>
        {label}
      </Mono>
    </div>
  );
}

/** Deterministic pseudo-random, so every render of frame N is identical. */
export function noise(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}
