/**
 * Shared by the Remotion composition AND the <Player> that embeds it, so the
 * two can never disagree about geometry or timing.
 */
export const FPS = 30;
export const W = 1280;
export const H = 720;

/** 20 seconds. 30fps rather than 60: the Player re-renders React every frame,
 *  and this content gains nothing from the extra 30. */
export const DURATION = 600;

/** Scene boundaries, in frames. The last scene decays back to scene 1's entry
 *  state so `loop` has nothing to cut. */
export const SCENE = {
  acquire: { from: 0, len: 120 },
  feature: { from: 120, len: 105 },
  calibrate: { from: 225, len: 135 },
  verify: { from: 360, len: 135 },
  deliver: { from: 495, len: 75 },
  seam: { from: 570, len: 30 },
} as const;

/** How many frames the outro takes to return to the baseline. */
export const SEAM_FRAMES = 30;
