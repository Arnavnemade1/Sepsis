// Types for the ported renderer (engine/twin3d.js). Only the surface the app calls.
export function start(el: HTMLElement): void;
export function attach(el: HTMLElement): void;
export function detach(el?: HTMLElement | null): void;
export function update(data: unknown): void;
export function resize(): void;
export function setOptical(regions: Record<string, unknown>): void;
export function setFocus(foci: string[]): void;
export function setParked(v: boolean): void;
export const isParked: () => boolean;
