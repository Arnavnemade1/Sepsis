/**
 * The mark: a shield outline whose interior is a single vitals deflection.
 * Aegis = shield; the trace is the only thing the system actually watches.
 * Hairline stroke, no fill, no glow — Rule 2.
 */
export function Wordmark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M10 1.5 17 4v6.2c0 4-2.9 7-7 8.3-4.1-1.3-7-4.3-7-8.3V4l7-2.5Z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
      <path
        d="M5.6 10.4h2.1l1.2-3 1.5 5.2 1.1-2.2h2.9"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
