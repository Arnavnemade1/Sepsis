import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";

import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist", display: "swap" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono", display: "swap" });
const serif = Instrument_Serif({ subsets: ["latin"], weight: "400", style: ["normal", "italic"],
  variable: "--font-serif-display", display: "swap" });

export const metadata: Metadata = {
  title: { default: "AegisOS — hospital paging that thinks", template: "%s · AegisOS" },
  description:
    "Every alert reaches the right clinician in seconds: triaged, routed, formally verified and escalated automatically. " +
    "Research and educational use only — not a medical device.",
};

export const viewport: Viewport = { themeColor: "#f6f6f3", colorScheme: "light", viewportFit: "cover" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
