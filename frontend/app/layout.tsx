import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist", display: "swap" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Aegis · Operations", template: "%s · Aegis" },
  description:
    "Hospital paging operations: alerts from nurses and the sepsis detector, triaged, routed, verified against policy and escalated. " +
    "Research and educational use only; not a medical device.",
};

export const viewport: Viewport = { themeColor: "#212429", colorScheme: "dark", viewportFit: "cover" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
