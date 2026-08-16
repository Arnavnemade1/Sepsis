import type { Metadata } from "next";
import type { ReactNode } from "react";

import { fontVars } from "@/lib/fonts";
import { site, backendUrl } from "@/lib/site-config";
import { MotionProvider } from "@/components/providers/motion-provider";
import { SiteHeader } from "@/components/chrome/site-header";
import { SiteFooter } from "@/components/chrome/site-footer";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: `${site.name} — ${site.tagline}`,
    template: `%s — ${site.name}`,
  },
  description: site.description,
  applicationName: site.name,
  openGraph: {
    title: `${site.name} — ${site.tagline}`,
    description: site.description,
    type: "website",
    siteName: site.name,
  },
  twitter: { card: "summary_large_image" },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={fontVars}>
      <head>
        {/* Cheap. The live-health probe may fire ~2 s after paint, or never;
            a full preconnect handshake isn't worth it for a maybe. */}
        <link rel="dns-prefetch" href={backendUrl} />

        {/* Motion renders its `initial` state as an inline style, so every
            reveal ships as opacity:0 in the SSR HTML and depends on JS to
            become visible. The prose is all there for crawlers, but a human
            with JS blocked or broken would get a blank page — unacceptable for
            a page whose entire purpose is showing its work. This forces
            everything visible when scripting is off. */}
        <noscript>
          <style>{`[style*="opacity:0"]{opacity:1!important}[style*="transform:translateY"]{transform:none!important}`}</style>
        </noscript>
      </head>
      <body className="bg-bg text-ink antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:border focus:border-line focus:bg-s2 focus:px-3 focus:py-2 focus:text-sm"
        >
          Skip to content
        </a>
        <MotionProvider>
          <SiteHeader />
          <main id="main">{children}</main>
          <SiteFooter />
        </MotionProvider>
      </body>
    </html>
  );
}
