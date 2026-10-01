"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

// There is no front page: the product is the console. Vercel redirects "/" at the
// edge (vercel.json); this covers every other host.
export default function Root() {
  const router = useRouter();
  useEffect(() => router.replace("/operator/"), [router]);
  return null;
}
