"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

// The ward board lives in the operator console now.
export default function CommandRedirect() {
  const router = useRouter();
  useEffect(() => router.replace("/operator/"), [router]);
  return null;
}
