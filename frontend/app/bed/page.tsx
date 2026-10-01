"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect } from "react";

// /bed/ became /patient/ when the operator console became the front door.
function Redirect() {
  const router = useRouter();
  const params = useSearchParams();
  useEffect(() => {
    const id = params.get("id");
    router.replace(id ? `/patient/?id=${encodeURIComponent(id)}` : "/patient/");
  }, [params, router]);
  return null;
}

export default function BedRedirect() {
  return (
    <Suspense>
      <Redirect />
    </Suspense>
  );
}
