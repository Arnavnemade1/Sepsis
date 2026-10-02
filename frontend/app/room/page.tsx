"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { WatchBoard } from "@/components/rooms/board";
import { RoomUnit } from "@/components/rooms/unit";
import { Page } from "@/components/shell";

/** /room/ is the watch board; /room/?id=512 is that room as a unit. */
export default function RoomRoute() {
  return (
    <Suspense>
      <Rooms />
    </Suspense>
  );
}

function Rooms() {
  const id = useSearchParams().get("id");
  return <Page active="rooms">{id ? <RoomUnit key={id} room={id.toUpperCase()} /> : <WatchBoard />}</Page>;
}
