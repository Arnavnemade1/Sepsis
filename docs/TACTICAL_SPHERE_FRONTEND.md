# AegisOS — Global ICU Tactical Sphere (frontend redesign spec)

> A hardware-accelerated window into machine memory. The cockpit for a hard
> real-time, zero-allocation bio-kernel. It must read like an aerospace weapon
> system / reactor-core monitor — instrumented, dense, deterministic — and emit
> that aura *naturally*, never as theming bolted onto a web app.

## 0. Design thesis

The backend now operates at kernel level (L1 `/dev/bio` bus → L2 zero-copy memory
fabric → L3 cognitive scheduler → L4 Z3 egress cage). The frontend's only job is
to be a **faithful, high-frame-rate instrument** reading that machine's live
state. Credibility is won or lost on *motion quality*: if a project claiming
microsecond determinism shows a laggy chart or an HTML spinner, the architecture
story dies. Therefore:

- **Everything that moves is drawn on `<canvas>`**, on a single
  `requestAnimationFrame` loop, decoupled from the data tick. No DOM-node
  animation, no SVG transitions, no CSS keyframes on hot data, no chart library.
- **Data arrives over WebSocket at ~1.6 Hz; the render runs at 60 Hz.** The UI
  interpolates between snapshots so it always feels alive between ticks — the
  instrument never "steps".
- **Fixed-precision monospace readouts** (JetBrains Mono) that update in place,
  tabular-nums, never reflow. Numbers tick; layout is dead still.
- **Restraint is the aura.** Near-black obsidian, hairline 1px borders, one
  accent family (cyan→blue→violet), color used only to mean something (cyan =
  nominal, amber = watch, crimson = critical). No gradients-as-decoration, no
  glassmorphism pile-up. The density + stillness + precision *is* the high-tech
  feel.

## 1. Reuse the existing design language (do not reinvent)

`webui/os.css` already defines the palette and type the shell uses — the cockpit
must extend it so it feels native to AegisOS, not like a second app:

| token | value | cockpit use |
|---|---|---|
| `--bg` | `#06080d` | base void |
| `--panel-solid` | `#12151d` | module fill |
| `--line` / `--line-soft` | `rgba(150,172,212,.10/.055)` | hairline grid + borders |
| `--cyan` `#5ccaf0` / `--blue` `#6f8cf5` / `--violet` `#9a8cf0` | accent ramp | nominal state, links, mesh |
| `--ok` `#39d9a2` | nominal | certified / stable |
| `--stress` `#f5c451` | watch | WATCH tier |
| `--crit` `#f2606e` | critical | CRITICAL tier, preempt owner |
| `--mono` JetBrains Mono | all numeric readouts |
| `--sans` Inter | labels only |

New cockpit-only tokens (add to a scoped `tactical.css`, not global): grid line
`rgba(92,202,240,.06)`, scanline `rgba(92,202,240,.03)`, crimson glow for crisis.

## 2. Data spine (already built — bind to this)

A live `BioCluster` (6 patient nodes, lane budget 96) runs server-side and
streams the whole machine state. **No mock data anywhere** — every pixel traces
to a real kernel structure.

- `WS  /aegis/cluster/stream` — pushes the full payload every 0.6 s.
- `GET /aegis/cluster/snapshot` — same payload on demand.
- `POST /aegis/cluster/chaos` — `{node, action: crisis|clear|noise, intensity}`.

Payload shape (`backend/aegis/server.py::_cluster_payload`):

```jsonc
{
  "topology":  [{ "node_id","name","weight","p","slope","tier","ring","lanes" }],  // Panel A
  "memfabric": { "arena": { "capacity_cells","used_cells","utilization",
                            "base_pointer","pointer_stable","bytes" },
                 "layout": [{ "name","offset","length","shape" }],
                 "rings":  { "<path>": { "capacity","write_index","write_ptr",
                                         "overwrites","fill" } },
                 "cursors":[{ "ring","read_ptr","lag","drops" }],
                 "published": 0 },                                                  // Panel B
  "lane_map":  { "budget", "mode", "owner",
                 "lanes": { "<node>": { "safety","ml_tracking","render","logging" } },
                 "classes": [...] },                                               // Panel C
  "devbio":    { "mounted","open_fds","nodes","total_reads",
                 "latency": { "p50_us","p95_us","max_us","mean_us" } },            // Panel D
  "scheduler": { "lane_budget","nodes","tick","mode","owner","global_cost","interrupts" },
  "safety":    { "certified","blocked","quarantined","pass_rate","recent":[...] }, // L4 cage
  "events":    [{ "tick","type":"HARD_PREEMPT","owner","reason","p","slope","ts" }]
}
```

## 3. The four panels (the "Macro System Workspace Grid")

Four deep-obsidian modular blocks, sharp single-pixel borders, a fifth thin
status spine across the top. Each panel is one `<canvas>` + a small monospace
readout overlay.

### Panel A — Global Node Topology Graph
- Hardware-accelerated force-directed mesh of the 6 `/dev/bio` patient nodes.
- Each node = a ring whose color is `tier` (cyan/amber/crimson) and whose radius
  scales with `lanes`. A soft cyan halo at nominal.
- The exact tick a crisis is injected, the node's ring snaps to crimson and the
  topology edges to it **thicken** to visualize the scheduler re-routing
  bandwidth (edge weight ∝ owner's lane share). Animate edge-flow with marching
  dashes drawn on canvas (not CSS).
- Readout per node: `name · p=0.xx · slope=±0.xxx · lanes=NN`.

### Panel B — Live Kernel Memory Address Matrix
- A hex grid of the preallocated arena (`memfabric.layout` regions over
  `capacity_cells`). Each ring's region is a colored band; the **write pointer**
  (`write_ptr`) and each **cursor's read pointer** (`cursors[].read_ptr`) cycle
  across their band every frame — interpolated so they sweep smoothly at 60 Hz.
- Header readout: `BASE 0x… · STABLE ✓ · util 2.6% · published 12,480` proving
  the zero-copy claim live (pointer never moves; pointers cycle without a frame
  drop). Overwrite/drasp counters tick in mono.

### Panel C — Asynchronous Thread-Lane Preemption Bars
- One horizontal stacked bar per node, segmented by lane class
  (`safety / ml_tracking / render / logging`), widths from `lane_map.lanes`.
- On `mode → PREEMPT`: render+logging segments collapse to flat lines
  cluster-wide while the owner's `ml_tracking` segment expands to consume the
  freed lanes — animated as a real reflow of the bar, eased over ~250 ms.
- `safety` segment is drawn in a locked accent and never collapses (visually
  encodes "safety floor preserved on every node").

### Panel D — High-Resolution Monotonic Instrumentation Canvas
- A rolling single-pixel vector trace of `devbio.latency.p50_us` (and a faint
  p95 band), 60 Hz, ring-buffered in a `Float32Array` (mirror the kernel's own
  zero-allocation discipline in the UI).
- A bold crimson horizontal rule at the `HARD DEADLINE` (e.g. 50,000 µs); the
  live trace hovers flat near ~2–4 µs far below it — the determinism money shot.
- Readout: `p50 3.1µs · p95 5.0µs · max … · DEADLINE 50,000µs · MARGIN 16,000×`.

### Status spine (top)
`AEGIS-OS // BIO-KERNEL` · global cost `C=Σwᵢ·Pᵢ` · `MODE NORMAL/PREEMPT` ·
`INTERRUPTS n` · L4 cage `CERT 1,240 / BLOCK 0 (100%)` · WS `● LIVE` · UTC clock.

## 4. The Chaos Controller (judge interaction)
A docked strip of the 6 nodes with `INJECT CRISIS / CLEAR / NOISE SURGE`
buttons → `POST /aegis/cluster/chaos`. The judge clicks a node; within ~2 s the
node reddens (A), its ring fills faster (B), its lane bar swells while the wing's
render/logging collapse (C), an interrupt event streams into the spine, and the
latency trace (D) stays flat — proving the kernel absorbs the crisis without
jitter. This is the demo climax.

## 5. Build plan (incremental, one panel per step)
1. `tactical.css` (scoped tokens + 2×2 grid shell) + `tactical.html` route + a
   `TacticalSphere` controller with the WS client, snapshot interpolation buffer,
   and the single rAF loop. Wire an entry from the OS shell dock.
2. Panel D first (simplest, highest-credibility): the monotonic latency canvas.
3. Panel C: lane-preemption bars (clear payoff on chaos).
4. Panel A: node topology mesh + edge re-routing.
5. Panel B: memory address matrix (most intricate).
6. Chaos Controller dock + interrupt event ticker + status spine polish.
7. Perf pass: confirm sustained 60 fps under chaos; verify with preview tools.

## 6. Non-negotiables (the credibility checklist)
- [ ] Sustained 60 fps during a chaos injection (no dropped frames).
- [ ] Zero chart libraries; all live visuals on canvas.
- [ ] All numbers monospace, tabular, fixed-width — no layout shift on update.
- [ ] Every value on screen traces to a real kernel field (no decorative fakes).
- [ ] Latency trace genuinely reads the kernel's own µs counters.
- [ ] Memory pointer is shown stable while read/write pointers cycle — live proof
      of the zero-copy/zero-allocation architecture.
