# Parity with the legacy shell (`webui/`)

`webui/` is deleted in one commit **when every row below is ported or explicitly dropped**
(the decision recorded 2026-09-29: retire after parity, not before). Until then both deploy.

| Legacy capability (`webui/`) | Where it lives now | Status |
|:--|:--|:--|
| Boot, sign-in, per-patient scoping (`AegisPatient`) | `/` launcher, top-bar patient selector on `/bed` and `/pharmacy` | ported |
| Ward / tactical console (`tactical.html`) | `/command` (bed board, KPIs, focus bundle, unit log, services) | ported (bed board); the 3D tactical sphere is not |
| Risk core, telemetry, organ systems, multi-agent panel | `/bed` (risk hero, vitals, organ systems, specialist panel) | ported |
| 3D anatomical twin (`twin3d.js`) | `engine/twin3d.js` + `components/twin.tsx` | ported (same renderer) |
| Risk timeline, decision (`/aegis/timeline`, `/aegis/decision`) | `/bed` risk chart + decision card | ported |
| Write syscalls: treat / escalate / crisis / clear | `/bed` Orders panel, capability-gated, with the bed's audit log | ported |
| Dr. Aegis typed consult (`doctor.js`) | `/bed` consult panel (read-only, `agentic: false`) | ported |
| Treatment options / PK comparison (`pktwin_ui.js`) | `/pharmacy` comparison table, curves, adverse events | ported (2D) |
| Pharmacopoeia interaction screen (`AegisPharma`) | `/pharmacy` interaction screen | ported |
| Lab (`lab.html`) | `/lab` refusal record | ported |
| Command shell, system monitor (`oskit/`) | `/os` shell, process table, journal, boot, services, drivers | ported (subset of verbs) |
| Validation report (`validation.html`) | `/lab` deployed-detector block | partial: no cross-hospital directions or decision curves yet |
| **Optical biosensor** (`camera.js`, `shal.js`): webcam rPPG → `/aegis/ingest`, the real-data path | — | **not ported** |
| **Voice consult** (Gemini Live relay, `/aegis/doctor/live`) | — | **not ported** |
| **3D drug-distribution twin** (`pktwin.js`) | — | **not ported** |
| Clinical Story: causal graph, incident trace-back, counterfactual replay, pattern mining (`AegisStory`) | — | not ported |
| Consensus chamber (`AegisConsensus`) | — | not ported |
| Orchestrator cascade + portable proof verify (`AegisOrchestrator`) | — | not ported |
| Identity guard (`AegisIdentity`) | — | not ported |
| Safety DSL editor (`AegisDSL`) | — | not ported |
| Federation topology (`AegisFederation`) | — | not ported |
| RBAC / access-control overlay (`AegisRBAC`), break-glass | sign-in only | partial |
| Hyperscan situation report (`AegisHyperscan`) | — | not ported |
| Simulator stage controls (Stable / Deteriorate / Sepsis / Recover) | — | not ported |
| ⌘K palette, ⌥A ask-anywhere (`palette.js`, `summon.js`) | — | not ported |
| Per-bed terminal, medical-history import, devices hub | — | not ported |

The bold rows carry capability nothing else has (a real sensor path, voice, the 3D PK
render); the rest are views over endpoints the new app already reaches.
