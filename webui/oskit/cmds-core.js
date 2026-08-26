/* ============================================================================
   oskit/cmds-core.js — the page-agnostic vocabulary
   ----------------------------------------------------------------------------
   Verbs that depend only on the kernel and the window manager, so any page that
   mounts a Shell can register them. The cohort-specific vocabulary (who, open,
   treat, radar, …) stays with the page that owns the cohort model; this is the
   part that is genuinely portable.

   Everything here is READ-ONLY except `open`/`layout`, which move windows. There
   is deliberately no write path to the kernel from this file: mutating syscalls
   go through the audited, Z3-gated trap gate, and a convenience alias into that
   from a general-purpose shell is exactly how an unaudited write happens.
   ============================================================================ */
(function (global) {
  "use strict";

  const esc = (s) => Shell.esc(s);
  const API = () => (global.AEGIS_BACKEND || "");

  const PROC = ["sched", "mem", "audit", "cage", "kthreads", "ksched", "llm", "models"];

  async function getJSON(url) {
    const r = await fetch(API() + url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    // /aegis/proc/<node> answers {path, value}; unwrap so verbs see the node.
    return (d && Object.prototype.hasOwnProperty.call(d, "value")) ? d.value : d;
  }

  function dump(sh, obj, depth) {
    if (obj == null) return sh.print("<i>null</i>", "dim");
    if (typeof obj !== "object") return sh.print(esc(String(obj)));
    const pad = "  ".repeat(depth || 0);
    Object.entries(obj).forEach(([k, v]) => {
      if (v && typeof v === "object" && !Array.isArray(v) && (depth || 0) < 1) {
        sh.print(pad + '<span class="sh-key">' + esc(k) + "</span>");
        dump(sh, v, (depth || 0) + 1);
      } else {
        const val = Array.isArray(v)
          ? "[" + v.length + " items]"
          : (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : String(v));
        sh.print(pad + '<span class="sh-key">' + esc(k) + "</span> " + esc(val));
      }
    });
  }

  function install(sh, ctx) {
    const wm = ctx && ctx.wm;
    const session = ctx && ctx.session;

    sh.register("help", function () {
      sh.print("<b>KERNEL</b>", "sys");
      [["ls [path]", "list the /proc namespace"],
       ["cat <node>", "read one /proc node"],
       ["ps", "process table — beds as scheduled processes"],
       ["top", "scheduler summary + load"],
       ["mem", "biometric arena occupancy"],
       ["audit", "hash-chained syscall log, refusals included"],
       ["sys", "system health snapshot"],
      ].forEach(([c, d]) => sh.print('<span class="sh-cmd">' + esc(c) + "</span><span class=\"sh-dim\">" + esc(d) + "</span>", "row"));
      sh.print("<b>WINDOWS</b>", "sys");
      [["win", "list windows and their state"],
       ["open <window>", "open and focus a window"],
       ["close <window>", "close a window"],
       ["float <window>", "detach a window from the tiling"],
       ["layout [name]", "focus | split | triple | quad"],
      ].forEach(([c, d]) => sh.print('<span class="sh-cmd">' + esc(c) + "</span><span class=\"sh-dim\">" + esc(d) + "</span>", "row"));
      sh.print("<b>SESSION</b>", "sys");
      [["whoami", "current role, ring and capabilities"],
       ["can <capability>", "would this session be permitted?"],
       ["clear", "clear the screen"],
      ].forEach(([c, d]) => sh.print('<span class="sh-cmd">' + esc(c) + "</span><span class=\"sh-dim\">" + esc(d) + "</span>", "row"));
    }, { help: "list commands", alias: ["?"] });

    sh.register("ls", async function (arg) {
      const p = (arg || "").replace(/^\/?proc\/?/, "").trim();
      if (!p) {
        const d = await getJSON("/aegis/proc").catch(() => null);
        const nodes = (d && (d.nodes || d.paths)) || PROC.map((n) => "/proc/" + n);
        nodes.forEach((n) => sh.print('<span class="sh-cmd">' + esc(n) + "</span>", "row"));
        return;
      }
      dump(sh, await getJSON("/aegis/proc/" + p), 0);
    }, { help: "list /proc", alias: ["dir"] });

    sh.register("cat", async function (arg) {
      if (!arg) return sh.print("usage: cat &lt;node&gt; — try <b>ls</b>", "dim");
      dump(sh, await getJSON("/aegis/proc/" + arg.replace(/^\/?proc\/?/, "")), 0);
    }, { help: "read a /proc node", alias: ["proc", "read"] });

    sh.register("ps", async function () {
      const d = await getJSON("/aegis/proc/kthreads");
      const rows = Array.isArray(d) ? d : (d.ps || []);
      if (!rows.length) return sh.print("no kernel threads reported", "dim");
      sh.print('<span class="sh-key">PID  TASK                 STATE      PRIO  CPU</span>', "row");
      rows.forEach((r) => sh.print(
        String(r.pid ?? r.id ?? "—").padEnd(5) +
        String(r.name ?? r.task ?? r.bed ?? "—").padEnd(21) +
        String(r.state ?? "—").padEnd(11) +
        String(r.prio ?? r.priority ?? "—").padEnd(6) +
        (r.cpu_us != null ? Number(r.cpu_us).toFixed(0) + "µs" : ""), "row"));
    }, { help: "process table" });

    sh.register("top", async function () {
      dump(sh, await getJSON("/aegis/proc/ksched"), 0);
    }, { help: "scheduler summary", alias: ["kstat", "sched"] });

    sh.register("mem", async function () {
      const d = await getJSON("/aegis/proc/mem");
      const used = d.used_cells ?? 0, cap = d.capacity_cells ?? 0;
      sh.print("arena " + used.toLocaleString() + " / " + cap.toLocaleString() + " cells (" +
        (cap ? (100 * used / cap).toFixed(1) : "0") + "% · " + ((d.bytes ?? cap * 8) / 1048576).toFixed(1) + " MiB)");
      if (d.base_pointer) sh.print("base " + d.base_pointer + " · " +
        (d.pointer_stable ? "pointer stable since boot" : "POINTER MOVED"), "dim");
    }, { help: "arena occupancy" });

    sh.register("audit", async function () {
      const d = await getJSON("/aegis/proc/audit");
      const rows = (d.recent || d.entries || d.tail) || (Array.isArray(d) ? d : []);
      if (!rows.length) return dump(sh, d, 0);
      rows.slice(-20).forEach((r) => {
        const ok = r.allowed ?? r.ok ?? (r.verdict === "allow");
        sh.print("#" + esc(r.seq ?? "—") + "  " + esc(r.call ?? r.op ?? "—") + "  " +
          '<span class="sh-' + (ok ? "ok" : "deny") + '">' + (ok ? "allowed" : "REFUSED") + "</span>  " +
          '<span class="sh-dim">' + esc(r.reason ?? "") + "</span>", "row");
      });
    }, { help: "syscall audit", alias: ["syslog"] });

    sh.register("sys", async function () {
      dump(sh, await getJSON("/aegis/system"), 0);
    }, { help: "system health" });

    /* ---- windows --------------------------------------------------------- */
    sh.register("win", function () {
      if (!wm) return sh.print("no window manager on this page", "dim");
      wm.list().forEach((w) => sh.print(
        '<span class="sh-cmd">' + esc(w.id) + "</span>" +
        '<span class="sh-dim">' + (w.open ? (w.floating ? "floating" : "tiled") : "closed") + "</span>", "row"));
    }, { help: "list windows", alias: ["windows"] });

    sh.register("open", function (a) {
      if (!wm) return sh.print("no window manager on this page", "dim");
      if (!wm.byId[a]) return sh.print("no window <b>" + esc(a) + "</b> — try <b>win</b>", "err");
      wm.open(a); sh.print("opened " + esc(a), "ok");
    }, { help: "open a window" });

    sh.register("close", function (a) {
      if (!wm || !wm.byId[a]) return sh.print("no window <b>" + esc(a) + "</b>", "err");
      wm.close(a); sh.print("closed " + esc(a), "ok");
    }, { help: "close a window" });

    sh.register("float", function (a) {
      if (!wm || !wm.byId[a]) return sh.print("no window <b>" + esc(a) + "</b>", "err");
      wm.toggleFloat(a);
      sh.print(esc(a) + " is now " + (wm.byId[a].floating ? "floating" : "tiled"), "ok");
    }, { help: "detach a window", alias: ["detach"] });

    sh.register("layout", function (a) {
      if (!wm) return sh.print("no window manager on this page", "dim");
      if (!a) { sh.print("layout is <b>" + esc(wm.layout) + "</b> · " + Object.keys(WM.LAYOUTS).join(" "), "dim"); return; }
      if (!WM.LAYOUTS[a]) return sh.print("unknown layout — " + Object.keys(WM.LAYOUTS).join(" "), "err");
      wm.setLayout(a); sh.print("layout " + esc(a), "ok");
    }, { help: "set the tiling" });

    /* ---- session --------------------------------------------------------- */
    sh.register("whoami", function () {
      if (!session) return sh.print("no session module on this page", "dim");
      if (!session.role) return sh.print("no session — no privilege ring held", "dim");
      sh.print(esc(session.role) + " · ring " + esc(session.ring));
      sh.print(session.caps().join(" ") || "(no capabilities)", "dim");
    }, { help: "current role and ring", alias: ["id"] });

    sh.register("can", function (a) {
      if (!session) return sh.print("no session module on this page", "dim");
      if (!a) return sh.print("usage: can &lt;capability&gt;", "dim");
      const v = session.may(a.trim());
      sh.print(v.ok
        ? '<span class="sh-ok">permitted</span> — ' + esc(a)
        : '<span class="sh-deny">refused</span> — ' + esc(v.reason), "row");
    }, { help: "test a capability" });

    sh.register("clear", function () { sh.clear(); sh.banner(); }, { help: "clear", alias: ["cls", "clr"] });

    // Tab completion over the nouns this vocabulary actually has.
    sh.completer(function (word, line) {
      const v = line.trim().split(/\s+/)[0];
      if (/^(open|close|float|detach)$/.test(v) && wm) return Object.keys(wm.byId);
      if (v === "layout") return Object.keys(WM.LAYOUTS);
      if (/^(cat|proc|read|ls|dir)$/.test(v)) return PROC;
      if (v === "can" && session && session.policy) return Object.keys(session.policy.cap_min_ring || {});
      return [];
    });

    return sh;
  }

  global.CoreCmds = { install, PROC };
})(window);
