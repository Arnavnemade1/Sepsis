/* ============================================================================
   oskit/boot.js — wire the shell layer into the console
   ----------------------------------------------------------------------------
   Composition only. Every module here is independent and testable on its own;
   this file is the one place that knows they exist together, which keeps the WM
   ignorant of RBAC, the shell ignorant of windows, and so on.

   Runs after os.js so the panels it registers already exist in the DOM.
   ============================================================================ */
(function () {
  "use strict";

  function boot() {
    if (!document.querySelector(".desk")) return;

    /* ---- window manager ------------------------------------------------- */
    const wm = WM.create(".desk");

    // The six desk panels, in the clinical order they already read in.
    [["twin", "Digital twin"], ["risk", "Sepsis risk"], ["vitals", "Telemetry"],
     ["aux", "Auxiliary"], ["agents", "Reasoning"], ["act", "Actions"]]
      .forEach(([k, title]) => {
        const el = document.querySelector(".win-" + k);
        if (el) wm.add(k, el, { title, kind: "panel", open: true });
      });

    // The system monitor is a first-class window, not a modal: it is the one
    // surface an operator wants beside a patient rather than instead of one.
    const smEl = document.getElementById("sysmon-win");
    if (smEl) wm.add("sysmon", smEl, { title: "System monitor", kind: "panel", open: false });

    wm.taskbar("#wm-taskbar").bindKeys();
    wm.restore();
    window.AegisWM = wm;

    /* ---- session --------------------------------------------------------- */
    const session = Session.create("#ses-mount");
    session.load();
    window.AegisSession = session;

    /* ---- system monitor -------------------------------------------------- */
    const sysmon = Sysmon.create("#sysmon-body").mount();
    window.AegisSysmon = sysmon;
    // Poll only while the window is actually open — a background monitor that
    // nobody is looking at is pure egress on a metered backend.
    const smWin = wm.byId.sysmon;
    if (smWin) {
      smWin.onOpen = () => {
        sysmon.start(3000);
        // The monitor has its own tiling; opening it in a layout with no slot for
        // it leaves a grid child with nowhere to go, which invents a phantom track.
        if (wm.layout !== "monitor" && wm.layout !== "focus") wm.setLayout("monitor");
      };
      smWin.onClose = () => {
        sysmon.stop();
        if (wm.layout === "monitor") wm.setLayout("triple");
      };
    }

    /* ---- shell ----------------------------------------------------------- */
    const sh = Shell.create({
      root: "#sh", out: "#sh-out", input: "#sh-in", prompt: "#sh-prompt",
      histKey: "aegis_console_hist",
    });
    CoreCmds.install(sh, { wm, session });

    // `ask` was the whole of what Cmd-K did before this: it opened an LLM prompt.
    // It survives as one verb among forty rather than being removed — the shell is
    // a superset of the panel it replaces, and Summon is still there for anyone
    // who wants the panel itself.
    sh.register("ask", function (q) {
      if (!q) return sh.print("usage: ask &lt;question&gt; — grounded, read-only", "dim");
      if (window.AegisSummon && window.AegisSummon.ask) { window.AegisSummon.ask(q); sh.hide(); return; }
      const b = document.getElementById("launch-summon");
      if (b) { b.click(); sh.hide(); return; }
      sh.print("the assistant panel is not loaded on this page", "err");
    }, { help: "grounded question", alias: ["q", "query"] });

    sh.register("sysmon", function () { wm.open("sysmon"); sh.hide(); },
      { help: "open the system monitor" });

    sh.mount((e) => (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k");
    window.AegisShell = sh;

    // The dock button and Cmd-K reach the same shell.
    const sb = document.getElementById("launch-shell");
    if (sb) sb.addEventListener("click", () => sh.toggle());
    const smb = document.getElementById("launch-sysmon");
    if (smb) smb.addEventListener("click", () => wm.toggle("sysmon"));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
