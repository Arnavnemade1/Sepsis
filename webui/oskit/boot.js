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
      frame: "#sh-frame", head: "#sh-head", grip: "#sh-grip", close: "#sh-close",
      status: "#sh-status", ghost: "#sh-ghost", titlePath: "#sh-title-path",
      chipRing: "#sh-chip-ring", segKernel: "#sh-seg-kernel", segHost: "#sh-seg-host",
      segWin: "#sh-seg-win", segCmds: "#sh-seg-cmds", segUp: "#sh-seg-up",
      histKey: "aegis_console_hist",
    });
    CoreCmds.install(sh, { wm, session });

    // The status line reads the running system, never a literal. `host` is where the
    // kernel actually is, which on a split deploy is not this page's origin — the one
    // fact an operator most needs when a verb comes back empty.
    sh.bar(function () {
      let host = "local";
      try {
        const b = (window.AEGIS && AEGIS.base && AEGIS.base()) || location.origin;
        host = new URL(b, location.href).host;
      } catch (e) {}
      // The kernel version is read from the system bar that already displays it,
      // rather than from a new global: one element owns that fact and stays live.
      const ver = (document.getElementById("sb-ver") || {}).textContent || "";
      return {
        kernel: ver.trim(),
        host: host,
        windows: wm.list().filter((w) => w.open).length,
        role: session.role || "",
        ring: session.ring,
      };
    });

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

    // Ctrl-` — the terminal gesture on every platform that has one. ⌘K now belongs
    // to the palette, which is where the terminal is also one row.
    sh.mount((e) => e.ctrlKey && (e.key === "`" || e.code === "Backquote"));
    window.AegisShell = sh;

    /* ---- command palette ------------------------------------------------- */
    const pal = Palette.create({
      root: "#pl", input: "#pl-in", list: "#pl-list", scrim: "#pl-scrim",
    });

    // Windows. Read live, so the verb offered is the one that applies right now —
    // a palette that says "open" for a window already on screen is worse than none.
    pal.source(() => wm.list().map((w) => ({
      group: "windows", icon: w.open ? "▣" : "▢",
      title: (w.open ? "Close " : "Open ") + (w.title || w.id).toLowerCase(),
      hint: w.open ? (w.floating ? "floating" : "tiled") : "closed",
      run: () => wm.toggle(w.id),
    })));
    pal.source(() => wm.list().filter((w) => w.open).map((w) => ({
      group: "windows", icon: "⧉",
      title: (w.floating ? "Dock " : "Float ") + (w.title || w.id).toLowerCase(),
      run: () => wm.toggleFloat(w.id),
    })));

    // Layouts.
    pal.source(() => Object.keys(WM.LAYOUTS).map((k) => ({
      group: "layout", icon: "▤", title: "Layout: " + k,
      hint: WM.LAYOUTS[k].label + (wm.layout === k ? " · current" : ""),
      run: () => wm.setLayout(k),
    })));

    // Patients. The top-bar selector is the only other way to switch scope, and it
    // is a mouse target in a corner; this makes it a keystroke.
    pal.source(() => {
      const P = window.AegisPatient;
      const all = (P && P.all && P.all()) || [];
      const cur = (P && P.current && P.current()) || null;
      return all.map((p) => ({
        group: "patient", icon: p.kind === "live" ? "◉" : "◍",
        title: "Go to " + p.name,
        hint: (p.kind === "live" ? "live monitor" : "bed " + String(p.id).replace("bed_", ""))
          + (p.septic ? " · septic course" : "") + (cur && cur.id === p.id ? " · current" : ""),
        run: () => P.set(p.id),
      }));
    });

    // Apps — read off the dock rather than listed again here, so a launcher added to
    // the dock is in the palette without anyone remembering to add it twice.
    pal.source(() => [].slice.call(document.querySelectorAll(".dock-grp-apps .dock-btn, .dock-grp-sys .dock-btn"))
      .map((btn) => ({
        group: "apps", icon: "◆",
        title: (btn.textContent || "").replace(/[◉◔◈◎▦❯_]/g, "").replace(/⌘K/g, "").trim(),
        hint: (btn.getAttribute("title") || "").split(" — ")[1] || "",
        run: () => btn.click(),
      })).filter((r) => r.title));

    // Kernel verbs. Every shell command is reachable without opening the shell; the
    // shell is opened to SHOW the result, which is what a palette entry cannot do.
    pal.source(() => Object.keys(sh.cmds).sort().map((name) => ({
      group: "kernel", icon: "❯", title: name, hint: sh.cmds[name].help || "",
      run: () => { sh.show(); sh.run(name); },
    })));

    pal.mount((e) => (e.metaKey || e.ctrlKey) && (e.key || "").toLowerCase() === "k");
    window.AegisPalette = pal;

    /* ---- keyboard sheet --------------------------------------------------- */
    // Every binding in this list is one that is actually registered somewhere in
    // this codebase; there is no aspirational row. ⌘\, ⌘1-9 and ⌘W in particular
    // have been live in wm.js since it was written with nothing anywhere naming them.
    const KEYS = [
      ["System", [
        ["Command palette", "⌘K"],
        ["Terminal", "⌃`"],
        ["Ask Dr. Aegis", "⌥A"],
        ["Keyboard shortcuts", "?"],
        ["Dismiss the top overlay", "esc"],
      ]],
      ["Windows", [
        ["Cycle the tiling", "⌘\\"],
        ["Focus the nth open window", "⌘1…9"],
        ["Close the focused window", "⌘W"],
        ["Window menu", "right-click its title bar", true],
        ["Workspace menu", "right-click the desk", true],
      ]],
      ["Terminal", [
        ["Complete", "tab"],
        ["Accept the suggestion", "→"],
        ["History", "↑ ↓"],
        ["Clear the screen", "⌃L"],
        ["Interrupt", "⌃C"],
      ]],
      ["Palette", [
        ["Select", "↑ ↓"],
        ["Run", "⏎"],
        ["Dismiss", "esc"],
      ]],
    ];
    const ks = document.getElementById("ks");
    function ksRender() {
      const body = document.getElementById("ks-body");
      if (!body) return;
      // A mouse gesture is not a key. Rendering "right-click its title bar" inside a
      // <kbd> chip both lies about what it is and overflows the chip, which clipped
      // the end of the sentence.
      body.innerHTML = KEYS.map(([sec, rows]) =>
        '<div class="ks-sec">' + sec + "</div>" + rows.map(([label, key, gesture]) =>
          '<div class="ks-row"><i>' + Palette.esc(label) + "</i>"
          + (gesture ? '<span class="ks-gest">' + Palette.esc(key) + "</span>"
                     : "<kbd>" + Palette.esc(key) + "</kbd>")
          + "</div>").join("")).join("");
    }
    function ksShow() { ksRender(); ks.classList.add("open"); ks.setAttribute("aria-hidden", "false"); }
    function ksHide() { ks.classList.remove("open"); ks.setAttribute("aria-hidden", "true"); }
    if (ks) {
      document.getElementById("ks-scrim").addEventListener("click", ksHide);
      document.addEventListener("keydown", (e) => {
        const t = e.target;
        const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
        if (e.key === "?" && !typing && !e.metaKey && !e.ctrlKey) { e.preventDefault(); ksShow(); return; }
        if (e.key === "Escape" && ks.classList.contains("open")) ksHide();
      });
    }

    // System rows — the verbs that are not a window, an app or a kernel node.
    pal.source(() => [
      { group: "system", icon: "⌨", title: "Keyboard shortcuts", kbd: "?", run: ksShow },
      { group: "system", icon: "❯", title: "Open terminal", kbd: "⌃`", run: () => sh.show() },
      { group: "system", icon: "▦", title: "System monitor", run: () => wm.open("sysmon") },
    ]);

    /* ---- context menus ---------------------------------------------------- */
    function layoutItems() {
      return Object.keys(WM.LAYOUTS).map((k) => ({
        label: "Layout: " + k, hint: wm.layout === k ? "current" : "",
        disabled: wm.layout === k, run: () => wm.setLayout(k),
      }));
    }
    document.addEventListener("contextmenu", function (e) {
      // A window's own title bar: verbs for THAT window. Until now float and close
      // existed only as 20px glyphs inside a 26px bar.
      const bar = e.target.closest(".wm-bar, .win-bar");
      const win = bar && bar.closest("[data-wm-id]");
      if (win && wm.byId[win.dataset.wmId]) {
        e.preventDefault();
        const w = wm.byId[win.dataset.wmId];
        Menu.open(e.clientX, e.clientY, [
          { label: w.floating ? "Dock into the tiling" : "Float this window",
            run: () => wm.toggleFloat(w.id) },
          { label: "Focus only this", hint: "layout", run: () => { wm.focus(w.id); wm.setLayout("focus"); } },
          { sep: true },
          { label: "Close", hint: "⌘W", run: () => wm.close(w.id) },
        ]);
        return;
      }
      // The desk itself: the workspace verbs. Right-clicking a chart or a reading
      // is left alone — the browser's own menu is the right answer over content.
      if (e.target.closest(".desk") && !e.target.closest(".win-body, .wm-win .win-body")) {
        e.preventDefault();
        Menu.open(e.clientX, e.clientY, layoutItems().concat([
          { sep: true },
          { label: "Open terminal", hint: "⌃`", run: () => sh.show() },
          { label: "Command palette", hint: "⌘K", run: () => pal.show() },
          { label: "System monitor", run: () => wm.open("sysmon") },
          { label: "Keyboard shortcuts", hint: "?", run: ksShow },
        ]));
      }
    });

    // The dock button and Cmd-K reach the same shell — and the button reflects its
    // state whichever of the two opened it, so the dock never claims the terminal is
    // closed while it is sitting on screen.
    const sb = document.getElementById("launch-shell");
    if (sb) {
      sb.addEventListener("click", () => sh.toggle());
      const markDock = () => sb.classList.toggle("on", sh.open);
      const prevShow = sh.o.onShow, prevHide = sh.o.onHide;
      sh.o.onShow = (s) => { markDock(); if (prevShow) prevShow(s); };
      sh.o.onHide = (s) => { markDock(); if (prevHide) prevHide(s); };
      markDock();
    }
    const smb = document.getElementById("launch-sysmon");
    if (smb) smb.addEventListener("click", () => wm.toggle("sysmon"));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
