/* ============================================================================
   tactical-os.js — mounts the OS shell on the tactical console
   ----------------------------------------------------------------------------
   oskit was extracted so that any page could mount it and register its own
   nouns; this is the second page to do so. The mechanism (window manager,
   palette, terminal, context menus, keyboard sheet) is shared with the clinical
   console. The VOCABULARY is not: beds, views and the interrupt log belong to
   this page's data layer and are registered here.

   Deliberately additive. tactical.js keeps ownership of the feed, the canvases
   and the dossier; nothing here reaches into its state except through the DOM
   contracts it already publishes (the tab buttons, the bed cards).
   ========================================================================== */
(function () {
  "use strict";

  function boot() {
    if (!window.WM || !window.Shell || !window.Palette) return;

    /* ---- windows ---------------------------------------------------------- */
    // The three operations panels become real windows: tileable, floatable,
    // closable, and listed in the dock's taskbar.
    const wm = WM.create(".board");
    [
      ["opgrid", ".span-ops", "ICU operations grid"],
      ["learning", ".span-learn", "Per-patient baseline learning"],
      ["personalization", ".span-blind", "Personalization map"],
    ].forEach(([id, sel, title]) => {
      const el = document.querySelector(sel);
      if (el) wm.add(id, el, { title, kind: "panel", open: true });
    });
    const term = document.getElementById("terminal");
    if (term) wm.add("protocol", term, { title: "Protocol terminal", kind: "panel", open: true });
    // taskbar() renders the dock's window list AND is what keeps it in sync — there
    // is no separate mount step; `wm.mount()` does not exist and the first version
    // of this file called it defensively, which is why the taskbar stayed empty.
    wm.taskbar("#wm-taskbar").bindKeys();
    wm.restore();
    window.AegisWM = wm;

    /* ---- terminal --------------------------------------------------------- */
    const sh = Shell.create({
      root: "#sh", out: "#sh-out", input: "#sh-in", prompt: "#sh-prompt",
      histKey: "aegis_tactical_hist",
    });
    if (window.CoreCmds) CoreCmds.install(sh, { wm, session: window.AegisSession });

    // This page's own nouns.
    const beds = () =>
      [].slice.call(document.querySelectorAll("[data-bed]"))
        .map((el) => el.getAttribute("data-bed"))
        .filter((v, i, a) => v && a.indexOf(v) === i);

    sh.register("beds", function () {
      const b = beds();
      if (!b.length) return sh.print("no beds in the current feed", "dim");
      b.forEach((id) => sh.print('<span class="sh-cmd">' + sh.esc(id) + "</span>", "row"));
    }, { help: "list beds in the feed", alias: ["ls-beds"] });

    sh.register("open", function (a) {
      const el = a && document.querySelector('[data-bed="' + CSS.escape(a) + '"]');
      if (!el) return sh.print("no bed <b>" + sh.esc(a || "") + "</b> — try <b>beds</b>", "err");
      sh.hide(); el.click();
      sh.print("opened dossier " + sh.esc(a), "ok");
    }, { help: "open a bed dossier" });

    sh.register("view", function (a) {
      const tab = a && document.querySelector('.tab[data-view="' + CSS.escape(a) + '"]');
      if (!a) return sh.print("views: ops · val · devices", "dim");
      if (!tab) return sh.print("unknown view <b>" + sh.esc(a) + "</b>", "err");
      tab.click(); sh.print("view " + sh.esc(a), "ok");
    }, { help: "switch view (ops | val | devices)" });

    sh.completer(function (word, line) {
      const v = line.trim().split(/\s+/)[0];
      if (v === "open") return beds();
      if (v === "view") return ["ops", "val", "devices"];
      return [];
    });

    sh.mount((e) => e.ctrlKey && (e.key === "`" || e.code === "Backquote"));
    window.AegisShell = sh;

    /* ---- palette ---------------------------------------------------------- */
    const pal = Palette.create({
      root: "#pl", input: "#pl-in", list: "#pl-list", scrim: "#pl-scrim",
    });

    pal.source(() => wm.list().map((w) => ({
      group: "windows", icon: w.open ? "▣" : "▢",
      title: (w.open ? "Close " : "Open ") + (w.title || w.id).toLowerCase(),
      hint: w.open ? (w.floating ? "floating" : "tiled") : "closed",
      run: () => wm.toggle(w.id),
    })));
    pal.source(() => Object.keys(WM.LAYOUTS).map((k) => ({
      group: "layout", icon: "▤", title: "Layout: " + k,
      hint: WM.LAYOUTS[k].label + (wm.layout === k ? " · current" : ""),
      run: () => wm.setLayout(k),
    })));
    // Beds are read live off the rendered grid, so the palette lists exactly the
    // cohort the feed is showing rather than a snapshot taken at boot.
    pal.source(() =>
      [].slice.call(document.querySelectorAll("[data-bed]")).map((el) => ({
        group: "beds", icon: "◍",
        title: "Open " + (el.getAttribute("data-name") || el.getAttribute("data-bed")),
        hint: el.getAttribute("data-bed") || "",
        run: () => el.click(),
      })));
    pal.source(() =>
      [].slice.call(document.querySelectorAll(".tab")).map((t) => ({
        group: "views", icon: "◈", title: "Go to " + t.textContent.trim(),
        hint: t.classList.contains("active") ? "current" : "",
        run: () => t.click(),
      })));
    pal.source(() => Object.keys(sh.cmds).sort().map((name) => ({
      group: "kernel", icon: "❯", title: name, hint: sh.cmds[name].help || "",
      run: () => { sh.show(); sh.run(name); },
    })));
    pal.source(() => [
      { group: "system", icon: "⌨", title: "Keyboard shortcuts", kbd: "?", run: ksShow },
      { group: "system", icon: "❯", title: "Open terminal", kbd: "⌃`", run: () => sh.show() },
    ]);

    pal.mount((e) => (e.metaKey || e.ctrlKey) && (e.key || "").toLowerCase() === "k");
    window.AegisPalette = pal;

    /* ---- keyboard sheet --------------------------------------------------- */
    const KEYS = [
      ["System", [["Command palette", "⌘K"], ["Terminal", "⌃`"],
                  ["Keyboard shortcuts", "?"], ["Dismiss the top overlay", "esc"]]],
      ["Windows", [["Cycle the tiling", "⌘\\"], ["Focus the nth open window", "⌘1…9"],
                   ["Close the focused window", "⌘W"],
                   ["Window menu", "right-click its title bar", true],
                   ["Workspace menu", "right-click the board", true]]],
      ["Terminal", [["Complete", "tab"], ["History", "↑ ↓"],
                    ["Clear the screen", "⌃L"], ["Interrupt", "⌃C"]]],
    ];
    const ks = document.getElementById("ks");
    function ksShow() {
      const body = document.getElementById("ks-body");
      if (body) {
        body.innerHTML = KEYS.map(([sec, rows]) =>
          '<div class="ks-sec">' + sec + "</div>" + rows.map(([l, k, gest]) =>
            '<div class="ks-row"><i>' + Palette.esc(l) + "</i>"
            + (gest ? '<span class="ks-gest">' + Palette.esc(k) + "</span>"
                    : "<kbd>" + Palette.esc(k) + "</kbd>") + "</div>").join(""),
        ).join("");
      }
      ks.classList.add("open"); ks.setAttribute("aria-hidden", "false");
    }
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

    /* ---- context menus ---------------------------------------------------- */
    document.addEventListener("contextmenu", function (e) {
      if (!window.Menu) return;
      const bar = e.target.closest(".card-hd, .term-bar");
      const win = bar && bar.closest("[data-wm-id]");
      if (win && wm.byId[win.dataset.wmId]) {
        e.preventDefault();
        const w = wm.byId[win.dataset.wmId];
        Menu.open(e.clientX, e.clientY, [
          { label: w.floating ? "Dock into the tiling" : "Float this window",
            run: () => wm.toggleFloat(w.id) },
          { label: "Focus only this", hint: "layout",
            run: () => { wm.focus(w.id); wm.setLayout("focus"); } },
          { sep: true },
          { label: "Close", hint: "⌘W", run: () => wm.close(w.id) },
        ]);
        return;
      }
      if (e.target.closest(".board") && !e.target.closest(".card-body")) {
        e.preventDefault();
        Menu.open(e.clientX, e.clientY,
          Object.keys(WM.LAYOUTS).map((k) => ({
            label: "Layout: " + k, hint: wm.layout === k ? "current" : "",
            disabled: wm.layout === k, run: () => wm.setLayout(k),
          })).concat([
            { sep: true },
            { label: "Open terminal", hint: "⌃`", run: () => sh.show() },
            { label: "Command palette", hint: "⌘K", run: () => pal.show() },
            { label: "Keyboard shortcuts", hint: "?", run: ksShow },
          ]));
      }
    });

    /* ---- dock wiring ------------------------------------------------------ */
    const on = (id, fn) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("click", fn);
    };
    on("cmd-open", () => sh.toggle());
    on("dock-palette", () => pal.toggle());
    on("dock-keys", ksShow);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
