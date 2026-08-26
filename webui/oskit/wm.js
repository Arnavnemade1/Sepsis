/* ============================================================================
   oskit/wm.js — window manager
   ----------------------------------------------------------------------------
   Tiled by default; floating only when asked.

   That order is a clinical decision, not a stylistic one. ISA-101 (the control-room
   HMI standard this console's design identity already independently matches) asks
   for a predictable, non-occluding layout: an operator must be able to look at the
   same place twice and find the same reading. Free-floating windows are the classic
   OS metaphor and they break exactly that — a dragged window can sit on top of the
   one patient who is deteriorating. So tiles are the ground state and `float` is an
   explicit, reversible act for ad-hoc comparison.

   The WM owns PLACEMENT only. Every surface keeps the markup and the open/close
   entry points it already had in os.js; nothing here re-implements a panel.
   ============================================================================ */
(function (global) {
  "use strict";

  const LS_KEY = "aegis_wm_layout_v1";

  /** Named tilings.
   *
   *  These are CSS attribute selectors, not inline styles. The desk's grid already
   *  carries responsive breakpoints in os.css that collapse three columns to two and
   *  then to one; writing `style.gridTemplateColumns` from JS would win the cascade
   *  at every width and silently destroy that. Setting `data-wm-layout` lets the
   *  layouts compose with the breakpoints instead of overriding them.
   *
   *  Slots are the desk's DIRECT children — the twin, the two panel columns, and the
   *  system monitor — because that is what the grid actually lays out. The panels
   *  inside a column are flex children of that column, not grid items.
   */
  const LAYOUTS = {
    triple:  { label: "three-column (default)" },
    wide:    { label: "twin + left column" },
    focus:   { label: "focused window only" },
    monitor: { label: "system monitor beside the twin" },
  };

  function WM(deskSel) {
    this.desk = document.querySelector(deskSel);
    this.wins = [];              // ordered: tile slot order
    this.byId = Object.create(null);
    this.layout = "triple";      // the console's historical arrangement
    this.focused = null;
    this.zTop = 40;
    this._taskbar = null;
  }

  WM.prototype = {
    /* ---- registry -------------------------------------------------------- */
    /** Register an existing element as a window. `kind` is "panel" (lives in the
     *  desk grid) or "app" (a full surface that takes a tile when opened). */
    add(id, el, meta) {
      if (!el) return null;
      const w = {
        id, el,
        title: (meta && meta.title) || id,
        kind: (meta && meta.kind) || "panel",
        floating: false,
        open: (meta && meta.open !== undefined) ? meta.open : !el.classList.contains("hidden"),
        onOpen: meta && meta.onOpen,
        onClose: meta && meta.onClose,
        rect: null,
      };
      this.wins.push(w);
      this.byId[id] = w;
      el.classList.add("wm-win");
      el.dataset.wmId = id;
      el.addEventListener("mousedown", () => this.focus(id), true);
      this._chrome(w);
      return w;
    },

    /** Give a window its float/close affordances.
     *
     *  The desk panels already ship a `.win-bar` with a title in the markup, so
     *  this appends to that bar rather than injecting a second one — two stacked
     *  title bars is the classic symptom of a window manager bolted onto a page
     *  that already had chrome. Only surfaces with no bar of their own get one.
     */
    _chrome(w) {
      let bar = w.el.querySelector(":scope > .win-bar") || w.el.querySelector(":scope > .wm-bar");
      if (!bar) {
        bar = document.createElement("div");
        bar.className = "wm-bar";
        bar.innerHTML = '<span class="wm-title">' + Shell.esc(w.title) + "</span>";
        w.el.insertBefore(bar, w.el.firstChild);
      }
      if (bar.querySelector(".wm-acts")) return;
      const acts = document.createElement("span");
      acts.className = "wm-acts";
      acts.innerHTML =
        '<button class="wm-act" data-act="float" title="Detach into a floating window">\u29c9</button>' +
        '<button class="wm-act" data-act="close" title="Close">\u2715</button>';
      acts.addEventListener("click", (e) => {
        const a = e.target.closest("[data-act]");
        if (!a) return;
        e.stopPropagation();
        if (a.dataset.act === "float") this.toggleFloat(w.id);
        else this.close(w.id);
      });
      bar.appendChild(acts);
      this._drag(w, bar);
    },

    /* ---- focus + z-order -------------------------------------------------- */
    focus(id) {
      const w = this.byId[id]; if (!w) return;
      if (this.focused === id) return;
      this.focused = id;
      this.wins.forEach((x) => x.el.classList.toggle("wm-focus", x.id === id));
      if (w.floating) w.el.style.zIndex = String(++this.zTop);
      this._renderTaskbar();
    },

    /* ---- open / close ----------------------------------------------------- */
    open(id) {
      const w = this.byId[id]; if (!w) return;
      w.open = true;
      w.el.classList.remove("hidden");
      if (w.onOpen) w.onOpen(w);
      this.apply();
      this.focus(id);
    },
    close(id) {
      const w = this.byId[id]; if (!w) return;
      w.open = false;
      w.floating = false;
      w.el.classList.add("hidden");
      w.el.classList.remove("wm-float");
      w.el.style.cssText = w.el.style.cssText.replace(/(left|top|width|height|z-index):[^;]*;?/g, "");
      if (w.onClose) w.onClose(w);
      if (this.focused === id) this.focused = null;
      this.apply();
    },
    toggle(id) { const w = this.byId[id]; if (w) (w.open ? this.close(id) : this.open(id)); },

    /* ---- floating --------------------------------------------------------- */
    toggleFloat(id) {
      const w = this.byId[id]; if (!w) return;
      w.floating = !w.floating;
      if (w.floating) {
        // Freeze the tile's current geometry as the float's starting rect, so the
        // window does not jump when it detaches — the motion should read as "lifted
        // off the grid", not "teleported".
        const r = w.el.getBoundingClientRect();
        w.el.classList.add("wm-float");
        w.el.style.left = r.left + "px";
        w.el.style.top = r.top + "px";
        w.el.style.width = r.width + "px";
        w.el.style.height = r.height + "px";
        w.el.style.zIndex = String(++this.zTop);
      } else {
        w.el.classList.remove("wm-float");
        ["left", "top", "width", "height", "zIndex"].forEach((k) => { w.el.style[k] = ""; });
      }
      this.apply();
      this.focus(id);
      this.save();
    },

    _drag(w, handle) {
      const self = this;
      handle.addEventListener("mousedown", function (e) {
        if (!w.floating || e.target.closest("[data-act]")) return;
        e.preventDefault();
        const sx = e.clientX, sy = e.clientY;
        const r = w.el.getBoundingClientRect();
        const move = (ev) => {
          // Clamp so a window can never be dragged fully off-screen and stranded.
          const nx = Math.max(8 - r.width + 80, Math.min(window.innerWidth - 80, r.left + ev.clientX - sx));
          const ny = Math.max(0, Math.min(window.innerHeight - 40, r.top + ev.clientY - sy));
          w.el.style.left = nx + "px";
          w.el.style.top = ny + "px";
        };
        const up = () => {
          document.removeEventListener("mousemove", move);
          document.removeEventListener("mouseup", up);
          self.save();
        };
        document.addEventListener("mousemove", move);
        document.addEventListener("mouseup", up);
        self.focus(w.id);
      });
    },

    /* ---- layout ----------------------------------------------------------- */
    setLayout(name) {
      if (!LAYOUTS[name]) return;
      this.layout = name;
      this.apply();
      this.save();
    },
    cycleLayout() {
      const keys = Object.keys(LAYOUTS);
      this.setLayout(keys[(keys.indexOf(this.layout) + 1) % keys.length]);
    },

    /** Write the tiling by attribute, and mark which slot holds focus so a
     *  focus-only layout knows what to keep. */
    apply() {
      if (!this.desk) return;
      this.desk.dataset.wmLayout = this.layout;
      const f = this.focused && this.byId[this.focused];
      const slots = [...this.desk.children];
      slots.forEach((c) => c.classList.remove("wm-slot-focus"));
      let marked = null;
      if (f && !f.floating) {
        const slot = f.el.parentElement === this.desk ? f.el : f.el.closest(".desk > *");
        if (slot && slot.parentElement === this.desk) { slot.classList.add("wm-slot-focus"); marked = slot; }
      }
      // A focus-only layout with nothing focused renders an EMPTY DESK — a blank
      // screen where a patient should be. Nothing about "the operator has not
      // clicked anything yet" should mean "show no vital signs", so the layout
      // falls back to the first slot that has something in it.
      if (!marked && this.layout === "focus") {
        const first = slots.find((c) => !c.classList.contains("hidden") && c.offsetParent !== null)
                   || slots.find((c) => !c.classList.contains("hidden"));
        if (first) first.classList.add("wm-slot-focus");
      }
      this.wins.forEach((w) => {
        if (w.kind === "overlay") return;
        w.el.classList.toggle("hidden", !w.open && !w.floating);
      });
      this._renderTaskbar();
      // Canvas and WebGL surfaces size from their box, and a box that was
      // display:none measures zero. Anything re-tiled has to be told.
      window.dispatchEvent(new Event("resize"));
    },

    /* ---- taskbar ---------------------------------------------------------- */
    taskbar(sel) { this._taskbar = document.querySelector(sel); this._renderTaskbar(); return this; },
    _renderTaskbar() {
      const tb = this._taskbar; if (!tb) return;
      const open = this.wins.filter((w) => w.open);
      tb.innerHTML = open.map((w) =>
        '<button class="wm-task' + (w.id === this.focused ? " on" : "") + (w.floating ? " float" : "") +
        '" data-wm="' + Shell.esc(w.id) + '">' + Shell.esc(w.title) + "</button>"
      ).join("") || '<span class="wm-task-empty">no windows</span>';
      if (!tb._wired) {
        tb.addEventListener("click", (e) => {
          const b = e.target.closest("[data-wm]");
          if (b) this.focus(b.dataset.wm);
        });
        tb._wired = true;
      }
    },

    /* ---- persistence ------------------------------------------------------ */
    save() {
      try {
        localStorage.setItem(LS_KEY, JSON.stringify({
          layout: this.layout,
          floating: this.wins.filter((w) => w.floating).map((w) => ({
            id: w.id, left: w.el.style.left, top: w.el.style.top,
            width: w.el.style.width, height: w.el.style.height,
          })),
        }));
      } catch (e) {}
    },
    restore() {
      let s = null;
      try { s = JSON.parse(localStorage.getItem(LS_KEY) || "null"); } catch (e) {}
      if (!s) { this.apply(); return; }
      if (LAYOUTS[s.layout]) this.layout = s.layout;
      (s.floating || []).forEach((f) => {
        const w = this.byId[f.id];
        if (!w || !w.open) return;
        w.floating = true;
        w.el.classList.add("wm-float");
        Object.assign(w.el.style, { left: f.left, top: f.top, width: f.width, height: f.height, zIndex: String(++this.zTop) });
      });
      this.apply();
    },

    /* ---- keyboard --------------------------------------------------------- */
    bindKeys() {
      const self = this;
      document.addEventListener("keydown", function (e) {
        const t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
        if (!(e.metaKey || e.ctrlKey)) return;
        if (e.key === "\\") { e.preventDefault(); self.cycleLayout(); return; }
        if (e.key.toLowerCase() === "w" && self.focused) { e.preventDefault(); self.close(self.focused); return; }
        if (/^[1-9]$/.test(e.key)) {
          const open = self.wins.filter((w) => w.open);
          const w = open[Number(e.key) - 1];
          if (w) { e.preventDefault(); self.focus(w.id); }
        }
      });
      return this;
    },

    list() { return this.wins.map((w) => ({ id: w.id, title: w.title, open: w.open, floating: w.floating })); },
  };

  global.WM = { create(deskSel) { return new WM(deskSel); }, LAYOUTS };
})(window);
