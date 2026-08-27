/* ============================================================================
   oskit/palette.js — one command gesture for the whole OS
   ----------------------------------------------------------------------------
   ⌘K used to open two different things. `summon.js` bound it to the assistant and
   `boot.js` bound it to the terminal, both called preventDefault, and both fired —
   so the single most-used key in the product opened two overlays on top of each
   other. Reassigning one of them would have been a coin flip about which surface
   "deserves" the gesture.

   This resolves it the way an OS does: ⌘K is not a shortcut to a panel, it is the
   command gesture. Everything the console can do is a row in here — open a window,
   switch a patient, change the tiling, run a kernel verb, wake the assistant, open
   the terminal — and the panels that used to fight over the key are two of those
   rows.

   The palette owns no vocabulary. Pages register sources; a source is a function
   returning rows, called fresh on every keystroke so the list reflects the system
   as it is now rather than as it was when the palette was built.
   ============================================================================ */
(function (global) {
  "use strict";

  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /** Subsequence score. Returns -1 for no match, higher is better.
   *
   *  Deliberately not a substring test: an operator typing "dt" should reach
   *  "Digital twin", which `indexOf` never finds. Word-start hits and runs of
   *  adjacent characters score higher, so exact prefixes still win outright and
   *  the ranking does not feel arbitrary. */
  function score(needle, hay) {
    if (!needle) return 0;
    const n = needle.toLowerCase(), h = hay.toLowerCase();
    let s = 0, hi = 0, run = 0;
    for (let i = 0; i < n.length; i++) {
      const c = n[i];
      const at = h.indexOf(c, hi);
      if (at < 0) return -1;
      if (at === hi && i > 0) { run += 1; s += 6 + run * 2; }   // contiguous
      else { run = 0; s += 1; }
      if (at === 0 || /[\s\-\/_.:]/.test(h[at - 1] || "")) s += 8;  // word start
      hi = at + 1;
    }
    return s - Math.max(0, h.length - n.length) * 0.05;          // prefer tight hits
  }

  function Palette(o) {
    this.o = o || {};
    this.sources = [];
    this.rows = [];
    this.sel = 0;
    this.open = false;
  }

  Palette.prototype = {
    _el(s) { return typeof s === "string" ? document.querySelector(s) : s; },
    get root() { return this._el(this.o.root); },
    get inEl() { return this._el(this.o.input); },
    get listEl() { return this._el(this.o.list); },

    /** Register a row source: fn() -> [{title, group, hint, kbd, run}] */
    source(fn) { this.sources.push(fn); return this; },

    _collect() {
      let out = [];
      this.sources.forEach((fn) => {
        try { out = out.concat(fn() || []); } catch (e) {}
      });
      return out;
    },

    _match(q) {
      const all = this._collect();
      if (!q) {
        // No query: show the whole vocabulary in source order rather than a
        // "recent" list nobody has taught this thing how to earn.
        return all.slice(0, 60);
      }
      return all
        .map((r) => ({ r, s: Math.max(score(q, r.title), score(q, (r.group || "") + " " + r.title) - 4) }))
        .filter((x) => x.s >= 0)
        .sort((a, b) => b.s - a.s)
        .slice(0, 40)
        .map((x) => x.r);
    },

    render() {
      const list = this.listEl;
      if (!list) return;
      const q = (this.inEl && this.inEl.value || "").trim();
      this.rows = this._match(q);
      if (!this.rows.length) {
        list.innerHTML = '<div class="pl-empty">no command matches <b>' + esc(q) + "</b></div>";
        return;
      }
      this.sel = Math.min(this.sel, this.rows.length - 1);
      let html = "", lastGroup = null;
      this.rows.forEach((r, i) => {
        if (r.group !== lastGroup) {
          html += '<div class="pl-group">' + esc(r.group || "") + "</div>";
          lastGroup = r.group;
        }
        html += '<button class="pl-row' + (i === this.sel ? " on" : "") + '" data-i="' + i + '">'
          + '<span class="pl-ico">' + esc(r.icon || "▸") + "</span>"
          + '<span class="pl-title">' + esc(r.title) + "</span>"
          + (r.hint ? '<span class="pl-hint">' + esc(r.hint) + "</span>" : "")
          + (r.kbd ? '<span class="pl-kbd">' + esc(r.kbd) + "</span>" : "")
          + "</button>";
      });
      list.innerHTML = html;
      this._scrollToSel();
    },

    _scrollToSel() {
      const el = this.listEl && this.listEl.querySelector(".pl-row.on");
      if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
    },

    move(d) {
      if (!this.rows.length) return;
      this.sel = (this.sel + d + this.rows.length) % this.rows.length;
      const list = this.listEl;
      list.querySelectorAll(".pl-row").forEach((el, i) => el.classList.toggle("on", i === this.sel));
      this._scrollToSel();
    },

    accept(i) {
      const r = this.rows[i == null ? this.sel : i];
      if (!r) return;
      this.hide();
      // After the overlay is down, so a row that opens another surface does not
      // race the palette's own teardown for focus.
      setTimeout(() => { try { r.run(); } catch (e) {} }, 0);
    },

    show() {
      this.open = true;
      const r = this.root;
      if (r) { r.classList.add("open"); r.setAttribute("aria-hidden", "false"); }
      const i = this.inEl;
      if (i) { i.value = ""; setTimeout(() => i.focus(), 20); }
      this.sel = 0;
      this.render();
    },
    hide() {
      this.open = false;
      const r = this.root;
      if (r) { r.classList.remove("open"); r.setAttribute("aria-hidden", "true"); }
    },
    toggle() { this.open ? this.hide() : this.show(); },

    mount(openMatch) {
      const self = this, i = this.inEl;
      if (i) {
        i.addEventListener("input", () => { self.sel = 0; self.render(); });
        i.addEventListener("keydown", (e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); self.move(1); return; }
          if (e.key === "ArrowUp") { e.preventDefault(); self.move(-1); return; }
          if (e.key === "Enter") { e.preventDefault(); self.accept(); return; }
          if (e.key === "Escape") { e.preventDefault(); self.hide(); return; }
        });
      }
      const list = this.listEl;
      if (list) {
        list.addEventListener("click", (e) => {
          const row = e.target.closest(".pl-row");
          if (row) self.accept(Number(row.dataset.i));
        });
        list.addEventListener("mousemove", (e) => {
          const row = e.target.closest(".pl-row");
          if (!row) return;
          const n = Number(row.dataset.i);
          if (n === self.sel) return;
          self.sel = n;
          list.querySelectorAll(".pl-row").forEach((el, k) => el.classList.toggle("on", k === n));
        });
      }
      const scrim = this._el(this.o.scrim);
      if (scrim) scrim.addEventListener("click", () => self.hide());

      document.addEventListener("keydown", function (e) {
        if (openMatch && openMatch(e)) { e.preventDefault(); self.toggle(); return; }
        if (e.key === "Escape" && self.open) self.hide();
      });
      return this;
    },
  };

  global.Palette = { create: (o) => new Palette(o), score: score, esc: esc };
})(window);
