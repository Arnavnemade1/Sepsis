/* ============================================================================
   oskit/shell.js — the terminal mechanism
   ----------------------------------------------------------------------------
   A shell, minus its vocabulary. This owns everything that is the same in every
   terminal — the output log, the echoed prompt, history with persistence, tab
   completion, a working directory, and the key bindings that open and drive it —
   and owns no commands at all. Pages register their own verbs.

   That split is deliberate and was found rather than assumed. The console's
   existing shell lives in tactical.js, and its ~40 verbs turned out to be bound
   tightly to that page's data layer (resolveBed, bedTag, tierOf, cleanName). The
   mechanism is genuinely page-agnostic; the vocabulary is not. Lifting the whole
   block would have dragged the cohort model into every page that wanted a prompt.

   Usage:
       const sh = Shell.create({ root: "#cmd", out: "#cmd-out", input: "#cmd-in" });
       sh.register("ps", fn, { help: "process table" });
       sh.alias("ls", "dir");
   ============================================================================ */
(function (global) {
  "use strict";

  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function Shell(opts) {
    this.o = opts || {};
    this.cmds = Object.create(null);
    this.aliases = Object.create(null);
    this.hist = [];
    this.hi = 0;
    this.cwd = "~";
    this.open = false;
    this.histKey = this.o.histKey || "aegis_shell_hist";
    this._completers = [];
    this._loadHist();
  }

  Shell.prototype = {
    /* ---- element access -------------------------------------------------- */
    _el(sel) { return typeof sel === "string" ? document.querySelector(sel) : sel; },
    get root() { return this._el(this.o.root); },
    get outEl() { return this._el(this.o.out); },
    get inEl() { return this._el(this.o.input); },

    /* ---- registry -------------------------------------------------------- */
    /** Register a verb. `handler(arg, shell)` — anything it returns is ignored;
     *  output goes through `print`, so a command is free to be async. */
    register(name, handler, meta) {
      this.cmds[name] = { fn: handler, help: (meta && meta.help) || "", usage: (meta && meta.usage) || name };
      if (meta && meta.alias) [].concat(meta.alias).forEach((a) => { this.aliases[a] = name; });
      return this;
    },
    alias(from, to) { this.aliases[from] = to; return this; },
    /** Extra tab-completion source: fn(currentWord, fullLine) -> [strings] */
    completer(fn) { this._completers.push(fn); return this; },

    /* ---- output ---------------------------------------------------------- */
    print(html, cls) {
      const o = this.outEl;
      if (!o) return;
      const d = document.createElement("div");
      d.className = "sh-line" + (cls ? " sh-" + cls : "");
      d.innerHTML = html;
      o.appendChild(d);
      // Only autoscroll when the reader is already at the bottom: yanking the view
      // down while someone is reading earlier output is how a log becomes unusable.
      const atBottom = o.scrollTop + o.clientHeight >= o.scrollHeight - d.offsetHeight - 8;
      if (atBottom) o.scrollTop = o.scrollHeight;
      return d;
    },
    echo(line) {
      this.print('<span class="sh-prompt">aegis@icu-wing:' + esc(this.cwd) + "$</span> " + esc(line), "echo");
    },
    clear() { const o = this.outEl; if (o) o.innerHTML = ""; },
    esc: esc,

    /* ---- dispatch -------------------------------------------------------- */
    run(raw) {
      const line = String(raw || "").trim();
      if (!line) return;
      this.echo(line);
      this.hist.push(line); this.hi = this.hist.length; this._saveHist();
      const sp = line.indexOf(" ");
      const verb = (sp < 0 ? line : line.slice(0, sp)).toLowerCase();
      const arg = sp < 0 ? "" : line.slice(sp + 1).trim();
      const name = this.cmds[verb] ? verb : this.aliases[verb];
      const h = name && this.cmds[name];
      if (!h) {
        this.print("unknown command <b>" + esc(verb) + "</b> — try <b>help</b>", "err");
        return;
      }
      try {
        const r = h.fn(arg, this);
        // A verb may return a promise; a rejection must surface in the terminal
        // rather than only in the browser console, where no operator is looking.
        if (r && typeof r.catch === "function") {
          r.catch((e) => this.print("error: " + esc(e && e.message ? e.message : e), "err"));
        }
      } catch (e) {
        this.print("error: " + esc(e && e.message ? e.message : e), "err");
      }
    },

    /* ---- history --------------------------------------------------------- */
    _loadHist() {
      try { this.hist = JSON.parse(localStorage.getItem(this.histKey) || "[]") || []; }
      catch (e) { this.hist = []; }
      this.hi = this.hist.length;
    },
    _saveHist() {
      try { localStorage.setItem(this.histKey, JSON.stringify(this.hist.slice(-200))); } catch (e) {}
    },
    histPrev() { if (!this.hist.length) return null; this.hi = Math.max(0, this.hi - 1); return this.hist[this.hi]; },
    histNext() {
      if (!this.hist.length) return null;
      this.hi = Math.min(this.hist.length, this.hi + 1);
      return this.hi >= this.hist.length ? "" : this.hist[this.hi];
    },

    /* ---- completion ------------------------------------------------------ */
    candidates(value) {
      const v = String(value || "");
      const word = v.slice(v.lastIndexOf(" ") + 1);
      // First word completes to verbs; later words go to whatever the page taught
      // this shell about its own nouns (beds, /proc paths, window names).
      let pool = v.indexOf(" ") < 0
        ? Object.keys(this.cmds).concat(Object.keys(this.aliases))
        : [];
      this._completers.forEach((fn) => {
        try { pool = pool.concat(fn(word, v) || []); } catch (e) {}
      });
      return [...new Set(pool)].filter((c) => c.startsWith(word)).sort();
    },
    complete() {
      const inp = this.inEl; if (!inp) return;
      const cands = this.candidates(inp.value);
      if (!cands.length) return;
      const head = inp.value.slice(0, inp.value.lastIndexOf(" ") + 1);
      if (cands.length === 1) { inp.value = head + cands[0] + " "; }
      else {
        // Extend to the longest common prefix, then show the options — the
        // behaviour every shell has, and the reason Tab feels fast.
        let pre = cands[0];
        cands.forEach((c) => { while (!c.startsWith(pre)) pre = pre.slice(0, -1); });
        if (pre.length > (inp.value.length - head.length)) inp.value = head + pre;
        this.print(cands.map((c) => '<span class="sh-cand">' + esc(c) + "</span>").join("  "), "dim");
      }
      this._render();
    },

    /* ---- visibility ------------------------------------------------------ */
    show() {
      this.open = true;
      const r = this.root; if (r) { r.classList.add("open"); r.setAttribute("aria-hidden", "false"); }
      if (!this.outEl || !this.outEl.childNodes.length) this.banner();
      setTimeout(() => { const i = this.inEl; if (i) { i.focus(); this._render(); } }, 20);
      if (this.o.onShow) this.o.onShow(this);
    },
    hide() {
      this.open = false;
      const r = this.root; if (r) { r.classList.remove("open"); r.setAttribute("aria-hidden", "true"); }
      if (this.o.onHide) this.o.onHide(this);
    },
    toggle() { this.open ? this.hide() : this.show(); },
    banner() {
      this.print("<b>AegisOS</b> shell — a direct line to the kernel: process table, scheduler, memory fabric, audit.", "sys");
      this.print("<b>help</b> lists commands · <b>Tab</b> completes · <b>Ctrl-L</b> clears · <b>↑/↓</b> history · <b>ls /proc</b> to browse kernel state.", "dim");
    },
    setCwd(p) { this.cwd = p || "~"; this._prompt(); },
    _prompt() {
      const p = this._el(this.o.prompt);
      if (p) p.innerHTML = "aegis@icu-wing<b>:" + esc(this.cwd) + "$</b>";
    },
    _render() { if (this.o.onRender) this.o.onRender(this); },

    /* ---- key bindings ---------------------------------------------------- */
    /** Bind the input plus a global opener. `openKey` is checked against the
     *  event; the caller decides the chord so pages can differ. */
    mount(openMatch) {
      const self = this;
      const inp = this.inEl;
      this._prompt();
      if (inp) {
        inp.addEventListener("keydown", function (e) {
          if (e.key === "Enter") { e.preventDefault(); const v = inp.value; inp.value = ""; self.run(v); self._render(); return; }
          if (e.key === "Tab") { e.preventDefault(); self.complete(); return; }
          if (e.key === "ArrowUp") { e.preventDefault(); const h = self.histPrev(); if (h != null) inp.value = h; self._render(); return; }
          if (e.key === "ArrowDown") { e.preventDefault(); const h = self.histNext(); if (h != null) inp.value = h; self._render(); return; }
          if (e.key === "l" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); self.clear(); self.banner(); return; }
          if (e.key === "Escape") { e.preventDefault(); self.hide(); return; }
          setTimeout(() => self._render(), 0);
        });
        inp.addEventListener("input", () => self._render());
      }
      document.addEventListener("keydown", function (e) {
        const t = e.target;
        const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
        if (openMatch && openMatch(e) && !typing) { e.preventDefault(); self.toggle(); return; }
        if (e.key === "Escape" && self.open) { self.hide(); }
      });
      return this;
    },
  };

  global.Shell = {
    create(opts) { return new Shell(opts); },
    esc: esc,
  };
})(window);
