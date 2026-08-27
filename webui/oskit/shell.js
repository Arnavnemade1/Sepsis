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
    this.status = "idle";        // last command's outcome: idle | busy | ok | err
    this.ran = 0;                // commands accepted this session
    this.bootedAt = Date.now();
    this._geoKey = (this.o.geoKey || this.histKey) + "_geo";
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
    clear() {
      const o = this.outEl; if (o) o.innerHTML = "";
      // A stale chevron over an empty screen claims an outcome for output nobody
      // can see any more.
      this.setStatus("idle");
    },
    esc: esc,

    /* ---- dispatch -------------------------------------------------------- */
    /** Accept a line. Commands are SERIALISED: while one is in flight the next is
     *  queued rather than run alongside it.
     *
     *  Not fussiness — an async verb whose fetch resolves after a later synchronous
     *  verb has already printed drops its output into the middle of someone else's,
     *  and the scrollback stops being a record of what happened in what order. Every
     *  real terminal blocks here; so does this one. `Ctrl-C` drops the queue. */
    run(raw) {
      const line = String(raw || "").trim();
      if (!line) return;
      if (this._busy) { (this._queue = this._queue || []).push(line); return; }
      this._exec(line);
    },
    _drain() {
      this._busy = false;
      const q = this._queue;
      if (q && q.length) this._exec(q.shift());
    },
    /** Abandon everything queued. The in-flight command is not killed — it holds no
     *  cancel handle — but it no longer gates what follows. */
    interrupt() {
      const n = (this._queue || []).length;
      this._queue = [];
      this._busy = false;
      this.print('<span class="sh-dim">^C</span>' +
        (n ? ' <span class="sh-dim">— dropped ' + n + " queued</span>" : ""));
      this.setStatus("idle");
    },
    _exec(raw) {
      const line = String(raw || "").trim();
      if (!line) return;
      this.echo(line);
      this.ran += 1;
      this.hist.push(line); this.hi = this.hist.length; this._saveHist();
      const sp = line.indexOf(" ");
      const verb = (sp < 0 ? line : line.slice(0, sp)).toLowerCase();
      const arg = sp < 0 ? "" : line.slice(sp + 1).trim();
      const name = this.cmds[verb] ? verb : this.aliases[verb];
      const h = name && this.cmds[name];
      if (!h) {
        this.print("unknown command <b>" + esc(verb) + "</b> — try <b>help</b>", "err");
        this.setStatus("err");
        return this._drain();
      }
      try {
        const r = h.fn(arg, this);
        // A verb may return a promise; a rejection must surface in the terminal
        // rather than only in the browser console, where no operator is looking.
        // While it is pending the chevron says so — an async verb against a cold
        // backend used to look exactly like a verb that printed nothing.
        if (r && typeof r.then === "function") {
          this._busy = true;
          this.setStatus("busy");
          r.then(() => this.setStatus("ok"),
                 (e) => { this.print("error: " + esc(e && e.message ? e.message : e), "err");
                          this.setStatus("err"); })
           .then(() => this._drain());
        } else {
          this.setStatus("ok");
          this._drain();
        }
      } catch (e) {
        this.print("error: " + esc(e && e.message ? e.message : e), "err");
        this.setStatus("err");
        this._drain();
      }
    },

    /** Exit status of the last command, shown on the prompt chevron. */
    setStatus(st) {
      this.status = st;
      const el = this._el(this.o.status);
      if (el) { el.dataset.st = st; el.textContent = st === "busy" ? "\u22ef" : "\u276f"; }
      this._bar();
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
      // The uptime field ticks only while the window is on screen; a status bar
      // repainting behind a closed terminal is pure wakeups for nobody.
      clearInterval(this._barTimer);
      this._barTimer = setInterval(() => this._bar(), 1000);
      if (this._markActive) setTimeout(this._markActive, 30);
      if (this.o.onShow) this.o.onShow(this);
    },
    hide() {
      this.open = false;
      clearInterval(this._barTimer); this._barTimer = null;
      const r = this.root; if (r) { r.classList.remove("open"); r.setAttribute("aria-hidden", "true"); }
      const f = this._el(this.o.frame); if (f) f.classList.remove("sh-active");
      if (this.o.onHide) this.o.onHide(this);
    },
    toggle() { this.open ? this.hide() : this.show(); },
    banner() {
      const f = this._barData();
      this.print("<b>AegisOS</b> " + esc(f.kernel || "") +
        " — a direct line to the kernel: process table, scheduler, memory fabric, audit.", "sys");
      this.print('<span class="sh-dim">' + [
        "<b>help</b> commands", "<b>Tab</b> complete", "<b>→</b> accept",
        "<b>↑/↓</b> history", "<b>Ctrl-L</b> clear", "<b>Ctrl-C</b> interrupt",
        "<b>ls /proc</b> browse kernel",
      ].join(" · ") + "</span>");
    },
    setCwd(p) { this.cwd = p || "~"; this._prompt(); },
    _prompt() {
      const p = this._el(this.o.prompt);
      if (p) p.innerHTML = "aegis@icu-wing<b>:" + esc(this.cwd) + "$</b>";
      const t = this._el(this.o.titlePath);
      if (t) t.textContent = ":" + this.cwd;
    },

    /* ---- status bar ------------------------------------------------------ */
    /** Feed the status line. Takes an object, or a FUNCTION returning one — pass a
     *  function for anything that changes while the window is open (open windows,
     *  the role, the kernel version), so the bar re-reads it each tick instead of
     *  displaying whatever happened to be true when the terminal was first shown. */
    bar(fields) {
      if (typeof fields === "function") this._barFn = fields;
      else this._barFields = Object.assign(this._barFields || {}, fields || {});
      this._bar();
      return this;
    },
    /** The resolved status fields. One path, so the banner and the bar can never
     *  disagree about the same fact — they did: the banner printed an empty kernel
     *  version while the bar right beneath it showed the real one. */
    _barData() {
      let f = this._barFields || {};
      if (this._barFn) { try { f = Object.assign({}, f, this._barFn() || {}); } catch (e) {} }
      return f;
    },
    _bar() {
      const f = this._barData();
      const set = (sel, txt) => { const e = this._el(sel); if (e) e.innerHTML = txt; };
      const up = Math.max(0, Math.round((Date.now() - this.bootedAt) / 1000));
      const hhmm = up < 60 ? up + "s"
        : up < 3600 ? Math.floor(up / 60) + "m " + (up % 60) + "s"
        : Math.floor(up / 3600) + "h " + Math.floor((up % 3600) / 60) + "m";
      set(this.o.segKernel, "AEGIS·OS <b>" + esc(f.kernel || "—") + "</b>");
      set(this.o.segHost, "<b>" + esc(f.host || "local") + "</b>");
      set(this.o.segWin, "<b>" + esc(f.windows == null ? "—" : f.windows) + "</b> win");
      set(this.o.segCmds, "<b>" + this.ran + "</b> cmd");
      set(this.o.segUp, "up <b>" + hhmm + "</b>");
      const chip = this._el(this.o.chipRing);
      if (chip) {
        chip.textContent = f.role ? f.role + " · ring " + f.ring : "no session";
        chip.classList.toggle("held", !!f.role);
      }
    },

    /* ---- inline completion ----------------------------------------------- */
    /** The single unambiguous completion, drawn behind the caret in dim text.
     *  Shown only when exactly one candidate matches: a ghost that guesses between
     *  several is worse than no ghost, because Tab would not honour it. */
    _ghost() {
      const g = this._el(this.o.ghost), inp = this.inEl;
      if (!g || !inp) return;
      const v = inp.value;
      if (!v || v.endsWith(" ")) { g.innerHTML = ""; return; }
      const cands = this.candidates(v);
      if (cands.length !== 1) { g.innerHTML = ""; return; }
      const word = v.slice(v.lastIndexOf(" ") + 1);
      const tail = cands[0].slice(word.length);
      // The typed prefix is rendered transparent rather than measured, so the
      // suggested tail lands exactly under the caret in any font at any zoom.
      g.innerHTML = tail ? "<i>" + esc(v) + "</i>" + esc(tail) : "";
    },
    _render() { this._ghost(); this._bar(); if (this.o.onRender) this.o.onRender(this); },

    /* ---- window geometry -------------------------------------------------- */
    /** Drag by the title bar, resize from the grip, both persisted. A terminal you
     *  cannot move is a dialog; the whole point of this window is that the ward
     *  behind it stays visible and usable while you type into it. */
    _geometry() {
      const frame = this._el(this.o.frame);
      if (!frame) return;
      const self = this;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

      try {
        const g = JSON.parse(localStorage.getItem(this._geoKey) || "null");
        if (g && window.innerWidth > 900) self._place(g);
      } catch (e) {}

      function save() {
        try {
          localStorage.setItem(self._geoKey, JSON.stringify({
            x: frame.offsetLeft, y: frame.offsetTop,
            w: frame.offsetWidth, h: frame.offsetHeight,
          }));
        } catch (e) {}
      }

      function drag(startEvt, mode) {
        if (window.innerWidth <= 900) return;          // pinned on narrow screens
        startEvt.preventDefault();
        const r = frame.getBoundingClientRect();
        // Freeze the box in absolute coordinates before the first move, so a drag
        // that starts on a still-centred frame does not jump by half its width.
        self._place({ x: r.left, y: r.top, w: r.width, h: r.height });
        const ox = startEvt.clientX, oy = startEvt.clientY;
        const move = (e) => {
          const dx = e.clientX - ox, dy = e.clientY - oy;
          if (mode === "move") {
            frame.style.left = clamp(r.left + dx, -r.width + 90, window.innerWidth - 90) + "px";
            frame.style.top = clamp(r.top + dy, 0, window.innerHeight - 34) + "px";
          } else {
            frame.style.width = clamp(r.width + dx, 380, window.innerWidth - r.left - 6) + "px";
            frame.style.height = clamp(r.height + dy, 210, window.innerHeight - r.top - 6) + "px";
          }
        };
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          save();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      }

      // Window activation. Tracked on the frame rather than the input, so clicking
      // the scrollback to select text keeps the window active — which is what an OS
      // does and what a bare input-focus test would get wrong.
      const mark = () => frame.classList.toggle("sh-active",
        self.open && frame.contains(document.activeElement));
      frame.addEventListener("focusin", mark);
      frame.addEventListener("focusout", () => setTimeout(mark, 0));
      frame.addEventListener("pointerdown", () => {
        const i = self.inEl;
        // Clicking dead space in the terminal focuses the prompt, the way clicking a
        // terminal window does — but not when the operator is selecting output.
        setTimeout(() => {
          if (i && !frame.contains(document.activeElement) &&
              String(window.getSelection() || "") === "") i.focus();
          mark();
        }, 0);
      });
      this._markActive = mark;

      const head = this._el(this.o.head);
      if (head) head.addEventListener("pointerdown", (e) => {
        if (e.target.closest("button")) return;        // the close box is not a handle
        drag(e, "move");
      });
      const grip = this._el(this.o.grip);
      if (grip) grip.addEventListener("pointerdown", (e) => drag(e, "resize"));
    },
    _place(g) {
      const frame = this._el(this.o.frame);
      if (!frame) return;
      frame.classList.add("sh-placed");
      frame.style.left = Math.round(g.x) + "px";
      frame.style.top = Math.round(g.y) + "px";
      if (g.w) frame.style.width = Math.round(g.w) + "px";
      if (g.h) frame.style.height = Math.round(g.h) + "px";
    },

    /* ---- key bindings ---------------------------------------------------- */
    /** Bind the input plus a global opener. `openKey` is checked against the
     *  event; the caller decides the chord so pages can differ. */
    mount(openMatch) {
      const self = this;
      const inp = this.inEl;
      this._prompt();
      this._geometry();
      this._bar();
      const x = this._el(this.o.close);
      if (x) x.addEventListener("click", () => self.hide());
      if (inp) {
        inp.addEventListener("keydown", function (e) {
          if (e.key === "Enter") { e.preventDefault(); const v = inp.value; inp.value = ""; self.run(v); self._render(); return; }
          if (e.key === "Tab") { e.preventDefault(); self.complete(); return; }
          if (e.key === "ArrowUp") { e.preventDefault(); const h = self.histPrev(); if (h != null) inp.value = h; self._render(); return; }
          if (e.key === "ArrowDown") { e.preventDefault(); const h = self.histNext(); if (h != null) inp.value = h; self._render(); return; }
          if (e.key === "l" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); self.clear(); self.banner(); return; }
          if (e.key === "c" && e.ctrlKey) { e.preventDefault(); inp.value = ""; self.interrupt(); self._render(); return; }
          if (e.key === "Escape") { e.preventDefault(); self.hide(); return; }
          // Right-arrow at end of line accepts the inline suggestion, the way every
          // shell with a ghost does. Only at the end, so it stays an ordinary
          // cursor key everywhere else in the line.
          if (e.key === "ArrowRight" && inp.selectionStart === inp.value.length
              && inp.selectionStart === inp.selectionEnd) {
            const c = self.candidates(inp.value);
            if (c.length === 1 && !inp.value.endsWith(" ")) {
              e.preventDefault();
              inp.value = inp.value.slice(0, inp.value.lastIndexOf(" ") + 1) + c[0] + " ";
              self._render();
              return;
            }
          }
          setTimeout(() => self._render(), 0);
        });
        inp.addEventListener("input", () => self._render());
      }
      document.addEventListener("keydown", function (e) {
        const t = e.target;
        const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
        if (openMatch && openMatch(e) && !typing) { e.preventDefault(); self.toggle(); return; }
        // Escape dismisses the TOPMOST surface. Without this the palette and the
        // terminal both took the same Escape, so dismissing a palette you opened over
        // your terminal also threw the terminal away — and the terminal is a place
        // you were working, not a momentary prompt.
        if (e.key === "Escape" && self.open &&
            !document.querySelector(".pl.open, .ks.open, .cm")) { self.hide(); }
      });
      return this;
    },
  };

  global.Shell = {
    create(opts) { return new Shell(opts); },
    esc: esc,
  };
})(window);
