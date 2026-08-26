/* ============================================================================
   oskit/sysmon.js — system monitor
   ----------------------------------------------------------------------------
   Makes the kernel legible.

   AegisOS has had a process table, a scheduler that allocates compute lanes per
   bed, a bump-allocated memory arena and a hash-chained syscall audit for a long
   time — all of it registered under /proc and all of it invisible from the console
   where the system actually operates. Everything here is read-only and comes from
   endpoints that already exist; nothing is computed in the browser.

   The arena pane earns its place now in particular: the arena used to be a fixed
   1Mi cells regardless of ward size, which capped the OS at 170 beds and surfaced
   as a MemoryError from a bump allocator at boot. It is now sized to the ward, so
   occupancy is a number that means something.
   ============================================================================ */
(function (global) {
  "use strict";

  const esc = (s) => Shell.esc(s);
  const API = () => (global.AEGIS_BACKEND || "");

  /** Read one /proc node.
   *  The endpoint answers `{path, value}` — an envelope, not the node. Unwrapping
   *  here means every pane receives the shape the kernel actually publishes. */
  async function readProc(path) {
    const r = await fetch(API() + "/aegis/proc/" + path.replace(/^\/?proc\/?/, ""));
    if (!r.ok) throw new Error(path + " → HTTP " + r.status);
    const d = await r.json();
    return (d && Object.prototype.hasOwnProperty.call(d, "value")) ? d.value : d;
  }

  function bar(frac, cls) {
    const p = Math.max(0, Math.min(1, frac || 0));
    return '<span class="sm-bar"><i class="' + (cls || "") + '" style="width:' + (p * 100).toFixed(1) + '%"></i></span>';
  }

  function table(rows, cols) {
    if (!rows || !rows.length) return '<div class="sm-empty">nothing reported</div>';
    return '<table class="sm-tbl"><thead><tr>' +
      cols.map((c) => "<th>" + esc(c.label) + "</th>").join("") +
      "</tr></thead><tbody>" +
      rows.map((r) => "<tr>" + cols.map((c) => '<td class="' + (c.cls || "") + '">' + (c.get(r) ?? "—") + "</td>").join("") + "</tr>").join("") +
      "</tbody></table>";
  }

  const PANES = {
    /* Beds are processes. That is not a metaphor the UI invents — the kernel
       genuinely schedules them, with a priority and a state. */
    processes: {
      label: "Processes",
      path: "kthreads",
      render(d) {
        const rows = Array.isArray(d) ? d : (d && d.ps) || [];
        return table(rows, [
          { label: "PID", get: (r) => esc(r.pid ?? r.id ?? "—") },
          { label: "Task", get: (r) => esc(r.name ?? r.task ?? r.bed ?? "—") },
          { label: "State", get: (r) => '<span class="sm-state sm-' + esc(String(r.state || "?").toLowerCase()) + '">' + esc(r.state ?? "—") + "</span>" },
          { label: "Prio", cls: "num", get: (r) => esc(r.prio ?? r.priority ?? "—") },
          { label: "CPU µs", cls: "num", get: (r) => (r.cpu_us != null ? Number(r.cpu_us).toFixed(0) : "—") },
          { label: "Ctx", cls: "num", get: (r) => esc(r.ctx ?? "—") },
        ]);
      },
    },
    scheduler: {
      label: "Scheduler",
      path: "ksched",
      render(d) {
        if (!d || typeof d !== "object") return '<div class="sm-empty">scheduler idle</div>';
        const entries = Object.entries(d).filter(([, v]) => typeof v !== "object");
        // cpu_shares IS the scheduler's lane allocation — the number this OS
        // computes every tick and has never shown anyone.
        const shares = Array.isArray(d.cpu_shares) ? d.cpu_shares : null;
        const lanes = shares
          ? Object.fromEntries(shares.map((c) => [(c.name || ("pid " + c.pid)), c.pct]))
          : (d.lanes || d.allocation || null);
        let html = '<div class="sm-kv">' + entries.map(([k, v]) =>
          '<div><span>' + esc(k.replace(/_/g, " ")) + "</span><b>" + esc(typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v) + "</b></div>"
        ).join("") + "</div>";
        if (lanes && typeof lanes === "object") {
          const tot = Object.values(lanes).reduce((a, b) => a + (Number(b) || 0), 0) || 1;
          html += '<div class="sm-lanes">' + Object.entries(lanes).map(([bed, n]) =>
            '<div class="sm-lane"><span>' + esc(bed) + "</span>" + bar(Number(n) / tot) +
            '<b class="num">' + esc(n) + "</b></div>"
          ).join("") + "</div>";
        }
        return html;
      },
    },
    memory: {
      label: "Memory",
      path: "mem",
      render(d) {
        if (!d) return '<div class="sm-empty">no arena</div>';
        const used = d.used_cells ?? d.used ?? 0, cap = d.capacity_cells ?? d.capacity ?? 0;
        const frac = d.utilization ?? (cap ? used / cap : 0);
        return '<div class="sm-arena">' +
          '<div class="sm-arena-head"><span>biometric arena</span>' +
          '<b class="num">' + used.toLocaleString() + " / " + cap.toLocaleString() + " cells</b></div>" +
          bar(frac, frac > 0.9 ? "hot" : "") +
          '<div class="sm-note">' + ((d.bytes ?? cap * 8) / 1048576).toFixed(1) + " MiB reserved · " +
          (frac * 100).toFixed(1) + "% occupied · sized to the ward at boot, never grown</div>" +
          (d.pointer_stable !== undefined
            ? '<div class="sm-note">base ' + esc(d.base_pointer || "—") + " · " +
              (d.pointer_stable ? "pointer stable since boot" : "POINTER MOVED") + "</div>"
            : "") +
          "</div>";
      },
    },
    audit: {
      label: "Audit",
      path: "audit",
      render(d) {
        const rows = (d && (d.recent || d.entries || d.tail)) || (Array.isArray(d) ? d : []);
        if (!rows.length) {
          return '<div class="sm-kv">' + Object.entries(d || {}).filter(([, v]) => typeof v !== "object")
            .map(([k, v]) => "<div><span>" + esc(k.replace(/_/g, " ")) + "</span><b>" + esc(v) + "</b></div>").join("") + "</div>";
        }
        return table(rows.slice(-40).reverse(), [
          { label: "Seq", cls: "num", get: (r) => esc(r.seq ?? "—") },
          { label: "Call", get: (r) => esc(r.call ?? r.op ?? "—") },
          // A refusal is the interesting row, so it gets the colour. Colour is
          // signal here exactly as it is for clinical state elsewhere.
          { label: "Verdict", get: (r) => {
              const ok = r.allowed ?? r.ok ?? (r.verdict === "allow");
              return '<span class="sm-verdict ' + (ok ? "ok" : "deny") + '">' + (ok ? "allowed" : "REFUSED") + "</span>";
            } },
          { label: "Reason", get: (r) => esc(r.reason ?? r.detail ?? "") },
        ]);
      },
    },
  };

  function Sysmon(rootSel) {
    this.root = document.querySelector(rootSel);
    this.pane = "processes";
    this.timer = null;
  }

  Sysmon.prototype = {
    mount() {
      if (!this.root) return this;
      this.root.innerHTML =
        '<div class="sm-tabs">' + Object.entries(PANES).map(([k, p]) =>
          '<button class="sm-tab' + (k === this.pane ? " on" : "") + '" data-pane="' + k + '">' + esc(p.label) + "</button>"
        ).join("") + '<span class="sm-src">read-only · /proc</span></div>' +
        '<div class="sm-body" id="sm-body"><div class="sm-empty">loading…</div></div>';
      this.root.addEventListener("click", (e) => {
        const b = e.target.closest("[data-pane]");
        if (!b) return;
        this.pane = b.dataset.pane;
        this.root.querySelectorAll(".sm-tab").forEach((t) => t.classList.toggle("on", t.dataset.pane === this.pane));
        this.refresh();
      });
      return this;
    },
    async refresh() {
      const body = this.root && this.root.querySelector("#sm-body");
      if (!body) return;
      const p = PANES[this.pane];
      try {
        const d = await readProc(p.path);
        body.innerHTML = p.render(d);
      } catch (e) {
        // An unreachable kernel is a real state and says so, rather than leaving
        // the last good frame on screen pretending to be live.
        body.innerHTML = '<div class="sm-empty sm-err">/proc/' + esc(p.path) + " unavailable — " + esc(e.message) + "</div>";
      }
    },
    start(ms) {
      this.refresh();
      clearInterval(this.timer);
      this.timer = setInterval(() => {
        if (!document.hidden) this.refresh();
      }, ms || 3000);
      return this;
    },
    stop() { clearInterval(this.timer); this.timer = null; },
  };

  global.Sysmon = { create: (sel) => new Sysmon(sel), readProc, PANES };
})(window);
