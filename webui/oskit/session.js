/* ============================================================================
   oskit/session.js — who you are, and what that lets you do
   ----------------------------------------------------------------------------
   AegisOS has had RBAC with five roles across four privilege rings, a deny-by-
   default capability model and a Z3 proof that the policy is sound — and the
   console never showed which role you were holding. An operator could not tell
   whether an action was unavailable because the system was busy, because the data
   was missing, or because they lacked the ring for it.

   This surfaces identity and, more importantly, makes a refusal legible: when a
   capability is denied, the reason names the capability and the ring it needs. A
   lock with a label is a different object from a door that will not open.

   Read-only against /aegis/rbac/policy; login is a POST the operator initiates.
   ============================================================================ */
(function (global) {
  "use strict";

  const esc = (s) => Shell.esc(s);
  const API = () => (global.AEGIS_BACKEND || "");
  const LS = "aegis_session_token";

  function Session(mountSel) {
    this.el = document.querySelector(mountSel);
    this.policy = null;
    this.token = null;
    this.role = null;
    this.ring = null;
    try { this.token = localStorage.getItem(LS); } catch (e) {}
  }

  Session.prototype = {
    async load() {
      try {
        const r = await fetch(API() + "/aegis/rbac/policy");
        if (r.ok) this.policy = await r.json();
      } catch (e) { this.policy = null; }
      this.render();
      return this;
    },

    /** Capabilities the current role holds, or [] when unauthenticated. */
    caps() {
      if (!this.policy || !this.role) return [];
      return (this.policy.grants && this.policy.grants[this.role]) || [];
    },
    ringOf(role) {
      return (this.policy && this.policy.rings && this.policy.rings[role]) ?? null;
    },
    /** Would `cap` be permitted? Returns {ok, reason} — the reason is the point.
     *
     *  This mirrors the kernel's actual rule, which is grant-based and NOT a ring
     *  comparison: `rbac.authorize` tests membership in the role's explicit grant
     *  set. `cap_min_ring` is a soundness invariant — the Z3 proof checks that no
     *  role below a capability's minimum ring holds it — so it constrains who MAY
     *  be granted something, and never by itself decides a call.
     *
     *  Getting this wrong produces a refusal that reads as nonsense: an earlier
     *  version said "annotate needs ring 1; chief holds ring 4", implying a chief
     *  was under-privileged for a nurse's capability. The truth is duller and more
     *  useful — a chief simply is not granted it, because chiefs do not chart.
     */
    may(cap) {
      if (!this.policy) return { ok: false, reason: "policy unavailable — the console cannot verify permissions" };
      if (!this.role) return { ok: false, reason: "no session — sign in to hold a privilege ring" };
      if (this.caps().indexOf(cap) >= 0) return { ok: true, reason: "" };
      const known = this.policy.cap_min_ring && (cap in this.policy.cap_min_ring);
      if (!known) return { ok: false, reason: "unknown capability " + cap };
      const holders = Object.entries(this.policy.grants || {})
        .filter(([, cs]) => (cs || []).indexOf(cap) >= 0).map(([r]) => r);
      return {
        ok: false,
        reason: this.role + " is not granted " + cap +
          " — grants are explicit, not inherited by ring" +
          (holders.length ? " (held by " + holders.join(", ") + ")" : " (no role holds it)"),
      };
    },

    async login(role) {
      const r = await fetch(API() + "/aegis/rbac/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || d.error) throw new Error(d.error || ("HTTP " + r.status));
      this.token = d.token || d.session || null;
      this.role = role;
      this.ring = this.ringOf(role);
      try { localStorage.setItem(LS, this.token || ""); } catch (e) {}
      this.render();
      return d;
    },
    logout() {
      this.token = null; this.role = null; this.ring = null;
      try { localStorage.removeItem(LS); } catch (e) {}
      this.render();
    },

    render() {
      if (!this.el) return;
      const roles = this.policy ? Object.keys(this.policy.rings || {}) : [];
      const unknown = !this.role;
      this.el.innerHTML =
        '<button class="ses-trigger" id="ses-trigger" aria-haspopup="listbox" aria-expanded="false" ' +
        'title="Session role and privilege ring">' +
          '<span class="ses-ring' + (unknown ? " none" : "") + '">' + (unknown ? "—" : "R" + this.ring) + "</span>" +
          '<span class="ses-role">' + esc(unknown ? "no session" : this.role) + "</span>" +
        "</button>" +
        '<div class="ses-menu hidden" id="ses-menu" role="listbox">' +
          '<div class="ses-head">privilege ring</div>' +
          roles.map((r) =>
            '<button class="ses-opt' + (r === this.role ? " on" : "") + '" data-role="' + esc(r) + '">' +
            '<span class="ses-ring">R' + esc(this.ringOf(r)) + "</span>" + esc(r) +
            '<span class="ses-caps">' + ((this.policy.grants[r] || []).length) + " caps</span></button>"
          ).join("") +
          (this.role ? '<button class="ses-opt ses-out" data-role="">sign out</button>' : "") +
          (this.policy && this.policy.soundness
            ? '<div class="ses-foot">policy proven sound · no grant exceeds its ring</div>' : "") +
        "</div>";
      const trig = this.el.querySelector("#ses-trigger");
      const menu = this.el.querySelector("#ses-menu");
      trig.addEventListener("click", (e) => {
        e.stopPropagation();
        menu.classList.toggle("hidden");
        trig.setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
      });
      menu.addEventListener("click", async (e) => {
        const b = e.target.closest("[data-role]");
        if (!b) return;
        menu.classList.add("hidden");
        const role = b.dataset.role;
        if (!role) return this.logout();
        try { await this.login(role); }
        catch (err) { console.warn("rbac login failed", err); }
      });
      document.addEventListener("click", () => menu.classList.add("hidden"));
    },
  };

  global.Session = { create: (sel) => new Session(sel) };
})(window);
