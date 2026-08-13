/* ============================================================
   AEGIS — SUMMON
   ------------------------------------------------------------
   Call the assistant from anywhere in the OS.

   The consult panel is a place you go. This is the assistant coming
   to you: one key (⌘K / Ctrl-K, or ⌥A) opens a bar over whatever you
   are already looking at, you ask, it answers in place, and it goes
   away. No context switch, no losing the panel you were reading.

   It is the SAME agent as the consult — same dossier, same 51 tools,
   same Z3-gated writes — reached through /aegis/doctor/consult. The
   only thing this adds is WHERE you can reach it from, and one piece
   of context the consult does not have: which panel you currently
   have open. That is what makes "what am I looking at?" answerable.

   When an answer wants a real conversation, "open full consult"
   hands off to the call with the question already asked.
   ============================================================ */
(function () {
  "use strict";

  var S = { open: false, busy: false, history: [], lastAnswer: "" };

  function $(s) { return document.querySelector(s); }
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  /* ---------------- what the user is actually looking at ---------------- */
  /* Without this, "what am I looking at?" is unanswerable and the assistant
     would have to guess — which is the exact failure this project is about. */
  var SURFACES = [
    ["#camera-app", "the optical biosensor (contactless rPPG / perfusion / distress signs)"],
    ["#story-app", "the Clinical Story — the temporal causal graph"],
    ["#consensus-app", "the consensus chamber"],
    ["#orch-app", "the autonomous orchestrator"],
    ["#fed-app", "the federated intelligence panel"],
    ["#rbac-app", "the clinical role/permission shell"],
    ["#dsl-app", "the clinical safety-rule DSL"],
    ["#pharma-app", "the living pharmacopoeia"],
    ["#identity-app", "the physiologic identity guard"],
    ["#doctor-app", "the consult panel"]
  ];

  function visibleSurface() {
    for (var i = 0; i < SURFACES.length; i++) {
      var n = document.querySelector(SURFACES[i][0]);
      if (n && !n.classList.contains("hidden")) return SURFACES[i][1];
    }
    var pk = document.querySelector("#pkt");
    if (pk && pk.getAttribute("aria-hidden") === "false") {
      return "the treatment twin — drug distribution over time";
    }
    return "the main cockpit: the 3D body twin, the sepsis risk core and live telemetry";
  }

  function bed() {
    try { return (window.AegisPatient && window.AegisPatient.bed()) || null; }
    catch (e) { return null; }
  }

  /* ---------------- shell ---------------- */
  function build() {
    if ($("#summon")) return;
    var root = el("div", "summon hidden");
    root.id = "summon";
    root.innerHTML =
      '<div class="summon-card" role="dialog" aria-label="Ask Dr. Aegis">' +
      '  <div class="summon-bar">' +
      '    <span class="summon-mark">◈</span>' +
      '    <input id="summon-input" type="text" autocomplete="off" spellcheck="false"' +
      '           placeholder="ask Dr. Aegis — about this patient, this panel, or the unit" />' +
      '    <span class="summon-hint" id="summon-hint">↵</span>' +
      '  </div>' +
      '  <div class="summon-ctx" id="summon-ctx"></div>' +
      '  <div class="summon-body hidden" id="summon-body">' +
      '    <div class="summon-answer" id="summon-answer"></div>' +
      '    <div class="summon-acts" id="summon-acts"></div>' +
      '  </div>' +
      '  <div class="summon-foot">' +
      '    <button class="summon-btn" id="summon-consult">open full consult</button>' +
      '    <span class="summon-esc">esc to dismiss</span>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(root);
  }

  function open(prefill) {
    build();
    var root = $("#summon");
    root.classList.remove("hidden");
    S.open = true;
    $("#summon-ctx").textContent = "looking at " + visibleSurface()
      + (bed() ? " · scoped to the selected patient" : "");
    var i = $("#summon-input");
    if (prefill) i.value = prefill;
    i.focus();
    i.select();
  }

  function close() {
    var root = $("#summon");
    if (root) root.classList.add("hidden");
    S.open = false;
  }

  function toggle() { S.open ? close() : open(); }

  /* ---------------- ask ---------------- */
  function ask() {
    var input = $("#summon-input");
    var q = (input.value || "").trim();
    if (!q || S.busy) return;
    S.busy = true;
    $("#summon-hint").textContent = "…";
    $("#summon-body").classList.remove("hidden");
    $("#summon-answer").textContent = "";
    $("#summon-answer").classList.add("waiting");
    $("#summon-acts").innerHTML = "";

    // The panel context rides along as part of the question, so the answer can
    // be about what is on screen without the agent having to guess.
    var scoped = "[The clinician is currently looking at " + visibleSurface()
      + ". Answer in two or three spoken sentences.] " + q;

    fetch(window.AEGIS.http("/aegis/doctor/consult"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: scoped, bed: bed(), history: S.history })
    }).then(function (r) { return r.json(); }).then(function (d) {
      S.busy = false;
      $("#summon-hint").textContent = "↵";
      $("#summon-answer").classList.remove("waiting");
      S.lastAnswer = d.answer || "";
      $("#summon-answer").textContent = S.lastAnswer || "(no answer)";
      S.history.push({ role: "user", text: q });
      S.history.push({ role: "model", text: S.lastAnswer });
      S.history = S.history.slice(-10);
      input.value = "";
      // Anything it drove, run against the real cockpit — same handler the
      // consult uses, so summon and call cannot drift apart.
      (d.actions || []).forEach(function (a) {
        if (a.ui && window.AegisDoctor && window.AegisDoctor.runUI) {
          window.AegisDoctor.runUI(a.ui.action, a.ui.args);
        }
        var chip = el("span", "summon-act", a.tool);
        if (!a.ok) chip.classList.add("bad");
        $("#summon-acts").appendChild(chip);
      });
    }).catch(function () {
      S.busy = false;
      $("#summon-hint").textContent = "↵";
      $("#summon-answer").classList.remove("waiting");
      $("#summon-answer").textContent =
        "Could not reach the assistant. On the free tier the backend sleeps after "
        + "15 minutes — the first request wakes it, so try again in a moment.";
    });
  }

  /* ---------------- wiring ---------------- */
  document.addEventListener("keydown", function (e) {
    var k = (e.key || "").toLowerCase();
    // ⌘K / Ctrl-K is the near-universal "command" gesture; ⌥A is a one-hand
    // alternative for when the cursor is already in a field.
    if ((e.metaKey || e.ctrlKey) && k === "k") { e.preventDefault(); toggle(); return; }
    if (e.altKey && k === "a") { e.preventDefault(); toggle(); return; }
    if (!S.open) return;
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key === "Enter" && e.target && e.target.id === "summon-input") {
      e.preventDefault(); ask();
    }
  });

  document.addEventListener("click", function (e) {
    if (e.target.closest("#summon-consult")) {
      var q = ($("#summon-input").value || "").trim();
      close();
      if (window.AegisDoctor) {
        window.AegisDoctor.open();
        if (q) setTimeout(function () {
          var i = document.querySelector("#doc-input");
          if (i) { i.value = q; document.querySelector("#doc-send").click(); }
        }, 700);
      }
      return;
    }
    if (e.target.closest("#launch-summon")) { toggle(); return; }
    // click-away dismiss, but not when the click is inside the card
    if (S.open && !e.target.closest(".summon-card")
        && !e.target.closest("#launch-summon")) close();
  });

  window.AegisSummon = { open: open, close: close, toggle: toggle, _state: S };
})();
