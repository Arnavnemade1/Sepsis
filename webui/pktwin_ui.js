/* ============================================================
   AEGIS-OS — treatment-projection overlay controller

   Binds the pharmacokinetic twin (pktwin.js, an ES module) to the DOM and the
   backend. Deliberately a classic script like tactical.js, so it can be called
   directly from the console and the dossier without module plumbing.

       window.AegisPK.open(bedId, opts)   fetch + show the projection
       window.AegisPK.close()

   Two rules this file exists to enforce, both about not lying to a clinician:

   1. ADMINISTERED vs HYPOTHETICAL. The backend distinguishes an order that was
      actually given from one the bundle engine merely would certify. That
      distinction is carried into the header as a state chip and into the HUD
      text, because a projection of a drug nobody gave, shown next to a body
      glowing with it, is the one way this view could do harm.

   2. THE NUMBERS COME FROM THE SAME PLACE AS THE PICTURE. The readouts are
      driven by the renderer's own per-frame level callback, not recomputed
      alongside it. There is no second interpolation that could drift out of step
      with what is on screen.
   ============================================================ */
(function () {
  "use strict";

  var open = false, data = null, bed = null, armIdx = 0, dragging = false;
  var mounted = false, lastLevels = null;

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  /* Hours are the model's unit; a clinician reads a clock. Every value carries a
     UNIT — an unlabelled "33:00" is read as thirty-three hours, which is exactly
     the wrong answer for a drug thirty-three minutes into its first infusion.
     Below six minutes the seconds matter (norepinephrine reaches steady state in
     about five), so that range gets m:ss with an explicit "min". */
  function clock(h) {
    if (!isFinite(h)) return "—";
    if (h < 0.1) {
      var m = Math.floor(h * 60), s = Math.floor((h * 60 - m) * 60);
      return m + ":" + (s < 10 ? "0" : "") + s + " min";
    }
    var hh = Math.floor(h), mm = Math.round((h - hh) * 60);
    if (mm === 60) { hh += 1; mm = 0; }
    return hh + "h" + (mm < 10 ? "0" : "") + mm;
  }
  function sig(v, unit) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a === 0) return "0";
    // No scientific notation on a clinical readout. A brain concentration of
    // 4.2e-3 µg/L is read as "essentially none", so say that instead of making
    // someone parse an exponent to reach the same conclusion.
    if (a < 0.001) return "<0.001";
    if (a < 1) return v.toFixed(3);
    if (a < 10) return v.toFixed(2);
    if (a < 1000) return v.toFixed(1);
    return Math.round(v).toLocaleString();
  }

  var ORGAN_LABEL = {
    lung: "Lungs", heart: "Heart", liver: "Liver", kidney: "Kidneys",
    spleen: "Spleen", pancreas: "Pancreas", intestine: "Intestine",
    brain: "Brain / CNS", muscle: "Muscle", skin: "Skin", adipose: "Adipose",
    rest: "Other tissue", arterial: "Arterial blood", venous: "Venous blood",
  };
  // Ordered as the drug arrives, not alphabetically: the list reads as the route.
  var ORGAN_ORDER = ["arterial", "lung", "heart", "kidney", "liver", "spleen",
                     "pancreas", "intestine", "brain", "muscle", "skin",
                     "adipose", "rest", "venous"];

  function drugRGB(cls) {
    var C = (window.AegisPKTwin && window.AegisPKTwin.DRUG_COLOR) || {};
    var hex = C[cls] != null ? C[cls] : 0x5fb3d4;
    return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255].join(",");
  }

  /* ---------------- rendering the panels ---------------- */
  function renderArms() {
    var host = $("pkt-arms"); if (!host) return;
    var arms = (data && data.arms) || [];
    host.innerHTML = arms.map(function (a, i) {
      return '<button class="pkt-arm' + (i === armIdx ? " on" : "") + '" data-i="' + i + '">' +
        esc(a.label.replace(/ \(balanced\)$/, "")) +
        '<i>' + (a.horizon_h < 2 ? Math.round(a.horizon_h * 60) + "m" : a.horizon_h + "h") + '</i></button>';
    }).join("");
    Array.prototype.forEach.call(host.querySelectorAll(".pkt-arm"), function (b) {
      b.addEventListener("click", function () { selectArm(+b.dataset.i); });
    });
  }

  function renderEvents(a) {
    var track = $("pkt-track"); if (!track) return;
    Array.prototype.forEach.call(track.querySelectorAll(".pkt-ev"), function (n) { n.remove(); });
    (a.events || []).forEach(function (ev) {
      var d = document.createElement("div");
      var key = ev.kind === "cmax" || ev.kind === "steady_state" || ev.kind === "infusion_start";
      d.className = "pkt-ev" + (key ? " key" : "");
      d.style.left = (100 * ev.t_h / (a.horizon_h || 1)) + "%";
      d.title = ev.label + " · " + clock(ev.t_h);
      track.appendChild(d);
    });
  }

  function renderMetrics(a) {
    var m = a.metrics || {}, u = a.unit, out = [];
    function tile(label, value, unit) {
      out.push('<div class="pkt-metric"><span>' + esc(label) + '</span><b>' +
        esc(value) + (unit ? '<em>' + esc(unit) + '</em>' : "") + "</b></div>");
    }
    if (a.kind === "volume") {
      tile("infused", sig(m.infused_ml), "mL");
      tile("peak plasma", sig(m.peak_plasma_expansion_ml), "mL");
      tile("still intravascular @1h", Math.round((m.plasma_retention_1h_post || 0) * 100), "%");
      tile("into tissue", sig(m.interstitial_ml_end), "mL");
    } else {
      if (m.continuous) {
        tile("steady state", sig(m.css), u);
        tile("90% of steady state", clock(m.t_to_90pct_steady_state_h || 0), "");
      } else {
        tile("Cmax", sig(m.cmax), u);
        tile("Tmax", clock(m.tmax_h || 0), "");
      }
      tile("AUC", sig(m.auc), u + "·h");
      tile("clearance", sig(m.clearance_l_h), "L/h");
      if (m.pct_ft_above_mic != null)
        tile("%fT>MIC", Math.round(m.pct_ft_above_mic), "% @ " + m.mic_mg_l + " mg/L");
      tile("dose given", sig(m.dose_total_mg), "mg");
    }
    $("pkt-metrics").innerHTML = out.join("");
    $("pkt-exp-meta").textContent = a.kind === "volume" ? "volume kinetics" : "over " + clock(a.horizon_h);
  }

  function renderProvenance(a) {
    var p = a.provenance || {};
    var bits = [];
    if (p.drug) bits.push("<b>Drug PK:</b> " + esc(p.drug));
    if (p.physiology) bits.push("<b>Physiology:</b> " + esc(p.physiology));
    if (p.target) bits.push("<b>Target:</b> " + esc(p.target));
    if (a.honesty) bits.push(esc(a.honesty));
    $("pkt-prov").innerHTML = bits.join("<br><br>");
  }

  /* The per-organ readout. Built once per arm; only the numbers and bars are
     touched per frame, so a 60fps scrub does not rebuild the DOM. */
  var organRows = {};
  function buildOrganRows(a) {
    organRows = {};
    var rows = ORGAN_ORDER.filter(function (k) { return a.organs && a.organs[k]; });
    $("pkt-organs").innerHTML = rows.map(function (k) {
      return '<div class="pkt-organ" data-k="' + k + '">' +
        '<div class="pkt-organ-n">' + esc(ORGAN_LABEL[k] || k) + "</div>" +
        '<div class="pkt-organ-v" data-v="' + k + '">—</div>' +
        '<div class="pkt-organ-bar"><i data-b="' + k + '"></i></div></div>';
    }).join("");
    rows.forEach(function (k) {
      organRows[k] = { row: $("pkt-organs").querySelector('[data-k="' + k + '"]'),
                       v: $("pkt-organs").querySelector('[data-v="' + k + '"]'),
                       b: $("pkt-organs").querySelector('[data-b="' + k + '"]') };
    });
    $("pkt-unit").textContent = a.kind === "volume" ? "mL" : a.unit;
    $("pkt-scale-hi").textContent = a.kind === "volume" ? "most loaded" : "peak organ";
  }

  /* Called by the renderer every frame with the levels IT is drawing. */
  function onFrame(t, levels, a) {
    lastLevels = levels;
    // The on-body labels format with the SAME function as the side rail, and the
    // renderer supplies the value it is currently drawing — so a number on the
    // anatomy and the same number in the list cannot drift apart.
    var unit = (a.kind === "volume" ? "mL" : a.unit);
    window.AegisPKTwin.setLabelFormat(function (v) { return sig(v) + " " + unit; });
    var horizon = a.horizon_h || 1;
    $("pkt-clock").textContent = clock(t);
    $("pkt-time").innerHTML = "<b>" + clock(t) + "</b> / " + clock(horizon);
    if (!dragging) {
      var pct = 100 * t / horizon;
      $("pkt-track-f").style.width = pct + "%";
      $("pkt-track-h").style.left = pct + "%";
    }
    for (var k in organRows) {
      var rec = levels[k], r = organRows[k];
      if (!rec || !r) continue;
      r.v.textContent = sig(rec.value);
      r.b.style.width = (100 * rec.level) + "%";
      r.row.classList.toggle("off", rec.level < 0.005);
    }
  }

  function selectArm(i) {
    armIdx = i;
    var a = data.arms[i]; if (!a) return;
    document.getElementById("pkt-frame").style.setProperty("--pkt-drug", drugRGB(a.drug_class));
    renderArms(); buildOrganRows(a); renderMetrics(a); renderProvenance(a); renderEvents(a);
    $("pkt-hud-t").textContent = a.label + " · " + a.regimen;
    $("pkt-hud-s").innerHTML = (data.administered
        ? "Projected distribution of the order now running."
        : "Projected distribution <b>if this order were given</b> — nothing is running.")
      + "<br>Brightness is drug concentration, not severity.";
    window.AegisPKTwin.setArm(i);
    window.AegisPKTwin.setTime(0);
    setPlaying(true);
  }

  function setPlaying(on) {
    var b = $("pkt-play");
    if (on) { window.AegisPKTwin.play(); b.textContent = "❚❚ pause"; b.classList.add("on"); }
    else { window.AegisPKTwin.pause(); b.textContent = "▶ play"; b.classList.remove("on"); }
  }

  /* ---------------- open / close ---------------- */
  function show() {
    var el = $("pkt"); el.classList.add("open"); el.setAttribute("aria-hidden", "false");
    open = true;
    // Park the clinical twin: two WebGL contexts each drawing ~20 MB of anatomy is
    // a real cost, and only one of them is visible.
    try { window.AegisTwin && window.AegisTwin.setParked && window.AegisTwin.setParked(true); } catch (e) {}
  }

  function close() {
    var el = $("pkt"); if (!el) return;
    el.classList.remove("open"); el.setAttribute("aria-hidden", "true");
    open = false;
    try { window.AegisPKTwin && window.AegisPKTwin.pause(); } catch (e) {}
    try { window.AegisTwin && window.AegisTwin.setParked && window.AegisTwin.setParked(false); } catch (e) {}
  }

  function fail(msg, detail) {
    $("pkt-load").style.display = "";
    $("pkt-load").innerHTML = esc(msg) + "<i>" + esc(detail || "") + "</i>";
  }

  function openBed(bedId, opts) {
    bed = bedId; opts = opts || {};
    show();
    $("pkt-load").style.display = "";
    $("pkt-load").innerHTML = "projecting…<i>solving the compartment model and assembling the anatomy</i>";
    $("pkt-name").textContent = opts.name || bedId;
    $("pkt-sub").textContent = "—";

    fetch(window.AEGIS.http("/aegis/pk/bed/" + encodeURIComponent(bedId)))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!open) return;
        if (d.status !== "ok") return fail("could not project", d.detail || "");
        if (!d.arms || !d.arms.length)
          return fail("nothing to project", "no drug is ordered for this bed — the "
            + "certified protocol has no antibiotic, vasopressor or fluid arm.");
        data = d;
        $("pkt-name").textContent = d.bed_name || bedId;
        var p = d.patient || {};
        $("pkt-sub").innerHTML = "<b>eGFR</b> " + Math.round(p.egfr) + " · <b>MAP</b> "
          + Math.round(p.map_mmhg) + " mmHg · <b>" + Math.round(p.weight_kg) + "</b> kg"
          + (p.shock_index > 0 ? " · shock redistribution active" : "");
        var st = $("pkt-state");
        st.textContent = d.administered ? "administered" : "not ordered";
        st.className = "pkt-state " + (d.administered ? "live" : "hypo");

        var un = (window.AegisPKTwin.UNMODELLED || {});
        var names = Object.keys(un).map(function (k) { return un[k]; });
        $("pkt-nodata").textContent = names.length
          ? "inert — no compartment in the model: " + names.join(", ") : "";

        mountTwin();
        window.AegisPKTwin.setProjection(d, { onFrame: onFrame });
        armIdx = 0; selectArm(0);
        $("pkt-load").style.display = "none";
      })
      .catch(function (e) { fail("projection unavailable", String(e && e.message || e)); });
  }

  function mountTwin() {
    var stage = $("pkt-stage");
    if (!mounted) {
      // The stage has just become visible; its box is only measurable after a
      // frame. Mounting on a 0x0 element is the classic way to get a twin that
      // renders into nothing (see the same trap in the clinical twin).
      requestAnimationFrame(function () {
        window.AegisPKTwin.mount(stage);
        requestAnimationFrame(function () { window.AegisPKTwin.resize(); });
      });
      mounted = true;
    } else {
      requestAnimationFrame(function () { window.AegisPKTwin.resize(); });
    }
  }

  /* ---------------- transport wiring ---------------- */
  function trackSeek(e) {
    var a = data && data.arms[armIdx]; if (!a) return;
    var r = $("pkt-track").getBoundingClientRect();
    var f = Math.max(0, Math.min(1, (e.clientX - r.left) / (r.width || 1)));
    $("pkt-track-f").style.width = (f * 100) + "%";
    $("pkt-track-h").style.left = (f * 100) + "%";
    window.AegisPKTwin.setTime(f * a.horizon_h);
  }

  function bind() {
    if (!$("pkt")) return;
    $("pkt-x").addEventListener("click", close);
    $("pkt").addEventListener("mousedown", function (e) { if (e.target === $("pkt")) close(); });
    document.addEventListener("keydown", function (e) {
      if (!open) return;
      if (e.key === "Escape") { close(); return; }
      if (e.key === " ") { e.preventDefault(); setPlaying(!window.AegisPKTwin.isPlaying()); return; }
      var a = data && data.arms[armIdx]; if (!a) return;
      // Arrow keys step by 1% of the arm's own horizon — the same gesture reads
      // 1.2 minutes on a vasopressor and 14 minutes on a vancomycin course.
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        setPlaying(false);
        var st = window.AegisPKTwin.state();
        window.AegisPKTwin.setTime(st.t_h + (e.key === "ArrowRight" ? 1 : -1) * a.horizon_h / 100);
      }
    });
    $("pkt-play").addEventListener("click", function () {
      var st = window.AegisPKTwin.state();
      var a = data && data.arms[armIdx];
      // Replay from the top if we are parked at the end.
      if (!st.playing && a && st.t_h >= a.horizon_h - 1e-6) window.AegisPKTwin.setTime(0);
      setPlaying(!st.playing);
    });
    var track = $("pkt-track");
    track.addEventListener("mousedown", function (e) {
      dragging = true; setPlaying(false); trackSeek(e);
    });
    window.addEventListener("mousemove", function (e) { if (dragging) trackSeek(e); });
    window.addEventListener("mouseup", function () { dragging = false; });
    var lb = $("pkt-labels-btn");
    if (lb) lb.addEventListener("click", function () {
      var on = !lb.classList.contains("on");
      lb.classList.toggle("on", on);
      window.AegisPKTwin.setLabels(on);
    });
    var fb = $("pkt-fit");
    if (fb) fb.addEventListener("click", function () { window.AegisPKTwin.fitToAtlas(); });
    Array.prototype.forEach.call($("pkt-rate").querySelectorAll("button"), function (b) {
      b.addEventListener("click", function () {
        Array.prototype.forEach.call($("pkt-rate").querySelectorAll("button"),
          function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        window.AegisPKTwin.setRate(+b.dataset.r);
      });
    });
    window.addEventListener("resize", function () {
      if (open) window.AegisPKTwin.resize();
    });
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", bind);
  else bind();

  window.AegisPK = { open: openBed, close: close, isOpen: function () { return open; } };
})();
