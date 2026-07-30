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
    if (window.AegisPKTwin.hasVessels && window.AegisPKTwin.hasVessels())
      bits.push("<b>Circulation:</b> " + esc(window.AegisPKTwin.VESSEL_SOURCE)
        + ". The route is schematic; the concentration it carries and the transit "
        + "delay along it are from the model.");
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
    tickBeats(t, a);
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
    clearBeats(); beatLast = 0;
    setPlaying(true);
  }

  function showPane(which) {
    $("pkt-pane-dist").classList.toggle("hidden", which !== "dist");
    $("pkt-pane-ae").classList.toggle("hidden", which !== "ae");
    $("pkt-pane-cmp").classList.toggle("hidden", which !== "cmp");
    Array.prototype.forEach.call($("pkt-tabs").querySelectorAll(".pkt-tab"), function (b) {
      b.classList.toggle("on", b.dataset.t === which);
    });
  }

  function setPlaying(on) {
    var b = $("pkt-play");
    if (on) { window.AegisPKTwin.play(); b.textContent = "❚❚ pause"; b.classList.add("on"); }
    else { window.AegisPKTwin.pause(); b.textContent = "▶ play"; b.classList.remove("on"); }
  }

  /* ---------------- narration beats ---------------- */
  /* Cards that surface as the playhead reaches each moment. They STACK: several
     beats are legitimately true at the same instant, and the backend deliberately
     does not shift them apart in time (doing so made a card describe a moment the
     body had already left). Presentation handles the overlap. */
  var narrateOn = true, beatsFired = {}, beatLast = 0;
  var BEAT_DWELL = 7000, BEAT_MAX = 2;

  function clearBeats() {
    beatsFired = {};
    var h = $("pkt-beats"); if (h) h.innerHTML = "";
  }

  function beatsFor(a) {
    var o = options && optKey
      ? (options.options || []).filter(function (x) { return x.key === optKey; })[0]
      : null;
    var n = (o && o.narration) || {};
    return (a && n[a.drug]) || [];
  }

  function showBeat(b) {
    var host = $("pkt-beats"); if (!host || !narrateOn) return;
    var el = document.createElement("div");
    el.className = "pkt-beat " + esc(b.kind || "");
    el.innerHTML = '<div class="pkt-beat-h"><span class="pkt-beat-t">'
      + esc(b.title) + '</span><span class="pkt-beat-at">' + esc(b.at) + "</span></div>"
      + '<div class="pkt-beat-b">' + esc(b.body) + "</div>";
    host.appendChild(el);
    // Oldest first out — the stack is newest-at-bottom via column-reverse.
    while (host.children.length > BEAT_MAX) host.removeChild(host.firstChild);
    setTimeout(function () {
      if (!el.parentNode) return;
      el.classList.add("out");
      setTimeout(function () { el.parentNode && el.parentNode.removeChild(el); }, 320);
    }, BEAT_DWELL);
    // Selecting the organ a beat is about points the eye at what it describes.
    if (b.organ && window.AegisPKTwin.pulseOrgan) window.AegisPKTwin.pulseOrgan(b.organ);
  }

  /* Fire any beat the playhead has just crossed. Scrubbing BACKWARD re-arms them,
     so replaying a window shows its explanations again rather than staying silent
     because they were seen once. */
  function tickBeats(t, a) {
    if (!narrateOn || !a) return;
    var list = beatsFor(a);
    if (t < beatLast - 1e-9) beatsFired = {};        // scrubbed back — re-arm
    for (var i = 0; i < list.length; i++) {
      var b = list[i];
      if (b.t_h <= t && !beatsFired[i]) { beatsFired[i] = 1; showBeat(b); }
    }
    beatLast = t;
  }

  /* ---------------- the comparison ---------------- */
  function renderCompare() {
    var c = options && options.comparison;
    if (!c || !c.rows) return;
    $("pkt-cmp-ref").textContent = c.reference_label || "—";
    $("pkt-cmp-note").textContent = c.honesty || "";

    // The explicit subtraction, selected option first so it reads as "this one
    // versus the reference" rather than as a list to search.
    var order = (c.columns || []).slice().sort(function (x, y) {
      return (x.key === optKey ? -1 : 0) - (y.key === optKey ? -1 : 0);
    });
    $("pkt-cmp-diff").innerHTML = '<div class="pkt-diff">' + order.map(function (col) {
      var d = (c.diff || {})[col.key] || [];
      var isRef = col.key === c.reference;
      return '<div class="pkt-diff-o' + (isRef ? " ref" : "") + '">'
        + '<div class="pkt-diff-n">' + esc(col.label)
        + (isRef ? "<span>REFERENCE</span>"
                 : (col.key === optKey ? "<span>SHOWING</span>" : "")) + "</div>"
        + (isRef ? '<div class="pkt-diff-l"><span class="pkt-diff-i">everything below '
                   + 'is measured against this path</span></div>'
                 : '<div class="pkt-diff-l">' + d.map(function (x) {
                     return '<span class="pkt-diff-i ' + esc(x.dir) + '">'
                       + esc(x.text) + "</span>"; }).join("") + "</div>")
        + "</div>";
    }).join("") + "</div>";

    // …and the full table, so nothing is hidden behind the summary.
    var head = "<thead><tr><th></th>" + (c.columns || []).map(function (col) {
      return '<th class="' + (col.key === c.reference ? "ref" : "")
        + (col.safe ? "" : " ref-bad") + '">' + esc(col.label.split(" ")[0]) + "</th>";
    }).join("") + "</tr></thead>";
    var body = "<tbody>" + c.rows.map(function (r) {
      var vals = (c.columns || []).map(function (col) { return r.cells[col.key]; });
      var nums = vals.map(function (v) { return v && typeof v.v === "number" ? v.v : null; })
                     .filter(function (v) { return v !== null; });
      var lowerBetter = vals.some(function (v) { return v && v.lower_better; });
      var best = nums.length ? (lowerBetter ? Math.min.apply(null, nums)
                                            : Math.max.apply(null, nums)) : null;
      return "<tr><th>" + esc(r.label) + "<em>" + esc(r.note) + "</em></th>"
        + (c.columns || []).map(function (col) {
            var v = r.cells[col.key] || {};
            var cls = [];
            if (col.key === c.reference) cls.push("is-ref");
            if (v.bad) cls.push("bad");
            else if (best !== null && typeof v.v === "number" && v.v === best
                     && nums.length > 1) cls.push("best");
            return '<td class="' + cls.join(" ") + '">' + esc(v.text || "—") + "</td>";
          }).join("") + "</tr>";
    }).join("") + "</tbody>";
    $("pkt-cmp-tbl").innerHTML = head + body;
  }

  /* ---------------- treatment options ---------------- */
  /* The overlay used to answer one question about one order. It now holds a set of
     candidate orders and re-answers everything for whichever one is selected: the
     body, the exposures, the side effects and the reasons all belong to the same
     option, so nothing on screen can describe a path you are not looking at. */
  var options = null, optKey = null;

  function renderOptions() {
    var host = $("pkt-opts-l"); if (!host || !options) return;
    var list = options.options || [];
    $("pkt-opts-n").textContent = list.length + " paths";
    host.innerHTML = list.map(function (o) {
      var rec = o.key === options.recommended;
      var flag = rec ? '<span class="pkt-opt-flag rec">REC</span>'
               : (!o.safe ? '<span class="pkt-opt-flag ref">REFUSED</span>' : "");
      var p = o.p_any_serious || 0;
      var drugs = [];
      if (o.protocol.antibiotic_code) drugs.push(o.antibiotic.split("-")[0]);
      if (o.protocol.vasopressor_dose > 0) drugs.push("pressor");
      if (o.protocol.fluid_rate > 0) drugs.push(Math.round(o.protocol.fluid_rate) + " mL/kg/h");
      return '<button class="pkt-opt' + (o.key === optKey ? " on" : "")
        + (o.safe ? "" : " refused") + '" data-k="' + esc(o.key) + '">'
        + '<span class="pkt-opt-t"><span class="pkt-opt-n">' + esc(o.label)
        + "</span>" + flag + "</span>"
        + '<span class="pkt-opt-s">' + esc(drugs.length ? drugs.join(" · ") : "nothing given") + "</span>"
        + '<span class="pkt-opt-m">'
        + (o.arms.length
            ? "serious risk <b>" + (p * 100).toFixed(0) + "%</b>"
            : "<i>no exposure</i>")
        + "</span></button>";
    }).join("");
    Array.prototype.forEach.call(host.querySelectorAll(".pkt-opt"), function (el) {
      el.addEventListener("click", function () { selectOption(el.dataset.k); });
    });
    $("pkt-opts-f").textContent = options.honesty || "";
  }

  function renderProsCons(o) {
    var host = $("pkt-pc"); if (!host) return;
    function row(pt, cls) {
      return '<div class="pkt-pc-i ' + cls + '"><span>'
        + (cls === "up" ? "+" : cls === "safety" ? "✕" : "−") + "</span><span>"
        + esc(pt.text) + (pt.evidence ? "<em>" + esc(pt.evidence) + "</em>" : "")
        + "</span></div>";
    }
    var up = (o.upsides || []).map(function (p) { return row(p, "up"); });
    var dn = (o.downsides || []).map(function (p) {
      return row(p, p.kind === "safety" ? "safety" : "down"); });
    host.innerHTML = up.concat(dn).join("")
      || '<div class="pkt-pc-none">Nothing crossed a reporting threshold for this '
         + 'option. That is not the same as no risk.</div>';
    host.className = "pkt-pc";
    $("pkt-pc-meta").textContent = (o.upsides || []).length + " up · "
      + (o.downsides || []).length + " down";
  }

  /* Selecting an option swaps in that option's arms wholesale — same renderer,
     different projection — and resets the side-effect selection, because an event
     belongs to the exposure that produced it. */
  function selectOption(key) {
    if (!options) return;
    var o = (options.options || []).filter(function (x) { return x.key === key; })[0];
    if (!o) return;
    optKey = key;
    data = { arms: o.arms, adverse: o.adverse, patient: options.patient,
             administered: options.administered && key === "recommended",
             bed_name: options.bed_name, protocol: o.protocol,
             source: options.administered && key === "recommended"
               ? "administered order" : "projected option" };
    aeSelected = null;
    try { window.AegisPKTwin.clearRiskFocus(); } catch (e) {}
    renderOptions();
    renderProsCons(o);
    renderCompare();
    clearBeats();
    var st = $("pkt-state");
    if (o.safe) {
      st.textContent = data.administered ? "administered"
        : (key === "recommended" ? "certified · not ordered" : "hypothetical");
      st.className = "pkt-state " + (data.administered ? "live" : "hypo");
    } else {
      st.textContent = "refused by cage";
      st.className = "pkt-state hypo";
    }
    if (!o.arms.length) {
      // Nothing is administered under this option, so there is no body to fill.
      // Say so rather than showing an inert twin that looks like a loading failure.
      $("pkt-load").style.display = "";
      $("pkt-load").innerHTML = esc(o.label) + " administers nothing"
        + "<i>There is no exposure to project. "
        + esc((o.downsides[0] || {}).text || "") + "</i>";
      renderAE();
      return;
    }
    $("pkt-load").style.display = "none";
    window.AegisPKTwin.setProjection(data, { onFrame: onFrame });
    armIdx = 0; selectArm(0);
    renderAE();
  }

  /* ---------------- the side-effect explorer ---------------- */
  var SEV_RGB = { "mild": "132,138,147", "moderate": "226,163,60",
                  "severe": "222,131,58", "life-threatening": "229,84,76" };
  var aeThreshold = 0, aeSelected = null;

  function pct(p) { return (p * 100).toFixed(p < 0.01 ? 2 : 1) + "%"; }

  /* An onset can fall well outside the arm being viewed — vancomycin AKI shows up
     around day three, and the antibiotic arm is a 24 h window. Saying "84.7 h" next
     to a scrubber that stops at 24 h invites the reading that the bar just has not
     got there yet, so the copy is explicit about which of the two is true. */
  function armForDrug(drug) {
    var list = (data && data.arms) || [];
    for (var i = 0; i < list.length; i++) if (list[i].drug === drug) return list[i];
    return null;
  }
  function onsetText(e) {
    // Judge the onset against the arm THIS EVENT'S DRUG belongs to, not against
    // whichever arm happens to be on screen. Fluid overload at 17 h is outside the
    // 4 h crystalloid window whether or not you are currently looking at a 12 h
    // antibiotic — and reading it against the wrong arm gets the answer backwards
    // in both directions.
    var a = armForDrug(e.drug);
    var within = a && e.onset_p50_h <= a.horizon_h;
    var t = e.onset_p50_h < 48 ? clock(e.onset_p50_h)
          : (e.onset_p50_h / 24).toFixed(1) + " d";
    return t + (within ? "" : " · after this arm's window");
  }

  function renderAE() {
    var ae = data && data.adverse;
    var host = $("pkt-ae-list"); if (!host) return;
    if (!ae || !ae.events || !ae.events.length) {
      host.innerHTML = '<div class="pkt-ae-empty">No catalogued adverse events for '
        + 'the drugs in this order.</div>';
      $("pkt-ae-n").textContent = "";
      return;
    }
    var s = ae.summary || {};
    $("pkt-ae-any").textContent = pct(s.p_any_serious || 0);
    $("pkt-ae-any-sub").textContent = "across " + (s.n_serious || 0)
      + " serious events · assumes independence, so correlated toxicity would put "
      + "the true joint risk below this";
    $("pkt-ae-n").textContent = "(" + ae.events.length + ")";

    var mix = ae.evidence_mix || {};
    $("pkt-ae-mix").textContent = mix.anchored + " anchored · " + mix.label_rate
      + " label · " + mix.mechanistic + " mechanistic";
    $("pkt-ae-note").innerHTML = esc(mix.note || "") + "<br><br>" + esc(ae.honesty || "");

    var shown = ae.events.filter(function (e) { return e.probability >= aeThreshold; });
    if (!shown.length) {
      host.innerHTML = '<div class="pkt-ae-empty">Nothing above this probability. '
        + 'The most likely event is ' + esc(ae.events[0].label) + ' at '
        + pct(ae.events[0].probability) + '.</div>';
      return;
    }
    host.innerHTML = shown.map(function (e) {
      var rgb = SEV_RGB[e.severity] || SEV_RGB.moderate;
      var open = aeSelected === e.key;
      var lo = Math.min(100, e.ci_low * 100), hi = Math.min(100, e.ci_high * 100);
      var mid = Math.min(100, e.probability * 100);
      var delta = e.probability - e.baseline;
      return '<div class="pkt-ae' + (open ? " on" : "") + '" data-k="' + esc(e.key)
        + '" style="--ae:' + rgb + '">'
        + '<div class="pkt-ae-hd"><i class="pkt-ae-dot"></i>'
        + '<span class="pkt-ae-name">' + esc(e.label) + "</span>"
        + '<span class="pkt-ae-p">' + pct(e.probability) + "</span></div>"
        + '<div class="pkt-ae-bar"><u style="left:' + lo + "%;width:" + Math.max(0.5, hi - lo)
        + '%"></u><i style="left:0;width:' + mid + '%"></i></div>'
        + '<div class="pkt-ae-meta">'
        + '<span>' + esc(e.severity) + "</span>"
        + '<span>◷ ' + esc(onsetText(e)) + "</span>"
        + '<span class="pkt-ae-delta">base ' + pct(e.baseline)
        + (Math.abs(delta) > 0.002
            ? ' → <b>' + (delta > 0 ? "+" : "") + (delta * 100).toFixed(1) + " pts</b>" : "")
        + "</span>"
        + '<span class="pkt-ae-ev ' + esc(e.evidence.toLowerCase()) + '">'
        + esc(e.evidence.replace("_", " ")) + "</span>"
        + "</div>"
        + (open ? aeDetail(e) : "")
        + "</div>";
    }).join("");
    Array.prototype.forEach.call(host.querySelectorAll(".pkt-ae"), function (el) {
      el.addEventListener("click", function () { selectAE(el.dataset.k); });
    });
  }

  function aeDetail(e) {
    var mods = (e.modifiers || []).map(function (m) {
      return "<li>" + esc(m.why) + " — <b>×" + m.factor + "</b></li>";
    }).join("");
    var exposure = e.exposure_metric
      ? "<b>Driven by:</b> " + esc(e.exposure_metric.replace("organ_auc:", "exposure in the "))
        + " — projected " + sig(e.exposure) + " vs " + sig(e.reference)
        + " at the reference dose (" + (e.exposure_ratio || 0).toFixed(2) + "×)<br>"
      : "<b>Not exposure-dependent</b> — an idiosyncratic reaction, so the dose does "
        + "not move it<br>";
    return '<div class="pkt-ae-more">'
      + exposure
      + "<b>95% interval:</b> " + pct(e.ci_low) + " – " + pct(e.ci_high)
      + " · <b>onset window:</b> " + clock(e.onset_p10_h) + " – " + clock(e.onset_p90_h) + "<br>"
      + "<b>Watch for it:</b> " + esc(e.monitoring)
      + (mods ? "<br><b>Raised here by:</b><ul>" + mods + "</ul>" : "")
      + '<span class="cite">' + esc(e.citation) + "</span></div>";
  }

  /* Selecting an event puts it on the BODY: the organ it lands in is ringed in its
     severity colour, and the transport jumps to the median onset when that moment
     is actually inside the arm being viewed. */
  function selectAE(key) {
    var ae = data && data.adverse; if (!ae) return;
    var e = ae.events.filter(function (x) { return x.key === key; })[0];
    if (!e) return;
    if (aeSelected === key) {                    // click again to clear
      aeSelected = null;
      window.AegisPKTwin.clearRiskFocus();
      renderAE();
      return;
    }
    aeSelected = key;
    // Show the arm that actually contains this drug — an event on the vasopressor
    // is meaningless while the antibiotic arm is on screen.
    var idx = (data.arms || []).map(function (a) { return a.drug; }).indexOf(e.drug);
    if (idx >= 0 && idx !== armIdx) selectArm(idx);
    window.AegisPKTwin.setRiskFocus(e.mesh, e.severity);
    var a = armForDrug(e.drug);
    if (a && e.onset_p50_h <= a.horizon_h) {
      setPlaying(false);
      window.AegisPKTwin.setTime(e.onset_p50_h);
    }
    renderAE();
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
    try { window.AegisPKTwin && window.AegisPKTwin.clearRiskFocus(); } catch (e) {}
    aeSelected = null; options = null; optKey = null;
    clearBeats(); beatLast = 0;
    try { window.AegisTwin && window.AegisTwin.setParked && window.AegisTwin.setParked(false); } catch (e) {}
  }

  function fail(msg, detail) {
    $("pkt-load").style.display = "";
    $("pkt-load").innerHTML = esc(msg) + "<i>" + esc(detail || "") + "</i>";
  }

  function openBed(bedId, opts) {
    // bedId null/"session" → the clinical console's single live patient, which has
    // no bed id. Same overlay, same contract, different endpoint.
    bed = bedId; opts = opts || {};
    var url = (!bedId || bedId === "session")
      ? "/aegis/pk/options"
      : "/aegis/pk/options?bed=" + encodeURIComponent(bedId);
    show();
    $("pkt-load").style.display = "";
    $("pkt-load").innerHTML = "projecting…<i>solving the compartment model for every "
      + "candidate order and assembling the anatomy</i>";
    $("pkt-name").textContent = opts.name || bedId || "Monitored patient";
    $("pkt-sub").textContent = "—";

    fetch(window.AEGIS.http(url))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!open) return;
        if (d.status !== "ok") return fail("could not project", d.detail || "");
        if (!d.options || !d.options.length)
          return fail("nothing to compare", "no candidate treatment paths were "
            + "returned for this patient.");
        options = d;
        $("pkt-name").textContent = d.bed_name || bedId || "Monitored patient";
        var p = d.patient || {};
        $("pkt-sub").innerHTML = "<b>eGFR</b> " + Math.round(p.egfr) + " · <b>MAP</b> "
          + Math.round(p.map_mmhg) + " mmHg · <b>" + Math.round(p.weight_kg) + "</b> kg"
          + " · risk <b>" + esc(d.risk_level || "—") + "</b>"
          + (p.shock_index > 0 ? " · shock redistribution active" : "");

        var un = (window.AegisPKTwin.UNMODELLED || {});
        var names = Object.keys(un).map(function (k) { return un[k]; });
        $("pkt-nodata").textContent = names.length
          ? "inert — no compartment in the model: " + names.join(", ") : "";

        mountTwin();
        // Open on the option that was asked for, else the recommended one, else
        // the first path that actually administers something.
        var want = (opts.option && d.options.some(function (o) { return o.key === opts.option; }))
          ? opts.option
          : (d.recommended
             || (d.options.filter(function (o) { return o.arms.length; })[0] || d.options[0]).key);
        selectOption(want);
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
        // Two settle passes: the overlay animates in (transform + opacity over
        // ~240ms), so the stage is still growing when the first frame lands.
        requestAnimationFrame(function () { window.AegisPKTwin.resize(); });
        setTimeout(function () { window.AegisPKTwin.resize(); }, 320);
      });
      mounted = true;
    } else {
      requestAnimationFrame(function () { window.AegisPKTwin.resize(); });
      setTimeout(function () { window.AegisPKTwin.resize(); }, 320);
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
    Array.prototype.forEach.call($("pkt-tabs").querySelectorAll(".pkt-tab"), function (b) {
      b.addEventListener("click", function () { showPane(b.dataset.t); });
    });
    Array.prototype.forEach.call($("pkt-ae-filter").querySelectorAll("button"), function (b) {
      b.addEventListener("click", function () {
        Array.prototype.forEach.call($("pkt-ae-filter").querySelectorAll("button"),
          function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        aeThreshold = +b.dataset.p;
        renderAE();
      });
    });
    var nb = $("pkt-narrate");
    if (nb) nb.addEventListener("click", function () {
      narrateOn = !nb.classList.contains("on");
      nb.classList.toggle("on", narrateOn);
      if (!narrateOn) clearBeats();
    });
    var lb = $("pkt-labels-btn");
    if (lb) lb.addEventListener("click", function () {
      var on = !lb.classList.contains("on");
      lb.classList.toggle("on", on);
      window.AegisPKTwin.setLabels(on);
    });
    var fb = $("pkt-fit"), wholeBody = false;
    if (fb) fb.addEventListener("click", function () {
      wholeBody = !wholeBody;
      fb.textContent = wholeBody ? "⤢ body" : "⤢ torso";
      if (wholeBody) window.AegisPKTwin.fitToBody();
      else window.AegisPKTwin.fitToAtlas();
    });
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
