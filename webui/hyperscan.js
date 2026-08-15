/* ============================================================
   AEGIS — HYPERSCAN
   ------------------------------------------------------------
   Sweep every subsystem, flag each finding on the body as it lands,
   then present the situation report.

   The body is the REAL twin, not a copy: the overlay borrows the
   live #twin-stage element and hands it back on close. Moving the
   host element keeps the same WebGL context, the same loaded
   anatomy and the same organ colours the desktop was showing —
   a second renderer would be 20 MB of duplicate geometry and, worse,
   a body that could disagree with the one behind it.

   Flags are anchored to the organ the finding came from, because a
   finding with no location is just a sentence. The backend tags every
   stage with a body region for exactly this.
   ============================================================ */
(function () {
  "use strict";

  var S = { open: false, running: false, bed: null, home: null, report: null,
           speechText: '', speechAudio: null, playing: null, speakState: 'idle' };

  function $(s) { return document.querySelector(s); }
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  function bedOf() {
    try { return (window.AegisPatient && window.AegisPatient.bed()) || null; }
    catch (e) { return null; }
  }

  /* ---------------- borrowing the body ---------------- */
  function takeBody() {
    var stage = $("#twin-stage"), slot = $("#hs-stage");
    if (!stage || !slot) return;
    S.home = stage.parentElement;                 // give it back exactly where it lived
    slot.appendChild(stage);
    stage.classList.add("hs-body");
    if (window.AegisTwin) {
      window.AegisTwin.setParked(false);
      window.AegisTwin.setHyperscan(true);
      window.AegisTwin.clearFlags();
      setTimeout(window.AegisTwin.resize, 60);
      setTimeout(window.AegisTwin.resize, 400);   // after the open transition settles
    }
  }
  function returnBody() {
    var stage = $("#twin-stage");
    if (stage && S.home) {
      stage.classList.remove("hs-body");
      S.home.appendChild(stage);
    }
    if (window.AegisTwin) {
      window.AegisTwin.setHyperscan(false);
      window.AegisTwin.clearFlags();
      setTimeout(window.AegisTwin.resize, 60);
    }
    S.home = null;
  }

  /* ---------------- lifecycle ---------------- */
  function open(bed) {
    if (S.open) return;
    S.bed = bed || bedOf();
    S.open = true;
    S.report = null;
    $("#hyperscan").classList.remove("hidden");
    $("#hs-stages").innerHTML = "";
    $("#hs-report").innerHTML = "";
    $("#hs-report").classList.add("hidden");
    $("#hs-progress").style.width = "0%";
    stopPlay(); S.speechAudio = null; S.speechText = "";
    setSpeak("idle");
    $("#hs-mute").textContent = muted() ? "◌ autoplay off" : "◉ autoplay on";
    $("#hs-mute").classList.toggle("on", !muted());
    setPhase("sweeping", "collecting");
    takeBody();
    run();
  }

  function close() {
    if (!S.open) return;
    stopPlay();
    S.open = false; S.running = false;
    returnBody();
    $("#hyperscan").classList.add("hidden");
  }

  function setPhase(cls, txt) {
    var h = $("#hs-phase");
    h.className = "hs-phase " + cls;
    h.textContent = txt.toUpperCase();
  }

  /* ---------------- the sweep ---------------- */
  function run() {
    S.running = true;
    fetch(window.AEGIS.http("/aegis/hyperscan/stream"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "hyperscan", bed: S.bed })
    }).then(function (r) {
      if (!r.ok || !r.body) throw new Error("stream unavailable");
      var reader = r.body.getReader(), dec = new TextDecoder(), buf = "";
      function pump() {
        return reader.read().then(function (res) {
          if (res.done) return;
          buf += dec.decode(res.value, { stream: true });
          var blocks = buf.split("\n\n"); buf = blocks.pop();
          blocks.forEach(function (b) {
            var line = b.split("\n").find(function (l) { return l.indexOf("data: ") === 0; });
            if (!line) return;
            var ev; try { ev = JSON.parse(line.slice(6)); } catch (e) { return; }
            onEvent(ev);
          });
          return pump();
        });
      }
      return pump();
    }).catch(function () {
      setPhase("error", "unreachable");
      $("#hs-report").classList.remove("hidden");
      $("#hs-report").appendChild(el("div", "hs-empty",
        "Could not reach the backend. On the free tier it sleeps after 15 minutes — "
        + "the first request wakes it, so try again in a moment."));
    });
  }

  function onEvent(ev) {
    if (ev.type === "stage") return onStage(ev);
    if (ev.type === "report") { S.report = ev.report; paintReport(ev.report); return; }
    if (ev.type === "speech_pending") { setSpeak("rendering"); return; }
    if (ev.type === "speech") return onSpeech(ev);
    if (ev.type === "error") { setPhase("error", "failed"); }
  }

  /* The report is read aloud because it reaches someone whose hands are busy —
     but never without asking twice: browsers block unprompted audio, and a
     report that starts talking at a bedside without warning is worse than one
     that stays quiet. Muting is remembered. */
  function onSpeech(ev) {
    S.speechText = ev.text || "";
    S.speechAudio = ev.audio || null;
    if (!ev.audio) { setSpeak(ev.available === false ? "unavailable" : "unavailable"); return; }
    setSpeak("ready");
    if (!muted()) play();
  }

  function muted() { try { return localStorage.getItem("aegis_hs_mute") === "1"; } catch (e) { return false; } }
  function setMuted(v) { try { localStorage.setItem("aegis_hs_mute", v ? "1" : "0"); } catch (e) {} }

  function play() {
    if (!S.speechAudio || !window.AegisDoctor || !window.AegisDoctor.speak) return;
    // A browser that has had no user gesture will refuse to start audio. The
    // hyperscan is always opened by a click, so this normally succeeds — and
    // when it does not, the button says "play" instead of failing silently.
    S.playing = window.AegisDoctor.speak(S.speechAudio, ev_rate());
    setSpeak("playing");
  }
  function ev_rate() { return 24000; }
  function stopPlay() {
    if (S.playing && S.playing.stop) S.playing.stop();
    S.playing = null;
    // "ready" means there is something to replay. Saying so before any audio has
    // arrived offers a play button that plays nothing.
    setSpeak(S.speechAudio ? "ready" : "idle");
  }

  function setSpeak(state) {
    var b = $("#hs-speak");
    if (!b) return;
    S.speakState = state;
    b.disabled = (state === "rendering" || state === "unavailable" || state === "idle");
    b.textContent = state === "rendering" ? "◍ rendering voice…"
      : state === "unavailable" ? "◍ voice unavailable"
      : state === "idle" ? "◍ voice"
      : state === "playing" ? "■ stop"
      : "▶ replay briefing";
    b.classList.toggle("on", state === "playing");
  }

  function onStage(ev) {
    var id = "hs-st-" + ev.key;
    var row = document.getElementById(id);
    if (!row) {
      row = el("div", "hs-row running");
      row.id = id;
      row.appendChild(el("i", "hs-dot"));
      row.appendChild(el("span", "hs-st-label", ev.label));
      row.appendChild(el("span", "hs-st-head", ""));
      $("#hs-stages").appendChild(row);
    }
    if (ev.state === "done") {
      row.className = "hs-row " + (ev.ok === false ? "bad" : "done");
      row.querySelector(".hs-st-head").textContent = ev.headline || "";
      // Pin the finding to the organ it came from, as it lands. Synthesis is
      // excluded: it is not a finding about a body part, and its headline is a
      // whole sentence that would trail off the edge of the stage.
      if (window.AegisTwin && window.AegisTwin.flag && ev.region
          && ev.ok !== false && ev.key !== "synthesis") {
        window.AegisTwin.flag(ev.region, ev.label,
                              String(ev.headline || "").slice(0, 34));
      }
    }
    var pct = Math.round(((ev.i + (ev.state === "done" ? 1 : 0)) / (ev.n || 1)) * 100);
    $("#hs-progress").style.width = Math.min(100, pct) + "%";
    if (ev.key === "synthesis" && ev.state === "running") setPhase("thinking", "synthesising");
  }

  /* ---------------- the report ---------------- */
  function paintReport(r) {
    setPhase("done", "report ready");
    if (window.AegisTwin) window.AegisTwin.setHyperscan(false);
    var out = $("#hs-report");
    out.innerHTML = "";
    out.classList.remove("hidden");

    out.appendChild(sec("Situation", el("p", "hs-p", r.situation || "—")));

    if ((r.key_findings || []).length) {
      var ul = el("div", "hs-finds");
      r.key_findings.forEach(function (f) {
        var d = el("div", "hs-find " + (f.weight || "medium"));
        d.appendChild(el("span", "hs-find-t", f.finding || ""));
        d.appendChild(el("span", "hs-find-s", (f.source || "") + (f.region ? " · " + f.region : "")));
        ul.appendChild(d);
      });
      out.appendChild(sec("Key findings", ul));
    }

    if ((r.consistent_with || []).length || (r.against || []).length) {
      var w = el("div", "hs-two");
      w.appendChild(listCard("Consistent with", r.consistent_with || []));
      w.appendChild(listCard("Argues against", r.against || []));
      out.appendChild(sec("Reading", w));
    }

    if ((r.options || []).length) {
      var ob = el("div", "hs-opts");
      r.options.forEach(function (o) {
        var d = el("div", "hs-opt" + (o.allowed ? "" : " refused")
          + (r.recommendation && r.recommendation.option === o.label ? " pick" : ""));
        var h = el("div", "hs-opt-h");
        h.appendChild(el("b", null, o.label || "—"));
        h.appendChild(el("span", "hs-opt-tag", o.allowed ? "permitted" : "refused by cage"));
        d.appendChild(h);
        if (o.tradeoff) d.appendChild(el("div", "hs-opt-t", o.tradeoff));
        if (!o.allowed && o.verdict) d.appendChild(el("div", "hs-opt-v", o.verdict));
        ob.appendChild(d);
      });
      out.appendChild(sec("Options considered", ob));
    }

    var rec = r.recommendation || {};
    var rc = el("div", "hs-rec" + (rec.option ? "" : " none"));
    rc.appendChild(el("div", "hs-rec-k", rec.option ? "Suggests" : "No recommendation"));
    rc.appendChild(el("b", "hs-rec-o", rec.option || "—"));
    if (rec.why) rc.appendChild(el("div", "hs-p", rec.why));
    if (rec.note) rc.appendChild(el("div", "hs-rec-note", rec.note));
    if (rec.watch) rc.appendChild(el("div", "hs-rec-w", "What would change this: " + rec.watch));
    out.appendChild(sec("Recommendation", rc));

    if ((r.not_measured || []).length) {
      out.appendChild(sec("Not measured by this system",
        listCard("", r.not_measured, true)));
    }
    if (r.confidence) out.appendChild(sec("How much weight this deserves",
      el("p", "hs-p", r.confidence)));

    // The notice is server-appended and rendered last and loudest — it is the
    // one thing on this screen that is not optional.
    var n = el("div", "hs-notice");
    n.appendChild(el("span", "hs-notice-k", "REVIEW REQUIRED"));
    n.appendChild(el("span", null, r.review_notice || ""));
    out.appendChild(n);
    out.scrollTop = 0;
  }

  function sec(title, node) {
    var s = el("section", "hs-sec");
    s.appendChild(el("h4", "hs-sec-h", title));
    s.appendChild(node);
    return s;
  }
  function listCard(title, items, bare) {
    var c = el("div", bare ? "hs-list bare" : "hs-list");
    if (title) c.appendChild(el("div", "hs-list-h", title));
    if (!items.length) c.appendChild(el("div", "hs-list-e", "—"));
    items.forEach(function (x) { c.appendChild(el("div", "hs-li", String(x))); });
    return c;
  }

  /* ---------------- wiring ---------------- */
  document.addEventListener("click", function (e) {
    if (e.target.closest("#launch-hyperscan")) { open(); return; }
    if (e.target.closest("#hs-close")) { close(); return; }
    if (e.target.closest("#hs-again")) {
      if (!S.running) { open(S.bed); }
      return;
    }
    if (e.target.closest("#hs-speak")) {
      if (S.speakState === "playing") stopPlay(); else play();
      return;
    }
    if (e.target.closest("#hs-mute")) {
      var m = !muted();
      setMuted(m);
      $("#hs-mute").textContent = m ? "◌ autoplay off" : "◉ autoplay on";
      $("#hs-mute").classList.toggle("on", !m);
      if (m) stopPlay();
      return;
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && S.open) { e.preventDefault(); close(); }
  });

  window.AegisHyperscan = { open: open, close: close, _state: S };
})();
