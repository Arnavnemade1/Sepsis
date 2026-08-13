/* ============================================================
   AEGIS — CONSULT CALL CLIENT
   ------------------------------------------------------------
   Call a colleague about a case. Microphone in, doctor's voice out,
   both sides transcribed, grounded in the selected bed's dossier.

   The browser never holds an API key. It speaks only to AegisOS
   (/aegis/doctor/live), which relays to the model and owns the
   credential. Everything clinical the doctor says came from the
   dossier the rail is displaying — that pairing is the point.

   Three details that this had to get right, all of them audio:

   1. PACING IS THE PROTOCOL. Voice-activity detection runs on the
      arrival rate of audio chunks. Buffering a couple of seconds and
      flushing them in a burst makes the model wait forever for a turn
      that already ended. So chunks go out the instant the worklet
      produces them, 1x realtime, never batched.
   2. RESAMPLE OURSELVES. Hardware gives us 44.1k or 48k; the model
      wants exactly 16k mono s16le. We convert in the audio thread
      rather than trusting a requested AudioContext rate, because
      browsers silently disagree about honouring it.
   3. BARGE-IN MUST FLUSH. When you talk over the doctor, the server
      says `interrupted` — and every already-scheduled buffer has to be
      killed. Miss this and the doctor keeps calmly answering a
      question you stopped asking, which is worse than no call at all.
   ============================================================ */
(function () {
  "use strict";

  var IN_RATE = 16000;              // uplink: mono s16le, fixed by the Live API
  var OUT_RATE = 24000;             // downlink: mono s16le
  var FRAME_MS = 100;               // one uplink chunk
  var VIDEO_MS = 1500;              // camera frame cadence when sharing video

  var S = {
    ws: null, open: false, connecting: false,
    bed: null, headline: "", facts: null,
    micOn: true, videoOn: false, camReady: false,
    startedAt: 0, timer: null,
    capCtx: null, capNode: null, capStream: null, capSrc: null,
    playCtx: null, playAt: 0, sources: [], flashT: null,
    level: 0, speaking: false, phase: "standby", voiceOk: null, voiceWhy: "",
    line: { role: null, text: "" },
    raf: null, statusTimer: null, thinkSince: 0,
    cursorT: null, asideT: null, viewT: null, prePhase: null
  };

  /* The assistant's phase. It drives the ring, the readout and the caption, so
     there is exactly one place that decides what the thing is doing — the old
     version inferred "speaking" from an audio packet and nothing tracked the gap
     between you finishing a sentence and the first word coming back, which is
     the moment a voice UI feels broken if it shows nothing. */
  var PHASES = {
    standby:    "STANDBY",
    waking:     "WAKING BACKEND",
    connecting: "ESTABLISHING LINK",
    renewing:   "RENEWING SESSION",
    listening:  "LISTENING",
    thinking:   "WORKING",
    speaking:   "SPEAKING",
    ended:      "LINK CLOSED",
    offline:    "VOICE OFFLINE"
  };

  function phase(p) {
    if (S.phase === p) return;
    S.phase = p;
    if (p === "thinking") S.thinkSince = Date.now();
    var app = $("#doctor-app");
    app.className = app.className.replace(/\bph-\w+/g, "").trim() + " ph-" + p;
    var r = $("#doc-phase");
    if (r) r.textContent = PHASES[p] || p.toUpperCase();
  }

  function $(s) { return document.querySelector(s); }
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  /* ---------------- lifecycle ---------------- */
  function open(bed) {
    S.bed = bed || currentBed();
    $("#doctor-app").classList.remove("hidden");
    resetUI();
    preflight();
    loadBrief();
    loadCaps();
    startOrb();
  }

  function close() {
    hangup("closed");
    stopOrb();
    $("#doctor-app").classList.add("hidden");
  }

  function currentBed() {
    // the shell's patient selector is the source of truth for "who are we on"
    try { return (window.AegisPatient && window.AegisPatient.bed()) || null; }
    catch (e) { return null; }
  }

  /* Ask the SERVER whether it can place a call before offering the button.
     Voice runs through the backend, so "can I talk to it" is a server fact, and
     a deploy with no key would otherwise present a live-looking call button that
     fails on click with a generic socket error. */
  function preflight() {
    phase("waking");
    fetch(window.AEGIS.http("/aegis/doctor/status"))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        S.voiceOk = !!d.voice;
        S.voiceWhy = d.voice_reason || "";
        var btn = $("#doc-call");
        if (S.voiceOk) {
          btn.disabled = false;
          btn.textContent = "◉ start call";
          phase("standby");
          setState("idle", "ready");
        } else {
          btn.disabled = true;
          btn.textContent = "◉ voice offline";
          phase("offline");
          setState("error", "voice offline");
          banner("Voice is unavailable on this deployment — " + S.voiceWhy
                 + " The typed consult below works and uses the same dossier and "
                 + "the same tools.");
        }
      })
      .catch(function () {
        // the backend sleeps on the free tier; a cold start is not a failure
        S.voiceOk = null;
        phase("standby");
        setState("error", "backend unreachable");
        banner("Could not reach the backend. On the free tier it sleeps after "
               + "15 minutes and the first request wakes it — try again in a moment.");
      });
  }

  function resetUI() {
    setState("idle", "not connected");
    $("#doc-log").innerHTML = "";
    $("#doc-log").appendChild(el("div", "doc-log-empty", "transcript will appear here"));
    $("#doc-acts").innerHTML = "";
    $("#doc-acts").appendChild(el("div", "doc-log-empty", "nothing run yet"));
    $("#doc-act-count").textContent = "0";
    caption("Open the call to talk this case through.", true);
    $("#doc-timer").textContent = "";
    $("#doc-banner").classList.add("hidden");
    $("#doc-call").disabled = false;
    $("#doc-end").disabled = true;
  }

  function setState(cls, txt) {
    var e = $("#doc-state");
    e.className = "doc-state " + cls;
    e.querySelector("span").textContent = txt;
  }

  function caption(txt, idle) {
    var c = $("#doc-caption");
    var grew = txt && txt.indexOf(c.textContent) === 0;   // same line, extended
    c.textContent = txt;
    c.classList.toggle("idle", !!idle);
    // A new line starts at its beginning; a line that is still being spoken
    // follows the words. Without this the docked strip opened halfway through a
    // sentence, because the box keeps the scroll position of the previous answer.
    c.scrollTop = grew ? c.scrollHeight : 0;
  }

  /* ---------------- grounding rail ---------------- */
  /* The capability roster. Shown because an agent that can drive the whole OS
     should not be a black box about what "the whole OS" means. */
  function loadCaps() {
    fetch(window.AEGIS.http("/aegis/agent/tools"))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var c = d.counts || {};
        $("#doc-cap-n").textContent = (d.bound || 0);
        var wrap = $("#doc-caps");
        wrap.innerHTML = "";
        [["read", c.read, "inspect any subsystem"],
         ["ui", c.ui, "drive the cockpit"],
         ["write", c.write, "act on the system"]].forEach(function (row) {
          var d0 = el("div", "doc-cap " + row[0]);
          d0.appendChild(el("b", null, String(row[1] || 0)));
          d0.appendChild(el("span", null, row[2]));
          wrap.appendChild(d0);
        });
        $("#doc-cap-note").textContent =
          "Every action that would reach a patient passes the Z3 egress gate and "
          + "is written to the append-only audit log. A refused action changes nothing.";
      }).catch(function () { $("#doc-cap-n").textContent = "—"; });
  }

  function loadBrief() {
    var url = window.AEGIS.http("/aegis/doctor/brief?opening=false" +
      (S.bed ? "&bed=" + encodeURIComponent(S.bed) : ""));
    fetch(url).then(function (r) { return r.json(); }).then(paintBrief).catch(function () {
      $("#doc-pt").textContent = "case unavailable";
    });
  }

  function paintBrief(b) {
    if (!b) return;
    S.headline = b.headline || "";
    S.facts = b.facts || {};
    paintGround(b.headline, b.facts);
    var ul = $("#doc-blind");
    ul.innerHTML = "";
    (b.unmeasured || []).forEach(function (u) { ul.appendChild(el("li", null, u)); });
    var p = b.provenance || {};
    if (p.auroc_range && p.false_alarm_range) {
      $("#doc-prov").textContent =
        "The detector under this conversation scores AUROC " +
        p.auroc_range[0].toFixed(3) + "–" + p.auroc_range[1].toFixed(3) +
        " across hospitals, with a " + Math.round(p.false_alarm_range[0] * 100) + "–" +
        Math.round(p.false_alarm_range[1] * 100) + "% false-alarm rate. " +
        "Dr. Aegis is required to say so when it quotes a risk.";
    }
    if (b.voice === false) {
      banner("Voice is unavailable on this deployment — the typed consult below uses the same dossier.");
    }
  }

  function paintGround(headline, facts) {
    facts = facts || {};
    // The headline ends "· TIER · P=x", both of which already have their own
    // element here; repeating them just pushed the patient's name out of the box.
    $("#doc-pt").textContent = (headline || "—").split(" · ").slice(0, 2).join(" · ");
    var tier = (facts.tier || "—").toUpperCase();
    var t = $("#doc-tier");
    t.textContent = tier;
    t.className = "doc-tier " + tier;
    var v = facts.vitals || {};
    var cells = [
      [fmt(facts.probability, 3), "P(sepsis)"],
      [facts.sirs != null ? facts.sirs + "/4" : "—", "SIRS"],
      [facts.qsofa != null ? facts.qsofa + "/3" : "—", "qSOFA"],
      [fmt(v.heart_rate, 0), "HR"],
      [fmt(v.respiratory_rate, 0), "RR"],
      [fmt(v.skin_temp, 1), "TEMP °C"],
      [fmt(v.spo2, 0), "SpO2"],
      [fmt(v.systolic_bp, 0), "SBP"],
      [facts.signal_quality != null ? Math.round(facts.signal_quality * 100) + "%" : "—", "SIGNAL"]
    ];
    var g = $("#doc-facts");
    g.innerHTML = "";
    cells.forEach(function (c) {
      var d = el("div", "doc-fact");
      d.appendChild(el("b", null, c[0]));
      d.appendChild(el("span", null, c[1]));
      g.appendChild(d);
    });
    $("#doc-recorded").textContent = facts.recorded
      ? "This bed is a replayed record from the validation cohort, not a live bedside monitor."
      : "";
  }

  function fmt(x, nd) {
    return (x == null || isNaN(x)) ? "—" : Number(x).toFixed(nd);
  }

  function banner(msg) {
    var b = $("#doc-banner");
    b.textContent = msg;
    b.classList.remove("hidden");
  }

  /* ---------------- transcript ---------------- */
  function logLine(role, text) {
    var log = $("#doc-log");
    var empty = log.querySelector(".doc-log-empty");
    if (empty) empty.remove();
    var cls = role === "doctor" ? "doctor" : role === "clinician" ? "you" : "sys";
    var last = log.lastElementChild;
    // Transcripts stream in fragments; keep appending to the open line so the
    // record reads as sentences rather than confetti.
    if (last && last.dataset && last.dataset.role === role && role !== "sys") {
      last.querySelector(".doc-text").textContent += text;
    } else {
      var line = el("div", "doc-line " + cls);
      line.dataset.role = role;
      line.appendChild(el("span", "doc-role", role === "doctor" ? "Dr. Aegis"
        : role === "clinician" ? "you" : "system"));
      line.appendChild(el("span", "doc-text", text));
      log.appendChild(line);
    }
    log.scrollTop = log.scrollHeight;
  }

  function closeLine() { S.line = { role: null, text: "" }; }

  /* ---------------- the call ---------------- */
  function callStart() {
    if (S.open || S.connecting) return;
    S.connecting = true;
    $("#doc-call").disabled = true;
    setState("connecting", "connecting…");
    phase("connecting");
    caption("Establishing link…", true);

    var url = window.AEGIS.base().replace(/^http/, "ws") + "/aegis/doctor/live" +
      (S.bed ? "?bed=" + encodeURIComponent(S.bed) : "");
    var ws;
    try { ws = new WebSocket(url); } catch (e) { return failCall(String(e)); }
    S.ws = ws;

    ws.onmessage = function (ev) {
      var m;
      try { m = JSON.parse(ev.data); } catch (e) { return; }
      onServer(m);
    };
    ws.onerror = function () { if (!S.open) failCall("could not reach the consult service"); };
    ws.onclose = function () {
      if (S.open) logLine("sys", "link closed");
      teardownAudio();
      S.open = false; S.connecting = false;
      setState("idle", "call ended");
      phase("ended");
      $("#doc-call").disabled = !(S.voiceOk !== false);
      $("#doc-end").disabled = true;
      stopTimer();
      caption("Link closed.", true);
    };
  }

  function failCall(msg) {
    S.connecting = false;
    S.open = false;
    setState("error", "unavailable");
    phase("offline");
    $("#doc-call").disabled = false;
    banner("Voice call unavailable — " + msg + ". The typed consult below still works and uses the same dossier.");
    caption("Voice is unavailable. Type a question below to consult by text.", true);
  }

  function onServer(m) {
    switch (m.type) {
      case "ready":
        S.connecting = false; S.open = true;
        setState("live", "on call");
        $("#doc-end").disabled = false;
        if (m.headline) { S.headline = m.headline; paintGround(m.headline, m.facts); }
        phase("listening");
        caption("Link established. Go ahead.", true);
        // a short, factual bring-up: what it is grounded in and what it can reach
        logLine("sys", "link established · " + (m.headline || "this case"));
        logLine("sys", "voice " + (m.voice || "—") + " · " + (m.tools || 0)
          + " tools · grounded in the dossier on the right");
        startMic();
        startTimer();
        break;
      case "transcript":
        if (m.role === "doctor") {
          if (S.line.role !== "doctor") { S.line = { role: "doctor", text: "" }; }
          S.line.text += m.text;
          caption(S.line.text);
        } else {
          if (S.line.role !== "clinician") { S.line = { role: "clinician", text: "" }; }
          S.line.text += m.text;
          // your words are transcribed the moment you stop — that is the handover
          // point, and it is where the ring should stop listening and start working
          if (S.phase === "listening") phase("thinking");
        }
        logLine(m.role, m.text);
        break;
      case "audio":
        playChunk(m.data);
        phase("speaking");
        break;
      case "turn_end":
        closeLine();
        S.speaking = false;
        $("#doctor-app").classList.remove("speaking");
        if (S.open) phase("listening");
        break;
      case "interrupted":
        flushPlayback();
        closeLine();
        break;
      case "case_update":
        paintGround(m.headline, m.facts);
        logLine("sys", "state moved: " + ((m.changes || []).join("; ")
                                          || ("tier → " + m.to_tier)));
        flash((m.changes || []).join(" · "));
        break;
      case "ui":
        logLine("sys", "▸ " + runUI(m.action, m.args));
        break;
      case "tool":
        logAction(m);
        break;
      case "renewing":
        // The server is rebuilding the realtime session underneath us and
        // replaying the conversation into it. Not an error, and not the end of
        // the call — just a gap.
        S.prePhase = S.phase;
        phase("renewing");
        break;
      case "renewed":
        phase(S.prePhase === "speaking" ? "listening" : (S.prePhase || "listening"));
        S.prePhase = null;
        flushPlayback();          // anything queued belonged to the dead session
        logLine("sys", "session renewed — conversation carried over");
        break;
      case "going_away":
        break;                    // handled server-side by the renewal above
      case "error":
        if (m.fallback === "text" || m.fatal) failCall(m.message || "service error");
        else logLine("sys", m.message || "error");
        break;
    }
  }

  function hangup(reason) {
    if (S.ws) {
      try { if (S.ws.readyState === 1) S.ws.send(JSON.stringify({ type: "bye" })); } catch (e) {}
      try { S.ws.close(); } catch (e) {}
      S.ws = null;
    }
    teardownAudio();
    S.open = false; S.connecting = false;
    stopTimer();
    if (reason !== "closed") setState("idle", "call ended");
  }

  function startTimer() {
    S.startedAt = Date.now();
    stopTimer();
    S.timer = setInterval(function () {
      var s = Math.floor((Date.now() - S.startedAt) / 1000);
      $("#doc-timer").textContent =
        String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
    }, 1000);
  }
  function stopTimer() { if (S.timer) { clearInterval(S.timer); S.timer = null; } }

  /* ---------------- microphone uplink ---------------- */
  function startMic() {
    if (S.capCtx) return;
    navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    }).then(function (stream) {
      S.capStream = stream;
      var Ctx = window.AudioContext || window.webkitAudioContext;
      S.capCtx = new Ctx();
      S.capSrc = S.capCtx.createMediaStreamSource(stream);
      buildCaptureNode(S.capCtx.sampleRate);
      S.capSrc.connect(S.capNode);
      // ScriptProcessor needs a sink to pump; a zero gain keeps it silent so the
      // clinician never hears themselves echoed back through the speakers.
      var sink = S.capCtx.createGain();
      sink.gain.value = 0;
      S.capNode.connect(sink);
      sink.connect(S.capCtx.destination);
      logLine("sys", "microphone live · " + Math.round(S.capCtx.sampleRate / 1000) + "kHz → 16kHz");
    }).catch(function (e) {
      banner("Microphone unavailable (" + (e.name || "denied") + ") — use the typed consult below.");
      logLine("sys", "microphone denied; typed consult still available");
    });
  }

  function buildCaptureNode(srcRate) {
    var frame = Math.max(256, Math.pow(2, Math.round(Math.log2(srcRate * FRAME_MS / 1000))));
    var node = S.capCtx.createScriptProcessor(frame, 1, 1);
    var carry = 0;                                  // fractional read position
    node.onaudioprocess = function (ev) {
      if (!S.open || !S.micOn) return;
      var input = ev.inputBuffer.getChannelData(0);
      var ratio = srcRate / IN_RATE;
      var outLen = Math.floor((input.length - carry) / ratio);
      if (outLen <= 0) return;
      var pcm = new Int16Array(outLen);
      var peak = 0, pos = carry;
      for (var i = 0; i < outLen; i++) {
        var i0 = Math.floor(pos), f = pos - i0;
        var a = input[i0] || 0, b = input[Math.min(i0 + 1, input.length - 1)] || 0;
        var v = a + (b - a) * f;                    // linear interpolation
        if (v > 1) v = 1; else if (v < -1) v = -1;
        pcm[i] = v * 0x7fff;
        var m = v < 0 ? -v : v;
        if (m > peak) peak = m;
        pos += ratio;
      }
      carry = pos - input.length;
      if (carry < 0) carry = 0;
      S.level = S.level * 0.7 + peak * 0.3;
      send({ type: "audio", data: b64(pcm.buffer) });   // out immediately — see (1)
    };
    S.capNode = node;
  }

  function teardownAudio() {
    if (S.capNode) { try { S.capNode.disconnect(); } catch (e) {} S.capNode = null; }
    if (S.capSrc) { try { S.capSrc.disconnect(); } catch (e) {} S.capSrc = null; }
    if (S.capStream) { S.capStream.getTracks().forEach(function (t) { t.stop(); }); S.capStream = null; }
    if (S.capCtx) { try { S.capCtx.close(); } catch (e) {} S.capCtx = null; }
    flushPlayback();
    S.level = 0;
  }

  function send(obj) {
    if (S.ws && S.ws.readyState === 1) {
      try { S.ws.send(JSON.stringify(obj)); } catch (e) {}
    }
  }

  function b64(buf) {
    var bytes = new Uint8Array(buf), s = "", CH = 0x8000;
    for (var i = 0; i < bytes.length; i += CH) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    }
    return btoa(s);
  }

  /* ---------------- playback downlink ---------------- */
  function playCtx() {
    if (!S.playCtx) {
      var Ctx = window.AudioContext || window.webkitAudioContext;
      S.playCtx = new Ctx();
      S.playAt = 0;
      // sources -> analyser -> speakers, so the ring can read the real waveform
      S.analyser = S.playCtx.createAnalyser();
      S.analyser.fftSize = 256;
      S.analyser.connect(S.playCtx.destination);
    }
    if (S.playCtx.state === "suspended") S.playCtx.resume();
    return S.playCtx;
  }

  function playChunk(b64data) {
    var ctx = playCtx();
    var bin = atob(b64data);
    var n = bin.length >> 1;
    // AudioBuffer at the model's rate; the graph resamples to the device rate.
    var buf = ctx.createBuffer(1, n, OUT_RATE);
    var ch = buf.getChannelData(0);
    for (var i = 0; i < n; i++) {
      var lo = bin.charCodeAt(i * 2), hi = bin.charCodeAt(i * 2 + 1);
      var v = (hi << 8) | lo;
      if (v >= 0x8000) v -= 0x10000;
      ch[i] = v / 0x8000;
    }
    var src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(S.analyser || ctx.destination);
    var at = Math.max(ctx.currentTime + 0.02, S.playAt);
    src.start(at);
    S.playAt = at + buf.duration;
    S.sources.push(src);
    src.onended = function () {
      var k = S.sources.indexOf(src);
      if (k >= 0) S.sources.splice(k, 1);
    };
    S.speaking = true;
    $("#doctor-app").classList.add("speaking");
  }

  function flushPlayback() {
    S.sources.forEach(function (s) { try { s.stop(); } catch (e) {} });
    S.sources = [];
    S.playAt = S.playCtx ? S.playCtx.currentTime : 0;
    S.speaking = false;
    $("#doctor-app").classList.remove("speaking");
  }

  /* ---------------- camera (self view + optional frames) ---------------- */
  function bindSelfView() {
    var proc = $("#cam-proc"), vid = $("#doc-self-video"), wrap = $("#doc-self");
    if (proc && proc.srcObject) {
      vid.srcObject = proc.srcObject;
      vid.play().catch(function () {});
      wrap.classList.remove("off");
      S.camReady = true;
    } else {
      wrap.classList.add("off");
      S.camReady = false;
    }
    // the stage reserves room for the PIP only while it is actually up
    $("#doctor-app").classList.toggle("selfview", S.camReady);
  }

  function toggleCamera() {
    if (!S.camReady && window.AegisCamera) {
      window.AegisCamera.enable();
      setTimeout(bindSelfView, 1200);
      return;
    }
    bindSelfView();
  }

  function toggleVideoShare() {
    S.videoOn = !S.videoOn;
    $("#doc-video").classList.toggle("on", S.videoOn);
    $("#doc-video").textContent = S.videoOn ? "◉ sharing view" : "◎ show patient";
    if (S.videoOn) {
      bindSelfView();
      if (!S.camReady) {
        S.videoOn = false;
        $("#doc-video").classList.remove("on");
        $("#doc-video").textContent = "◎ show patient";
        banner("Enable the optical feed first — the camera is not running.");
        return;
      }
      logLine("sys", "sharing the camera view with Dr. Aegis");
      pumpVideo();
    }
  }

  function pumpVideo() {
    if (!S.videoOn || !S.open) return;
    var proc = $("#cam-proc");
    if (proc && proc.videoWidth) {
      var c = document.createElement("canvas");
      var w = 640, h = Math.round(640 * proc.videoHeight / proc.videoWidth);
      c.width = w; c.height = h;
      c.getContext("2d").drawImage(proc, 0, 0, w, h);
      var data = c.toDataURL("image/jpeg", 0.6).split(",")[1];
      send({ type: "video", data: data, mime: "image/jpeg" });
    }
    setTimeout(pumpVideo, VIDEO_MS);
  }

  /* ---------------- the agent's hands ---------------- */
  /* The server decides WHAT to do; this decides HOW, against the real shell
     modules. Nothing here is a private copy of the cockpit — `select_patient`
     drives the same AegisPatient the top bar drives, so the agent and the
     clinician are always looking at one system. */
  var PANELS = {
    camera:       function () { return window.AegisCamera; },
    story:        function () { return window.AegisStory; },
    consensus:    function () { return window.AegisConsensus; },
    orchestrator: function () { return window.AegisOrchestrator; },
    federation:   function () { return window.AegisFederation; },
    rbac:         function () { return window.AegisRBAC; },
    dsl:          function () { return window.AegisDSL; },
    pharma:       function () { return window.AegisPharma; },
    identity:     function () { return window.AegisIdentity; }
  };

  /* ---------------- watching it work ---------------- */
  /* The consult overlay is fullscreen, so when the agent opens a panel it lands
     UNDERNEATH and the clinician sees nothing happen. Three pieces fix that:
     a pointer that travels to whatever is being driven, a pulse on the target,
     and the consult stepping aside long enough to watch. It is the difference
     between "it says it opened the causal graph" and seeing the causal graph
     open. */
  function agentPointer() {
    var p = $("#doc-cursor");
    if (!p) {
      p = el("div", "doc-cursor");
      p.id = "doc-cursor";
      p.innerHTML = '<i class="doc-cursor-ring"></i><i class="doc-cursor-dot"></i>'
        + '<span class="doc-cursor-tag"></span>';
      document.body.appendChild(p);
    }
    return p;
  }

  function pointAt(target, label) {
    var p = agentPointer();
    var r = (typeof target === "string" ? document.querySelector(target) : target);
    var x, y;
    if (r && r.getBoundingClientRect) {
      var b = r.getBoundingClientRect();
      if (!b.width && !b.height) r = null;
      else { x = b.left + b.width / 2; y = b.top + Math.min(b.height / 2, 60); }
    }
    if (!r) { x = window.innerWidth * 0.5; y = window.innerHeight * 0.42; }
    p.querySelector(".doc-cursor-tag").textContent = label || "";
    p.style.transform = "translate(" + Math.round(x) + "px," + Math.round(y) + "px)";
    p.classList.add("on");
    p.classList.remove("tap");
    // reflow so the tap animation restarts even on back-to-back commands
    void p.offsetWidth;
    p.classList.add("tap");
    if (r && r.classList) {
      r.classList.add("doc-target");
      setTimeout(function () { r.classList.remove("doc-target"); }, 2200);
    }
    clearTimeout(S.cursorT);
    S.cursorT = setTimeout(function () { p.classList.remove("on"); }, 4200);
  }

  /* Shrink the consult into a corner so the panel it just opened is visible,
     then come back. Never while it is mid-sentence — losing the caption you are
     reading to watch a panel appear is a bad trade. */
  /* Sticky, not timed. A timer that restored after five seconds snapped the
     consult back over the panel while you were still reading it — which is the
     same bug as never stepping aside, just delayed. It stays docked until you
     ask for it back: tap the strip, or ask another question. */
  function stepAside() {
    var app = $("#doctor-app");
    if (app.classList.contains("hidden")) return;
    app.classList.add("aside");
    clearTimeout(S.asideT);
  }

  function stepBack() {
    var app = $("#doctor-app");
    if (app) app.classList.remove("aside");
  }

  function isAside() {
    var app = $("#doctor-app");
    return !!(app && app.classList.contains("aside"));
  }

  function agentView(text, detail) {
    var v = $("#doc-agentview");
    if (!v) return;
    v.classList.remove("idle");
    $("#doc-agentview-act").textContent = text || "—";
    $("#doc-agentview-detail").textContent = detail || "";
    clearTimeout(S.viewT);
    S.viewT = setTimeout(function () {
      v.classList.add("idle");
      $("#doc-agentview-act").textContent = "idle";
      $("#doc-agentview-detail").textContent = "";
    }, 6000);
  }

  // where each UI command should point on screen
  var POINT_AT = {
    select_patient: "#patient-trigger",
    enable_camera: "#launch-camera",
    open_panel: "#launch-story",
    focus_organ: "#twin-stage",
    note: ".win-aux",
    show_treatment_options: "#launch-camera"
  };

  function showDriving(action, args) {
    args = args || {};
    var label = ({
      open_panel: "opening " + (args.panel || "panel"),
      close_panel: "closing " + (args.panel || "panel"),
      select_patient: "switching patient",
      enable_camera: "starting optical feed",
      focus_organ: "focusing " + (args.organ || "organ"),
      highlight: "pointing something out",
      note: "writing an observation",
      play_treatment_twin: "driving the treatment twin",
      set_camera_sensitivity: "retuning the optical feed",
      show_treatment_options: "pulling up treatment options"
    })[action] || action;

    agentView(label, summarizeArgs(args));
    pointAt(POINT_AT[action] || "#doctor-app", label);
    // Only step aside for things there is actually something to SEE.
    //
    // This used to be skipped while the doctor was speaking, which is precisely
    // when it opens panels — it narrates as it acts — so on a voice call the
    // consult never got out of the way and the panel opened behind it. The
    // caption is no longer a reason to stay fullscreen: the docked strip keeps
    // it readable, so you can watch the panel AND follow what it is saying.
    if (action === "open_panel" || action === "show_treatment_options"
        || action === "play_treatment_twin" || action === "enable_camera") {
      stepAside();
    }
  }

  function runUI(action, args) {
    args = args || {};
    showDriving(action, args);
    try {
      switch (action) {
        case "open_panel":   return openPanel(args.panel, args.bed);
        case "close_panel":  return closePanel(args.panel);
        case "select_patient":
          if (window.AegisPatient) window.AegisPatient.set(args.bed);
          S.bed = args.bed;
          return "switched to " + args.bed;
        case "enable_camera":
          if (window.AegisCamera) window.AegisCamera.enable();
          setTimeout(bindSelfView, 1500);
          return "camera requested";
        case "focus_organ":
          if (window.AegisTwin && window.AegisTwin.setFocus) window.AegisTwin.setFocus(args.organ);
          return "focused " + args.organ;
        case "highlight":
          flash(args.text);
          return "highlighted";
        case "play_treatment_twin":  return driveTwin(args);
        case "set_camera_sensitivity":
          if (window.AegisCamera && window.AegisCamera.setSensitivity) {
            window.AegisCamera.setSensitivity(args.mode);
            return "camera sensitivity → " + args.mode;
          }
          return "optical biosensor unavailable";
        case "note":
          return addNote(args.text);
        case "show_treatment_options":
          // the treatment twin IS the options view — same overlay, scoped to the bed
          if (window.AegisPK) { window.AegisPK.open(args.bed || S.bed || "session"); }
          return "treatment options opened";
        default:
          return "unknown command " + action;
      }
    } catch (e) {
      return "failed: " + (e && e.message);
    }
  }

  function openPanel(panel, bed) {
    if (panel === "consult") return "already open";
    if (panel === "tactical") { window.open("tactical.html", "_blank"); return "tactical opened"; }
    if (panel === "twin") {
      if (window.AegisPK) window.AegisPK.open(bed || S.bed || "session");
      return "treatment twin opened";
    }
    var mod = (PANELS[panel] || function () {})();
    if (mod && mod.open) { mod.open(); return panel + " opened"; }
    return panel + " unavailable";
  }

  function closePanel(panel) {
    if (panel === "consult") { close(); return "consult closed"; }
    if (panel === "twin") { if (window.AegisPK) window.AegisPK.close(); return "twin closed"; }
    var mod = (PANELS[panel] || function () {})();
    if (mod && mod.close) { mod.close(); return panel + " closed"; }
    return panel + " unavailable";
  }

  /* The PK twin is a separate overlay with its own player; open it first if the
     agent reaches for playback before anyone has opened it. */
  function driveTwin(args) {
    var pk = window.AegisPKTwin;
    if (!window.AegisPK) return "treatment twin unavailable";
    if (!window.AegisPK.isOpen || !window.AegisPK.isOpen()) {
      window.AegisPK.open(S.bed || "session");
      if (args.command !== "play") return "treatment twin opened";
    }
    if (!pk) return "treatment twin opened";
    var did = [];
    if (args.arm && pk.setArm) { pk.setArm(args.arm); did.push("arm " + args.arm); }
    if (args.at_hour != null && pk.setTime) { pk.setTime(+args.at_hour); did.push("t=" + args.at_hour + "h"); }
    if (args.rate && pk.setRate) { pk.setRate(+args.rate); did.push(args.rate + "×"); }
    if (args.command === "play" && pk.play) { pk.play(); did.push("playing"); }
    if (args.command === "pause" && pk.pause) { pk.pause(); did.push("paused"); }
    if (args.command === "restart" && pk.setTime) { pk.setTime(0); if (pk.play) pk.play(); did.push("restarted"); }
    return "treatment twin: " + (did.join(", ") || "no change");
  }

  /* An agent-authored observation is marked as such in the log — a note the
     clinician keeps must say who wrote it. */
  function addNote(text) {
    var log = document.querySelector("#notes-log");
    if (!log) return "observations pane unavailable";
    // match the pane's own row markup (.note > .ni + time + .nt) so an agent note
    // is styled like every other observation rather than as unstyled stray text
    var row = el("div", "note");
    row.appendChild(el("span", "ni"));
    row.appendChild(el("time", null, new Date().toLocaleTimeString("en-GB")));
    row.appendChild(el("span", "nt", "Dr. Aegis · " + (text || "")));
    log.prepend(row);
    while (log.children.length > 40) log.removeChild(log.lastChild);
    return "noted";
  }

  function flash(text) {
    var el0 = $("#doc-flash");
    if (!el0) return;
    el0.textContent = text || "";
    el0.classList.remove("hidden");
    clearTimeout(S.flashT);
    S.flashT = setTimeout(function () { el0.classList.add("hidden"); }, 6000);
  }

  /* ---------------- activity feed ---------------- */
  /* Every tool call the agent makes, shown as it makes it. An agent with this
     much reach that works invisibly is not auditable, and this project's whole
     argument is that you can check it. Gate refusals render as refusals. */
  function logAction(a) {
    var feed = $("#doc-acts");
    if (!feed) return;
    var empty = feed.querySelector(".doc-log-empty");
    if (empty) empty.remove();
    var row = el("div", "doc-act " + (a.kind || "read") + (a.ok ? "" : " bad"));
    var head = el("div", "doc-act-h");
    head.appendChild(el("b", null, a.tool));
    head.appendChild(el("span", "doc-act-kind", a.kind || "read"));
    row.appendChild(head);
    var argstr = summarizeArgs(a.args);
    if (argstr) row.appendChild(el("div", "doc-act-args", argstr));
    var v = a.verdict;
    if (v) {
      var verdict = el("div", "doc-act-verdict " + (v.allowed ? "ok" : "deny"));
      verdict.textContent = (v.allowed ? "✓ certified · " : "✕ " + (v.errno || "denied") + " · ")
        + (v.message || "");
      row.appendChild(verdict);
    } else if (!a.ok && a.error) {
      row.appendChild(el("div", "doc-act-verdict deny", a.error));
    }
    feed.appendChild(row);
    feed.scrollTop = feed.scrollHeight;
    var n = $("#doc-act-count");
    if (n) n.textContent = feed.querySelectorAll(".doc-act").length;
  }

  function summarizeArgs(args) {
    if (!args) return "";
    var out = [];
    for (var k in args) {
      if (!Object.prototype.hasOwnProperty.call(args, k)) continue;
      var v = args[k];
      out.push(k + "=" + (typeof v === "object" ? JSON.stringify(v) : String(v)));
    }
    return out.join(" · ").slice(0, 120);
  }

  /* ---------------- typed consult (no mic, or voice unavailable) --------- */
  function askText() {
    var input = $("#doc-input");
    var q = (input.value || "").trim();
    if (!q) return;
    input.value = "";
    stepBack();                 // a new question means they are back with you
    logLine("clinician", q);
    if (S.open) { send({ type: "text", text: q }); return; }
    // Streamed REST: same dossier, same persona, no socket — but each tool is
    // announced as it runs, so a multi-step answer shows its work instead of
    // sitting silent for the eight seconds it genuinely takes.
    setState("connecting", "working…");
    phase("thinking");
    caption("", true);
    streamConsult(q, textHistory(), {
      onRunning: function (ev) {
        caption("· " + describeTool(ev.tool, ev.args), true);
      },
      onAction: function (a) {
        logAction({ tool: a.tool, args: a.args, ok: a.ok, kind: a.ui ? "ui" : "read" });
        if (a.ui) logLine("sys", "▸ " + runUI(a.ui.action, a.ui.args));
      },
      onAnswer: function (d) {
        setState("idle", "typed consult");
        phase("standby");
        logLine("doctor", d.answer || "(no answer)");
        caption(d.answer || "");
      },
      onError: function () {
        setState("error", "unreachable");
        phase("standby");
        logLine("sys", "consult service unreachable");
        caption("Could not reach the assistant.", true);
      }
    });
  }

  /* Shared SSE reader for the typed consult. The summon bar uses it too, so both
     typed entry points narrate identically and cannot drift. */
  function streamConsult(question, history, cb, bedOverride) {
    var url = window.AEGIS.http("/aegis/doctor/consult/stream");
    fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question,
                             bed: bedOverride !== undefined ? bedOverride : S.bed,
                             history: history })
    }).then(function (r) {
      if (!r.ok || !r.body) throw new Error("stream unavailable");
      var reader = r.body.getReader(), dec = new TextDecoder(), buf = "";
      function pump() {
        return reader.read().then(function (res) {
          if (res.done) return;
          buf += dec.decode(res.value, { stream: true });
          var parts = buf.split("\n\n");
          buf = parts.pop();                       // keep the incomplete tail
          parts.forEach(function (block) {
            var line = block.split("\n").find(function (l) { return l.indexOf("data: ") === 0; });
            if (!line) return;
            var ev;
            try { ev = JSON.parse(line.slice(6)); } catch (e) { return; }
            if (ev.type === "running" && cb.onRunning) cb.onRunning(ev);
            else if (ev.type === "action" && cb.onAction) cb.onAction(ev);
            else if (ev.type === "answer" && cb.onAnswer) cb.onAnswer(ev);
            else if (ev.type === "error" && cb.onError) cb.onError(ev);
          });
          return pump();
        });
      }
      return pump();
    }).catch(function () { if (cb.onError) cb.onError({}); });
  }

  var TOOL_VERB = {
    list_patients: "checking the unit", get_patient: "reading the chart",
    get_patient_history: "pulling the prior record",
    get_causal_graph: "tracing the causal graph", run_consensus: "convening consensus",
    show_treatment_options: "pulling up treatment options",
    get_vitals_trend: "reading the trend", select_patient: "switching patient",
    open_panel: "opening a panel", get_validation: "checking how good this model is",
    syscall: "asking the gate", counterfactual_replay: "running the counterfactual"
  };
  function describeTool(name, args) {
    var v = TOOL_VERB[name] || name.replace(/_/g, " ");
    if (name === "open_panel" && args && args.panel) v = "opening the " + args.panel;
    return v + "…";
  }

  function textHistory() {
    var out = [];
    document.querySelectorAll("#doc-log .doc-line").forEach(function (l) {
      var role = l.dataset.role;
      if (role === "clinician") out.push({ role: "user", text: l.querySelector(".doc-text").textContent });
      else if (role === "doctor") out.push({ role: "model", text: l.querySelector(".doc-text").textContent });
    });
    return out.slice(-12);
  }

  /* ---------------- the voice ring ---------------- */
  /* Neutral by rule (see doctor.css): it shows who is talking, which is a device
     state, not a clinical one. Outer arc = the doctor, inner = your mic. */
  /* An instrument bezel, not a light show.
     Three layers: a static tick ring (the bezel), one sweeping arc whose speed
     says what the assistant is doing, and a waveform ring whose amplitude is
     REAL — the doctor's actual output level from an analyser on the playback
     graph, and your actual mic level on the way in. It is monochrome by rule:
     saturated colour in this product means a patient is in trouble, so the
     assistant's own chrome never borrows it. */
  function startOrb() {
    var cv = $("#doc-orb-canvas");
    if (!cv) return;
    var ctx = cv.getContext("2d");
    var spin = 0, sweep = 0;

    function frame() {
      S.raf = requestAnimationFrame(frame);
      var dpr = window.devicePixelRatio || 1;
      var w = cv.clientWidth, h = cv.clientHeight;
      if (!w || !h) return;
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      var cx = w / 2, cy = h / 2, base = Math.min(w, h) / 2 - 6;
      var css = getComputedStyle(document.documentElement);
      var faint = css.getPropertyValue("--faint").trim() || "#4b4f56";
      var dim = css.getPropertyValue("--dim").trim() || "#6d7178";
      var muted = css.getPropertyValue("--muted").trim() || "#a0a4ac";
      var ink = css.getPropertyValue("--ink").trim() || "#f2f3f5";

      var p = S.phase;
      var busy = (p === "connecting" || p === "waking");
      var live = (p === "listening" || p === "thinking" || p === "speaking");
      var amp = 0;
      if (p === "speaking") amp = outLevel();
      else if (p === "listening") amp = Math.min(1, S.level * 2.4);

      spin += busy ? 0.055 : (p === "thinking" ? 0.028 : 0.006);
      sweep += busy ? 0.09 : 0.035;

      // ---- bezel: fixed ticks, so motion elsewhere has something to move against
      ctx.strokeStyle = faint;
      ctx.lineWidth = 1;
      ctx.globalAlpha = live ? 0.55 : 0.3;
      for (var i = 0; i < 60; i++) {
        var a0 = (i / 60) * Math.PI * 2;
        var long = i % 5 === 0;
        var r0 = base, r1 = base - (long ? 6 : 3);
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a0) * r0, cy + Math.sin(a0) * r0);
        ctx.lineTo(cx + Math.cos(a0) * r1, cy + Math.sin(a0) * r1);
        ctx.stroke();
      }

      // ---- sweeping arc: the "it is doing something" signal
      var arcR = base - 11;
      ctx.globalAlpha = busy ? 0.9 : (live ? 0.6 : 0.25);
      ctx.strokeStyle = busy ? muted : dim;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, arcR, spin, spin + (busy ? 1.1 : 0.55));
      ctx.stroke();
      ctx.globalAlpha *= 0.45;
      ctx.beginPath();
      ctx.arc(cx, cy, arcR, spin + Math.PI, spin + Math.PI + (busy ? 0.7 : 0.35));
      ctx.stroke();

      // ---- waveform rings: amplitude is real audio, not decoration
      ctx.lineWidth = 1;
      for (var r = 0; r < 3; r++) {
        // amplitude scales with the ring, not in fixed pixels — at the small orb
        // size a fixed 13px wobble braided the three rings into a knot
        var a1 = live ? base * (0.012 + amp * 0.085 + r * amp * 0.022) : base * 0.01;
        ctx.beginPath();
        for (var a = 0; a <= Math.PI * 2 + 0.02; a += 0.055) {
          var wob = Math.sin(a * (3 + r) + sweep * (1 + r * 0.35)) * a1;
          var rad = base - 20 - r * 8 + wob;
          var x = cx + Math.cos(a) * rad, y = cy + Math.sin(a) * rad;
          if (a === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = p === "speaking" ? ink : (live ? muted : dim);
        ctx.globalAlpha = (p === "speaking" ? 0.8 : live ? 0.45 : 0.22) - r * 0.13;
        ctx.stroke();
      }

      // ---- mic ring, inside: reports the input device only
      if (S.open && S.micOn && S.level > 0.01) {
        ctx.beginPath();
        ctx.arc(cx, cy, base * 0.34 + S.level * 22, 0, Math.PI * 2);
        ctx.strokeStyle = muted;
        ctx.globalAlpha = Math.min(0.45, S.level * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      paintLevel();
    }
    frame();
  }

  /* RMS of what is actually coming out of the speaker. Without this the ring
     "speaks" as a fixed animation whenever a packet arrives, which looks alive
     but tells you nothing; with it, the ring tracks the voice. */
  function outLevel() {
    if (!S.analyser) return 0.35;
    var buf = S.analyBuf || (S.analyBuf = new Uint8Array(S.analyser.fftSize));
    S.analyser.getByteTimeDomainData(buf);
    var sum = 0;
    for (var i = 0; i < buf.length; i++) {
      var v = (buf[i] - 128) / 128;
      sum += v * v;
    }
    var rms = Math.sqrt(sum / buf.length);
    S.outSmooth = (S.outSmooth || 0) * 0.6 + rms * 0.4;
    return Math.min(1, S.outSmooth * 3.2);
  }

  function stopOrb() { if (S.raf) { cancelAnimationFrame(S.raf); S.raf = null; } }

  function paintLevel() {
    var bars = document.querySelectorAll("#doc-level i");
    for (var i = 0; i < bars.length; i++) {
      var t = (i + 1) / bars.length;
      var on = S.open && S.micOn && S.level > t * 0.45;
      bars[i].style.height = (on ? 4 + t * 14 : 4) + "px";
      bars[i].classList.toggle("hot", on);
    }
  }

  /* ---------------- wiring ---------------- */
  document.addEventListener("click", function (e) {
    if (e.target.closest("#launch-doctor")) { open(); }
    else if (e.target.closest("#doc-close")) { close(); }
    else if (e.target.closest("#doc-call")) { callStart(); }
    else if (e.target.closest("#doc-end")) { hangup(); }
    else if (e.target.closest("#doc-mic")) {
      S.micOn = !S.micOn;
      var b = $("#doc-mic");
      b.classList.toggle("muted", !S.micOn);
      b.textContent = S.micOn ? "◉ mic" : "◌ muted";
    }
    else if (e.target.closest("#doc-video")) { toggleVideoShare(); }
    else if (e.target.closest("#doc-cam")) { toggleCamera(); }
    else if (e.target.closest("#doc-send")) { askText(); }
    // tapping the docked strip (anywhere that is not a control) brings it back
    else if (isAside() && e.target.closest("#doctor-app")
             && !e.target.closest("input") && !e.target.closest("button")) {
      stepBack();
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && e.target && e.target.id === "doc-input") askText();
    if (e.key === "Escape" && !$("#doctor-app").classList.contains("hidden")) close();
  });

  // Follow the shell's patient selector: if the bed changes while the panel is
  // open but idle, re-ground. Mid-call the server's own watcher handles it.
  document.addEventListener("aegis:patient", function (e) {
    S.bed = (e.detail && e.detail.id) || S.bed;   // detail is the patient record
    if (!$("#doctor-app").classList.contains("hidden") && !S.open) loadBrief();
  });

  // runUI is exported so the summon bar drives the cockpit through the SAME
  // handler the call does — two entry points, one implementation, no drift.
  window.AegisDoctor = { open: open, close: close, call: callStart, hangup: hangup,
                         runUI: runUI, streamConsult: streamConsult,
                         describeTool: describeTool, _state: S };
})();
