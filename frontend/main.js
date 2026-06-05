// HyperSepsisTwin-Ω Control Console Driver

let socket = null;
let telemetryChart = null;
const chartLimit = 30;
let timeTicks = [];
let hrData = [];
let spo2Data = [];
let mapData = [];

// HDC Trajectory Trail History
let trajectoryHistory = [];
const maxTrailLength = 20;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    connectWebSocket();
    setupFormHandlers();
});

// 1. Establish WebSocket Connection
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host || 'localhost:8000';
    const wsUrl = `${wsProtocol}//${wsHost}/ws`;
    
    console.log(`[WS] Connecting to ${wsUrl}...`);
    const badge = document.getElementById("connection-badge");
    badge.innerText = "CONNECTING...";
    badge.className = "connection-status";

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("[WS] Connection established.");
        badge.innerText = "ONLINE";
        badge.className = "connection-status connected";
        document.getElementById("system-status-dot").classList.add("active");
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        } catch (err) {
            console.error("[WS] Error parsing message:", err);
        }
    };

    socket.onclose = () => {
        console.warn("[WS] Socket disconnected. Reconnecting in 3 seconds...");
        badge.innerText = "OFFLINE (RETRYING)";
        badge.className = "connection-status";
        document.getElementById("system-status-dot").classList.remove("active");
        setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = (error) => {
        console.error("[WS] WebSocket error:", error);
    };
}

// 2. Setup Form Handlers for Invariants
function setupFormHandlers() {
    const form = document.getElementById("history-form");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        
        const payload = {
            chf: document.getElementById("chf-toggle").checked,
            renal_failure: document.getElementById("renal-toggle").checked,
            penicillin_allergy: document.getElementById("penicillin-allergy-toggle").checked,
            wbc: parseFloat(document.getElementById("wbc-input").value),
            gcs: parseInt(document.getElementById("gcs-input").value)
        };

        const host = window.location.host || 'localhost:8000';
        const protocol = window.location.protocol;
        
        fetch(`${protocol}//${host}/patient/history`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            console.log("[REST] Invariants updated:", data);
            showConsoleMessage(`[SYSTEM] Safety invariants updated: CHF=${payload.chf}, Renal=${payload.renal_failure}, Penicillin Allergy=${payload.penicillin_allergy}`);
        })
        .catch(err => console.error("[REST] Error updating history:", err));
    });
}

// 3. Set Patient Physiology State
function setPatientStage(stage) {
    const host = window.location.host || 'localhost:8000';
    const protocol = window.location.protocol;
    
    fetch(`${protocol}//${host}/patient/stage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage: stage })
    })
    .then(res => res.json())
    .then(data => {
        console.log(`[REST] Patient stage set to ${stage}:`, data);
        showConsoleMessage(`[SYSTEM] Manually setting patient physiology to stage: ${stage}`);
    })
    .catch(err => console.error("[REST] Error setting stage:", err));
}

// 4. Update DOM Elements with Socket Telemetry
function updateDashboard(data) {
    // A. Patient Demographics & Stage
    document.getElementById("patient-name").innerText = data.patient.name;
    document.getElementById("patient-age-gender").innerText = `${data.patient.age} / ${data.patient.gender}`;
    
    const stageIndicator = document.getElementById("current-stage-text");
    stageIndicator.innerText = data.simulator_stage;
    stageIndicator.className = `${data.simulator_stage.toLowerCase()}-text`;
    
    // Auto sync check boxes on initial load
    if (!document.activeElement || document.activeElement.tagName !== 'INPUT') {
        document.getElementById("chf-toggle").checked = data.patient.clinical_history.chf;
        document.getElementById("renal-toggle").checked = data.patient.clinical_history.renal_failure;
        document.getElementById("penicillin-allergy-toggle").checked = data.patient.clinical_history.penicillin_allergy;
        document.getElementById("wbc-input").value = data.patient.clinical_history.wbc;
        document.getElementById("gcs-input").value = data.patient.clinical_history.gcs;
    }

    // B. Vitals Panels
    const v = data.vitals;
    document.getElementById("val-hr").innerText = Math.round(v.heart_rate);
    document.getElementById("val-spo2").innerText = v.spo2.toFixed(1);
    document.getElementById("val-temp").innerText = v.skin_temp.toFixed(1);
    document.getElementById("val-map").innerText = Math.round(v.map);
    document.getElementById("val-bp").innerText = `BP: ${Math.round(v.systolic_bp)}/${Math.round(v.diastolic_bp)}`;
    document.getElementById("val-lactate").innerText = v.sweat_lactate ? v.sweat_lactate.toFixed(1) : "N/A";
    document.getElementById("val-hrv").innerText = Math.round(v.hrv);

    // Vitals Trends
    document.getElementById("trend-hr").innerText = v.heart_rate > 90 ? "TACHYCARDIA" : "NORMAL";
    document.getElementById("trend-hr").className = `vital-trend ${v.heart_rate > 90 ? 'danger-text' : 'stable-text'}`;
    
    document.getElementById("trend-spo2").innerText = v.spo2 < 92 ? "HYPOXIC" : "NORMAL";
    document.getElementById("trend-spo2").className = `vital-trend ${v.spo2 < 92 ? 'danger-text' : 'stable-text'}`;
    
    document.getElementById("trend-temp").innerText = data.trends.temp;
    document.getElementById("trend-temp").className = `vital-trend ${data.trends.temp !== 'STABLE' ? 'warning-text' : 'stable-text'}`;
    
    document.getElementById("trend-hrv").innerText = data.trends.hrv;
    document.getElementById("trend-hrv").className = `vital-trend ${data.trends.hrv !== 'STABLE' ? 'warning-text' : 'stable-text'}`;
    
    document.getElementById("trend-lactate").innerText = v.sweat_lactate > 2.0 ? "ELEVATED" : "NORMAL";
    document.getElementById("trend-lactate").className = `vital-trend ${v.sweat_lactate > 2.0 ? 'danger-text' : 'stable-text'}`;

    // C. Update Charts
    updateChart(new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'}), v.heart_rate, v.spo2, v.map);

    // D. Neurosymbolic Risk Status & Rules
    const evalData = data.evaluation;
    const banner = document.getElementById("risk-alert-banner");
    const statusText = document.getElementById("risk-status");
    
    banner.className = `risk-banner ${evalData.risk_level.toLowerCase()}-risk`;
    statusText.innerText = `${evalData.risk_level} RISK`;
    
    document.getElementById("sirs-score").innerText = `${evalData.sirs_score} / 4`;
    document.getElementById("qsofa-score").innerText = `${evalData.qsofa_score} / 3`;

    // Triggered rules list
    const rulesList = document.getElementById("triggered-rules-list");
    rulesList.innerHTML = "";
    if (evalData.triggered_rules.length === 0) {
        rulesList.innerHTML = '<li class="empty-rules">No clinical rules triggered. Patient physiology is stable.</li>';
    } else {
        evalData.triggered_rules.forEach(rule => {
            const li = document.createElement("li");
            li.innerText = rule;
            // Set styling based on severity
            if (rule.includes("SIRS")) li.style.borderLeftColor = "var(--accent-amber)";
            else if (rule.includes("qSOFA")) li.style.borderLeftColor = "var(--accent-red)";
            else if (rule.includes("HDC")) li.style.borderLeftColor = "var(--accent-blue)";
            rulesList.appendChild(li);
        });
    }

    // E. HDC Similarity & Trajectory Map
    document.getElementById("val-healthy-sim").innerText = data.healthy_similarity.toFixed(4);
    document.getElementById("bar-healthy-sim").style.width = `${Math.max(0, data.healthy_similarity) * 100}%`;
    
    document.getElementById("val-sepsis-sim").innerText = data.sepsis_similarity.toFixed(4);
    document.getElementById("bar-sepsis-sim").style.width = `${Math.max(0, data.sepsis_similarity) * 100}%`;

    updateTrajectoryMap(data.healthy_similarity, data.sepsis_similarity);

    // F. Treatment Resuscitation Protocol
    const treatment = data.active_treatment;
    document.getElementById("prop-fluid").innerText = treatment.fluid_rate.toFixed(2);
    document.getElementById("prop-vaso").innerText = treatment.vasopressor_dose.toFixed(3);
    
    let abName = "None";
    if (treatment.antibiotic_code === 1) abName = "Pip/Tazo (Penicillin)";
    else if (treatment.antibiotic_code === 2) abName = "Vancomycin (MRSA)";
    else if (treatment.antibiotic_code === 3) abName = "Cefepime (Gram-Neg)";
    else if (treatment.antibiotic_code === 4) abName = "Meropenem (Carbapenem)";
    else if (treatment.antibiotic_code === 5) abName = "Levofloxacin";
    document.getElementById("prop-antibiotic").innerText = abName;

    // G. Z3 Solver Proof Verifier Output
    const z3 = data.verification_log;
    const z3Badge = document.getElementById("z3-badge");
    const formulaBox = document.getElementById("formula-box");
    
    z3Badge.innerText = z3.status;
    z3Badge.className = `verification-badge ${z3.status.toLowerCase()}`;
    
    if (z3.proof_details && z3.proof_details.formula) {
        formulaBox.innerText = z3.proof_details.formula;
    } else {
        formulaBox.innerText = "Assert(fluid_rate == proposed) ∧ Assert(vaso_dose == proposed) ∧ Assert(ab_selection == proposed)...";
    }

    if (z3.status !== "IDLE") {
        showConsoleMessage(z3.message, z3.status === "VERIFIED");
    }
}

// 5. Console Message Logger
function showConsoleMessage(msg, isVerified = true) {
    const logBox = document.getElementById("verifier-log");
    const timestamp = new Date().toLocaleTimeString();
    const prefix = isVerified ? "[Z3_SMT_PROVER:OK]" : "[Z3_SMT_PROVER:VIOLATION]";
    
    // Check if duplicate to prevent flooding
    const lines = logBox.innerText.split("\n");
    if (lines[0] && lines[0].includes(msg)) return;

    const color = isVerified ? "#00ff87" : "#ff4757";
    logBox.innerHTML = `<span style="color: ${color}">${timestamp} ${prefix} ${msg}</span><br>` + logBox.innerHTML;
}

// 6. Chart.js Configurations
function initChart() {
    const ctx = document.getElementById("telemetryChart").getContext("2d");
    
    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeTicks,
            datasets: [
                {
                    label: 'Heart Rate (bpm)',
                    data: hrData,
                    borderColor: '#ff4757',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'SpO2 (%)',
                    data: spo2Data,
                    borderColor: '#00d2ff',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y1'
                },
                {
                    label: 'MAP (mmHg)',
                    data: mapData,
                    borderColor: '#a55eea',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#a4b0be', font: { size: 9 } }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#ff4757', font: { size: 10 } },
                    min: 40,
                    max: 160
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#00d2ff', font: { size: 10 } },
                    min: 80,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f1f2f6', font: { size: 9, family: 'Outfit' } },
                    position: 'top'
                }
            }
        }
    });
}

function updateChart(label, hr, spo2, map) {
    if (!telemetryChart) return;
    
    timeTicks.push(label);
    hrData.push(hr);
    spo2Data.push(spo2);
    mapData.push(map);

    if (timeTicks.length > chartLimit) {
        timeTicks.shift();
        hrData.shift();
        spo2Data.shift();
        mapData.shift();
    }

    telemetryChart.update();
}

// 7. HDC Trajectory Map Drawing
function updateTrajectoryMap(healthySim, sepsisSim) {
    const canvas = document.getElementById("trajectoryMap");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    const w = canvas.width;
    const h = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = "#030508";
    ctx.fillRect(0, 0, w, h);
    
    // Draw grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Map similarities to coordinates:
    // Left side (x = 30) is HEALTHY (healthySim = 1.0)
    // Right side (x = w - 30) is SEPSIS (sepsisSim = 1.0)
    // Dynamic coordinate mapping:
    // Healthy Similarity governs how far left, Sepsis similarity governs how far right/up
    // Let's model coordinate:
    // x coordinate: center + (sepsisSim - healthySim) * (w/2.5)
    // y coordinate: bottom - (sepsisSim * h * 0.7) - 15
    const centerX = w / 2;
    const xVal = centerX + (sepsisSim - healthySim) * (w / 2.6);
    const yVal = h - 25 - (sepsisSim * (h - 40));

    // Append to trail
    trajectoryHistory.push({ x: xVal, y: yVal });
    if (trajectoryHistory.length > maxTrailLength) {
        trajectoryHistory.shift();
    }

    // Draw reference clusters
    // Healthy Cluster (Green)
    ctx.fillStyle = "rgba(0, 255, 135, 0.15)";
    ctx.beginPath();
    ctx.arc(35, h - 25, 25, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "var(--accent-green)";
    ctx.stroke();
    ctx.fillStyle = "var(--text-secondary)";
    ctx.font = "8px Share Tech Mono";
    ctx.fillText("HEALTHY CLUSTER", 10, h - 8);

    // Sepsis Cluster (Red)
    ctx.fillStyle = "rgba(255, 71, 87, 0.15)";
    ctx.beginPath();
    ctx.arc(w - 35, 25, 25, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "var(--accent-red)";
    ctx.stroke();
    ctx.fillStyle = "var(--text-secondary)";
    ctx.fillText("SEPSIS CLUSTER", w - 85, 12);

    // Draw safety barrier boundary line (at Sepsis Sim = 0.48 threshold)
    // mapped to Y-coordinate
    const thresholdY = h - 25 - (0.48 * (h - 40));
    ctx.strokeStyle = "rgba(255, 71, 87, 0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, thresholdY);
    ctx.lineTo(w, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]); // Reset
    ctx.fillStyle = "rgba(255, 71, 87, 0.6)";
    ctx.fillText("CRITICAL SEPSIS THRESHOLD", 10, thresholdY - 4);

    // Draw historical trajectory path line
    if (trajectoryHistory.length > 1) {
        ctx.beginPath();
        ctx.moveTo(trajectoryHistory[0].x, trajectoryHistory[0].y);
        for (let i = 1; i < trajectoryHistory.length; i++) {
            ctx.lineTo(trajectoryHistory[i].x, trajectoryHistory[i].y);
        }
        ctx.strokeStyle = "rgba(0, 210, 255, 0.45)";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Draw historical path fading dots
    trajectoryHistory.forEach((pt, index) => {
        const opacity = (index + 1) / trajectoryHistory.length;
        ctx.fillStyle = `rgba(0, 210, 255, ${opacity})`;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, index === trajectoryHistory.length - 1 ? 5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
        
        // Glow for current position
        if (index === trajectoryHistory.length - 1) {
            ctx.shadowColor = "#00d2ff";
            ctx.shadowBlur = 10;
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0; // Reset shadow
        }
    });
}
