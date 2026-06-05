import asyncio
import uuid
import os
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core.hdc import HDCEngine
from backend.core.twin import PatientDigitalTwin
from backend.core.neurosymbolic import NeurosymbolicEngine
from backend.core.evolution import EvolutionaryOptimizer
from backend.core.verifier import ProtocolVerifier
from backend.core.database import DatabaseClient
from backend.core.simulator import PatientTelemetrySimulator

app = FastAPI(title="HyperSepsisTwin-Ω Backend", version="1.0.0")

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
hdc_engine = HDCEngine()
db_client = DatabaseClient()
neuro_engine = NeurosymbolicEngine()
evolution_optimizer = EvolutionaryOptimizer()
verifier = ProtocolVerifier()
simulator = PatientTelemetrySimulator()

# Global state for default patient
DEFAULT_PATIENT_ID = "patient-001"
patient_info = {
    "name": "Jane Doe",
    "age": 67,
    "gender": "Female",
    "clinical_history": {
        "chf": True,                  # Congestive Heart Failure (fluid limit)
        "renal_failure": False,       # Renal Failure (fluid limit)
        "penicillin_allergy": True,   # Penicillin Allergy (antibiotic violation)
        "wbc": 7.5,                   # WBC count (k/uL)
        "gcs": 15                     # Glasgow Coma Scale
    }
}

# Ensure patient exists in DB
db_client.save_patient(
    DEFAULT_PATIENT_ID, 
    patient_info["name"], 
    patient_info["age"], 
    patient_info["gender"], 
    patient_info["clinical_history"]
)

# Initialize patient digital twin
digital_twin = PatientDigitalTwin(DEFAULT_PATIENT_ID, hdc_engine)

# Active treatment applied to the patient
active_treatment = {
    "fluid_rate": 0.0,
    "vasopressor_dose": 0.0,
    "antibiotic_code": 0
}

# Latest evaluation results
latest_evaluation = {
    "risk_level": "LOW",
    "sirs_score": 0,
    "qsofa_score": 0,
    "triggered_rules": []
}

latest_verification_log = {
    "status": "IDLE",
    "message": "Waiting for sepsis alert triggers...",
    "proof_details": {}
}

# WebSocket client registry
connected_clients: List[WebSocket] = []

# Models for API requests
class ClinicalHistoryUpdate(BaseModel):
    chf: bool
    renal_failure: bool
    penicillin_allergy: bool
    wbc: float
    gcs: int

class SimulatorStageUpdate(BaseModel):
    stage: str # 'STABLE', 'DETERIORATING', 'SEPSIS', 'RECOVERING'

async def broadcast_status():
    """Broadcasts current digital twin and telemetry state to all connected WebSockets."""
    if not connected_clients:
        return
        
    twin_state = digital_twin.get_current_twin_state()
    message = {
        "vitals": digital_twin.latest_vitals,
        "healthy_similarity": digital_twin.healthy_similarity,
        "sepsis_similarity": digital_twin.sepsis_similarity,
        "trends": twin_state["trends"],
        "evaluation": latest_evaluation,
        "active_treatment": active_treatment,
        "verification_log": latest_verification_log,
        "patient": {
            "id": DEFAULT_PATIENT_ID,
            "name": patient_info["name"],
            "age": patient_info["age"],
            "gender": patient_info["gender"],
            "clinical_history": patient_info["clinical_history"]
        },
        "simulator_stage": simulator.stage
    }
    
    # Broadcast to all clients
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
            
    for ws in disconnected:
        if ws in connected_clients:
            connected_clients.remove(ws)

async def telemetry_loop():
    """Background loop that drives telemetry gathering, HDC state compilation, and rules evaluation."""
    global latest_evaluation, active_treatment, latest_verification_log
    
    print("[API] Background telemetry monitoring loop started.")
    while True:
        try:
            # 1. Fetch vitals from simulator (or BLE if connected)
            vitals = simulator.next_tick(active_treatment)
            
            # 2. Update Digital Twin (and HDC state vector representation)
            twin_update = digital_twin.update_telemetry(vitals)
            
            # Save vitals to DB
            db_client.save_telemetry(DEFAULT_PATIENT_ID, vitals)

            # 3. Evaluate neurosymbolic clinical rules
            risk_level, sirs, qsofa, rules = neuro_engine.evaluate_patient(
                vitals, 
                twin_update["sepsis_similarity"], 
                patient_info["clinical_history"]
            )
            
            # Save alert change if severity shifts
            alert_id = str(uuid.uuid4())
            if risk_level != latest_evaluation["risk_level"] or risk_level == "HIGH":
                db_client.save_alert(
                    alert_id,
                    DEFAULT_PATIENT_ID,
                    risk_level,
                    twin_update["sepsis_similarity"],
                    sirs,
                    qsofa,
                    rules
                )
                
            latest_evaluation = {
                "risk_level": risk_level,
                "sirs_score": sirs,
                "qsofa_score": qsofa,
                "triggered_rules": rules
            }

            # 4. If Patient is in HIGH Risk Sepsis, run Evolutionary Protocol Synthesis + Z3 verification
            if risk_level == "HIGH":
                print("[API] HIGH SEPSIS RISK DETECTED. Triggering Evolutionary Optimization...")
                
                # Run Genetic Algorithm
                proposed_protocol, fitness = evolution_optimizer.optimize(
                    vitals, 
                    patient_info["clinical_history"], 
                    risk_level
                )
                
                print(f"[API] Evolutionary Optimizer proposed: {proposed_protocol} (Fitness: {fitness:.2f})")
                
                # Formally verify the proposed treatment parameters using Z3
                is_safe, msg, proof_details = verifier.verify_protocol(
                    patient_info["clinical_history"],
                    vitals,
                    risk_level,
                    proposed_protocol
                )
                
                protocol_id = str(uuid.uuid4())
                proof_id = str(uuid.uuid4())
                
                # Save protocol and verifier logs
                db_client.save_protocol(
                    protocol_id,
                    DEFAULT_PATIENT_ID,
                    alert_id,
                    proposed_protocol["fluid_rate"],
                    proposed_protocol["vasopressor_dose"],
                    verifier.ANTIBIOTICS.get(proposed_protocol["antibiotic_code"], "Unknown"),
                    fitness,
                    "VERIFIED" if is_safe else "REJECTED"
                )
                
                db_client.save_verification_proof(
                    proof_id,
                    protocol_id,
                    proof_details.get("formula", ""),
                    proof_details.get("z3_status", ""),
                    proof_details
                )
                
                latest_verification_log = {
                    "status": "VERIFIED" if is_safe else "REJECTED",
                    "message": msg,
                    "proof_details": proof_details,
                    "proposed_protocol": proposed_protocol
                }

                if is_safe:
                    print(f"[API] Z3 CERTIFIED PROTOCOL: {proposed_protocol}. Applying to patient...")
                    active_treatment = proposed_protocol
                else:
                    print(f"[API] Z3 REJECTED PROTOCOL: {msg}. Safety lockout engaged. Retrying next tick.")
                    # Keep active treatment at current level (or zero it if unsafe)
            else:
                # If risk level drops to LOW or MEDIUM and patient is stable, gradually taper treatment
                if risk_level == "LOW" and active_treatment["fluid_rate"] > 0:
                    active_treatment = {
                        "fluid_rate": max(0.0, active_treatment["fluid_rate"] - 1.0),
                        "vasopressor_dose": max(0.0, active_treatment["vasopressor_dose"] - 0.05),
                        "antibiotic_code": active_treatment["antibiotic_code"] # maintain active course
                    }
                    latest_verification_log = {
                        "status": "IDLE",
                        "message": "Sepsis resolved. Tapering resuscitation protocols.",
                        "proof_details": {}
                    }
            
            # 5. Broadcast live telemetry data to Web UI
            await broadcast_status()

        except Exception as e:
            print(f"[API] Error in background telemetry loop: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(1.5)

@app.on_event("startup")
async def startup_event():
    # Start background loop task
    asyncio.create_task(telemetry_loop())

# --- REST ENDPOINTS ---

@app.get("/patient")
def get_patient():
    return {
        "id": DEFAULT_PATIENT_ID,
        "name": patient_info["name"],
        "age": patient_info["age"],
        "gender": patient_info["gender"],
        "clinical_history": patient_info["clinical_history"]
    }

@app.post("/patient/history")
def update_clinical_history(update: ClinicalHistoryUpdate):
    global patient_info
    patient_info["clinical_history"] = {
        "chf": update.chf,
        "renal_failure": update.renal_failure,
        "penicillin_allergy": update.penicillin_allergy,
        "wbc": update.wbc,
        "gcs": update.gcs
    }
    
    # Save patient update
    db_client.save_patient(
        DEFAULT_PATIENT_ID, 
        patient_info["name"], 
        patient_info["age"], 
        patient_info["gender"], 
        patient_info["clinical_history"]
    )
    
    # Force digital twin state update
    print(f"[API] Patient history updated: {patient_info['clinical_history']}")
    return {"status": "SUCCESS", "clinical_history": patient_info["clinical_history"]}

@app.post("/patient/stage")
def update_simulator_stage(update: SimulatorStageUpdate):
    if update.stage not in ["STABLE", "DETERIORATING", "SEPSIS", "RECOVERING"]:
        raise HTTPException(status_code=400, detail="Invalid stage.")
        
    simulator.set_stage(update.stage)
    
    # If resetting, reset active treatments as well
    global active_treatment
    if update.stage == "STABLE":
        active_treatment = {
            "fluid_rate": 0.0,
            "vasopressor_dose": 0.0,
            "antibiotic_code": 0
        }
        
    return {"status": "SUCCESS", "stage": simulator.stage}

@app.post("/protocol/optimize")
def manual_optimize():
    """Manually triggers evolutionary protocol optimization and verification."""
    global active_treatment, latest_verification_log
    
    vitals = digital_twin.latest_vitals
    risk_level = latest_evaluation["risk_level"]
    
    proposed_protocol, fitness = evolution_optimizer.optimize(
        vitals, 
        patient_info["clinical_history"], 
        risk_level
    )
    
    is_safe, msg, proof_details = verifier.verify_protocol(
        patient_info["clinical_history"],
        vitals,
        risk_level,
        proposed_protocol
    )
    
    latest_verification_log = {
        "status": "VERIFIED" if is_safe else "REJECTED",
        "message": msg,
        "proof_details": proof_details,
        "proposed_protocol": proposed_protocol
    }
    
    if is_safe:
        active_treatment = proposed_protocol
        
    return {
        "proposed_protocol": proposed_protocol,
        "fitness": fitness,
        "is_safe": is_safe,
        "message": msg,
        "proof_details": proof_details
    }

# --- WEBSOCKET CHANNELS ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] Web client connected. Total clients: {len(connected_clients)}")
    
    try:
        while True:
            # Keep connection alive by listening for message (heartbeats)
            data = await websocket.receive_text()
            # If client sends a ping, respond with pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"[WS] Web client disconnected. Total clients: {len(connected_clients)}")
    except Exception as e:
        print(f"[WS] WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# Mount static files for dashboard frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"[API] Mounted frontend static files from: {frontend_dir}")
else:
    print(f"[API] Warning: Frontend directory not found at {frontend_dir}")
