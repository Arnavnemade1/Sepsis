import os
import json
import sqlite3
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load env variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class DatabaseClient:
    def __init__(self):
        self.use_supabase = bool(SUPABASE_URL and SUPABASE_KEY)
        self.supabase_client = None
        self.sqlite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sepsis_twin.db")
        
        if self.use_supabase:
            try:
                from supabase import create_client, Client
                self.supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("[DATABASE] Supabase client initialized successfully.")
            except Exception as e:
                print(f"[DATABASE] Failed to initialize Supabase ({e}). Falling back to SQLite.")
                self.use_supabase = False
                self._init_sqlite()
        else:
            print("[DATABASE] Supabase credentials missing. Utilizing local SQLite database.")
            self._init_sqlite()

    def _init_sqlite(self):
        """Initializes local SQLite database and creates tables mirroring Supabase."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            clinical_history TEXT NOT NULL, -- JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            heart_rate REAL NOT NULL,
            hrv REAL NOT NULL,
            spo2 REAL NOT NULL,
            skin_temp REAL NOT NULL,
            motion REAL NOT NULL,
            sweat_lactate REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sepsis_alerts (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            hdc_similarity REAL NOT NULL,
            sirs_score INTEGER NOT NULL,
            qsofa_score INTEGER NOT NULL,
            triggered_rules TEXT NOT NULL, -- JSON list
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS treatment_protocols (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            alert_id TEXT,
            fluid_rate REAL NOT NULL,
            vasopressor_dose REAL NOT NULL,
            antibiotic_selection TEXT NOT NULL,
            evolution_fitness REAL NOT NULL,
            verification_status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
            FOREIGN KEY (alert_id) REFERENCES sepsis_alerts(id) ON DELETE SET NULL
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS verification_proofs (
            id TEXT PRIMARY KEY,
            protocol_id TEXT NOT NULL,
            formula_evaluated TEXT NOT NULL,
            z3_status TEXT NOT NULL,
            proof_model TEXT, -- JSON string
            certified_by TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (protocol_id) REFERENCES treatment_protocols(id) ON DELETE CASCADE
        );
        """)
        
        conn.commit()
        conn.close()
        print(f"[DATABASE] SQLite database initialized at {self.sqlite_path}")

    # --- Save Methods ---
    
    def save_patient(self, patient_id: str, name: str, age: int, gender: str, clinical_history: Dict[str, Any]) -> str:
        history_json = json.dumps(clinical_history)
        
        if self.use_supabase:
            try:
                res = self.supabase_client.table("patients").upsert({
                    "id": patient_id, "name": name, "age": age, "gender": gender, "clinical_history": clinical_history
                }).execute()
                return patient_id
            except Exception as e:
                print(f"[DATABASE] Supabase save_patient failed: {e}. Falling back to SQLite.")
                
        # SQLite Fallback
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO patients (id, name, age, gender, clinical_history) VALUES (?, ?, ?, ?, ?)",
            (patient_id, name, age, gender, history_json)
        )
        conn.commit()
        conn.close()
        return patient_id

    def save_telemetry(self, patient_id: str, vitals: Dict[str, float]) -> None:
        if self.use_supabase:
            try:
                self.supabase_client.table("telemetry_logs").insert({
                    "patient_id": patient_id,
                    "heart_rate": vitals["heart_rate"],
                    "hrv": vitals["hrv"],
                    "spo2": vitals["spo2"],
                    "skin_temp": vitals["skin_temp"],
                    "motion": vitals["motion"],
                    "sweat_lactate": vitals.get("sweat_lactate")
                }).execute()
                return
            except Exception as e:
                print(f"[DATABASE] Supabase save_telemetry failed: {e}")
                
        # SQLite Fallback
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO telemetry_logs (patient_id, heart_rate, hrv, spo2, skin_temp, motion, sweat_lactate) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (patient_id, vitals["heart_rate"], vitals["hrv"], vitals["spo2"], vitals["skin_temp"], vitals["motion"], vitals.get("sweat_lactate"))
        )
        conn.commit()
        conn.close()

    def save_alert(
        self, 
        alert_id: str,
        patient_id: str, 
        risk_level: str, 
        hdc_similarity: float, 
        sirs_score: int, 
        qsofa_score: int, 
        triggered_rules: List[str]
    ) -> None:
        if self.use_supabase:
            try:
                self.supabase_client.table("sepsis_alerts").upsert({
                    "id": alert_id,
                    "patient_id": patient_id,
                    "risk_level": risk_level,
                    "hdc_similarity": hdc_similarity,
                    "sirs_score": sirs_score,
                    "qsofa_score": qsofa_score,
                    "triggered_rules": triggered_rules
                }).execute()
                return
            except Exception as e:
                print(f"[DATABASE] Supabase save_alert failed: {e}")

        # SQLite Fallback
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO sepsis_alerts (id, patient_id, risk_level, hdc_similarity, sirs_score, qsofa_score, triggered_rules) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (alert_id, patient_id, risk_level, hdc_similarity, sirs_score, qsofa_score, json.dumps(triggered_rules))
        )
        conn.commit()
        conn.close()

    def save_protocol(
        self,
        protocol_id: str,
        patient_id: str,
        alert_id: Optional[str],
        fluid_rate: float,
        vasopressor_dose: float,
        antibiotic_selection: str,
        evolution_fitness: float,
        verification_status: str
    ) -> None:
        if self.use_supabase:
            try:
                self.supabase_client.table("treatment_protocols").upsert({
                    "id": protocol_id,
                    "patient_id": patient_id,
                    "alert_id": alert_id,
                    "fluid_rate": fluid_rate,
                    "vasopressor_dose": vasopressor_dose,
                    "antibiotic_selection": antibiotic_selection,
                    "evolution_fitness": evolution_fitness,
                    "verification_status": verification_status
                }).execute()
                return
            except Exception as e:
                print(f"[DATABASE] Supabase save_protocol failed: {e}")

        # SQLite Fallback
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO treatment_protocols (id, patient_id, alert_id, fluid_rate, vasopressor_dose, antibiotic_selection, evolution_fitness, verification_status) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (protocol_id, patient_id, alert_id, fluid_rate, vasopressor_dose, antibiotic_selection, evolution_fitness, verification_status)
        )
        conn.commit()
        conn.close()

    def save_verification_proof(
        self,
        proof_id: str,
        protocol_id: str,
        formula_evaluated: str,
        z3_status: str,
        proof_model: Optional[Dict[str, Any]],
        certified_by: str = "Z3_SMT_PROVER"
    ) -> None:
        model_json = json.dumps(proof_model) if proof_model else None
        
        if self.use_supabase:
            try:
                self.supabase_client.table("verification_proofs").upsert({
                    "id": proof_id,
                    "protocol_id": protocol_id,
                    "formula_evaluated": formula_evaluated,
                    "z3_status": z3_status,
                    "proof_model": proof_model,
                    "certified_by": certified_by
                }).execute()
                return
            except Exception as e:
                print(f"[DATABASE] Supabase save_verification_proof failed: {e}")

        # SQLite Fallback
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO verification_proofs (id, protocol_id, formula_evaluated, z3_status, proof_model, certified_by) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (proof_id, protocol_id, formula_evaluated, z3_status, model_json, certified_by)
        )
        conn.commit()
        conn.close()

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if self.use_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").eq("id", patient_id).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                print(f"[DATABASE] Supabase get_patient failed: {e}")
                
        # SQLite Fallback
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            patient = dict(row)
            patient["clinical_history"] = json.loads(patient["clinical_history"])
            return patient
        return None
