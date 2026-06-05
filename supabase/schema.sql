-- Supabase Schema for HyperSepsisTwin-Ω
-- Enables real-time tracking of patient state, telemetry logs, sepsis alerts, protocols, and verification proofs

-- Create tables
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    clinical_history JSONB NOT NULL DEFAULT '{}'::jsonb, -- e.g. {"chf": true, "copd": false}
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    id BIGSERIAL PRIMARY KEY,
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE NOT NULL,
    heart_rate NUMERIC(5, 2) NOT NULL,
    hrv NUMERIC(5, 2) NOT NULL,
    spo2 NUMERIC(5, 2) NOT NULL,
    skin_temp NUMERIC(4, 2) NOT NULL,
    motion NUMERIC(5, 2) NOT NULL,
    sweat_lactate NUMERIC(4, 2),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE TABLE IF NOT EXISTS sepsis_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE NOT NULL,
    risk_level TEXT NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH'
    hdc_similarity NUMERIC(6, 4) NOT NULL,
    sirs_score INTEGER NOT NULL,
    qsofa_score INTEGER NOT NULL,
    triggered_rules TEXT[] NOT NULL DEFAULT '{}'::text[],
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE TABLE IF NOT EXISTS treatment_protocols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE NOT NULL,
    alert_id UUID REFERENCES sepsis_alerts(id) ON DELETE SET NULL,
    fluid_rate NUMERIC(5, 2) NOT NULL, -- mL/kg/hr
    vasopressor_dose NUMERIC(5, 3) NOT NULL, -- mcg/kg/min (Norepinephrine)
    antibiotic_selection TEXT NOT NULL,
    evolution_fitness NUMERIC(6, 2) NOT NULL,
    verification_status TEXT NOT NULL, -- 'PENDING', 'VERIFIED', 'REJECTED'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_proofs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    protocol_id UUID REFERENCES treatment_protocols(id) ON DELETE CASCADE NOT NULL,
    formula_evaluated TEXT NOT NULL,
    z3_status TEXT NOT NULL, -- 'sat', 'unsat', 'unknown'
    proof_model JSONB, -- contains SMT model bindings or counterexamples
    certified_by TEXT NOT NULL DEFAULT 'Z3_SMT_PROVER',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_telemetry_patient_timestamp ON telemetry_logs (patient_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_patient_timestamp ON sepsis_alerts (patient_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_protocols_patient_created ON treatment_protocols (patient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_proofs_protocol_id ON verification_proofs (protocol_id);
