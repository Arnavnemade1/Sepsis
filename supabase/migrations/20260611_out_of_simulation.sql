-- ============================================================================
--  HyperSepsisTwin-Ω — "out of simulation" migration (2026-06-11)
--  Applied to the Supabase project `oyukwcuwiyiuslmmbvou` (Sepsis).
--
--  Goal: run the live system on REAL ingested sensor data (the optical camera)
--  instead of the in-memory simulator, persist it, and keep database size +
--  egress safely under the free-tier limits (500 MB / 5 GB) with NO loss of
--  displayed fidelity (the live stream is full-resolution over the WebSocket;
--  only raw history older than 7 days is downsampled).
-- ============================================================================

-- 1) Provenance: tag every reading so simulator demo data can never be confused
--    with real sensor data (the backend persists ONLY real data, tagged 'camera').
ALTER TABLE telemetry_logs ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'camera';
ALTER TABLE sepsis_alerts  ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'camera';

-- 2) A contactless camera measures a SUBSET of vitals (HR + HRV reliably). Make
--    the channels it cannot measure nullable instead of fabricating values.
ALTER TABLE telemetry_logs ALTER COLUMN hrv       DROP NOT NULL;
ALTER TABLE telemetry_logs ALTER COLUMN spo2      DROP NOT NULL;
ALTER TABLE telemetry_logs ALTER COLUMN skin_temp DROP NOT NULL;
ALTER TABLE telemetry_logs ALTER COLUMN motion    DROP NOT NULL;

-- 3) Long-term, low-footprint history. 1-minute rollups keep trend fidelity
--    forever at ~1/60th the rows, so raw telemetry can be pruned past 7 days.
CREATE TABLE IF NOT EXISTS telemetry_rollup_1m (
    patient_id    UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    bucket        TIMESTAMP WITH TIME ZONE NOT NULL,
    source        TEXT NOT NULL DEFAULT 'camera',
    n             INTEGER NOT NULL,
    hr_avg        NUMERIC(6,2), hr_min NUMERIC(6,2), hr_max NUMERIC(6,2),
    hrv_avg       NUMERIC(6,2),
    spo2_avg      NUMERIC(6,2),
    skin_temp_avg NUMERIC(5,2),
    motion_avg    NUMERIC(6,2),
    PRIMARY KEY (patient_id, bucket, source)
);
ALTER TABLE telemetry_rollup_1m ENABLE ROW LEVEL SECURITY;   -- locked: backend-only via service-role
CREATE INDEX IF NOT EXISTS idx_rollup_patient_bucket ON telemetry_rollup_1m (patient_id, bucket DESC);

-- 4) Retention: roll up raw telemetry older than 7 days into 1-minute buckets,
--    delete the rolled-up raw, and trim alert events older than 90 days.
--    Idempotent; safe to run repeatedly.
CREATE OR REPLACE FUNCTION aegis_retain() RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  INSERT INTO telemetry_rollup_1m AS r
      (patient_id, bucket, source, n, hr_avg, hr_min, hr_max, hrv_avg, spo2_avg, skin_temp_avg, motion_avg)
  SELECT patient_id,
         date_trunc('minute', timestamp) AS bucket,
         COALESCE(source, 'camera') AS source,
         count(*), avg(heart_rate), min(heart_rate), max(heart_rate),
         avg(hrv), avg(spo2), avg(skin_temp), avg(motion)
  FROM telemetry_logs
  WHERE timestamp < now() - interval '7 days'
  GROUP BY patient_id, date_trunc('minute', timestamp), COALESCE(source, 'camera')
  ON CONFLICT (patient_id, bucket, source) DO UPDATE SET
      hr_avg   = (COALESCE(r.hr_avg,0)*r.n  + COALESCE(EXCLUDED.hr_avg,0)*EXCLUDED.n)/(r.n+EXCLUDED.n),
      hr_min   = LEAST(r.hr_min, EXCLUDED.hr_min),
      hr_max   = GREATEST(r.hr_max, EXCLUDED.hr_max),
      hrv_avg  = (COALESCE(r.hrv_avg,0)*r.n + COALESCE(EXCLUDED.hrv_avg,0)*EXCLUDED.n)/(r.n+EXCLUDED.n),
      spo2_avg = (COALESCE(r.spo2_avg,0)*r.n+ COALESCE(EXCLUDED.spo2_avg,0)*EXCLUDED.n)/(r.n+EXCLUDED.n),
      n        = r.n + EXCLUDED.n;

  DELETE FROM telemetry_logs WHERE timestamp < now() - interval '7 days';
  DELETE FROM sepsis_alerts  WHERE timestamp < now() - interval '90 days';
END;
$$;

-- Security: aegis_retain() is SECURITY DEFINER and prunes data, so it must NOT be
-- reachable from the public REST RPC surface (anon/authenticated). Backend only.
REVOKE EXECUTE ON FUNCTION public.aegis_retain() FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION public.aegis_retain() TO service_role;

-- 5) Automate it: pg_cron runs retention hourly (the backend also calls it hourly
--    as a belt-and-suspenders backup when it is up).
CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA pg_catalog;
DO $$ BEGIN PERFORM cron.unschedule('aegis-retain-hourly'); EXCEPTION WHEN OTHERS THEN NULL; END $$;
SELECT cron.schedule('aegis-retain-hourly', '7 * * * *', $$ SELECT aegis_retain(); $$);
