-- The kernel journal: one ordered, hash-chained record of what the OS did.
--
-- Until this is applied, the journal still works — it writes to the local SQLite
-- file — but on Render the disk is ephemeral, so the chain restarts with every
-- deploy and every spin-down. Applying this is what makes "the chain survives a
-- reboot" true in production rather than only on a developer's machine.
--
-- Apply: Supabase dashboard -> SQL editor -> run. RLS is on with no policies, the
-- same deny-all posture as every other table here; the backend reaches it with
-- the service-role key.
--
-- Volume: writes, denials, auth and lifecycle only — never reads. At the current
-- tick rates that is a few thousand rows a day, and the retention job below keeps
-- 30 days of detail.

create table if not exists aegis_journal (
  hash        text primary key,
  seq         bigint not null,
  ts_us       bigint not null,
  facility    text not null,
  severity    int not null,
  event       text not null,
  uid         text not null,
  pid         bigint not null default 0,
  detail      jsonb not null default '{}'::jsonb,
  prev_hash   text not null,
  created_at  timestamptz default now()
);

create index if not exists aegis_journal_seq on aegis_journal (seq);
create index if not exists aegis_journal_facility on aegis_journal (facility, seq);
create index if not exists aegis_journal_created on aegis_journal (created_at);

alter table aegis_journal enable row level security;   -- deny-all to anon/authenticated

-- Retention. Deleting old rows BREAKS the chain for those entries by design: the
-- chain proves nothing was silently altered within what is retained, and the head
-- is what carries forward. Keep 30 days.
create or replace function aegis_journal_retain() returns void
language sql security definer as $$
  delete from aegis_journal where created_at < now() - interval '30 days';
$$;

revoke all on function aegis_journal_retain() from public, anon, authenticated;
grant execute on function aegis_journal_retain() to service_role;

select cron.schedule('aegis-journal-retain', '23 * * * *',
                     $$ select aegis_journal_retain(); $$)
where not exists (select 1 from cron.job where jobname = 'aegis-journal-retain');
