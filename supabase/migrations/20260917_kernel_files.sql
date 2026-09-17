-- The kernel's persistent files: /home/<user>/… and /etc/policy.dsl.
--
-- Until this is applied the file store still works — it writes the local SQLite
-- file — but on Render the disk is ephemeral, so a clinician's saved scripts and an
-- edited safety policy would vanish at the next deploy or spin-down.
--
-- Apply: Supabase dashboard -> SQL editor -> run. Deny-all RLS like every other
-- table here; the backend reaches it with the service-role key.
--
-- Size: files are capped at 64 KiB each and 1 MiB / 256 files per user in the
-- kernel (kernel/vfs.py), so this table stays small by construction.

create table if not exists aegis_files (
  owner    text not null,
  path     text not null,
  content  text,
  is_dir   int  not null default 0,
  size     int  not null default 0,
  mtime    double precision not null,
  primary key (owner, path)
);

create index if not exists aegis_files_owner on aegis_files (owner);

alter table aegis_files enable row level security;   -- deny-all to anon/authenticated
