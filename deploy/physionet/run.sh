#!/bin/bash
# On the instance: `sudo deploy/physionet/run.sh up`, `... status`, `... doctor`.
#
# The login comes from Parameter Store into a file only root can read, on a memory-backed
# filesystem, for the life of this command. Never on a command line, never in the image.
set -euo pipefail
REGION="$(TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); \
  curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)"
ENVFILE="$(mktemp -p /dev/shm)"
trap 'rm -f "$ENVFILE"' EXIT
chmod 600 "$ENVFILE"
{
  printf 'PHYSIONET_USERNAME=%s\n' "$(aws ssm get-parameter --region "$REGION" --name /physionet/username --query Parameter.Value --output text)"
  printf 'PHYSIONET_PASSWORD=%s\n' "$(aws ssm get-parameter --region "$REGION" --name /physionet/password --with-decryption --query Parameter.Value --output text)"
} > "$ENVFILE"
# This machine does nothing else: one booster thread per physical core (m7i cores are
# hyperthreaded), the cloud scale (every negative it can afford, more tuning), and a time budget
# the planner fits into. HOURS=30 at ~$0.81/h caps the compute near $25.
THREADS="${GENESIS_THREADS:-$(( $(nproc) / 2 ))}"
exec docker run --rm --env-file "$ENVFILE" -e GENESIS_THREADS="$THREADS" -v /srv/lake:/lake aegis-physionet \
  --scale "${SCALE:-cloud}" --hours "${HOURS:-30}" "$@"
