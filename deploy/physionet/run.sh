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
# This machine does nothing else: one booster thread per physical core on a big machine (m7i
# cores are hyperthreaded), every vCPU on a small one; 80% of RAM for the planner; the cloud
# scale (every negative it can afford, more tuning); and a time budget the planner fits into.
# HOURS=30 caps an m7i.4xlarge (~$0.81/h) near $25; a free-tier m7i-flex.large is ~$0.10/h.
N="$(nproc)"
THREADS="${GENESIS_THREADS:-$(( N > 4 ? N / 2 : N ))}"
exec docker run --rm --env-file "$ENVFILE" -e GENESIS_THREADS="$THREADS" \
  -e MIMIC_RAM_FRACTION="${MIMIC_RAM_FRACTION:-0.8}" -v /srv/lake:/lake aegis-physionet \
  --scale "${SCALE:-cloud}" --hours "${HOURS:-30}" --parallel "${PARALLEL:-4}" "$@"
