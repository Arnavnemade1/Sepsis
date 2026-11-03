#!/bin/bash
# First boot: Docker, the repository, the image. It starts no job and fetches no data.
set -euo pipefail
dnf install -y docker git tmux
systemctl enable --now docker
usermod -aG docker ssm-user 2>/dev/null || true
mkdir -p /srv/aegis /srv/lake
chown 1000:1000 /srv/lake
chmod 700 /srv/lake
git clone --depth 1 "${repo_url}" /srv/aegis/Sepsis
docker build -f /srv/aegis/Sepsis/deploy/physionet/Dockerfile -t aegis-physionet /srv/aegis/Sepsis
