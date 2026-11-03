#!/bin/bash
# First boot: Docker, the repository, the image. It starts no job and fetches no data.
set -euo pipefail
dnf install -y docker git tmux
systemctl enable --now docker
# Swap on the encrypted root disk, so a free-tier machine (8 GB) survives building the grid.
# The planner never counts it: modelling is fitted into real RAM.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=24576 status=none
  chmod 600 /swapfile && mkswap /swapfile && echo '/swapfile none swap defaults 0 0' >> /etc/fstab
fi
swapon -a
usermod -aG docker ssm-user 2>/dev/null || true
mkdir -p /srv/aegis /srv/lake
chown 1000:1000 /srv/lake
chmod 700 /srv/lake
git clone --depth 1 "${repo_url}" /srv/aegis/Sepsis
docker build -f /srv/aegis/Sepsis/deploy/physionet/Dockerfile -t aegis-physionet /srv/aegis/Sepsis
