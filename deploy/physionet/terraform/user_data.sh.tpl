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
# Idle = no job container for two hours, then stop (a stop keeps the disk and the lake).
cat > /usr/local/sbin/aegis-idle-stop <<'IDLE'
#!/bin/bash
mark=/run/aegis-idle-since
if [ -n "$(docker ps -q)" ]; then rm -f "$mark"; exit 0; fi
# a job holding the one heavy-job lock is not idle, even between its containers
if [ -x /usr/local/sbin/aegis-heavy ] && /usr/local/sbin/aegis-heavy status | grep -q "^busy"; then rm -f "$mark"; exit 0; fi
[ -f "$mark" ] || { date +%s > "$mark"; exit 0; }
[ $(( $(date +%s) - $(cat "$mark") )) -ge 7200 ] && shutdown -h now "aegis: no job for two hours" || true
IDLE
chmod 755 /usr/local/sbin/aegis-idle-stop
printf '[Service]\nType=oneshot\nExecStart=/usr/local/sbin/aegis-idle-stop\n' > /etc/systemd/system/aegis-idle-stop.service
printf '[Timer]\nOnBootSec=10min\nOnUnitActiveSec=10min\n[Install]\nWantedBy=timers.target\n' > /etc/systemd/system/aegis-idle-stop.timer
systemctl daemon-reload && systemctl enable --now aegis-idle-stop.timer
usermod -aG docker ssm-user 2>/dev/null || true
mkdir -p /srv/aegis /srv/lake
chown 1000:1000 /srv/lake
chmod 700 /srv/lake
git clone --depth 1 "${repo_url}" /srv/aegis/Sepsis
docker build -f /srv/aegis/Sepsis/deploy/physionet/Dockerfile -t aegis-physionet /srv/aegis/Sepsis
