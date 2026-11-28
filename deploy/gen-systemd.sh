#!/usr/bin/env bash
# Generate & install systemd units for crypto-chatbot (scalp_v1 cadence, SCALP-SPEC-v1 §11).
# Usage: sudo deploy/gen-systemd.sh
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="$(stat -c %U "$APP_DIR")"
UNIT_DIR=/etc/systemd/system

# name | OnCalendar (UTC) | command (in deploy/bin)
# Scalp v1 cadence (§11):
#   - alerts run (scalp): *:00/5:20
#   - settle-setups --strategy scalp_v1: every 1 min
#   - contract-stats poll: *:01/5:00 (before alerts)
#   - universe refresh: weekly Monday 00:30
#   - digest: daily 00:05 (07:05 WIB)
JOBS=(
  "alerts|*:00/5:20|cc alerts run"
  "settle|*:*:00|cc settle-setups --strategy scalp_v1"
  "cstats|*:01/5:00|cc-universe contract-stats poll --interval 5m --limit 12"
  "news|*:10/30:00|cc sync-news"
  "funding|00/8:12:00|cc-universe sync-gate-funding --days 3"
  "feargreed|*-*-* 00:20:00|cc sync-fear-greed"
  "universe|Mon *-*-* 00:30:00|cc universe refresh --strategy scalp_v1"
  "digest|*-*-* 00:05:00|cc alerts digest"
  "backup|*-*-* 18:30:00|cc backup-db"
)

cat > "$UNIT_DIR/cc-bot.service" <<EOF
[Unit]
Description=crypto-chatbot telegram bot
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/deploy/bin/cc run-bot
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

for job in "${JOBS[@]}"; do
  IFS='|' read -r name cal cmd <<< "$job"
  cat > "$UNIT_DIR/cc-$name.service" <<EOF
[Unit]
Description=crypto-chatbot $name
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=oneshot
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/deploy/bin/$cmd
Nice=5
TimeoutStartSec=4min
EOF
  cat > "$UNIT_DIR/cc-$name.timer" <<EOF
[Unit]
Description=crypto-chatbot $name timer

[Timer]
OnCalendar=$cal UTC
Persistent=true
AccuracySec=1s

[Install]
WantedBy=timers.target
EOF
done

systemctl daemon-reload
systemctl enable --now cc-bot.service
for job in "${JOBS[@]}"; do systemctl enable --now "cc-${job%%|*}.timer"; done
systemctl list-timers 'cc-*' --no-pager
