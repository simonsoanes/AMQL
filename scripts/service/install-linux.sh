#!/usr/bin/env bash
# install-linux.sh — installs AMQL as a systemd service: amql-manager's supervisor, which
# runs the current version's manager and applies updates.
#
#   sudo scripts/service/install-linux.sh <amql-<version>-<rid>.tar.gz | published folder>
#
# Environment: ROOT (default /opt/amql), URLS (default http://127.0.0.1:8100), API_KEY
# (generated if unset), SERVICE (default amql), USER_NAME (default amql).
#
#   $ROOT/versions/<version>/   this install
#   $ROOT/current               the version run
#   $ROOT/supervisor/           the service's own copy of amql-manager (never replaced by an update)
#   $ROOT/manager.json          the manager's config (written once; add models to it)
#   $ROOT/data/                 logs, job output, which models were running
set -euo pipefail

SOURCE="${1:?give a release archive (amql-<version>-<rid>.tar.gz) or a published folder}"
ROOT="${ROOT:-/opt/amql}"
URLS="${URLS:-http://127.0.0.1:8100}"
SERVICE="${SERVICE:-amql}"
USER_NAME="${USER_NAME:-amql}"
[ "$(id -u)" -eq 0 ] || { echo "run as root (it creates a user and a systemd unit)" >&2; exit 1; }

work=""
if [ -f "$SOURCE" ]; then
    work="$(mktemp -d)"
    tar -xzf "$SOURCE" -C "$work"
    SOURCE="$work"
    [ "$(ls -A "$work" | wc -l)" -eq 1 ] && [ -d "$work/$(ls -A "$work")" ] && SOURCE="$work/$(ls -A "$work")"
fi
for exe in amql-manager amql-server amql-cli; do
    [ -f "$SOURCE/$exe" ] || { echo "$exe is not in $SOURCE" >&2; exit 1; }
    chmod +x "$SOURCE/$exe"
done
VERSION="$("$SOURCE/amql-manager" version | tr -d '[:space:]')"

id -u "$USER_NAME" >/dev/null 2>&1 || useradd --system --home-dir "$ROOT" --shell /usr/sbin/nologin "$USER_NAME"
# The GPU device nodes belong to the video/render groups on most distributions.
for g in video render; do getent group "$g" >/dev/null && usermod -aG "$g" "$USER_NAME"; done

mkdir -p "$ROOT/versions" "$ROOT/data" "$ROOT/containers"
rm -rf "$ROOT/versions/$VERSION"
cp -a "$SOURCE" "$ROOT/versions/$VERSION"
systemctl stop "$SERVICE" 2>/dev/null || true
rm -rf "$ROOT/supervisor"
cp -a "$SOURCE" "$ROOT/supervisor"
[ -f "$ROOT/current" ] && cp "$ROOT/current" "$ROOT/previous"
printf '%s' "$VERSION" > "$ROOT/current"

print_key=0
if [ ! -f "$ROOT/manager.json" ]; then
    if [ -z "${API_KEY:-}" ]; then
        API_KEY="amql-$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        print_key=1
    fi
    cat > "$ROOT/manager.json" <<EOF
{
  "urls": "$URLS",
  "api_key": "$API_KEY",
  "containers_dir": "$ROOT/containers",
  "models": [],
  "update": { "enabled": true, "auto_apply": true, "interval_hours": 24 }
}
EOF
fi
chown -R "$USER_NAME:$USER_NAME" "$ROOT"
chmod 600 "$ROOT/manager.json"

cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=AMQL inference and model management (amql-manager)
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$ROOT
Environment=AMQL_ROOT=$ROOT
ExecStart=$ROOT/supervisor/amql-manager supervise --root $ROOT
Restart=always
RestartSec=5
TimeoutStopSec=60
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now "$SERVICE"
[ -n "$work" ] && rm -rf "$work"

echo "AMQL $VERSION installed in $ROOT and running as the '$SERVICE' service (journalctl -u $SERVICE)."
echo "Management API: $URLS (see docs/manager.md). Add models in $ROOT/manager.json, then systemctl restart $SERVICE."
[ "$print_key" -eq 1 ] && echo "API key (shown once, also in $ROOT/manager.json): $API_KEY"
