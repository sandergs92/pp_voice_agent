#!/usr/bin/env bash
# install.sh — set up pp_voice_agent systemd user service
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_FILE="$REPO_DIR/pp-tts.service"

echo "pp_voice_agent installer"
echo "  repo:   $REPO_DIR"
echo "  units:  $SYSTEMD_USER_DIR"
echo ""

# Verify the service file exists
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "FATAL: $SERVICE_FILE not found" >&2
    exit 1
fi

# Verify tts-server.py exists
if [[ ! -f "$REPO_DIR/tts-server.py" ]]; then
    echo "FATAL: $REPO_DIR/tts-server.py not found" >&2
    exit 1
fi

# Verify the project venv has sherpa_onnx
if ! "$REPO_DIR/.venv/bin/python3" -c "import sherpa_onnx" 2>/dev/null; then
    echo "FATAL: sherpa_onnx not importable from $REPO_DIR/.venv" >&2
    exit 1
fi

mkdir -p "$SYSTEMD_USER_DIR"
ln -sf "$SERVICE_FILE" "$SYSTEMD_USER_DIR/pp-tts.service"
echo "  linked pp-tts.service → $SERVICE_FILE"

# Verify symlink resolves
if [[ ! -e "$SYSTEMD_USER_DIR/pp-tts.service" ]]; then
    echo "WARN: symlink is broken" >&2
fi

chmod +x "$REPO_DIR/tts-server.py" "$REPO_DIR/tts-client.sh" 2>/dev/null || true

systemctl --user daemon-reload

echo ""
echo "Done. To start TTS daemon:"
echo "  systemctl --user enable --now pp-tts.service"
echo ""
echo "To test:"
echo "  echo 'Hello from PinePhone' | $REPO_DIR/tts-client.sh"
