#!/usr/bin/env bash
# install.sh — set up pp_voice_agent systemd user services
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
WP_CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/wireplumber/wireplumber.conf.d"

echo "pp_voice_agent installer"
echo "  repo:   $REPO_DIR"
echo "  units:  $SYSTEMD_USER_DIR"
echo ""

# Verify critical files exist
for f in tts-server.py services/pp-tts.service services/pp-llm.service services/pp-denoise.service; do
    if [[ ! -f "$REPO_DIR/$f" ]]; then
        echo "FATAL: $REPO_DIR/$f not found" >&2
        exit 1
    fi
done

# Verify the project venv has sherpa_onnx
if ! "$REPO_DIR/.venv/bin/python3" -c "import sherpa_onnx" 2>/dev/null; then
    echo "FATAL: sherpa_onnx not importable from $REPO_DIR/.venv" >&2
    exit 1
fi

# Build denoise_source if needed
if [[ ! -f "$REPO_DIR/denoise/denoise_source" ]]; then
    echo "  building denoise_source..."
    make -C "$REPO_DIR/denoise" denoise_source
fi

# ── Install systemd user services ──
mkdir -p "$SYSTEMD_USER_DIR"

for svc in pp-tts.service pp-llm.service pp-denoise.service pp-denoise-default.service; do
    ln -sf "$REPO_DIR/services/$svc" "$SYSTEMD_USER_DIR/$svc"
    echo "  linked $svc"
done

# ── Install WirePlumber config (clean_mic as default source) ──
if [[ -f "$REPO_DIR/audio/50-clean-mic-default.conf" ]]; then
    mkdir -p "$WP_CONF_DIR"
    ln -sf "$REPO_DIR/audio/50-clean-mic-default.conf" "$WP_CONF_DIR/50-clean-mic-default.conf"
    echo "  linked WirePlumber clean_mic config"
fi

chmod +x "$REPO_DIR/tts-server.py" "$REPO_DIR/tts-client.sh" \
         "$REPO_DIR/audio/fix-audio.sh" 2>/dev/null || true

systemctl --user daemon-reload

echo ""
echo "Done. To enable all services:"
echo "  systemctl --user enable --now pp-tts pp-llm pp-denoise pp-denoise-default"
echo ""
echo "To test:"
echo "  echo 'Hello from PinePhone' | $REPO_DIR/tts-client.sh"
echo "  wpctl status | grep Sources   # verify Clean Mic is default"
