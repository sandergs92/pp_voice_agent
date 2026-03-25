#!/usr/bin/env bash
# tts-client.sh — send text to pp-tts daemon
# Usage: echo "Hello world" | ./tts-client.sh
#        ./tts-client.sh "Hello world"
set -euo pipefail
 
SOCKET_PATH="${SOCKET_PATH:-${XDG_RUNTIME_DIR:-/tmp}/pp-tts.sock}"
 
if [[ ! -S "$SOCKET_PATH" ]]; then
    echo "ERR: pp-tts not running (no socket at $SOCKET_PATH)" >&2
    exit 1
fi
 
if [[ $# -gt 0 ]]; then
    text="$*"
else
    read -r text
fi
 
[[ -z "$text" ]] && { echo "ERR: empty text" >&2; exit 1; }
 
# socat: connect, send text + newline, read response, 30s timeout
response=$(printf '%s\n' "$text" | socat -T 30 - UNIX-CONNECT:"$SOCKET_PATH" 2>/dev/null)
 
if [[ "$response" == "OK" ]]; then
    exit 0
else
    echo "$response" >&2
    exit 1
fi
 
