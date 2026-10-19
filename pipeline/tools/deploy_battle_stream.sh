#!/usr/bin/env bash
# Push cr_battle_stream to a MuMu instance and run a standby sanity test.
# Standby expectation (no battle running): battle_active:false frames, no crash.
# Usage:
#   bash deploy_battle_stream.sh              # actual_game (default ${CR_ADB_SERIAL:-127.0.0.1:5555})
#   bash deploy_battle_stream.sh 127.0.0.1:16384
set -euo pipefail

SERIAL="${1:-${CR_ADB_SERIAL:-127.0.0.1:5555}}"
ADB="${ADB:-D:/Software/MuMu Player 12/shell/adb.exe}"
LOCAL_BIN="$(cd "$(dirname "$0")/../native/build" && pwd)/cr_battle_stream"
command -v cygpath >/dev/null 2>&1 && LOCAL_BIN="$(cygpath -m "$LOCAL_BIN")"
REMOTE_DIR="/data/local/tmp/crai"
REMOTE_BIN="$REMOTE_DIR/cr_battle_stream"
PKG=nullsroyale.rel.free

[ -f "$LOCAL_BIN" ] || { echo "binary missing: $LOCAL_BIN (build first)" >&2; exit 1; }

export MSYS_NO_PATHCONV=1
"$ADB" -s "$SERIAL" shell "mkdir -p $REMOTE_DIR"
"$ADB" -s "$SERIAL" push "$LOCAL_BIN" "$REMOTE_BIN" >/dev/null
"$ADB" -s "$SERIAL" shell "chmod 755 $REMOTE_BIN"

PID=$("$ADB" -s "$SERIAL" shell "ps -A -o PID,NAME 2>/dev/null | grep -w $PKG" | awk '{print $1}' | head -1)
echo "game pid: ${PID:-<not running>}"
if [ -z "${PID:-}" ]; then
  echo "game not running; launching standby test without a battle anyway is pointless" >&2
  exit 0
fi

echo "--- standby capture (6 s, expect battle_active frames, then ok) ---"
"$ADB" -s "$SERIAL" shell "su 0 sh -c 'timeout 6 $REMOTE_BIN $PID 200; echo EXIT=\$?'" | tee /tmp/crai_standby.txt | head -12
echo "--- frame summary:"
grep -c '"event":"battle_frame"' /tmp/crai_standby.txt || true
grep -o '"battle_active":[a-z]*' /tmp/crai_standby.txt | sort | uniq -c || true
