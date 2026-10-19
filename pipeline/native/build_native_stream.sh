#!/usr/bin/env bash
# Build cr_battle_stream — a static Linux x86_64 ELF that runs INSIDE the MuMu
# emulator (Android/Linux kernel, invoked via `su 0`). This is a cross-
# compilation from Windows: MSVC cannot target Linux ELF, so the build goes
# through WSL gcc (already installed; no extra toolchain introduced).
# Usage: bash pipeline/native/build_native_stream.sh  (from the workspace root)
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$NATIVE_DIR/build"

# map Git Bash /d/... to WSL /mnt/d/...
WSL_DIR="$(printf '%s' "$NATIVE_DIR" | sed 's|^/\([a-z]\)/|/mnt/\1/|')"

echo "wsl gcc: src=$WSL_DIR/cr_battle_stream.c"
wsl.exe -e bash -c "set -e; gcc -static -O3 -std=c17 -s -Wall -Wextra \
  -o '$WSL_DIR/build/cr_battle_stream' '$WSL_DIR/cr_battle_stream.c'"

ls -l "$NATIVE_DIR/build/cr_battle_stream"
file "$NATIVE_DIR/build/cr_battle_stream" 2>/dev/null || true
