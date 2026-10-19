#!/system/bin/sh
# P0.1 libg.so runtime fingerprint (pipeline/tools, run via: adb shell "su 0 sh /data/local/tmp/crai_fp_libg.sh")
# Output: process identity, libg.so load base, ELF header (e_machine), sha256, size.
PKG=nullsroyale.rel.free
PID=$(pidof $PKG)
echo "PID=$PID"
[ -z "$PID" ] && { echo GAME_NOT_RUNNING; exit 1; }
echo "--- exe:"
readlink /proc/$PID/exe
echo "--- libg.so mappings (first 6):"
grep libg.so /proc/$PID/maps | head -6
MAPLINE=$(grep libg.so /proc/$PID/maps | head -1)
LIBPATH=$(echo "$MAPLINE" | awk '{print $6}')
echo "LIBPATH=$LIBPATH"
[ -z "$LIBPATH" ] && { echo LIBPATH_NOT_FOUND; exit 1; }
BASE=$(echo "$MAPLINE" | cut -d- -f1)
echo "BASE=0x$BASE"
echo "PERM=$(echo "$MAPLINE" | awk '{print $2}')"
echo "--- ELF header (20 bytes, bytes 18-19 = e_machine LE):"
dd if="$LIBPATH" bs=1 count=20 2>/dev/null | od -An -tx1
echo "--- sha256:"
sha256sum "$LIBPATH"
echo "--- bytes:"
wc -c < "$LIBPATH"
