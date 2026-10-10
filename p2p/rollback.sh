#!/bin/bash
# ROLL BACK to the modules + config captured by backup-current.sh (the install before P2P).
#   sudo ./rollback.sh --yes      then cold boot
# Also usable from a boot with module_blacklist=nvidia,... (see docs/p2p.md, recovery).
set -euo pipefail
cd "$(dirname "$0")"
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
[[ ${1:-} == --yes ]] || { echo "refusing without --yes"; exit 2; }
KVER=${KVER:-$(uname -r)}
DST=${MODULE_DIR:-/lib/modules/${KVER}/updates/cmpunlocker}
CONF=${CONF:-/etc/modprobe.d/cmp-pcie-gen2.conf}
B=rollback/current
(cd "$B" && sha256sum -c --quiet SHA256SUMS) || { echo "backup fails checksum"; exit 1; }
for f in "$B"/modules/*; do install -m 0644 "$f" "${DST}/$(basename "$f")"; done
install -m 0644 "$B/etc/$(basename "$CONF")" "$CONF"
(cd "$B/modules" && for f in *; do cmp -s "$f" "${DST}/$f" || { echo "restore mismatch $f"; exit 1; }; done)
depmod -a "$KVER"
update-initramfs -u -k "$KVER"
cat "$CONF"
echo "restored. Cold boot now (power fully off, then on)."
