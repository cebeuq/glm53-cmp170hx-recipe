#!/bin/bash
# Snapshot the CURRENTLY INSTALLED (cmpunlocker) NVIDIA modules + driver config into p2p/rollback/current/.
# Read-only on the system (copies files out). Safe to run any time.  usage: sudo ./backup-current.sh
set -euo pipefail
cd "$(dirname "$0")"
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
KVER=${KVER:-$(uname -r)}
SRC=${MODULE_DIR:-/lib/modules/${KVER}/updates/cmpunlocker}
CONF=${CONF:-/etc/modprobe.d/cmp-pcie-gen2.conf}
DST=rollback/current
if [[ -f ${DST}/SHA256SUMS ]]; then
  echo "${DST} already exists; verifying instead of overwriting"
  (cd "${DST}" && sha256sum -c --quiet SHA256SUMS) && echo "backup OK"; exit $?
fi
mkdir -p "${DST}/modules" "${DST}/etc"
cp -a "${SRC}"/. "${DST}/modules/"
cp -a "$CONF" "${DST}/etc/"
{
  echo "taken_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) kver=${KVER}"
  echo "proc_version=$(head -1 /proc/driver/nvidia/version)"
  echo "loaded_srcversion=$(cat /sys/module/nvidia/srcversion)"
  echo "params_registry=$(grep RegistryDwords: /proc/driver/nvidia/params)"
  echo "modinfo_resolves=$(modinfo -k "${KVER}" -n nvidia)"
} > "${DST}/STATE.txt"
(cd "${DST}" && find modules etc -type f -print0 | sort -z | xargs -0 sha256sum) > "${DST}/SHA256SUMS"
chown -R "${SUDO_UID:-1000}:${SUDO_GID:-1000}" rollback
cat "${DST}/STATE.txt" "${DST}/SHA256SUMS"
