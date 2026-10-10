#!/bin/bash
# INSTALL the P2P-enabled modules built by build.sh. Read docs/p2p.md first, and have a way to
# recover a machine that does not boot (BMC/IPMI console, or physical access) before you run it.
#   sudo ./install.sh --yes                 stage modules + ForceP2P config; then cold boot
#   sudo ./install.sh --yes --live-reload   same, then try unloading/reloading the driver (no reboot)
# Changes exactly: /lib/modules/$KVER/updates/cmpunlocker/*.ko, $CONF, depmod, initramfs.
# No GRUB/kernel-cmdline change, no BAR1 resize. Undo: sudo ./rollback.sh --yes
# env: KVER ($(uname -r)), MODULE_DIR (/lib/modules/$KVER/updates/cmpunlocker),
#      CONF (/etc/modprobe.d/cmp-pcie-gen2.conf, the file cmpunlocker installed),
#      REGISTRY_DWORDS (the existing cmpunlocker dwords + ForceP2P=0x11),
#      POWERCAP_SERVICE (optional systemd unit that re-applies GPU power/clock caps after a live reload)
set -euo pipefail
cd "$(dirname "$0")"
YES=0; LIVE=0
for a in "$@"; do case $a in --yes) YES=1 ;; --live-reload) LIVE=1 ;; *) echo "unknown arg $a"; exit 2 ;; esac; done
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
[[ $YES == 1 ]] || { echo "refusing without --yes (read docs/p2p.md first)"; exit 2; }
KVER=${KVER:-$(uname -r)}
DST=${MODULE_DIR:-/lib/modules/${KVER}/updates/cmpunlocker}
CONF=${CONF:-/etc/modprobe.d/cmp-pcie-gen2.conf}
DWORDS=${REGISTRY_DWORDS:-RmForceEnableGen2=1;RMPcieLinkSpeed=0x1;ForceP2P=0x11}
LINE="options nvidia NVreg_RegistryDwords=\"${DWORDS}\""

# Preconditions
(cd out/modules && sha256sum -c --quiet ../SHA256SUMS) || { echo "out/ artifacts fail checksum"; exit 1; }
grep -q "kver=${KVER} " out/BUILD-INFO.txt || { echo "artifacts were built for another kernel"; exit 1; }
(cd rollback/current && sha256sum -c --quiet SHA256SUMS) || { echo "no verified backup; run ./backup-current.sh"; exit 1; }
if pgrep -f 'loop/(queue|run_candidate)\.sh' >/dev/null || docker ps --format '{{.Names}}' | grep -qx glm53-vllm; then
  echo "engine/queue still running; stop it first (bash loop/stop_all.sh)"; exit 1
fi

echo "== installing modules into ${DST}"
for ko in out/modules/*.ko; do install -m 0644 "$ko" "${DST}/$(basename "$ko")"; done
(cd out/modules && for ko in *.ko; do cmp -s "$ko" "${DST}/$ko" || { echo "copy mismatch $ko"; exit 1; }; done)
echo "== ${CONF}"
printf '%s\n' "$LINE" > "$CONF"
cat "$CONF"
depmod -a "$KVER"
update-initramfs -u -k "$KVER"
resolved=$(modinfo -k "$KVER" -n nvidia)
echo "modinfo resolves nvidia -> ${resolved}"
[[ $resolved == "${DST}/nvidia.ko" ]] || { echo "WARNING: nvidia does not resolve to ${DST}"; exit 1; }

if [[ $LIVE == 0 ]]; then
  echo "== staged. Next: cold boot (power fully off, then on). Then run the checks in docs/p2p.md."
  exit 0
fi
echo "== live reload"
systemctl stop nvidia-persistenced || true
nvidia-smi -pm 0 >/dev/null 2>&1 || true
for m in nvidia_peermem nvidia_drm nvidia_uvm nvidia_modeset nvidia; do
  grep -q "^$m " /proc/modules && modprobe -r "$m" || true
done
if grep -q '^nvidia ' /proc/modules; then
  echo "could not unload nvidia (in use). Modules are staged; cold boot instead."; exit 3
fi
modprobe nvidia && modprobe nvidia_modeset && modprobe nvidia_uvm && modprobe nvidia_drm
systemctl start nvidia-persistenced || true
[[ -n ${POWERCAP_SERVICE:-} ]] && systemctl restart "$POWERCAP_SERVICE"   # caps reset on reload
grep RegistryDwords: /proc/driver/nvidia/params
nvidia-smi --query-gpu=index,memory.total,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv
echo "If memory.total is not 65536 MiB on every GPU: cold boot (power fully off)."
