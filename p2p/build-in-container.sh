#!/bin/bash
# Container entrypoint: extract open-gpu-kernel-modules, apply the cmpunlocker-p2p patch set
# exactly as cmpunlocker's driver/build.sh does, compile the modules, write them to /out.
# It never installs or loads anything. Inputs (read-only mounts):
#   /src/cmpunlocker  cmpunlocker-p2p tree     /cache/<tarball>   NVIDIA source tarball
#   $SYSSRC           kernel headers for $KVER
# Env: KVER, SYSSRC, DRIVER_VERSION, PROFILE (8gb), JOBS (8), PREP_ONLY (0|1)
set -euo pipefail

: "${KVER:?}" "${SYSSRC:?}" "${DRIVER_VERSION:?}"
PROFILE=${PROFILE:-8gb}
JOBS=${JOBS:-8}
PREP_ONLY=${PREP_ONLY:-0}
SRC=/src/cmpunlocker
TARBALL=/cache/open-gpu-kernel-modules-${DRIVER_VERSION}.tar.gz
WORK=/build
TREE=${WORK}/open-gpu-kernel-modules-${DRIVER_VERSION}
OUT=/out
LOG=${OUT}/build.log

log() { echo "[build] $*" | tee -a "${LOG}"; }
die() { echo "[build] FAIL: $*" | tee -a "${LOG}" >&2; exit 1; }

mkdir -p "${OUT}/modules"
: > "${LOG}"
grep -qx "${DRIVER_VERSION}" "${SRC}/driver/VERSION" || die "driver ${DRIVER_VERSION} not in cmpunlocker driver/VERSION"
[[ -f ${TARBALL} ]] || die "missing ${TARBALL}"
[[ -f ${SYSSRC}/Makefile && -f ${SYSSRC}/Module.symvers ]] || die "kernel headers not usable at ${SYSSRC}"

# PATCH_ORDER is taken from the tree's own driver/build.sh so this script cannot drift from it.
mapfile -t PATCH_ORDER < <(python3 - "${SRC}/driver/build.sh" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r"PATCH_ORDER=\(\n(.*?)\n\)", t, re.S)
print("\n".join(l.strip() for l in m.group(1).splitlines() if l.strip()))
PY
)
log "patch order: ${PATCH_ORDER[*]}"
[[ " ${PATCH_ORDER[*]} " == *" p2p-unlock.patch "* ]] || die "p2p-unlock.patch not in PATCH_ORDER"
# The BAR1-resize patch hung our board (Gigabyte G481-HA0, two PLX switches) at boot; P2P does not need it.
if [[ ${ALLOW_BAR1_RESIZE:-0} != 1 ]]; then
  [[ " ${PATCH_ORDER[*]} " != *" bar1-resize-unlock.patch "* ]] || die "bar1-resize-unlock.patch is in PATCH_ORDER (set ALLOW_BAR1_RESIZE=1 to build it anyway)"
fi

# Same validation cmpunlocker runs before building.
CONSTANTS_ENV="$(python3 "${SRC}/tools/read-constants.py" "${SRC}/common/constants.yaml" \
    "${SRC}/driver/patches" "${SRC}/driver/build.sh" "${PROFILE}")" || die "constants.yaml rejected"
eval "${CONSTANTS_ENV}"
log "profile ${PROFILE}: label=${UNLOCK_LABEL} skip_geometry_rewrite=${SKIP_GEOMETRY_REWRITE}"

rm -rf "${TREE}"
tar -xzf "${TARBALL}" -C "${WORK}"
[[ -d ${TREE} ]] || die "extracted tree not found"
cd "${TREE}"
for name in "${PATCH_ORDER[@]}"; do
    p="${SRC}/driver/patches/${name}"
    if [[ ${name} == p2p-unlock.patch ]]; then
        # Morrowmake's own test applies this one with zero fuzz; require the same.
        patch -p1 -F0 --dry-run --batch < "${p}" >> "${LOG}" 2>&1 || die "${name} does not apply with -F0"
    fi
    log "apply ${name} ($(sha256sum "${p}" | cut -c1-16))"
    patch -p1 --batch < "${p}" >> "${LOG}" 2>&1 || die "patch ${name} failed"
done
if find . -name '*.rej' | grep -q .; then die "reject files present"; fi

GSP_C="${TREE}/src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c"
# Geometry step copied verbatim from cmpunlocker driver/build.sh.
if [[ "${SKIP_GEOMETRY_REWRITE}" -ne 1 ]]; then
python3 - "${GSP_C}" "${CFG1}" "${LMR}" "${FB_BYTES}" "${UNLOCK_LABEL}" <<'PY' | tee -a "${LOG}"
import pathlib, re, sys
path, cfg1, lmr, fb, label = sys.argv[1:6]
text = pathlib.Path(path).read_text()
if (
    "SEC2_POSTBL_TIMING_CMP_170HX_8GB_PCI_DEVICE_ID" in text
    and "SEC2_POSTBL_TIMING_CMP_170HX_10GB_PCI_DEVICE_ID" in text
    and "0x02779000U" in text
    and "0x02669000U" in text
    and "0x0000001000000000ULL" in text
    and "0x0000000A00000000ULL" in text
):
    print(f"runtime device-id geometry (profile metadata={label})")
    raise SystemExit(0)

text2, n1 = re.subn(r"(NvU32 cfg1Value = )0x[0-9A-Fa-f]+(U;)", rf"\g<1>{cfg1}\g<2>", text, count=1)
text2, n2 = re.subn(r"(NvU32 lmrValue\s*=\s*)0x[0-9A-Fa-f]+(U;)", rf"\g<1>{lmr}\g<2>", text2, count=1)
text2, n3 = re.subn(r"(NvU64 targetFbBytes = )0x[0-9A-Fa-f]+ULL;\s*/\*[^*]*\*/", rf"\g<1>{fb}ULL;  /* {label} */", text2, count=1)
if n1 != 1 or n2 != 1 or n3 != 1:
    raise SystemExit(f"geometry rewrite failed (cfg1={n1} lmr={n2} fb={n3}); check kernel_gsp.c markers")
pathlib.Path(path).write_text(text2)
print(f"cfg1={cfg1} lmr={lmr} fb={fb} ({label})")
PY
fi

# Source-level proof that the P2P pieces are in (checked again on the binary below).
grep -q 'TRAP31_PLM' "${GSP_C}" || die "TRAP31_PLM entry missing from kernel_gsp.c"
grep -q '_cmpP2PTrapArm' src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm200.c || die "trap arm missing"
grep -q 'NvU32 trapPlm = GPU_REG_RD32(pGpu, 0x0012277cU);' src/nvidia/src/kernel/gpu/gpu.c || die "caps gate missing"
log "patched tree OK"
[[ ${PREP_ONLY} == 1 ]] && { log "PREP_ONLY=1: stopping before compile"; exit 0; }

log "gcc: $(gcc --version | head -1)"
log "kernel built with: $(grep CONFIG_CC_VERSION_TEXT "${SYSSRC}/.config")"
log "make -j${JOBS} modules SYSSRC=${SYSSRC}"
start=$(date +%s)
make -j"${JOBS}" modules SYSSRC="${SYSSRC}" CC=gcc >> "${LOG}" 2>&1 || die "make failed (see build.log)"
log "compile took $(( $(date +%s) - start )) s"

mapfile -t KOS < <(find "${TREE}" -type f \( -name nvidia.ko -o -name nvidia-modeset.ko -o -name nvidia-uvm.ko \
    -o -name nvidia-drm.ko -o -name nvidia-peermem.ko \) ! -path '*/conftest/*' | sort -u)
[[ ${#KOS[@]} -ge 4 ]] || die "expected >=4 modules, got ${#KOS[@]}"
rm -f "${OUT}"/modules/*.ko
for ko in "${KOS[@]}"; do install -m 0644 "${ko}" "${OUT}/modules/"; done

# Binary checks: right kernel, right driver, P2P strings present.
NV="${OUT}/modules/nvidia.ko"
[[ $(modinfo -F vermagic "${NV}" | cut -d' ' -f1) == "${KVER}" ]] || die "vermagic mismatch"
[[ $(modinfo -F version "${NV}") == "${DRIVER_VERSION}" ]] || die "module version mismatch"
grep -aq 'CMPUNLOCK: PCIe P2P caps forced to OK' "${NV}" || die "P2P caps string missing from nvidia.ko"
grep -aq 'TRAP31_PLM' "${NV}" || die "TRAP31_PLM string missing from nvidia.ko"

{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kver=${KVER} driver=${DRIVER_VERSION} profile=${PROFILE} unlock_geometry=${UNLOCK_LABEL}"
  echo "tarball_sha256=$(sha256sum "${TARBALL}" | cut -d' ' -f1)"
  echo "gcc=$(gcc --version | head -1)"
  echo "patch_order=${PATCH_ORDER[*]}"
  for name in "${PATCH_ORDER[@]}"; do echo "patch_sha256 ${name} $(sha256sum "${SRC}/driver/patches/${name}" | cut -d' ' -f1)"; done
  for ko in "${OUT}"/modules/*.ko; do
    echo "module $(basename "${ko}") version=$(modinfo -F version "${ko}") srcversion=$(modinfo -F srcversion "${ko}") vermagic='$(modinfo -F vermagic "${ko}")'"
  done
} > "${OUT}/BUILD-INFO.txt"
(cd "${OUT}/modules" && sha256sum ./*.ko) > "${OUT}/SHA256SUMS"
cat "${OUT}/BUILD-INFO.txt" "${OUT}/SHA256SUMS" | tee -a "${LOG}"
log "done"
