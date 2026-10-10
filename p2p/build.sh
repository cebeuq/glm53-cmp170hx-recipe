#!/bin/bash
# Build the cmpunlocker + P2P (TRAP31) NVIDIA kernel modules in Docker -> p2p/out/.
# BUILD ONLY: installs nothing, loads nothing, touches no boot config.
#
# usage (on the GPU server, from this directory):
#   sudo CMPUNLOCKER_DIR=/path/to/cmpunlocker ./build.sh [--prep-only] [--force]
#   --prep-only  extract + patch + verify only (seconds, CPU-light)
#   --force      build even while the engine/queue is running (default: refuse; a parallel
#                -j8 compile adds ~100 W and disturbs benchmarks)
# env: CMPUNLOCKER_DIR (required), JOBS (8), CPUSET (all), KVER ($(uname -r)),
#      DRIVER_VERSION (loaded driver), PROFILE (8gb), TARBALL (optional local copy of
#      open-gpu-kernel-modules-<version>.tar.gz; otherwise downloaded from NVIDIA's GitHub),
#      TARBALL_SHA256 (optional expected checksum)
set -euo pipefail
cd "$(dirname "$0")"

PREP_ONLY=0; FORCE=0
for a in "$@"; do
  case $a in
    --prep-only) PREP_ONLY=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown arg $a"; exit 2 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run with sudo (reads kernel headers, runs docker)"; exit 1; }
OWNER="${SUDO_UID:-1000}:${SUDO_GID:-1000}"
export UID_GID=$OWNER
: "${CMPUNLOCKER_DIR:?set CMPUNLOCKER_DIR to a cmpunlocker tree with the TRAP31 P2P patch}"
export CMPUNLOCKER_DIR; CMPUNLOCKER_DIR=$(readlink -f "$CMPUNLOCKER_DIR")

export KVER=${KVER:-$(uname -r)}
export DRIVER_VERSION=${DRIVER_VERSION:-$(modinfo -F version nvidia 2>/dev/null || true)}
export PROFILE=${PROFILE:-8gb}
export JOBS=${JOBS:-8}
export PREP_ONLY
[[ -n $DRIVER_VERSION ]] || { echo "cannot determine driver version; set DRIVER_VERSION"; exit 1; }

# Kernel headers: the -generic dir plus the common dir its symlinks point into.
export HDR_DIR; HDR_DIR=$(readlink -f "/lib/modules/${KVER}/build")
export HDR_COMMON_DIR; HDR_COMMON_DIR=$(dirname "$(readlink -f "${HDR_DIR}/Kbuild")")
[[ -f ${HDR_DIR}/Module.symvers && -d ${HDR_COMMON_DIR} ]] || { echo "kernel headers for ${KVER} missing"; exit 1; }

if [[ $PREP_ONLY == 0 && $FORCE == 0 ]]; then
  busy=""
  pgrep -f 'loop/queue\.sh' >/dev/null && busy+=" queue"
  pgrep -f 'loop/run_candidate\.sh' >/dev/null && busy+=" candidate"
  docker ps --format '{{.Names}}' | grep -qx glm53-vllm && busy+=" engine"
  [[ -z $busy ]] || { echo "refusing to compile while busy:${busy} (use --prep-only now, or --force)"; exit 3; }
fi

TARBALL_NAME="open-gpu-kernel-modules-${DRIVER_VERSION}.tar.gz"
# Known checksum of the tarball this recipe was built and tested with.
declare -A KNOWN_SHA=( [615.71.09]=caf48cda7d5fe7374d1d24ca4a90efb4dc783eefabfad5fa4b64fdc24213b62a )
mkdir -p cache out
if [[ ! -f cache/${TARBALL_NAME} ]]; then
  if [[ -n ${TARBALL:-} && -f ${TARBALL} ]]; then cp "$TARBALL" "cache/${TARBALL_NAME}"
  else curl -L --fail -o "cache/${TARBALL_NAME}.partial" \
         "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/refs/tags/${DRIVER_VERSION}.tar.gz" \
       && mv "cache/${TARBALL_NAME}.partial" "cache/${TARBALL_NAME}"; fi
fi
want=${TARBALL_SHA256:-${KNOWN_SHA[$DRIVER_VERSION]:-}}
got=$(sha256sum "cache/${TARBALL_NAME}" | cut -d' ' -f1)
[[ -z $want || $want == "$got" ]] || { echo "tarball sha256 mismatch: $got != $want"; exit 1; }
chown -R "$OWNER" cache out

echo "build: kver=$KVER driver=$DRIVER_VERSION profile=$PROFILE jobs=$JOBS prep_only=$PREP_ONLY hdr=$HDR_DIR src=$CMPUNLOCKER_DIR"
docker compose config --quiet
nice -n 10 docker compose run --rm --build modbuild
chown -R "$OWNER" out
if [[ $PREP_ONLY == 0 ]]; then
  echo "--- artifacts (out/modules) ---"; cat out/SHA256SUMS
fi
