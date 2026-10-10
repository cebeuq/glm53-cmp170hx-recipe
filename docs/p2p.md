# PCIe peer-to-peer on CMP 170HX

**This replaces NVIDIA kernel modules on your machine.** Read all of it first, make a backup
(`p2p/backup-current.sh`), and have a way to recover a server that does not boot (BMC/IPMI
remote console or physical access). Everything else in this recipe works without P2P; see
[`engine/variants/no-p2p-128k.vars.example`](../engine/variants/no-p2p-128k.vars.example).

## Why

The CMP 170HX firmware reports "P2P not supported" (`nvidia-smi topo -p2p r` shows `GNS` for
every pair), so every collective goes GPU -> host memory -> GPU. TP2 x PP4 runs ~160 small
(12-150 KB) all-reduces per decode step, one after another. With P2P a GPU writes directly
into its peer's memory: NCCL's P2P transport and vLLM's one-kernel custom all-reduce become
usable.

## What the patch does

Morrowmake's TRAP31 change to the [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
driver patch set ([Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker)), on top
of the cmpunlocker maintainer's P2P work:

1. **`p2p-unlock.patch`.** For CMP device IDs only: read the PRI trap-31 privilege mask. If it
   is writable, force the GPU's PCIe P2P read/write capabilities to OK; otherwise log
   "P2P left unsupported" and change nothing (a part with a different mask was seen to "copy"
   99.6% garbage). Around the two P2P mailbox setup calls, arm PRI decode trap 31 so the
   mailbox writes the CMP refuses at their normal privilege level are re-stamped at an accepted
   level, then restore the trap. This runs only when a P2P pair is first set up, never at boot.
2. **Booter PLM list.** Adds the trap-31 mask as one more entry in the privilege masks the
   Booter already opens at GSP bootstrap, only if the `ForceP2P` registry key enables P2P.
3. **`ForceP2P=0x11`** (a stock NVIDIA registry key) forces P2P read and write capabilities for
   every PCIe pair, skipping the chipset checks. All 56 ordered pairs are advertised, including
   pairs across the two CPU sockets.

No BAR1 resize is needed: GA100 PCIe P2P uses mailboxes, which reserve 64 KiB x 8 peers =
512 KiB of BAR1 per GPU (our BAR1 is 64 MiB).

**Advertised is not verified.** `cudaDeviceCanAccessPeer` returns true for every pair once
`ForceP2P` is set. Only a content check proves that bytes land where they should: run the
byte-exact probe below before letting NCCL or vLLM use P2P, and set `VLLM_SKIP_P2P_CHECK=0` so
vLLM runs its own IPC write/read test instead of trusting the driver.

## Our build

Our server ran cmpunlocker
[`2d10f891`](https://github.com/amoghmunikote/cmpunlocker/commit/2d10f891a58824f0591c3332c4eccf37ff36b086)
with the open kernel modules 615.71.09. Morrowmake's tree
([`1f2c498f`](https://github.com/Morrowmake/cmpunlocker/commit/1f2c498f577cfe846f866e3e8151ae448f64b456))
is based on an older upstream, so we merged it onto our installed tree: kept the ECC patches,
appended `p2p-unlock.patch` last in `PATCH_ORDER`, and removed `bar1-resize-unlock.patch`
(it hung our board at boot; P2P does not need it). If your installed cmpunlocker matches
Morrowmake's base, use their tree and installer directly.

[`p2p/`](../p2p) builds the modules for the running kernel in Docker, without touching the
host:

```bash
cd p2p
sudo ./backup-current.sh                               # copy of the installed modules + modprobe config
sudo CMPUNLOCKER_DIR=/path/to/cmpunlocker-with-p2p ./build.sh --prep-only   # patch + verify only
sudo CMPUNLOCKER_DIR=/path/to/cmpunlocker-with-p2p ./build.sh               # compile -> out/
```

`build.sh` runs the cmpunlocker patch sequence in an Ubuntu 24.04 + gcc-13 container (match your
kernel's compiler), checks that `p2p-unlock.patch` applies with zero fuzz, and checks the
result: right `vermagic`, right driver version, and the P2P strings present in `nvidia.ko`. On
our machine every module's `srcversion` matched the installed build, i.e. the same
kernel-interface sources.

## Install

```bash
sudo bash loop/stop_all.sh          # nothing may use the GPUs
cd p2p && sudo ./install.sh --yes   # copies the modules, writes ForceP2P=0x11, depmod, initramfs
# cold boot: power fully off, then on
```

`install.sh` changes exactly the cmpunlocker module directory, `/etc/modprobe.d/cmp-pcie-gen2.conf`
(the registry dwords: cmpunlocker's Gen2 settings plus `ForceP2P=0x11`; override with
`REGISTRY_DWORDS`), `depmod` and the initramfs. No kernel command line change. After any
driver change use a cold boot: a warm reboot can leave the unlocked 64 GB geometry falling
back to 8 GB.

After boot, check: `dmesg` shows the trap-31 PLM opened and `CMPUNLOCK: PCIe P2P caps forced to
OK` for every GPU, with no `FAILED to open`, DMAR fault or Xid; `/proc/driver/nvidia/params` has
`ForceP2P=0x11`; `nvidia-smi` shows 65,536 MiB, ECC on, Gen2 x16 and your power caps on every
card. Re-apply power caps if they are set by a boot-time service.

**Rollback:** `sudo ./rollback.sh --yes`, then cold boot. It restores the backed-up modules and
modprobe config byte for byte. **If the machine does not boot**: from the remote console, edit
the kernel line in GRUB and append
`module_blacklist=nvidia,nvidia_drm,nvidia_modeset,nvidia_uvm,nvidia_peermem`, boot, run
`rollback.sh`, cold boot.

## Test

```bash
mkdir -p p2p/test-out
docker compose -f p2p/tests/compose.yaml run --rm p2ptools            # all stages
docker compose -f p2p/tests/compose.yaml run --rm p2ptools topo probe # just the content check
```

The image is the serving image plus Morrowmake's byte-exact `p2p_probe` (fetched at a pinned
commit and checksum), CUDA samples' `p2pBandwidthLatencyTest` and nccl-tests. Stages:
`topo` (nvidia-smi), `probe` (cudaMemcpyPeer, SM peer write, SM peer read and a cross-process
IPC write at 9 sizes, every byte compared, on GPU sets {0,1}, {0-3}, {4-7}, {0-7}), `bw`, `nccl`
(all-reduce 12 KB-1.5 MB, P2P on vs off, LL and Simple), `ar` (vLLM custom all-reduce vs NCCL,
exact sums). Anything that sends P2P traffic runs only after the probe passed for that GPU set
in the same run. It also works before installing, as a no-P2P baseline.

## Our measurements

Server: Gigabyte G481-HA0, GPUs 0-3 behind one PLX PEX8796 switch on CPU socket 0, GPUs 4-7
behind another on socket 1, every GPU at Gen2 x16, IOMMU on, ACS redirect on the switch ports
(so every peer request goes up to the CPU root complex and back down).

| Check | Result |
|---|---|
| Byte-exact probe, {0,1}, {0-3}, {4-7}, {0-7} | pass on all |
| P2P write bandwidth | 5.79 GB/s, every pair, cross-socket the same |
| P2P latency | 2.1-2.6 us |
| NCCL LL all-reduce, 12 KB, TP2 | 22.8 -> 11.4 us |
| same, TP4 (one switch) | 58 -> 22 us |
| same, TP8 | 80 -> 74 us (`NCCL_P2P_LEVEL=PXB`), 65 us (`SYS`) |
| Chat, TP2 x PP4, NCCL P2P only | +4-7% |
| Chat, TP2 x PP4, + vLLM custom all-reduce | +12-16% |
| TP4 x PP2, custom all-reduce inside one switch | 30.1 -> 61.1 tok/s (harness) |
| TP8, custom all-reduce across sockets | 15.1 tok/s (unusable) |
| dmesg after all tests | clean |

More in [all-reduce measurements](comm.md). We have not tried "stage 2" (`iommu=pt` and
`pci=disable_acs_redir=...` on the kernel command line, so switch-local pairs talk inside the
PLX switch instead of hairpinning through the root complex); it may lower latency further.

## Engine settings that go with it

- `engine/compose.p2p.yaml`: `p2p_custom_all_reduce.py` adds `CMP_PCIE_CUSTOM_AR=1`, which
  treats a TP group as fully connected so TP > 2 may use custom all-reduce (vLLM only checks
  NVLink; TP2 is exempt anyway).
- `DISABLE_CAR=` (empty) in the vars file, so `--disable-custom-all-reduce` is not passed.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`: custom all-reduce needs legacy CUDA IPC
  handles on graph buffers and fails at startup with "invalid argument" otherwise.
- `NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS` (PP hand-offs also cross sockets),
  `VLLM_SKIP_P2P_CHECK=0`.

All in [`engine/env/p2p-tp2pp4.env`](../engine/env/p2p-tp2pp4.env).
