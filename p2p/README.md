# P2P driver build, install, rollback and tests

Read [docs/p2p.md](../docs/p2p.md) first. Summary:

| File | Purpose |
|---|---|
| `backup-current.sh` | copy the installed cmpunlocker modules + modprobe config to `rollback/current/` (read-only on the system) |
| `build.sh` (+ `Dockerfile`, `compose.yaml`, `build-in-container.sh`) | patch and compile the modules for the running kernel in a container -> `out/`; installs nothing |
| `install.sh --yes` | copy `out/` modules into place, add `ForceP2P=0x11`, depmod, initramfs; then cold boot |
| `rollback.sh --yes` | restore the backup; then cold boot |
| `tests/` | byte-exact probe, bandwidth/latency, NCCL and vLLM all-reduce battery (`docker compose -f p2p/tests/compose.yaml run --rm p2ptools`) |

Input for `build.sh`: `CMPUNLOCKER_DIR`, a [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
tree that includes Morrowmake's TRAP31 `p2p-unlock.patch`
([Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker)). The build refuses a tree
whose `PATCH_ORDER` contains `bar1-resize-unlock.patch` unless `ALLOW_BAR1_RESIZE=1`.
`out/`, `cache/`, `rollback/` and `test-out/` are local artifacts (git-ignored).
