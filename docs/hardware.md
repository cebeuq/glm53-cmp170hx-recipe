# Hardware notes

## The server

| | |
|---|---|
| Chassis | Gigabyte G481-HA0 |
| CPUs | 2x Intel Xeon Gold 6150 (Skylake-SP) |
| GPUs | 8x NVIDIA CMP 170HX (GA100, sm_80), 64 GB HBM2e each after the cmpunlocker memory unlock (65,536 MiB in `nvidia-smi`), ECC on |
| PCIe | every GPU at Gen2 x16; GPUs 0-3 behind one PLX PEX8796 switch on CPU socket 0, GPUs 4-7 behind another on socket 1; switch uplinks Gen3 x16; `nvidia-smi topo -m`: PIX inside a group of four, SYS between groups |
| Driver | NVIDIA open kernel modules 615.71.09 with the cmpunlocker patch set (+ TRAP31 P2P, [P2P](p2p.md)) |
| OS | Ubuntu 24.04, Docker with the NVIDIA Container Toolkit |
| Power | two PSUs on separate circuits; GPUs capped at 160 W with the SM clock at up to 1,695 MHz |

## Things we learned about these cards

- **Memory clock is fixed** at 1,728 MHz and cannot be raised (NVML offset range 0..0). It does
  not drop with the power cap, so a low power cap costs little memory bandwidth by itself.
- **The SM clock still matters for decode.** At a 450 MHz lock a GPU reached only 0.55-0.70 TB/s
  on decode-shaped matrix-vector products, against 1.37-1.41 TB/s at ~1,200 MHz. Going from
  100 W / 450 MHz to 160 W / 1,695 MHz took PP8 from 11.0 to 31.0 tok/s.
- **Under sustained load the 160 W cap binds** (SM clock 750-1,100 MHz), so multi-stream decode
  and prefill are power-limited; single-stream decode is not.
- **Power transitions are risky on these marginal cards.** Raising one card's limit sharply
  (to 250 W) knocked it off the bus once. Change caps in small steps, one card at a time,
  watching the kernel log for Xid / AER messages.
- **Watch the wall, not just the GPUs.** Full-pipeline decode at 8 streams draws as much as
  prefill. Our production stack peaked at 2,340 W at the wall (4 streams) and 2,236 W
  (8 streams), with a mean around 1,800-1,900 W; one PSU circuit was not enough. Do not
  run CPU-heavy jobs (checkpoint conversion, kernel builds) on the server while the GPUs are
  loaded: that added ~380 W and tripped our wall-power guard once.
- **Errors:** on this board the firmware owns PCIe AER, so the OS AER counters always read 0;
  NVIDIA Xid messages in the kernel log are the only signal. `loop/guard.sh` stops the engine on
  Xid 31/48/79/94/95/154, "fallen off the bus" or a soft lockup.
- **Cold boot after any driver change.** A warm reboot can leave the unlocked memory geometry
  falling back to 8 GB without an error. Check for 65,536 MiB on every card before loading.
- **No profiler.** CUPTI is fused off; Nsight and `torch.profiler` fail. Time with CUDA events.
- **PCIe Gen3** is fuse-blocked; an experimental Gen3 driver branch knocked a card off the bus.
- **BAR1 resize** (a cmpunlocker option) hung this board at boot; P2P does not need it.
