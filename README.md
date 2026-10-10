<h1 align="center">GLM-5.3 (753B MoE) on 8× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://github.com/cebeuq">cebeuq</a></strong>
  <br><br>
  <img alt="model" src="https://img.shields.io/badge/model-GLM--5.3%20~753B%20MoE-2ea44f?style=flat">
  &nbsp;
  <a href="https://hub.docker.com/r/lazymio/vllm-backport"><img alt="engine" src="https://img.shields.io/badge/engine-vllm--backport%20sm80%20%2B%20patches-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**The full GLM-5.3, a ~753B-parameter mixture-of-experts model, served with a 524,288-token
context on eight CMP 170HX mining cards. One user gets about 68 tok/s on code, 73-75 tok/s on
agentic web-browsing tasks and 60 tok/s on prose (measured at 358K context); eight users get 135.5 tok/s together; the KV
pool holds 569,792 tokens and a 519K-token prompt is answered coherently. Before this work the
same machine did 34.4 tok/s. The routed experts are the published NVFP4 bytes, untouched; the
rest of the model is INT8, the KV cache 8-bit, and nothing is offloaded to the CPU.**

This repository is the recipe: Docker Compose files for a pinned sm_80 vLLM image, the patches
we mount into it, the offline checkpoint conversion, the peer-to-peer driver build, the
benchmarks, and everything we measured, including what did not work. It builds on Morrowmake's
[GLM-5.3-Flash recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) for the
smaller sibling model, and on its TRAP31 P2P driver patch.

> **Which setups this is for.** Eight CMP 170HX cards unlocked to 64 GB each, on PCIe x16
> links, in one server. The production configuration also needs the P2P driver patch
> ([docs/p2p.md](docs/p2p.md)); without it, use
> [`engine/variants/no-p2p-128k.vars.example`](engine/variants/no-p2p-128k.vars.example)
> (58 / 59 / 53 tok/s chat, 128K context).

## What you need

- **8x NVIDIA CMP 170HX, each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB; the memory
  unlock is the [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) driver patch set),
  PCIe Gen2 x16 links
- Linux with the cmpunlocker-patched NVIDIA open driver (we used 615.71.09), Docker and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For production speed: the TRAP31 P2P driver build ([docs/p2p.md](docs/p2p.md))
- Disk: 433 GiB for the source checkpoint plus about as much again for the converted one (and
  room for the intermediate step while converting)
- Power: our production stack peaks at ~2.3 kW at the wall with the GPUs capped at 160 W
  ([hardware](docs/hardware.md))

## Headline numbers

Measured 2026-10-09/10 on 8x CMP 170HX, 160 W per card, SM clock up to 1,695 MHz, PCIe Gen2
x16, P2P on. One user: mean streaming speed on 36 chat prompts (temperature 0, thinking on,
384 tokens), measured at 358K context / 4 sequences. Details and methods:
[Results](docs/results.md).

| | TP2 x PP4 + P2P + MTP k=2 + PipeSpec |
|---|---:|
| **1 user** · coding | **68.3-68.6 tok/s** |
| **1 user** · web-browsing agent tasks | **72.8-74.9 tok/s** |
| **1 user** · prose | **59.9-60.0 tok/s** |
| **8 users, total** (raw prose) | 135.5 tok/s |
| Time to first token, 16K prompt | 8.2 s |
| Context / KV pool | 524,288 / 569,792 tokens |
| GSM8K-250, 5-shot | 82.8-87.6% (three runs) |

The 524K / 8-sequence production configuration measured 4% lower than the 358K / 4-sequence one
on our single-stream prose harness; its chat speed was not measured separately.

## Quick start

```bash
git clone https://github.com/cebeuq/glm53-cmp170hx-recipe.git
cd glm53-cmp170hx-recipe

# 1. Weights: download the NVFP4 checkpoint and build the hybrid (CPU only, see requant/README.md)
hf download dealignai/GLM-5.3-ABLITERATED-NVFP4 --local-dir /data/models/glm53-nvfp4
SRC=/data/models/glm53-nvfp4 RQ=/data/models/glm53-step1 \
  docker compose -f requant/compose.yaml run --rm requant
SRC=/data/models/glm53-nvfp4 RQ=/data/models/glm53-step1 DST=/data/models/glm53-hybrid \
  docker compose -f requant/compose.yaml run --rm hybrid

# 2. Optional but needed for production speed: P2P driver (docs/p2p.md)

# 3. Configure and start
cp engine/prod.vars.example engine/prod.vars        # set MODEL_DIR=/data/models/glm53-hybrid
cd engine
docker compose --env-file prod.vars config --quiet
docker compose --env-file prod.vars up -d
docker logs -f glm53-vllm                            # loading takes a while

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 400}'

docker compose --env-file prod.vars down             # stop
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1` and model
`glm-5.3`; tool calls and reasoning are parsed (`glm47` / `glm45`). The API listens on this
machine only and has no API key;
[`engine/compose.publish.example.yaml`](engine/compose.publish.example.yaml) shows how to add a
private address. Other configurations, all measured: [`engine/variants/`](engine/variants).

## What makes it fast

1. **A working MTP drafter under pipeline parallelism.** The fork's MTP layer ran with an
   unloaded embedding table under PP (GLM-5.x keeps `embed_tokens` only at the top level), so
   drafts were accepted 22% of the time. A one-hunk fix in `v32_mtp.py` took acceptance to 86%
   and made speculation pay: 34.4 -> ~56-59 tok/s on chat with k=2. [MTP](docs/mtp.md)
2. **PCIe peer-to-peer + vLLM custom all-reduce.** Morrowmake's TRAP31 driver patch opens GPU
   peer access on these cards; with byte-exact verification, NCCL P2P and vLLM's one-kernel
   all-reduce cut each TP2 all-reduce roughly in half: +12-16% on chat. [P2P](docs/p2p.md),
   [all-reduce](docs/comm.md)
3. **int8-tile latent KV cache.** DeepSeek's `fp8_ds_mla` layout with an int8 payload and a
   Triton read path for Ampere (which has no FP8 math): 656 instead of 1,152 bytes per token
   per layer, 2.6x less attention error than e4m3, no speed cost. With a 22,20,20,16 layer split
   this takes the KV pool from 171K to 570K tokens, which is what makes 512K context fit.
   [KV cache](docs/kv-cache.md)
4. **PipeSpec.** With one request, three of four pipeline stages sit idle while the last one
   verifies. PipeSpec starts the next speculative chunk before the current one is verified and
   rolls back if the bet fails: +6% coding, +9-13% browsing, +3-5% prose. [PipeSpec](docs/pipespec.md)
5. **Exact NVFP4 experts + INT8 residue.** Most of the bytes read per token are the BF16
   non-expert weights, not the 4-bit experts. Converting those to INT8-g128 and keeping the
   NVFP4 experts bit-exact passes the quality gate (96.4% teacher-forced agreement); converting
   the experts too does not. [How it works](docs/how-it-works.md)
6. **TP2 x PP4.** Two GPUs per stage read weights in parallel, all-reduces stay inside one PCIe
   switch, and the KV pool stays large. [Layouts](docs/results.md#layout-comparison)

## Documentation

- [How it works](docs/how-it-works.md): the model, the stack, what runs
- [Results](docs/results.md): all measurements, progression, layouts, quality, power
- [MTP](docs/mtp.md): the PP embedding bug, choosing k, drafter experiments
- [PipeSpec](docs/pipespec.md): pipelined speculation across pipeline stages
- [KV cache](docs/kv-cache.md): int8-tile latent KV on Ampere
- [P2P](docs/p2p.md): the TRAP31 driver build, install, rollback, tests
- [All-reduce](docs/comm.md) and [kernels](docs/kernels.md): microbenchmarks
- [Roofline](docs/roofline.md): bytes per token and the speed limit of this layout
- [What didn't work](docs/what-didnt-work.md)
- [Hardware notes](docs/hardware.md), [methodology](docs/methodology.md),
  [known limits and open questions](docs/known-limits.md)
- [Patches](patches/README.md): every mounted file, its switch and a diff against the image

## Limitations

- Measured on one server; other boards and topologies may differ, especially for P2P.
- Greedy output is not reproducible run to run on this engine.
- PipeSpec helps only a lone request, and at long agentic contexts it may lower draft
  acceptance unless `CMP_PIPESPEC_REFILL=1` (not measured yet).
- Fair chunked prefill (`CMP_FAIR_CHUNK`) is experimental and off by default.
- An unverified finding from reading the fork's source: the TP2 pipeline hand-off may combine
  per-GPU partial sums incorrectly at three layers. Details and how to test it:
  [known limits](docs/known-limits.md#open-question-the-pipeline-hand-off-at-tp2-unverified).
- The patches are whole-file copies for one pinned image; port them with the diffs.
- The checkpoint is abliterated; add whatever safeguards your use needs.

---

## Licences

**This recipe** (scripts, compose files, tools and documentation) is MIT, © 2026 cebeuq.
See [LICENSE](LICENSE). Everything under `patches/` is vLLM code or written to be mounted into
vLLM, and keeps the Apache-2.0 SPDX header each file carries.

The engine image, the driver patches and the model have their own licences:

- **Engine:** [`lazymio/vllm-backport`](https://hub.docker.com/r/lazymio/vllm-backport), built
  from [vllm-backport](https://github.com/wtdcode/vllm-backport) on vLLM (Apache-2.0).
- **Model:** [`dealignai/GLM-5.3-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/GLM-5.3-ABLITERATED-NVFP4),
  derived from [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3); see their model cards.
- **Driver patches:** [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) and
  [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker), applied to NVIDIA's
  [open GPU kernel modules](https://github.com/NVIDIA/open-gpu-kernel-modules); this repository
  contains build scripts only, no driver source.
- **Benchmark data:** GSM8K from [openai/grade-school-math](https://github.com/openai/grade-school-math)
  (downloaded by `bench/fetch_gsm8k.py`); browse prompts from DeepSearchQA, AA-Omniscience and
  GDPval.

## Credits

- **[Morrowmake](https://github.com/Morrowmake)** for the
  [GLM-5.3-Flash CMP 170HX recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe),
  which showed what these cards can do and whose findings shaped this work (host-staged and
  P2P all-reduce, fair prefill, verified P2P, the "never trust an advertised peer" rule), for
  the **TRAP31 P2P driver patch**, and for the byte-exact `p2p_probe` our test image builds.
  The Flash recipe's [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx) was the
  reference for our host-staged all-reduce and fair chunked prefill ports.
- **lazymio** and [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) for the
  sm_80 vLLM image this recipe runs unmodified underneath its patches.
- **[dealignai](https://huggingface.co/dealignai)** for the abliterated NVFP4 checkpoint.
- **[Z.ai](https://huggingface.co/zai-org)** for GLM-5.3.
- **The [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) maintainer** for the memory
  unlock and the P2P groundwork the TRAP31 patch builds on.
- **DeepSeek** for the sparse-attention (DSA) and MTP designs GLM-5.3 uses, and the
  `fp8_ds_mla` cache layout our int8 KV format reuses.
- **Red Hat AI** and the vLLM speculative-decoding contributors for the drafter paths in the
  fork (MTP, EAGLE, DSpark), which we studied while choosing a drafter.
- The **[vLLM](https://github.com/vllm-project/vllm)** project, **[Marlin](https://github.com/IST-DASLab/marlin)**
  (IST Austria / Neural Magic) for the kernels that do most of the work, and NVIDIA's
  [nccl-tests](https://github.com/NVIDIA/nccl-tests) and
  [cuda-samples](https://github.com/NVIDIA/cuda-samples) used by the P2P test image.
- The PipeSpec, PipeDec, FlowSpec, SPD and PPSD papers ([PipeSpec](docs/pipespec.md#related-work))
  for the ideas around speculation inside a pipeline.
