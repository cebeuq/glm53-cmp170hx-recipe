# How it works

## The problem

GLM-5.3 is a ~753B-parameter mixture-of-experts model (`GlmMoeDsaForCausalLM`): 78 layers
(3 dense, 75 MoE with 256 routed experts, 8 active per token, plus one shared expert), hidden
size 6,144, DeepSeek-V3.2-style sparse attention (MLA with a 512+64 latent KV per token and an
indexer that picks the 2,048 most relevant keys), a 154,880-token vocabulary, and one extra
multi-token-prediction (MTP) layer that can draft tokens for speculative decoding.

The CMP 170HX is a mining card built on the A100's GA100 chip (sm_80). With an unlocked driver
it exposes 64 GB of HBM2e per card, so 8 cards hold 512 GB. That is enough for a 4-bit
checkpoint of this model with room left for KV cache, but:

- **Ampere has no FP4/FP8 tensor cores**, and upstream vLLM's sparse-attention kernels need
  Hopper. The model does not run on these cards without sm_80 kernels.
- **PCIe Gen2 x16 only, and no peer-to-peer** with the stock driver: every all-reduce between
  GPUs goes through host memory.
- **Decode reads ~32 GB of weights per token**, so single-stream speed is bounded by HBM
  bandwidth and by how many GPUs read in parallel ([roofline](roofline.md)).

## The stack

### Engine: an sm_80 vLLM fork, patched by bind mounts

We use lazymio's [`vllm-backport`](https://hub.docker.com/r/lazymio/vllm-backport) `latest-sm80`
image (pinned by digest). It carries the sm_80 paths this model needs: a Triton sparse-MLA
attention backend (`TRITON_MLA_SPARSE`), software FP8 stores for the indexer, and an
NVFP4 -> Marlin weight-only path for the 4-bit experts. Nothing is rebuilt. Our changes are
whole-file copies mounted read-only over the image's files ([patches/](../patches/README.md)),
each behind an environment switch that defaults to off.

### Weights: exact NVFP4 experts + INT8 residue

The starting point is dealignai's abliterated NVFP4 checkpoint of GLM-5.3 (ModelOpt NVFP4 for
the routed experts, BF16 for everything else; 432.9 GiB). On Ampere, NVFP4 is a weight-only
format: Marlin expands it to BF16 inside the kernel, so it only saves memory bytes.

In that checkpoint the BF16 "residue" (attention projections, shared experts, the three dense
layers, `lm_head`) is more than half of the bytes read per token, so the residue, not the
experts, is the bandwidth lever. [requant/](../requant/README.md) converts it offline:

1. `requant_full.py`: experts NVFP4 -> INT4-g128 and residue BF16 -> INT8-g128 (round to
   nearest, no calibration, CPU only).
2. `hybrid_nvfp4e_int8r.py`: keep step 1's INT8 residue but put the **original NVFP4 expert
   bytes back**.

Converting the experts to INT4 failed the quality gate (93.65% teacher-forced top-1 agreement
against the original), so the experts stay exactly as published. The hybrid passes (96.4%) and
vLLM loads it as one compressed-tensors "mixed-precision" checkpoint (NVFP4 experts through
Marlin FP4, INT8-g128 residue through Marlin W8A16). Never quantized: the sparse-attention
indexer, router gates, norms, embeddings, `lm_head` and the MTP layer.

### Layout: TP2 x PP4

Eight GPUs as four pipeline stages of two tensor-parallel GPUs each, layers split 22,20,20,16
(the last stage also holds the BF16 MTP layer). Why this layout:

- **PP8** needs no all-reduce at all, but each token passes through 8 GPUs one after another, and
  each GPU reads 1/8 of the weights alone.
- **TP8** reads with all 8 GPUs at once but needs ~160 small all-reduces per step across both
  CPU sockets, and its KV pool is small (the MLA latent is replicated on every TP rank).
- **TP2 x PP4**: two GPUs read each stage's weights in parallel, all-reduces stay between two
  GPUs on the same PCIe switch, and the KV pool still reaches 570K tokens. Measured fastest
  at long context ([results](results.md#layout-comparison)).

With the default device order the two GPUs of each pair share a PLX switch; stages 0-1 are on
CPU socket 0, stages 2-3 on socket 1.

### MTP speculative decoding (k=2), with a fix for pipeline parallelism

GLM-5.3's MTP layer drafts the next tokens; the full model verifies 1 + k tokens per step.
Out of the box, under pipeline parallelism, the drafter ran with an **unloaded embedding
table**: GLM-5.x stores `embed_tokens` only once at the top level, the fork's MTP loader skipped
it, and the code that shares the target model's embedding returns early under PP because the
last stage does not hold the embedding. Draft acceptance was 20-22% (useless). `v32_mtp.py`
loads the table, the same way upstream's `deepseek_mtp.py` does, and acceptance went to 86%
for k=1. Production drafts k=2 tokens per step. Details: [MTP](mtp.md).

### PCIe peer-to-peer and custom all-reduce

The stock CMP 170HX driver refuses GPU peer access. Morrowmake's TRAP31 patch to the
cmpunlocker driver opens it; we rebuilt the modules for our kernel and driver in Docker and
verified the transfers byte for byte on every GPU pair. With P2P, NCCL uses direct GPU writes
and vLLM's custom all-reduce (one kernel, no host round trip) becomes usable. NCCL's 12 KB TP2
all-reduce went from 22.8 to 11.4 us; under CUDA graphs the custom all-reduce takes 16.0 us
for the 3-token verify step (NCCL: 27.7 us). Single-stream chat speed rose 12-16%. Custom all-reduce is kept inside one PCIe switch: across sockets it is much
slower. Details: [P2P](p2p.md), [all-reduce measurements](comm.md).

### int8-tile latent KV cache

The MLA cache stores one 576-value latent per token per layer (1,152 bytes in BF16). We keep
vLLM's `fp8_ds_mla` page layout (656 bytes: 512 8-bit values, 4 fp32 scales per 128 values,
and the 64 RoPE values in BF16) but store **int8** instead of e4m3 in it, and added a Triton
read path that expands the bytes inside the attention kernel (Ampere has no FP8 math).
On the one layer measured with exact inputs, int8 tiles had 2.6x less attention error than
e4m3 (0.48% vs 1.23%; the BF16 cache itself is at 0.35%), and in the engine they cost no
speed. The KV pool went from 171K tokens (BF16) to 520K-570K, which is what makes 512K
context possible on this layout. Details: [KV cache](kv-cache.md).

### PipeSpec: pipelined speculation across pipeline stages

With one request, a pipeline-parallel step walks stage 1 -> 2 -> 3 -> 4, and stage 4 verifies,
samples, drafts and sends the result back before stage 1 can start again: three of four
stages are idle. PipeSpec lets stage 1 start the next chunk (chunk B, made of further MTP
drafts) before stage 4 has verified the current one (chunk A), on the bet that chunk A is fully
accepted. If the bet fails, stage 4 skips chunk B and everyone rolls back; nothing is copied,
because positions past the committed length are always overwritten before they are read.
It runs only when exactly one request is decoding. Measured: +6% coding, +9-13% browse,
+3-5% prose on chat. Details: [PipeSpec](pipespec.md).

### Long context

- 8-bit KV (above) and a smaller indexer prefill buffer (`CMP_INDEXER_PREFILL_BUF_FACTOR=4`
  instead of `max_model_len x 40`; at 512K the x40 buffer is 2.6 GiB per GPU, x4 is 0.26 GiB).
- `mqa_logits_triton.py` clamps the logits width that a FULL CUDA graph captures; without it a
  replay whose context grew past the captured width read out of bounds (Xid 31).
- `marlin_utils_fp4.py` repacks NVFP4 experts in bounded chunks with the originals parked on the
  CPU, so loading fits in 64 GB cards.
- CUDA graphs: `FULL_AND_PIECEWISE`; `--max-num-batched-tokens 2048` chunked prefill.

### Fair chunked prefill (experimental)

A long new prompt is prefilled in chunks of up to 2,048 tokens per scheduler step. While other
requests are decoding, every such step also delays them, so an agent that sends a 100K-token
prompt can stall everyone else's streams for a long time. `CMP_FAIR_CHUNK=N` (a port of the
Flash recipe's `prefill_chunk_with_decodes` / fair prefill idea) caps the total prefill tokens
per step at N **only while some request is decoding**; decode tokens are never capped, and a
prompt that arrives at an idle server keeps the full chunk size. It is in `scheduler.py` and
`pipespec_scheduler.py`, off by default. Try it with
[`engine/variants/fair-chunk.vars.example`](../engine/variants/fair-chunk.vars.example) (384)
and measure with `bench/multiagent_stall.py`. Results: pending ([results](results.md#fair-chunked-prefill-experimental)).

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1`, no API key; model id `glm-5.3` (`SERVED_MODEL_NAME`) |
| Tools and reasoning | `--enable-auto-tool-choice`, `glm47` tool-call parser, `glm45` reasoning parser |
| Engine image | `lazymio/vllm-backport@sha256:2627d2a9...` (vLLM `g29e66dad4`, torch 2.13, CUDA 13.0) + [patches](../patches/README.md) |
| Weights | hybrid: NVFP4 routed experts (exact) + INT8-g128 residue, from `dealignai/GLM-5.3-ABLITERATED-NVFP4` |
| Layout | TP2 x PP4, layers 22,20,20,16, MTP layer on the last stage |
| Attention | `TRITON_MLA_SPARSE`, int8-tile latent KV (`fp8_ds_mla` layout, `CMP_KV_DSMLA_INT8=1`) |
| Context | 524,288 tokens; KV pool 569,792 tokens at `--gpu-memory-utilization 0.97` |
| Concurrency | `--max-num-seqs 8` |
| Speculation | MTP, `num_speculative_tokens` 2, stock head; PipeSpec on (`CMP_PIPESPEC=1`) |
| All-reduce | vLLM custom all-reduce over PCIe P2P (TP2), NCCL P2P (`NCCL_P2P_LEVEL=SYS`) for the rest |
| CUDA graphs | `FULL_AND_PIECEWISE` |
| Prefill | chunked, 2,048 tokens per step |
