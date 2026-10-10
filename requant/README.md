# Building the hybrid checkpoint

Input: [`dealignai/GLM-5.3-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/GLM-5.3-ABLITERATED-NVFP4)
(ModelOpt NVFP4 routed experts, BF16 everything else, 432.9 GiB).
Output: the same model with exact NVFP4 routed experts and INT8-g128 residue, as one
compressed-tensors "mixed-precision" checkpoint that vLLM loads directly. CPU only; no
calibration data.

```bash
hf download dealignai/GLM-5.3-ABLITERATED-NVFP4 --local-dir /data/models/glm53-nvfp4

# step 1: experts -> INT4-g128, residue -> INT8-g128 (only the residue is kept)
SRC=/data/models/glm53-nvfp4 RQ=/data/models/glm53-step1 \
  docker compose -f requant/compose.yaml run --rm requant
# step 2: step 1's residue + the original NVFP4 expert bytes
SRC=/data/models/glm53-nvfp4 RQ=/data/models/glm53-step1 DST=/data/models/glm53-abliterated-nvfp4e-int8r \
  docker compose -f requant/compose.yaml run --rm hybrid
# step 1's output is no longer needed afterwards
```

The source, the step-1 output and the hybrid must fit on disk at the same time. Set
`WORKERS`, `CPUS` and `MEM` for your machine (defaults 8 / 8 / 40g). Run it somewhere other
than the GPU server if that server is serving: a CPU-heavy job next to GPU load can push wall
power over a PSU limit.

What is converted (step 1, `requant_full.py`):

- INT8-g128, symmetric, round to nearest: attention projections `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` (layers 0-77), shared experts, and the MLPs of the
  3 dense layers.
- Never quantized: the sparse-attention indexer, router gates, norms, biases, embeddings,
  `lm_head`, and the MTP layer (78).

Step 2 (`hybrid_nvfp4e_int8r.py`) drops step 1's INT4 experts for layers 3-77 and inserts the
source's NVFP4 tensors unchanged, renamed to compressed-tensors NVFP4 names
(`weight` -> `weight_packed`; `weight_scale_2` and `input_scale` become their reciprocal
`*_global_scale`, which vLLM feeds to the same Marlin FP4 path). `--int4-layers 5,8,...` keeps
INT4 experts for chosen layers (not used in production).

Self-test of the packing conventions (no checkpoint needed):

```bash
docker compose -f requant/compose.yaml build requant   # needs SRC/RQ set to any value
docker run --rm --network none glm53-recipe/requant:1 /work/requant_full.py --selftest
```

Quality of the result against the original NVFP4 checkpoint: teacher-forced top-1 agreement
96.4% (gate 95%), NLL within budget. Converting the experts to INT4 as well fails (93.65%).
