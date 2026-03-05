# Major TOM Benchmark Workflow

This folder contains a three-script workflow:

1. `01_build_collection.py` - build and persist a scene-level Rasteret cache.
2. `02_run_benchmark_from_cache.py` - run benchmark scenarios from that cache.
3. `03_hf_vs_rasteret_benchmark.py` - standalone benchmark script (optional `--json-out`).

All scripts assume **HF streaming only** for benchmark execution.

## Build cache

```bash
uv run python examples/major_tom_benchmark/01_build_collection_sharded.py \
  --name major-tom-benchmark-scenes-20k \
  --target-rows 20000 \
  --bands B02 B08 \
  --date-range 2020-01-01 2024-02-06
```

## Run benchmark from cache

```bash
uv run python examples/major_tom_benchmark/02_run_benchmark_from_cache.py \
  --collection-path ~/rasteret_workspace/major-tom-benchmark-scenes_records \
  --samples 100 1000 \
  --bands B02 B08
```

## Run standalone (shareable) benchmark

```bash
HF_TOKEN=... uv run python examples/major_tom_benchmark/03_hf_vs_rasteret_benchmark.py \
  --collection-path ~/rasteret_workspace/major-tom-benchmark-scenes_records \
  --samples 1000 \
  --bands B02 B08 \
  --sample-strategy random \
  --random-seed 42 \
  --json-out ./hf_vs_rasteret_results.json
```
