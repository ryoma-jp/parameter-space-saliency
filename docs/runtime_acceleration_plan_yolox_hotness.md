# Runtime Acceleration Plan for YOLOX Hotness Workflow

## Goal

Reduce total processing time for 100 images while keeping the following constraints:

- Do not skip any currently enabled processing steps
- Keep image-wise execution of `parameter_and_input_saliency.py`

---

## Scope and Assumptions

- Target script: `scripts/run_yolox_tiny_custom_model.sh`
- Current behavior: one process per image, sequential loop
- Runtime device is confirmed as CUDA in execution logs
- Objective mode includes `hotness_unified` and all related outputs remain enabled

---

## Status Legend

Use one of these values in the status column:

- `not-started`: not implemented yet
- `in-progress`: currently being implemented or tested
- `done`: implemented and validated
- `blocked`: pending dependency or environment prerequisite

---

## Acceleration Backlog (with Implementation Status)

| ID | Acceleration Item | Expected Impact | Implementation Status | Owner | Notes |
|---|---|---:|---|---|---|
| A1 | Switch inference/execution from CPU to GPU | High (2.5x - 8x) | done | user | Confirmed by runtime logs (`Runtime device selected: cuda`, RTX 4070 Ti) |
| A2 | Add image-wise process-level parallelism (multi-worker) | Medium-High (1.3x - 2.2x on single GPU) | not-started | TBD | Keep one image per process; adjust worker count by memory |
| A3 | Optimize annotation lookup path (COCO JSON indexing) | Medium (10% - 30%) | not-started | TBD | Same outputs, less overhead per image |
| A4 | Optimize I/O path (faster disk / mount strategy) | Medium (1.1x - 1.4x) | not-started | TBD | Preserve all artifacts |
| A5 | Optimize visualization/write pipeline (same outputs) | Low-Medium (5% - 20%) | not-started | TBD | Keep file set and semantics unchanged |
| A6 | CUDA runtime tuning (`cudnn.benchmark`, etc.) | Low-Medium (5% - 25%) | not-started | TBD | Verify reproducibility requirements |
| A7 | Mixed precision (AMP) for speed-up | Medium (1.2x - 1.8x after GPU) | not-started | TBD | Only if numerical tolerance is acceptable |

---

## Recommended Execution Order

1. A1: GPU enablement
2. A2: process-level parallelism
3. A3: annotation lookup optimization
4. A4: storage/mount optimization
5. A5: visualization/write optimization
6. A6: CUDA runtime tuning
7. A7: AMP (optional)

---

## Practical Configuration Template

### 1) GPU Enablement (A1)

Status: completed.

Validation logs:

- `Runtime device selected: cuda`
- `CUDA device count: 1`
- `CUDA current device: 0`
- `CUDA device name: NVIDIA GeForce RTX 4070 Ti`

### 2) Worker Parallelism (A2)

Introduce configurable workers while preserving image-wise execution.

Suggested env variables:

- `PARALLEL_JOBS=1` (default)
- `PARALLEL_JOBS=2` (start point on single GPU)

### 3) Modes for Controlled Rollout

- `RUN_MODE=test` with `TEST_MAX_FILES=100` for benchmark consistency
- `RUN_MODE=full` for production runs

---

## Time-Reduction Estimate for 100 Images

These are planning estimates before measurement.

| Scenario | Relative Runtime (baseline = 1.00) | Approx. Speed-up |
|---|---:|---:|
| Baseline (current) | 1.00 | 1.0x |
| A1 only (GPU) | 0.40 - 0.12 | 2.5x - 8.3x |
| A1 + A2 | 0.30 - 0.07 | 3.3x - 14.3x |
| A1 + A2 + A3 + A4 | 0.24 - 0.05 | 4.2x - 20.0x |

Notes:

- Real gains depend on GPU model, VRAM size, storage throughput, and container mount performance.
- A7 (AMP) can add speed but may change numeric behavior slightly.

---

## Benchmark Protocol (Status-Trackable)

Use this checklist and fill status per run.

| Step | Description | Status | Result Summary |
|---|---|---|---|
| B1 | Baseline timing on 100 images (current config) | not-started | |
| B2 | Timing after A1 | done | CUDA runtime confirmed (RTX 4070 Ti). Collect wall-time metrics next. |
| B3 | Timing after A1 + A2 | not-started | |
| B4 | Timing after A1 + A2 + A3 | not-started | |
| B5 | Timing after A1 + A2 + A3 + A4 | not-started | |
| B6 | Optional: A6/A7 evaluation | not-started | |

Recommended metrics:

- total wall time for 100 images
- mean time per image
- p50 / p90 time per image
- GPU memory peak (if available)

---

## Change Log Section (for Ongoing Updates)

Append updates in this format:

```text
[YYYY-MM-DD] [ID] [status before -> after]
- What changed:
- Validation:
- Observed impact:
- Risks / follow-up:
```

---

## Current Snapshot

- Script-level run mode (`test` / `full`) is already available.
- A1 is complete: runtime device is confirmed as CUDA (NVIDIA GeForce RTX 4070 Ti).
- Main remaining bottlenecks are process-level parallelism and data/IO overhead.
- No processing step should be removed for this acceleration track.
