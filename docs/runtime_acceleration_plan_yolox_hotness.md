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
| A2 | Add image-wise process-level parallelism (multi-worker) | Medium-High (1.3x - 2.2x on single GPU) | done | user | `PARALLEL_JOBS` env var (integer or `auto`); auto-detect uses nvidia-smi probe on first image |
| A2a | Resume from checkpoint on interrupted runs | Low (0x time, preserves work) | done | user | `RESUME_ENABLED=1` (default); keeps `.resume_state/`.done files; completed images skipped |
| A3 | Optimize annotation lookup path (COCO JSON indexing) | Medium (10% - 30%) | not-started | TBD | Same outputs, less overhead per image |
| A4 | Optimize I/O path (faster disk / mount strategy) | Medium (1.1x - 1.4x) | not-started | TBD | Preserve all artifacts |
| A5 | Optimize visualization/write pipeline (same outputs) | Low-Medium (5% - 20%) | not-started | TBD | Keep file set and semantics unchanged |
| A6 | CUDA runtime tuning (`cudnn.benchmark`, etc.) | Low-Medium (5% - 25%) | not-started | TBD | Verify reproducibility requirements |
| A7 | Mixed precision (AMP) for speed-up | Medium (1.2x - 1.8x after GPU) | not-started | TBD | Only if numerical tolerance is acceptable |
| A8 | Disable DataParallel when running on a single GPU | Low (5% - 15%) | not-started | TBD | Change condition to `device == 'cuda' and torch.cuda.device_count() > 1`; eliminates scatter/gather overhead |

---

## Recommended Execution Order

1. A1: GPU enablement
2. A2: process-level parallelism (with auto-detect)
3. A2a: resume/checkpoint support (included in A2)
4. A8: disable DataParallel on single GPU (low risk, quick win)
5. A3: annotation lookup optimization
6. A4: storage/mount optimization
7. A5: visualization/write optimization
8. A6: CUDA runtime tuning
9. A7: AMP (optional)

---

## Practical Configuration Template

### 1) GPU Enablement (A1)

Status: completed.

Validation logs:

- `Runtime device selected: cuda`
- `CUDA device count: 1`
- `CUDA current device: 0`
- `CUDA device name: NVIDIA GeForce RTX 4070 Ti`

### 2) Disable DataParallel on Single GPU (A8)

Background:

`torch.nn.DataParallel` is designed for multi-GPU setups. When only one GPU is present, it still runs `scatter` (split input) and `gather` (merge output) on every forward pass, adding CUDA stream synchronization overhead. With batch size 1 (this project processes one image per process), the entire batch lands on GPU 0 anyway — so scatter/gather is pure overhead.

Change in `parameter_and_input_saliency.py`:

```python
# Before
if device == 'cuda':
    net = torch.nn.DataParallel(net)

# After
if device == 'cuda' and torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
```

Additional simplification: the `.module` unwrap guards throughout the code (`net.module if isinstance(net, torch.nn.DataParallel) else net`) remain valid and correct — no other changes needed.

### 3) Worker Parallelism (A2)
### 3) Worker Parallelism (A2) with Auto-Detection

Implementation:

- Env variable `PARALLEL_JOBS`:
    - Integer (e.g., 1, 2, 4): use directly
    - `"auto"` (default): auto-probe on startup

- Auto-probe:
    - Runs first pending image, measures peak GPU VRAM usage
    - Calculates `PARALLEL_JOBS = floor(free_vram_after_probe × GPU_MEMORY_SAFETY / per_image_vram)`
    - Uses `nvidia-smi` polling (0.3 sec intervals) to capture peak memory
    - Falls back to 1 if nvidia-smi unavailable or VRAM delta unmeasurable

- Env variable `GPU_MEMORY_SAFETY` (default 0.8):
    - Fraction of post-probe free VRAM considered safe to use
    - Reduce to 0.5-0.7 if OOM occurs

### 3a) Resume and Checkpoint Support (A2a)

Implementation:

- Env variable `RESUME_ENABLED` (default 1):
    - `1`: keep existing OUTPUT_ROOT; reprocess only pending images
    - `0`: delete OUTPUT_ROOT; start fresh

- Env variable `RESUME_RESET` (default 0):
    - `1`: clear OUTPUT_ROOT once on startup, then enable resume
    - `0`: read existing checkpoint state and resume

- Checkpoint state:
    - Directory `.resume_state/` inside OUTPUT_ROOT stores completed image markers
    - Each completed image creates `00000042.done` (0-padded index); contains image path
    - On re-run, startup scans for these files and skips completed images
    - Parallelism respected: no race conditions (1 image = 1 file)

- Usage examples:
    ```bash
    # Normal: auto-parallelism, resume on interrupt
    PARALLEL_JOBS=auto bash scripts/run_yolox_tiny_custom_model.sh

    # Force reset (delete old output, then resume-enabled)
    RESUME_ENABLED=1 RESUME_RESET=1 PARALLEL_JOBS=auto bash scripts/run_yolox_tiny_custom_model.sh

    # No resume (always start fresh)
    RESUME_ENABLED=0 PARALLEL_JOBS=2 bash scripts/run_yolox_tiny_custom_model.sh
    ```

### 4) Modes for Controlled Rollout

- `RUN_MODE=test` with `TEST_MAX_FILES=100` for benchmark consistency
- `RUN_MODE=full` for production runs

---

## Time-Reduction Estimate for 100 Images

These are planning estimates before measurement.

| Scenario | Relative Runtime (baseline = 1.00) | Approx. Speed-up |
|---|---:|---:|
| Baseline (current) | 1.00 | 1.0x |
| A1 only (GPU) | 0.40 - 0.12 | 2.5x - 8.3x |
| A1 + A2 (parallel=2) | 0.30 - 0.06 | 3.3x - 16.7x |
| A1 + A2 + A8 (parallel=2 + no DP) | 0.28 - 0.05 | 3.6x - 20.0x |
| A1 + A2 + A8 + A3 + A4 | 0.22 - 0.04 | 4.5x - 25.0x |

Notes:

- Real gains depend on GPU model, VRAM size, storage throughput, and container mount performance.
- A7 (AMP) can add speed but may change numeric behavior slightly.

---

## Benchmark Protocol (Status-Trackable)

Use this checklist and fill status per run.

| Step | Description | Status | Result Summary |
|---|---|---|---|
| B1 | Baseline timing on 100 images (current config) | not-started | |
| B2 | Timing after A1 | done | CUDA runtime confirmed (RTX 4070 Ti). Wall-time baseline needed. |
| B2a | Timing after A1 + A2 (parallel=auto) | not-started | |
| B3 | Timing after A1 + A2 + A8 | not-started | |
| B4 | Timing after A1 + A2 + A8 + A3 | not-started | |
| B5 | Timing after A1 + A2 + A8 + A3 + A4 | not-started | |
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

- **A1**: ✅ GPU runtime confirmed (CUDA, RTX 4070 Ti)
- **A2**: ✅ Parallelism implemented:
    - `PARALLEL_JOBS` env var (integer or `auto`)
    - Auto-detect via GPU memory probe on pending images
    - `GPU_MEMORY_SAFETY` tuning (default 0.8)
- **A2a**: ✅ Resume/checkpoint support implemented:
    - `RESUME_ENABLED` control (default 1 = resume mode)
    - `RESUME_RESET` for forced restart
    - `.resume_state/` directory tracks completed images
- **A8**: ❌ Not yet implemented (disable DataParallel on single GPU)
- **A3–A7**: Not implemented; queued for next phase
- Main remaining bottlenecks are A8 (DataParallel overhead), A3 (COCO indexing), and A4 (I/O optimization).
