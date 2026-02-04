# CLIP4Clip × RTime Benchmark Plan (Text→Video Retrieval)

This plan targets CLIP4Clip-style reporting (**R@1 / R@5 / R@10 / MdR / MnR**) for **text-to-video retrieval**, training on **RTime `train.json`** and evaluating on **`test.json`**, for the three CLIP4Clip methods: `meanP`, `seqTransf`, `tightTransf`.

---

## 1) Lock down the target protocol (so results are comparable)

### 1.1 Metrics (match CLIP4Clip Tables 1–5)
Report, for **Text→Video** retrieval:
- **R@1, R@5, R@10**
- **MdR** (median rank)
- **MnR** (mean rank)

### 1.2 RTime evaluation settings (Origin vs Hard)
RTime defines two candidate-set settings relevant here:

- **RTime-Origin**: retrieve among *original* videos only (**~1,000** pairs).
- **RTime-Hard**: retrieve among *original + reversed* videos (**~2,000** pairs).

**Implementation approach (recommended):**
- Run evaluation twice per trained checkpoint:
  1) **Hard**: all videos in `test.json`.
  2) **Origin**: filter to “original-only” entries (e.g., create `rtime_test_origin.csv`).

### 1.3 Caption usage (avoid leakage, follow RTime intent)
RTime uses:
- **Training**: GPT-generated captions (often multiple per video) are used for training.
- **Val/Test**: human/manual captions are used for evaluation.

**Operational rule:**
- Train with **all captions** present in `train.json`.
- Validate/test with **manual/human captions only** from `valid.json` / `test.json`.

---

## 2) GPU hardware plan (EC2)

### 2.1 Recommended instance tiers
Pick one family and use it for **all** runs to keep comparisons clean.

**Tier A (default, cost-efficient):**
- **G5** (NVIDIA A10G, 24 GB per GPU), up to 8 GPUs/instance.

**Tier B (more VRAM headroom):**
- **G6e** (NVIDIA L40S, 48 GB per GPU).

**Tier C (fastest for heavy sweeps):**
- **P4d** (A100 40 GB) or **P5** (H100 80 GB).

### 2.2 Batch and GPU normalization
For fairness, keep **global batch size** constant across methods and seeds:
- If 4 GPUs: target **global batch 128** (≈32/GPU), matching CLIP4Clip’s common recipe.
- If fewer GPUs: use **gradient accumulation** to preserve global batch 128.

---

## 3) Environment + reproducibility controls

### 3.1 Software baseline
Use a single fixed environment across all runs:
- Prefer a container or DLAMI.
- Record exact versions of: Python, PyTorch, CUDA, cuDNN, FFmpeg/OpenCV.

### 3.2 Determinism checklist
- Fix seeds: `python`, `numpy`, `torch`, `torch.cuda`.
- Set:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- Log: git commit hash, full CLI args, dataset hash, GPU model, driver/CUDA versions.

---

## 4) Data preparation (RTime → CLIP4Clip input)

### 4.1 Original + reversed videos
Verify local paths exist for:
- **Original** videos.
- **Reversed** counterparts (needed for Hard candidate set).

If reversed videos are not prebuilt, generate deterministically (e.g., ffmpeg) and name consistently (e.g., `_rev.mp4`).

### 4.2 Optional but recommended: compress for stable throughput
Use CLIP4Clip’s provided compression utility to normalize decoding cost:
- compress to ~3 fps and size ~224-side (or equivalent).
Apply to both original and reversed videos.

### 4.3 Convert JSON splits into CLIP4Clip-friendly CSVs
Create minimal CSVs:

- `rtime_train.csv`: rows `(video_id, caption)` including **all training captions**.
- `rtime_valid.csv`: rows `(video_id, caption)` using **manual captions only**.
- `rtime_test.csv`: rows `(video_id, caption)` using **manual captions only**.

Also create:
- `rtime_test_origin.csv`: filter to original-only entries.

**Important:** ensure unique IDs for original vs reversed, e.g.:
- `abc123` (original)
- `abc123_rev` (reversed)

---

## 5) Minimal CLIP4Clip integration steps

### 5.1 Add a dataset loader
Add something like:
- `dataloaders/dataloader_rtime_retrieval.py`

Requirements:
- Train: sample `(video_id, caption)` → load frames → tokenize caption.
- Eval: load all captions (queries) and all videos (candidates) → compute similarity matrix.

### 5.2 Register the dataset type
Add `--datatype rtime` routing to:
- create train/val/test loaders
- select text→video evaluation path

### 5.3 Origin vs Hard evaluation
Simplest: run evaluation twice using different test CSVs:
- Hard: `rtime_test.csv`
- Origin: `rtime_test_origin.csv`

---

## 6) Training plan for `meanP`, `seqTransf`, `tightTransf`

### 6.1 Common hyperparameter baseline (start from CLIP4Clip retrieval recipe)
Use CLIP4Clip’s defaults as a base:
- epochs ~5 (then adjust if validation is still improving)
- lr ~1e-4
- `max_words=32`
- `max_frames=12` (see validation sweep below)
- `feature_framerate=1` (or consistent decoding scheme)
- consistent CLIP backbone (freeze choice across all runs)

### 6.2 One-time tiny sweep (then freeze)
Before main benchmarking, run a small sweep (using `meanP` only) on validation:
- `max_frames ∈ {12, 16, 24}`
Pick best on validation **Hard** (e.g., maximize R@1+R@5+R@10).
Freeze `max_frames` for all methods.

### 6.3 Method-specific flags (only controlled difference)
Keep everything identical except `--sim_header` (and any required tight/loose toggle).

Templates (adapt to your CLIP4Clip fork’s CLI exactly):
- **meanP**: `--sim_header meanP` (often with `--loose_type`)
- **seqTransf**: `--sim_header seqTransf` (often with `--loose_type`)
- **tightTransf**: `--sim_header tightTransf` (often without `--loose_type`)

Keep `--linear_patch` and other shared flags constant across methods.

---

## 7) Evaluation + statistics (make it “statistically sound”)

### 7.1 Core evaluation
For every trained checkpoint, evaluate:
- **Hard** (all test videos; ~2k candidates)
- **Origin** (original-only; ~1k candidates)

Report:
- R@1, R@5, R@10, MdR, MnR

### 7.2 Multiple training seeds
For each method (`meanP`, `seqTransf`, `tightTransf`):
- Train **≥3 seeds** (recommended: 3 minimum; 5 if budget allows).
Use the same seed list for all methods.

Report:
- mean ± std across seeds for each metric (Origin and Hard).

### 7.3 Confidence intervals and pairwise significance (paired bootstrap)
Because retrieval metrics are query-based, use **paired** procedures:

- **Bootstrap CIs:** sample text queries with replacement; recompute metrics; report 95% percentile CI.
- **Method comparisons:** paired bootstrap of the **difference** in metrics between two methods.
  - If 95% CI for the difference excludes 0, the difference is supported.
  - Optionally correct for multiple comparisons (Holm–Bonferroni over 3 pairwise tests).

---

## 8) Run matrix (what you actually execute)

### 8.1 One-time setup runs
1) **Pipeline sanity check**: tiny subset (e.g., 200 videos) to ensure it overfits and the loader is correct.
2) **Validation sweep**: `meanP` with `max_frames ∈ {12,16,24}` on validation Hard; pick best; freeze.

### 8.2 Main runs (the ones you will report)
For each method ∈ {`meanP`, `seqTransf`, `tightTransf`}:
- Train full `train.json`
- Select best checkpoint using validation
- Evaluate on test Hard + Origin
- Repeat for seeds ∈ {0,1,2} (or 0..4)

Total: **3 methods × 3 seeds = 9** full runs (+ small sweep + sanity run).

---

## 9) Reporting format (mirrors CLIP4Clip tables)

Produce **two tables**:

### Table A — RTime-Origin (test; original-only candidates)
Columns:
- Method | Train (RTime) | R@1 | R@5 | R@10 | MdR | MnR | (mean±std across seeds)

### Table B — RTime-Hard (test; original+reversed candidates)
Same columns.

Add footnotes:
- #videos/#queries per setting
- backbone, `max_frames`, framerate/decoding setup
- instance type + #GPUs
- seed count, CI method, paired comparison method

---

## 10) Practical EC2 starting point
If you want a clean “paper-like” workflow:
- **One G5 instance with 4 GPUs** (A10G) and distributed training with 4 processes.

If you want fewer OOM surprises (bigger `max_frames` sweep, larger batches):
- **G6e** (L40S, 48 GB) while keeping global batch constant.

---
