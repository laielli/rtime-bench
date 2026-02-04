# CLIP4Clip × RTime Benchmark Plan (Frozen CLIP Backbone, Offline Embeddings)

Goal: produce CLIP4Clip-style **text→video retrieval** results (**R@1 / R@5 / R@10 / MdR / MnR**) on **RTime**, training on **`train.json`** and evaluating on **`test.json`**, for:
- `meanP` (no learnable params; “training” ≈ zero)
- `seqTransf` (train a temporal Transformer encoder only)
- `tightTransf` (train a cross-modal Transformer module only)

Key modification vs the original plan: **freeze the CLIP backbone** (text + image encoders) and **precompute CLIP embeddings once** so all experiments reuse the same cached features.

---

## 0) Core idea & constraints

### 0.1 Freeze CLIP, train only the “head”
- **No gradients** through CLIP visual encoder or CLIP text encoder.
- All learnable parameters live in:
  - nothing for `meanP`, or at most an optional scalar temperature (see below),
  - the temporal Transformer for `seqTransf`,
  - the cross-modal Transformer for `tightTransf`.

### 0.2 One-time feature extraction
You will perform a single pass over the dataset to extract:
- **Video frame embeddings**: per-video sequence `V ∈ ℝ[T, D]` (T frames, D=CLIP dim).
- **Text embeddings**:
  - pooled caption embedding `t_pool ∈ ℝ[D]` (needed for `meanP`, `seqTransf`)
  - optionally token-level embeddings `t_tok ∈ ℝ[L, D]` (needed if `tightTransf` consumes tokens)

Then all training/eval uses these arrays and **never decodes videos** or runs CLIP again.

---

## 1) Decide the “feature contract” (must be fixed before extraction)

This is the most important decision because it defines what you cache.

### 1.1 Freeze a specific CLIP checkpoint
Pick *one* CLIP variant and freeze it for the entire benchmark:
- e.g., CLIP ViT-B/32 vs ViT-B/16, etc.
Record:
- checkpoint name / hash
- tokenizer version
- preprocessing transforms (resize/crop/normalize)

### 1.2 Fix a single frame sampling policy
Choose once and reuse everywhere:
- **feature_fps** (e.g., 1 fps or 3 fps)
- **T_max** (max frames cached per video; recommend **24** to allow subsetting)
- deterministic sampling rule: e.g., uniform over duration with fixed rounding

**Recommendation:** cache at `T_max=24` and for runs needing fewer frames (12/16) use a deterministic subset (e.g., evenly-spaced indices from the 24).

### 1.3 Fix what to cache for text
- `meanP` / `seqTransf` need only **pooled** CLIP text embeddings.
- `tightTransf` usually benefits from **token-level** embeddings.

**Practical tradeoff:**
- Caching pooled text embeddings is always cheap.
- Caching token-level embeddings can be large; store in **fp16** and/or only for splits you actually need.
  - If storage is tight, you can compute token embeddings on-the-fly on CPU/GPU because text encoding is much cheaper than video encoding—but that breaks the “compute once” principle.

---

## 2) Feature cache design (fast + reproducible)

### 2.1 Recommended on-disk layout (simple, robust)
Create a versioned cache directory, e.g.:
```
rtime_cache/
  cache_meta.json
  video_feats/
    {video_id}.pt          # [T_max, D] fp16 or fp32
  text_feats/
    {caption_id}.pt        # pooled: [D] and optional tokens: [L, D]
  splits/
    train_pairs.csv
    valid_pairs.csv
    test_pairs.csv
    test_origin_pairs.csv
```

`cache_meta.json` must include:
- CLIP checkpoint identifier
- image preprocessing parameters
- fps, T_max, sampling rule
- embedding dtype (fp16/fp32)
- dataset commit/hash + the exact `train/valid/test` JSON hashes
- code commit hash used for extraction

### 2.2 Treat reversed videos as “free”
If a reversed video is *literally the original frames in reverse order*, then:
- You do **not** need to store a second copy of CLIP frame embeddings.
- Create reversed candidates by `flip(V, dim=0)` when assembling candidate sets.

This halves extraction time/storage and ensures perfect consistency.

**Important evaluation nuance:** `meanP` is order-invariant, so original and reversed have identical mean-pooled embeddings. Expect ties in RTime-Hard.

### 2.3 Sanity check (must pass before training)
Pick ~20 videos and verify:
- On-the-fly CLIP embeddings ≈ cached embeddings (max abs diff within expected fp16 tolerance).
- Reversed features produced by flipping match “true” reversed-video extraction (if you have reversed files), within tolerance.

---

## 3) One-time extraction job (compute-heavy stage)

### 3.1 What you extract
For each *original* video:
- decode frames according to your sampling policy (T_max)
- run frozen CLIP visual encoder → store `V ∈ ℝ[T_max, D]`

For each caption string (train/val/test):
- tokenize once, run frozen CLIP text encoder → store:
  - pooled `t_pool ∈ ℝ[D]`
  - optional tokens `t_tok ∈ ℝ[L, D]`

### 3.2 Hardware recommendation for extraction
Extraction is dominated by video decode + CLIP forward:
- Use **multi-GPU** if available (e.g., 4× or 8× GPUs) to finish quickly.
- Use fast local storage (NVMe instance store or provisioned IOPS EBS).
- Write cache to EBS and/or sync to S3 once complete.

Suggested EC2 choices for extraction:
- G5 (A10G) multi-GPU instance for cost-efficient throughput
- G6e (L40S) for more VRAM headroom / speed

### 3.3 Make extraction deterministic
- fixed seed for sampling
- fixed decode backend and ffmpeg version
- stable ordering of video_ids and captions
- store mapping files so downstream jobs can reproduce indices exactly

---

## 4) Dataset preparation for CLIP4Clip-style training/eval

Create pair lists that reference the cache:

- `train_pairs.csv`: `(video_id, caption_id, is_original=1)`
  - include **all training captions**
- `valid_pairs.csv`: manual captions only
- `test_pairs.csv`: manual captions only (**Hard** candidate set)
- `test_origin_pairs.csv`: manual captions + original videos only (**Origin** candidate set)

Make sure every `caption_id` maps to a cached text embedding.

---

## 5) Model definitions under frozen CLIP

All methods use cached CLIP embeddings and train only small modules (or none).

### 5.1 `meanP` (training ≈ zero)
Inputs:
- video frame embeddings `V ∈ ℝ[T, D]`
- pooled text embedding `t_pool ∈ ℝ[D]`

Compute:
- `v_pool = mean(V, dim=time)`
- similarity = cosine(`t_pool`, `v_pool`)

**No training.** (Optionally: calibrate a single scalar temperature on train/val; keep optional to preserve “training ≈ zero”.)

**Tie handling warning (RTime-Hard):**
- original vs reversed will often tie exactly (same `v_pool`).
- Use a deterministic stable sort (`argsort(stable=True)`) and/or define tie-breaking explicitly.
- Consider also reporting tie-aware expected R@K (optional).

### 5.2 `seqTransf` (train temporal Transformer encoder only)
Inputs:
- cached `V ∈ ℝ[T, D]` per video
- cached pooled `t_pool ∈ ℝ[D]` per caption

Trainable module:
- temporal Transformer encoder `Enc_v` that maps `V → v̂ ∈ ℝ[D]`
  - e.g., add positional encodings + CLS token or attention pooling

Loss:
- standard in-batch contrastive loss between `v̂` and `t_pool`
- CLIP encoders remain frozen

### 5.3 `tightTransf` (train cross-modal Transformer only)
Inputs:
- cached video frames `V ∈ ℝ[T, D]`
- cached text tokens `t_tok ∈ ℝ[L, D]` (preferred) or pooled text if code requires

Trainable module:
- cross-modal Transformer producing a similarity score between (text, video)
  - e.g., cross-attention layers or “tight” interaction module

Loss:
- contrastive / ranking loss computed from cross-modal scores

---

## 6) Training protocol (fast now, but keep it fair)

### 6.1 Global controls (shared across methods)
Keep constant for comparability:
- training split: `train.json`
- validation split: `valid.json`
- optimizer type, LR schedule
- global batch size (can be larger now due to cheap forward)
- number of epochs or early stopping rule
- same negative sampling regime (in-batch negatives)

### 6.2 Suggested two-phase workflow
1) **Feature extraction** (once)
2) **Head training** (multiple short runs)
   - `meanP`: run eval only
   - `seqTransf`: train temporal encoder
   - `tightTransf`: train cross-modal encoder

### 6.3 Updated hardware recommendation (training stage)
Because CLIP is frozen and not in the training graph:
- `seqTransf` / `tightTransf` can often train on **1 GPU** comfortably
- multi-GPU still helps for faster sweeps / more seeds

---

## 7) Evaluation + statistics (unchanged in spirit, faster in practice)

### 7.1 Evaluate both settings per checkpoint
For each method (and each seed where applicable):
- **RTime-Hard**: candidates = original + reversed (≈2,000 videos)
- **RTime-Origin**: candidates = original only (≈1,000 videos)

Metrics:
- R@1, R@5, R@10, MdR, MnR

### 7.2 Multiple seeds for trainable heads
- `meanP`: deterministic given cache (no seeds needed, except for tie-breaking if you randomize)
- `seqTransf`: train **≥3 seeds** (recommend 3–5)
- `tightTransf`: train **≥3 seeds** (recommend 3–5)

Report mean ± std across seeds for trainable methods.

### 7.3 Confidence intervals + paired comparisons
Use query-level bootstrap on the test captions:
- 95% CI for each metric
- paired bootstrap CI for differences (e.g., `tightTransf - seqTransf`)
Optionally apply Holm–Bonferroni across pairwise comparisons.

---

## 8) Run matrix (what you actually execute)

### 8.1 One-time jobs
1) Build cache (video + text embeddings) at `T_max=24`.
2) Verify cache correctness (spot checks, small eval parity test).

### 8.2 Main benchmarking
- `meanP`:
  - eval on test Hard + Origin (no training)
- `seqTransf`:
  - train on `train.json` with cached feats for seeds {0,1,2}
  - choose best checkpoint via `valid.json`
  - eval on test Hard + Origin
- `tightTransf`:
  - same as `seqTransf`

Total full training runs: **2 methods × 3 seeds = 6**, plus `meanP` eval-only.

---

## 9) Reporting format (CLIP4Clip-style tables)

Produce **two tables**:

### Table A — RTime-Origin (test; original-only candidates)
`Method | R@1 | R@5 | R@10 | MdR | MnR | (mean±std across seeds where applicable)`

### Table B — RTime-Hard (test; original+reversed candidates)
Same columns.

Footnotes:
- CLIP backbone (frozen), preprocessing, fps, T_max, sampling policy
- cache dtype (fp16/fp32)
- tie-breaking rule (especially for `meanP` on Hard)
- number of seeds and CI method

---

## 10) Practical “best bang for buck” EC2 approach

**Stage 1 (extraction):**
- one multi-GPU instance (e.g., 4× or 8× GPUs) + fast disk
- run extraction once, store cache on EBS and mirror to S3

**Stage 2 (training):**
- smaller single-GPU instances per run, possibly parallelized across multiple cheap instances
- training becomes fast enough that running 3–5 seeds is usually affordable

---

## Quick checklist (do not skip)
- [ ] CLIP backbone + preprocessing fixed and logged  
- [ ] frame sampling fixed (fps, T_max, deterministic)  
- [ ] cache meta/versioning written and immutable  
- [ ] reversed videos handled by feature flip (or validated if extracted separately)  
- [ ] stable tie-breaking defined (critical for `meanP` on Hard)  
- [ ] ≥3 seeds for `seqTransf` and `tightTransf`  
- [ ] Origin + Hard tables produced with identical metrics and query sets  

