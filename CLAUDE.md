# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rtime-bench is a benchmarking framework integrating CLIP4Clip (video-text retrieval model) with the Reversed-in-Time (RTime) dataset to evaluate temporal understanding in cross-modal video-text retrieval.

## Common Commands

### Training (Distributed)
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  CLIP4Clip/main_task_retrieval.py --do_train \
  --train_csv data/rtime_train.csv \
  --val_csv data/rtime_valid.csv \
  --features_path data/videos \
  --output_dir ckpts/rtime_meanP_seed0 \
  --datatype rtime \
  --sim_header meanP \
  --max_frames 12 \
  --epochs 5 \
  --batch_size 128 \
  --lr 1e-4
```

### Evaluation Only
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  CLIP4Clip/main_task_retrieval.py --do_eval \
  --test_csv data/rtime_test.csv \
  --features_path data/videos \
  --init_model ckpts/rtime_meanP_seed0/pytorch_model.bin.1
```

### Video Preprocessing
```bash
python CLIP4Clip/preprocess/compress_video.py \
  --input_root raw_videos/ \
  --output_root compressed_videos/
```

## Architecture

### Core Components

**CLIP4Clip Model Pipeline:**
```
Text → CLIP Text Encoder → Text Embeddings
Video Frames → CLIP Visual Encoder → Visual Embeddings
                    ↓
        Cross-Modal Transformer (4 layers, 512 hidden)
                    ↓
        Similarity Computation (meanP | seqTransf | tightTransf)
```

**Three Similarity Methods:**
- `meanP`: Parameter-free mean pooling (fastest, baseline)
- `seqTransf`: Sequential transformer on frame sequence
- `tightTransf`: Tight transformer with cross-attention (most powerful)

### Key Files

| File | Purpose |
|------|---------|
| `CLIP4Clip/main_task_retrieval.py` | Entry point: training loop, evaluation, CLI |
| `CLIP4Clip/modules/modeling.py` | CLIP4Clip model architecture |
| `CLIP4Clip/modules/module_clip.py` | CLIP backbone (ViT-B/32 or ViT-B/16) |
| `CLIP4Clip/metrics.py` | R@1, R@5, R@10, MedianR, MeanR computation |
| `CLIP4Clip/dataloaders/data_dataloaders.py` | Dataset router for different benchmarks |
| `CLIP4Clip/dataloaders/rawvideo_util.py` | Video frame extraction |

### RTime Dataset

Located in `Reversed-in-Time/` with train/valid/test JSON splits.

**Data Format:**
```json
{
  "video_id": {
    "temporal": true,
    "reverse": false,
    "forward_captions": ["..."],
    "reverse_captions": ["..."],
    "forward_rewrites": ["..."],  // GPT-4 rewrites
    "reverse_rewrites": ["..."]
  }
}
```

**Evaluation Settings:**
- **RTime-Origin**: Original videos only (~1,000 candidates)
- **RTime-Hard**: Original + reversed videos (~2,000 candidates)

## Key CLI Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--datatype` | msrvtt, msvd, lsmdc, activity, didemo, rtime | Dataset type |
| `--sim_header` | meanP, seqTransf, tightTransf | Similarity computation method |
| `--pretrained_clip_name` | ViT-B/32, ViT-B/16 | CLIP backbone |
| `--freeze_layer_num` | 0-12 | CLIP layers to freeze |
| `--max_frames` | int | Video frames to sample (default: 12) |
| `--max_words` | int | Text token limit (default: 32) |
| `--loose_type` | flag | Use loose similarity (required for meanP) |

## Default Training Settings

- Global batch size: 128 (with gradient accumulation across GPUs)
- Learning rate: 1e-4
- Warmup: 10% of training steps
- Max gradient norm: 1.0
- Feature framerate: 1 fps

## Adding New Datasets

1. Create `CLIP4Clip/dataloaders/dataloader_<name>_retrieval.py`
2. Register in `CLIP4Clip/dataloaders/data_dataloaders.py` (DATALOADER_DICT)
3. Create CSV with columns: `video_id`, `sentence`

## Benchmarking Protocol

See `clip4clip_rtime_benchmark_plan.md` for:
- Multi-seed experiments (≥3 seeds with mean ± std)
- Paired bootstrap for confidence intervals
- Holm-Bonferroni correction for multiple comparisons
