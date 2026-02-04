#!/bin/bash
# Train CLIP4Clip with frozen CLIP backbone using cached embeddings
#
# Usage:
#   bash scripts/train_frozen.sh <method> <seed> [extra_args]
#
# Examples:
#   bash scripts/train_frozen.sh seqTransf 0
#   bash scripts/train_frozen.sh tightTransf 1 --epochs 15
#   bash scripts/train_frozen.sh meanP 0  # meanP has no trainable params, just eval

set -e

METHOD=${1:-seqTransf}
SEED=${2:-0}
shift 2 || shift $#

# Configuration
CACHE_PATH=${CACHE_PATH:-rtime_cache}
OUTPUT_BASE=${OUTPUT_BASE:-ckpts}
TRAIN_CSV=${TRAIN_CSV:-data/rtime_train.csv}
VAL_CSV=${VAL_CSV:-data/rtime_valid.csv}
TEST_CSV=${TEST_CSV:-data/rtime_test.csv}

# Training hyperparameters
MAX_FRAMES=${MAX_FRAMES:-12}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-4}
NUM_GPUS=${NUM_GPUS:-4}

OUTPUT_DIR="${OUTPUT_BASE}/frozen_${METHOD}_seed${SEED}"

echo "=============================================="
echo "Training Frozen CLIP4Clip"
echo "=============================================="
echo "Method: ${METHOD}"
echo "Seed: ${SEED}"
echo "Output: ${OUTPUT_DIR}"
echo "Cache path: ${CACHE_PATH}"
echo "=============================================="

# Set loose_type for meanP and seqTransf
LOOSE_FLAG=""
if [[ "$METHOD" == "meanP" ]] || [[ "$METHOD" == "seqTransf" ]] || [[ "$METHOD" == "seqLSTM" ]]; then
    LOOSE_FLAG="--loose_type"
fi

# Check if we should skip training for meanP (no trainable params)
if [[ "$METHOD" == "meanP" ]]; then
    echo "Note: meanP has no trainable parameters. Running evaluation only."

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
        CLIP4Clip/main_task_retrieval.py \
        --do_eval \
        --use_cached_embeddings \
        --cache_path ${CACHE_PATH} \
        --datatype rtime_cached \
        --sim_header ${METHOD} \
        ${LOOSE_FLAG} \
        --train_csv ${TRAIN_CSV} \
        --val_csv ${VAL_CSV} \
        --test_csv ${TEST_CSV} \
        --max_frames ${MAX_FRAMES} \
        --batch_size_val 1000 \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        "$@"
else
    # Train with frozen CLIP
    python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
        CLIP4Clip/main_task_retrieval.py \
        --do_train \
        --use_cached_embeddings \
        --cache_path ${CACHE_PATH} \
        --datatype rtime_cached \
        --sim_header ${METHOD} \
        ${LOOSE_FLAG} \
        --train_csv ${TRAIN_CSV} \
        --val_csv ${VAL_CSV} \
        --test_csv ${TEST_CSV} \
        --max_frames ${MAX_FRAMES} \
        --batch_size ${BATCH_SIZE} \
        --batch_size_val 1000 \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --warmup_proportion 0.1 \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR} \
        "$@"
fi

echo "=============================================="
echo "Training complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
