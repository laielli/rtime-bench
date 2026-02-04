#!/bin/bash
# Evaluate frozen CLIP4Clip model on RTime-Hard and RTime-Origin
#
# Usage:
#   bash scripts/evaluate_frozen.sh <method> <seed> [checkpoint]
#
# Examples:
#   bash scripts/evaluate_frozen.sh seqTransf 0
#   bash scripts/evaluate_frozen.sh tightTransf 1 ckpts/my_model.bin

set -e

METHOD=${1:-seqTransf}
SEED=${2:-0}
CHECKPOINT=${3:-}

# Configuration
CACHE_PATH=${CACHE_PATH:-rtime_cache}
OUTPUT_BASE=${OUTPUT_BASE:-results}
TEST_CSV=${TEST_CSV:-data/rtime_test.csv}
TEST_ORIGIN_CSV=${TEST_ORIGIN_CSV:-data/rtime_test_origin.csv}

MAX_FRAMES=${MAX_FRAMES:-12}
NUM_GPUS=${NUM_GPUS:-4}

# Find checkpoint if not specified
if [[ -z "$CHECKPOINT" ]]; then
    CKPT_DIR="ckpts/frozen_${METHOD}_seed${SEED}"
    # Find the best model (latest epoch)
    CHECKPOINT=$(ls -t ${CKPT_DIR}/pytorch_model.bin.* 2>/dev/null | head -1)
    if [[ -z "$CHECKPOINT" ]] && [[ "$METHOD" != "meanP" ]]; then
        echo "Error: No checkpoint found in ${CKPT_DIR}"
        echo "Please run training first or specify checkpoint path"
        exit 1
    fi
fi

# Set loose_type for meanP and seqTransf
LOOSE_FLAG=""
if [[ "$METHOD" == "meanP" ]] || [[ "$METHOD" == "seqTransf" ]] || [[ "$METHOD" == "seqLSTM" ]]; then
    LOOSE_FLAG="--loose_type"
fi

# Optional init_model flag
INIT_FLAG=""
if [[ -n "$CHECKPOINT" ]] && [[ -f "$CHECKPOINT" ]]; then
    INIT_FLAG="--init_model ${CHECKPOINT}"
fi

echo "=============================================="
echo "Evaluating Frozen CLIP4Clip"
echo "=============================================="
echo "Method: ${METHOD}"
echo "Seed: ${SEED}"
echo "Checkpoint: ${CHECKPOINT:-'(using pretrained CLIP)'}"
echo "=============================================="

# =====================================
# Evaluate on RTime-Hard (original + reversed)
# =====================================
OUTPUT_DIR="${OUTPUT_BASE}/frozen_${METHOD}_seed${SEED}_hard"
mkdir -p ${OUTPUT_DIR}

echo ""
echo ">>> Evaluating on RTime-Hard..."
echo ""

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
    CLIP4Clip/main_task_retrieval.py \
    --do_eval \
    --use_cached_embeddings \
    --cache_path ${CACHE_PATH} \
    --datatype rtime_cached \
    --sim_header ${METHOD} \
    ${LOOSE_FLAG} \
    ${INIT_FLAG} \
    --test_csv ${TEST_CSV} \
    --val_csv ${TEST_CSV} \
    --max_frames ${MAX_FRAMES} \
    --batch_size_val 1000 \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR}

# =====================================
# Evaluate on RTime-Origin (original only)
# =====================================
OUTPUT_DIR="${OUTPUT_BASE}/frozen_${METHOD}_seed${SEED}_origin"
mkdir -p ${OUTPUT_DIR}

echo ""
echo ">>> Evaluating on RTime-Origin..."
echo ""

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
    CLIP4Clip/main_task_retrieval.py \
    --do_eval \
    --use_cached_embeddings \
    --cache_path ${CACHE_PATH} \
    --datatype rtime_cached \
    --sim_header ${METHOD} \
    ${LOOSE_FLAG} \
    ${INIT_FLAG} \
    --test_csv ${TEST_ORIGIN_CSV} \
    --val_csv ${TEST_ORIGIN_CSV} \
    --max_frames ${MAX_FRAMES} \
    --batch_size_val 1000 \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR}

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_BASE}/frozen_${METHOD}_seed${SEED}_*"
echo "=============================================="
