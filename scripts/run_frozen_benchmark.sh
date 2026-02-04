#!/bin/bash
# Run complete frozen CLIP benchmark on RTime dataset
#
# This script orchestrates all experiments:
# - meanP: eval only (no trainable params)
# - seqTransf: train + eval
# - tightTransf: train + eval
#
# Usage:
#   bash scripts/run_frozen_benchmark.sh
#
# Environment variables:
#   SEEDS: Space-separated list of seeds (default: "0 1 2")
#   METHODS: Space-separated list of methods (default: "meanP seqTransf tightTransf")
#   NUM_GPUS: Number of GPUs (default: 4)
#   CACHE_PATH: Path to feature cache (default: rtime_cache)

set -e

# Configuration
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"meanP seqTransf tightTransf"}
NUM_GPUS=${NUM_GPUS:-4}
CACHE_PATH=${CACHE_PATH:-rtime_cache}

export NUM_GPUS
export CACHE_PATH

echo "=============================================="
echo "Frozen CLIP Benchmark for RTime"
echo "=============================================="
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"
echo "GPUs: ${NUM_GPUS}"
echo "Cache: ${CACHE_PATH}"
echo "=============================================="
echo ""

# Check if cache exists
if [[ ! -d "${CACHE_PATH}" ]]; then
    echo "Error: Cache directory not found: ${CACHE_PATH}"
    echo "Please run feature extraction first:"
    echo "  python scripts/extract_features.py --output_dir ${CACHE_PATH}"
    exit 1
fi

# Track timing
START_TIME=$(date +%s)

# Run experiments
for method in ${METHODS}; do
    echo ""
    echo "======================================"
    echo "Processing method: ${method}"
    echo "======================================"
    echo ""

    for seed in ${SEEDS}; do
        echo ""
        echo "--- Seed: ${seed} ---"
        echo ""

        # Training (or eval-only for meanP)
        echo ">>> Training ${method} with seed ${seed}..."
        bash scripts/train_frozen.sh ${method} ${seed}

        # Evaluation on both settings
        echo ">>> Evaluating ${method} with seed ${seed}..."
        bash scripts/evaluate_frozen.sh ${method} ${seed}

        echo ""
        echo "Completed: ${method} seed ${seed}"
        echo ""
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Total time: ${DURATION} seconds"
echo ""
echo "Results are saved in:"
echo "  - ckpts/frozen_<method>_seed<seed>/ (model checkpoints)"
echo "  - results/frozen_<method>_seed<seed>_hard/ (RTime-Hard results)"
echo "  - results/frozen_<method>_seed<seed>_origin/ (RTime-Origin results)"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/aggregate_frozen_results.py"
echo "=============================================="
