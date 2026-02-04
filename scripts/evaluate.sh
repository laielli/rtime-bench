#!/bin/bash
# evaluate.sh - Evaluate trained CLIP4Clip models on RTime test sets
#
# Usage:
#   bash scripts/evaluate.sh [OPTIONS]
#
# Options:
#   --methods METHOD1,METHOD2,...   Methods to evaluate (default: all found in ckpts/)
#   --seeds SEED1,SEED2,...         Seeds to evaluate (default: all found in ckpts/)
#   --settings SETTING1,...         Test settings: origin, hard, or both (default: origin,hard)
#   --epoch N                       Specific epoch to evaluate (default: best by validation)
#   --num_gpus N                    Number of GPUs (default: 4)
#   --dry_run                       Print commands without executing
#
# Example:
#   bash scripts/evaluate.sh --methods meanP --seeds 0 --settings hard --dry_run

set -e

# Default configuration
METHODS=""
SEEDS=""
SETTINGS="origin,hard"
EPOCH=""
NUM_GPUS=4
MAX_FRAMES=12
MAX_WORDS=32
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_cmd() { echo -e "${BLUE}[CMD]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --settings)
            SETTINGS="$2"
            shift 2
            ;;
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: bash scripts/evaluate.sh [OPTIONS]"
            echo ""
            echo "Evaluate trained CLIP4Clip models on RTime test sets."
            echo ""
            echo "Options:"
            echo "  --methods M1,M2,...    Methods to evaluate (default: auto-detect)"
            echo "  --seeds S1,S2,...      Seeds to evaluate (default: auto-detect)"
            echo "  --settings S1,S2,...   Test settings: origin,hard (default: both)"
            echo "  --epoch N              Specific epoch (default: best by validation)"
            echo "  --num_gpus N           Number of GPUs (default: 4)"
            echo "  --dry_run              Print commands without executing"
            echo "  --help                 Show this help message"
            echo ""
            echo "Test Settings:"
            echo "  origin - Original videos only (~1000 candidates)"
            echo "  hard   - Original + reversed videos (~2000 candidates)"
            exit 0
            ;;
        *)
            echo_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Auto-detect methods and seeds if not specified
if [ -z "$METHODS" ]; then
    # Find all methods from checkpoint directories
    METHODS=$(ls -d ckpts/rtime_*_seed* 2>/dev/null | sed 's/.*rtime_\(.*\)_seed.*/\1/' | sort -u | tr '\n' ',' | sed 's/,$//')
    if [ -z "$METHODS" ]; then
        echo_error "No checkpoint directories found in ckpts/"
        echo_error "Run 'bash scripts/run_benchmark.sh' first to train models."
        exit 1
    fi
fi

if [ -z "$SEEDS" ]; then
    # Find all seeds from checkpoint directories
    SEEDS=$(ls -d ckpts/rtime_*_seed* 2>/dev/null | sed 's/.*_seed\([0-9]*\)/\1/' | sort -u | tr '\n' ',' | sed 's/,$//')
fi

# Parse arrays
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=',' read -ra SETTING_ARRAY <<< "$SETTINGS"

echo_info "Configuration:"
echo "  Methods: ${METHOD_ARRAY[*]}"
echo "  Seeds: ${SEED_ARRAY[*]}"
echo "  Settings: ${SETTING_ARRAY[*]}"
echo "  Num GPUs: $NUM_GPUS"
echo "  Dry run: $DRY_RUN"
echo ""

# Create results directory
mkdir -p results

# Function to find best checkpoint by validation R@1
find_best_checkpoint() {
    local ckpt_dir="$1"
    local log_file="${ckpt_dir}/log.txt"

    if [ ! -f "$log_file" ]; then
        # Fallback: find latest checkpoint
        local latest=$(ls -t "${ckpt_dir}"/pytorch_model.bin.* 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "$latest"
        fi
        return
    fi

    # Parse log for best R@1 and corresponding epoch
    # Log format: "The best model is: <path>, the R1 is: <score>"
    local best_line=$(grep "The best model is:" "$log_file" | tail -1)
    if [ -n "$best_line" ]; then
        local best_model=$(echo "$best_line" | sed 's/.*The best model is: \([^,]*\).*/\1/')
        if [ -f "$best_model" ]; then
            echo "$best_model"
            return
        fi
    fi

    # Fallback: use latest checkpoint
    local latest=$(ls -t "${ckpt_dir}"/pytorch_model.bin.* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    fi
}

# Run evaluations
eval_count=0
total_evals=$((${#METHOD_ARRAY[@]} * ${#SEED_ARRAY[@]} * ${#SETTING_ARRAY[@]}))

echo_info "Starting evaluation: $total_evals runs"
echo ""

for method in "${METHOD_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        ckpt_dir="ckpts/rtime_${method}_seed${seed}"

        # Check if checkpoint directory exists
        if [ ! -d "$ckpt_dir" ]; then
            echo_warn "Checkpoint directory not found: $ckpt_dir, skipping..."
            continue
        fi

        # Find best checkpoint or use specified epoch
        if [ -n "$EPOCH" ]; then
            model_path="${ckpt_dir}/pytorch_model.bin.${EPOCH}"
        else
            model_path=$(find_best_checkpoint "$ckpt_dir")
        fi

        if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
            echo_warn "No checkpoint found in $ckpt_dir, skipping..."
            continue
        fi

        echo_info "Method: $method, Seed: $seed"
        echo "  Using checkpoint: $model_path"

        for setting in "${SETTING_ARRAY[@]}"; do
            eval_count=$((eval_count + 1))

            # Select test CSV based on setting
            if [ "$setting" = "origin" ]; then
                test_csv="data/rtime_test_origin.csv"
            else
                test_csv="data/rtime_test.csv"
            fi

            result_file="results/eval_${method}_seed${seed}_${setting}.txt"

            echo_info "  [$eval_count/$total_evals] Evaluating on $setting setting"

            # Build evaluation command
            cmd="python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
            cmd+=" CLIP4Clip/main_task_retrieval.py"
            cmd+=" --do_eval"
            cmd+=" --val_csv $test_csv"
            cmd+=" --features_path data/videos"
            cmd+=" --output_dir $ckpt_dir"
            cmd+=" --init_model $model_path"
            cmd+=" --datatype rtime"
            cmd+=" --sim_header $method"
            cmd+=" --max_frames $MAX_FRAMES"
            cmd+=" --max_words $MAX_WORDS"

            # Add --loose_type for meanP and seqTransf
            if [ "$method" = "meanP" ] || [ "$method" = "seqTransf" ]; then
                cmd+=" --loose_type"
            fi

            echo_cmd "$cmd"

            if [ "$DRY_RUN" = false ]; then
                # Run evaluation and capture output
                $cmd 2>&1 | tee "$result_file"

                # Extract and display key metrics
                echo ""
                echo "  Results saved to: $result_file"

                # Extract metrics from output
                if grep -q "Text-to-Video:" "$result_file"; then
                    t2v_metrics=$(grep -A1 "Text-to-Video:" "$result_file" | tail -1)
                    echo "  Text-to-Video: $t2v_metrics"
                fi
                if grep -q "Video-to-Text:" "$result_file"; then
                    v2t_metrics=$(grep -A1 "Video-to-Text:" "$result_file" | tail -1)
                    echo "  Video-to-Text: $v2t_metrics"
                fi
            fi

            echo ""
        done
    done
done

# Summary
echo "========================================"
echo_info "Evaluation complete!"
echo "========================================"
echo "  Total evaluations: $eval_count"
echo "  Results saved to: results/"
echo ""
echo "Next step:"
echo "  python scripts/aggregate_results.py"
