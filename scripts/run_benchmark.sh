#!/bin/bash
# run_benchmark.sh - Run CLIP4Clip benchmark on RTime dataset
#
# Usage:
#   bash scripts/run_benchmark.sh [OPTIONS]
#
# Options:
#   --methods METHOD1,METHOD2,...   Similarity methods to train (default: meanP,seqTransf,tightTransf)
#   --seeds SEED1,SEED2,...         Random seeds to use (default: 0,1,2)
#   --epochs N                      Number of training epochs (default: 5)
#   --batch_size N                  Batch size per GPU (default: 32)
#   --num_gpus N                    Number of GPUs to use (default: 4)
#   --max_frames N                  Max video frames to sample (default: 12)
#   --lr RATE                       Learning rate (default: 1e-4)
#   --dry_run                       Print commands without executing
#   --resume                        Skip completed runs
#   --single METHOD SEED            Run only a single configuration
#
# Example:
#   bash scripts/run_benchmark.sh --methods meanP --seeds 0 --epochs 1 --dry_run

set -e

# Default configuration
METHODS="meanP,seqTransf,tightTransf"
SEEDS="0,1,2"
EPOCHS=5
BATCH_SIZE=32  # Per-GPU batch size, total = BATCH_SIZE * NUM_GPUS
NUM_GPUS=4
MAX_FRAMES=12
MAX_WORDS=32
LR="1e-4"
DRY_RUN=false
RESUME=false
SINGLE_METHOD=""
SINGLE_SEED=""

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
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max_frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --single)
            SINGLE_METHOD="$2"
            SINGLE_SEED="$3"
            shift 3
            ;;
        --help)
            echo "Usage: bash scripts/run_benchmark.sh [OPTIONS]"
            echo ""
            echo "Run CLIP4Clip training on RTime dataset with multiple methods and seeds."
            echo ""
            echo "Options:"
            echo "  --methods M1,M2,...    Similarity methods (default: meanP,seqTransf,tightTransf)"
            echo "  --seeds S1,S2,...      Random seeds (default: 0,1,2)"
            echo "  --epochs N             Training epochs (default: 5)"
            echo "  --batch_size N         Batch size per GPU (default: 32)"
            echo "  --num_gpus N           Number of GPUs (default: 4)"
            echo "  --max_frames N         Max video frames (default: 12)"
            echo "  --lr RATE              Learning rate (default: 1e-4)"
            echo "  --dry_run              Print commands without executing"
            echo "  --resume               Skip completed runs"
            echo "  --single METHOD SEED   Run single configuration"
            echo "  --help                 Show this help message"
            echo ""
            echo "Methods:"
            echo "  meanP       - Mean pooling (parameter-free, fastest)"
            echo "  seqTransf   - Sequential transformer on frame sequence"
            echo "  tightTransf - Tight transformer with cross-attention (most powerful)"
            echo ""
            echo "Examples:"
            echo "  # Full benchmark (3 methods x 3 seeds = 9 runs)"
            echo "  bash scripts/run_benchmark.sh"
            echo ""
            echo "  # Quick test with single configuration"
            echo "  bash scripts/run_benchmark.sh --single meanP 0 --epochs 1"
            echo ""
            echo "  # Dry run to see all commands"
            echo "  bash scripts/run_benchmark.sh --dry_run"
            exit 0
            ;;
        *)
            echo_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Verify required files exist
echo_info "Verifying setup..."

required_files=(
    "data/rtime_train.csv"
    "data/rtime_valid.csv"
    "CLIP4Clip/main_task_retrieval.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo_error "Required file not found: $file"
        if [[ "$file" == data/rtime_* ]]; then
            echo_error "Run 'python scripts/prepare_data.py' first to create CSV files."
        fi
        exit 1
    fi
done

# Check for videos directory
if [ ! -d "data/videos" ] || [ -z "$(ls -A data/videos 2>/dev/null)" ]; then
    echo_warn "data/videos/ is empty or missing."
    echo_warn "Please place video files there before training."
    if [ "$DRY_RUN" = false ]; then
        echo_error "Cannot proceed without video files."
        exit 1
    fi
fi

# Create output directories
mkdir -p ckpts logs

# Parse methods and seeds
if [ -n "$SINGLE_METHOD" ] && [ -n "$SINGLE_SEED" ]; then
    IFS=',' read -ra METHOD_ARRAY <<< "$SINGLE_METHOD"
    IFS=',' read -ra SEED_ARRAY <<< "$SINGLE_SEED"
else
    IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
    IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
fi

# Calculate total batch size
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

echo_info "Configuration:"
echo "  Methods: ${METHOD_ARRAY[*]}"
echo "  Seeds: ${SEED_ARRAY[*]}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE x $NUM_GPUS GPUs = $TOTAL_BATCH_SIZE total"
echo "  Max frames: $MAX_FRAMES"
echo "  Learning rate: $LR"
echo "  Dry run: $DRY_RUN"
echo "  Resume: $RESUME"
echo ""

# Function to check if run is complete
is_run_complete() {
    local output_dir="$1"
    local epochs="$2"

    # Check if final model checkpoint exists
    if [ -f "${output_dir}/pytorch_model.bin.$((epochs - 1))" ]; then
        return 0  # Complete
    fi
    return 1  # Incomplete
}

# Run training
run_count=0
skip_count=0
total_runs=$((${#METHOD_ARRAY[@]} * ${#SEED_ARRAY[@]}))

echo_info "Starting benchmark: $total_runs training runs"
echo ""

for method in "${METHOD_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
        run_count=$((run_count + 1))
        output_dir="ckpts/rtime_${method}_seed${seed}"
        log_file="logs/train_${method}_seed${seed}.log"

        echo_info "[$run_count/$total_runs] Method: $method, Seed: $seed"
        echo "  Output: $output_dir"

        # Check if run is complete (for --resume)
        if [ "$RESUME" = true ] && is_run_complete "$output_dir" "$EPOCHS"; then
            echo_warn "  Skipping: run already complete"
            skip_count=$((skip_count + 1))
            continue
        fi

        # Build command
        cmd="python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
        cmd+=" CLIP4Clip/main_task_retrieval.py"
        cmd+=" --do_train"
        cmd+=" --train_csv data/rtime_train.csv"
        cmd+=" --val_csv data/rtime_valid.csv"
        cmd+=" --features_path data/videos"
        cmd+=" --output_dir $output_dir"
        cmd+=" --datatype rtime"
        cmd+=" --sim_header $method"
        cmd+=" --max_frames $MAX_FRAMES"
        cmd+=" --max_words $MAX_WORDS"
        cmd+=" --epochs $EPOCHS"
        cmd+=" --batch_size $TOTAL_BATCH_SIZE"
        cmd+=" --lr $LR"
        cmd+=" --seed $seed"

        # Add --loose_type for meanP and seqTransf (NOT for tightTransf)
        # Note: main_task_retrieval.py automatically sets loose_type=False for tightTransf
        # but we explicitly add the flag for meanP and seqTransf for clarity
        if [ "$method" = "meanP" ] || [ "$method" = "seqTransf" ]; then
            cmd+=" --loose_type"
        fi

        echo_cmd "$cmd"

        if [ "$DRY_RUN" = false ]; then
            echo "  Logging to: $log_file"

            # Create output directory
            mkdir -p "$output_dir"

            # Run training with logging
            $cmd 2>&1 | tee "$log_file"

            echo_info "  Training complete for $method seed $seed"
        fi

        echo ""
    done
done

# Summary
echo "========================================"
echo_info "Benchmark complete!"
echo "========================================"
echo "  Total runs: $total_runs"
echo "  Completed: $((run_count - skip_count))"
echo "  Skipped: $skip_count"
echo ""
echo "Next steps:"
echo "  1. Evaluate models: bash scripts/evaluate.sh"
echo "  2. Aggregate results: python scripts/aggregate_results.py"
