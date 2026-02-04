#!/bin/bash
# setup.sh - Environment setup for RTime benchmark with CLIP4Clip
#
# Usage: bash scripts/setup.sh
#
# This script:
#   1. Creates a conda environment with Python 3.8 and PyTorch 1.7.1
#   2. Installs required pip dependencies
#   3. Downloads CLIP ViT-B/32 weights
#   4. Creates necessary directory structure
#   5. Verifies RTime dataset files exist

set -e  # Exit on error

# Configuration
ENV_NAME="clip4clip"
PYTHON_VERSION="3.8"
PYTORCH_VERSION="1.7.1"
CUDA_VERSION="11.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
SKIP_CONDA=false
SKIP_WEIGHTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --skip-weights)
            SKIP_WEIGHTS=true
            shift
            ;;
        --help)
            echo "Usage: bash scripts/setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-conda     Skip conda environment creation"
            echo "  --skip-weights   Skip CLIP weights download"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"
echo_info "Project directory: $PROJECT_DIR"

# Step 1: Create conda environment
if [ "$SKIP_CONDA" = false ]; then
    echo_info "Step 1: Creating conda environment '$ENV_NAME'..."

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo_error "conda not found. Please install Anaconda or Miniconda first."
        exit 1
    fi

    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo_warn "Environment '$ENV_NAME' already exists. Skipping creation."
        echo_info "To recreate, run: conda env remove -n $ENV_NAME"
    else
        echo_info "Creating conda environment with Python $PYTHON_VERSION..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    fi

    echo_info "Installing PyTorch $PYTORCH_VERSION with CUDA $CUDA_VERSION..."

    # Activate environment and install packages
    # Note: Using conda run to avoid issues with shell initialization
    conda run -n "$ENV_NAME" pip install torch==${PYTORCH_VERSION}+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

    echo_info "Installing pip dependencies..."
    conda run -n "$ENV_NAME" pip install \
        ftfy \
        regex \
        tqdm \
        opencv-python \
        boto3 \
        requests \
        pandas \
        ffmpeg-python \
        scipy \
        scikit-learn
else
    echo_info "Step 1: Skipping conda environment creation (--skip-conda)"
fi

# Step 2: Create directory structure
echo_info "Step 2: Creating directory structure..."

directories=(
    "data/videos"
    "ckpts"
    "logs"
    "results"
    "weights"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo_info "  Created: $dir/"
    else
        echo_info "  Exists: $dir/"
    fi
done

# Step 3: Download CLIP weights
CLIP_WEIGHTS_URL="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
CLIP_WEIGHTS_PATH="weights/ViT-B-32.pt"

if [ "$SKIP_WEIGHTS" = false ]; then
    echo_info "Step 3: Downloading CLIP ViT-B/32 weights..."

    if [ -f "$CLIP_WEIGHTS_PATH" ]; then
        echo_info "  CLIP weights already exist at $CLIP_WEIGHTS_PATH"
    else
        echo_info "  Downloading from: $CLIP_WEIGHTS_URL"
        if command -v wget &> /dev/null; then
            wget -O "$CLIP_WEIGHTS_PATH" "$CLIP_WEIGHTS_URL"
        elif command -v curl &> /dev/null; then
            curl -L -o "$CLIP_WEIGHTS_PATH" "$CLIP_WEIGHTS_URL"
        else
            echo_error "Neither wget nor curl found. Please install one of them."
            exit 1
        fi
        echo_info "  Downloaded CLIP weights to $CLIP_WEIGHTS_PATH"
    fi
else
    echo_info "Step 3: Skipping CLIP weights download (--skip-weights)"
fi

# Step 4: Verify RTime dataset files
echo_info "Step 4: Verifying RTime dataset files..."

rtime_files=(
    "Reversed-in-Time/train.json"
    "Reversed-in-Time/valid.json"
    "Reversed-in-Time/test.json"
)

all_files_exist=true
for file in "${rtime_files[@]}"; do
    if [ -f "$file" ]; then
        echo_info "  Found: $file"
    else
        echo_error "  Missing: $file"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo_warn "Some RTime dataset files are missing."
    echo_warn "Please ensure the Reversed-in-Time directory contains train.json, valid.json, and test.json"
fi

# Step 5: Verify CLIP4Clip code
echo_info "Step 5: Verifying CLIP4Clip code..."

clip4clip_files=(
    "CLIP4Clip/main_task_retrieval.py"
    "CLIP4Clip/modules/modeling.py"
    "CLIP4Clip/dataloaders/data_dataloaders.py"
)

for file in "${clip4clip_files[@]}"; do
    if [ -f "$file" ]; then
        echo_info "  Found: $file"
    else
        echo_error "  Missing: $file"
    fi
done

# Final summary
echo ""
echo "========================================"
echo_info "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: conda activate $ENV_NAME"
echo "  2. Prepare the data: python scripts/prepare_data.py"
echo "  3. Place videos in: data/videos/"
echo "  4. Run the benchmark: bash scripts/run_benchmark.sh"
echo ""

# Verification command
if [ "$SKIP_CONDA" = false ]; then
    echo "To verify CUDA is available, run:"
    echo "  conda activate $ENV_NAME"
    echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
fi
