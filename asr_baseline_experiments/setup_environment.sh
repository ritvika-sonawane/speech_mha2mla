#!/bin/bash

echo "=========================================="
echo "Setting up ASR Baseline Experiments"
echo "=========================================="

# Check if we're in the correct directory
if [ ! -f "setup_environment.sh" ]; then
    echo "Error: Please run this script from the asr_baseline_experiments directory"
    exit 1
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p models
mkdir -p scripts
mkdir -p configs
mkdir -p results
mkdir -p data
mkdir -p logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --break-system-packages torch torchaudio
pip install --break-system-packages transformers
pip install --break-system-packages datasets
pip install --break-system-packages sentencepiece
pip install --break-system-packages jiwer
pip install --break-system-packages tensorboard
pip install --break-system-packages pandas
pip install --break-system-packages matplotlib
pip install --break-system-packages seaborn
pip install --break-system-packages pyyaml
pip install --break-system-packages einops
pip install --break-system-packages tqdm
pip install --break-system-packages soundfile
pip install --break-system-packages librosa
pip install --break-system-packages omegaconf

# Clone MHA2MLA repository if not exists
if [ ! -d "MHA2MLA" ]; then
    echo "Cloning MHA2MLA repository..."
    git clone https://github.com/JT-Ushio/MHA2MLA.git
else
    echo "MHA2MLA repository already exists, skipping clone..."
fi

# Create Python package init files
touch models/__init__.py
touch scripts/__init__.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare data: bash scripts/prepare_data.sh"
echo "2. Run baselines: bash run_all_baselines.sh"
echo ""
