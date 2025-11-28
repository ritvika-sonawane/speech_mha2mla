#!/bin/bash

# Training script for ASR models
# Usage: bash scripts/train_model.sh <model_type> <attention_type>
# Example: bash scripts/train_model.sh conformer mha

MODEL_TYPE=$1
ATTENTION_TYPE=$2

if [ -z "$MODEL_TYPE" ] || [ -z "$ATTENTION_TYPE" ]; then
    echo "Usage: bash scripts/train_model.sh <model_type> <attention_type>"
    echo ""
    echo "Available model types: conformer, branchformer"
    echo "Available attention types: mha, mla, gqa, linear"
    echo ""
    echo "Example: bash scripts/train_model.sh conformer mha"
    exit 1
fi

# Validate inputs
if [[ ! "$MODEL_TYPE" =~ ^(conformer|branchformer)$ ]]; then
    echo "Error: Invalid model type '$MODEL_TYPE'"
    echo "Available: conformer, branchformer"
    exit 1
fi

if [[ ! "$ATTENTION_TYPE" =~ ^(mha|mla|gqa|linear)$ ]]; then
    echo "Error: Invalid attention type '$ATTENTION_TYPE'"
    echo "Available: mha, mla, gqa, linear"
    exit 1
fi

CONFIG_FILE="configs/${MODEL_TYPE}_${ATTENTION_TYPE}.yaml"
OUTPUT_DIR="results/${MODEL_TYPE}_${ATTENTION_TYPE}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Training $MODEL_TYPE with $ATTENTION_TYPE"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run training
python scripts/train.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/train_${MODEL_TYPE}_${ATTENTION_TYPE}.log"

echo ""
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
