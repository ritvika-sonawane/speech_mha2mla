#!/bin/bash

# Evaluation script for ASR models
# Usage: bash scripts/evaluate_model.sh <model_type> <attention_type>
# Example: bash scripts/evaluate_model.sh conformer mha

MODEL_TYPE=$1
ATTENTION_TYPE=$2

if [ -z "$MODEL_TYPE" ] || [ -z "$ATTENTION_TYPE" ]; then
    echo "Usage: bash scripts/evaluate_model.sh <model_type> <attention_type>"
    echo ""
    echo "Available model types: conformer, branchformer"
    echo "Available attention types: mha, mla, gqa, linear"
    exit 1
fi

MODEL_DIR="results/${MODEL_TYPE}_${ATTENTION_TYPE}"
CHECKPOINT="$MODEL_DIR/best_model.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please train the model first using: bash scripts/train_model.sh $MODEL_TYPE $ATTENTION_TYPE"
    exit 1
fi

echo "=========================================="
echo "Evaluating $MODEL_TYPE with $ATTENTION_TYPE"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Profile KV cache and evaluate WER
python scripts/profile_kv_cache.py \
    --checkpoint "$CHECKPOINT" \
    --output "$MODEL_DIR/evaluation_results.json" \
    --num_cache_samples 100 \
    --num_wer_samples 500 \
    --eval_wer

echo ""
echo "Evaluation complete!"
echo "Results saved to: $MODEL_DIR/evaluation_results.json"
