#!/bin/bash

echo "=========================================="
echo "Running All Baseline Experiments"
echo "=========================================="
echo ""
echo "This will train and evaluate all combinations of:"
echo "  - Models: Conformer, Branchformer"
echo "  - Attention: MHA, MLA, GQA, Linear"
echo ""
echo "Total: 8 models"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create logs directory
mkdir -p logs
mkdir -p results

# Define model and attention combinations
MODELS=("conformer" "branchformer")
ATTENTIONS=("mha" "mla" "gqa" "linear")

# Track timing
START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "Starting experiments..."
echo "=========================================="

# Counter
TOTAL=$((${#MODELS[@]} * ${#ATTENTIONS[@]}))
CURRENT=0

# Run all combinations
for MODEL in "${MODELS[@]}"; do
    for ATTENTION in "${ATTENTIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "=========================================="
        echo "Experiment $CURRENT/$TOTAL: $MODEL with $ATTENTION"
        echo "=========================================="
        
        # Train model
        echo "Training..."
        bash scripts/train_model.sh "$MODEL" "$ATTENTION"
        
        if [ $? -ne 0 ]; then
            echo "Error: Training failed for $MODEL-$ATTENTION"
            continue
        fi
        
        # Evaluate model
        echo ""
        echo "Evaluating..."
        bash scripts/evaluate_model.sh "$MODEL" "$ATTENTION"
        
        if [ $? -ne 0 ]; then
            echo "Error: Evaluation failed for $MODEL-$ATTENTION"
            continue
        fi
        
        echo "Completed: $MODEL with $ATTENTION"
    done
done

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Generating comparison report..."

# Compare results
python scripts/compare_results.py \
    --results_dir results \
    --output_dir comparison

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo ""
echo "Results available in:"
echo "  - Individual results: results/"
echo "  - Comparison report: comparison/"
echo ""
echo "View summary report:"
echo "  cat comparison/summary_report.txt"
echo ""
