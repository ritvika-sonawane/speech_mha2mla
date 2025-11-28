#!/bin/bash

echo "=========================================="
echo "Evaluating All Models"
echo "=========================================="

MODELS=("conformer" "branchformer")
ATTENTIONS=("mha" "mla" "gqa" "linear")

TOTAL=$((${#MODELS[@]} * ${#ATTENTIONS[@]}))
CURRENT=0

for MODEL in "${MODELS[@]}"; do
    for ATTENTION in "${ATTENTIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "[$CURRENT/$TOTAL] Evaluating $MODEL with $ATTENTION..."
        
        bash scripts/evaluate_model.sh "$MODEL" "$ATTENTION"
        
        if [ $? -ne 0 ]; then
            echo "Warning: Evaluation failed for $MODEL-$ATTENTION (model may not be trained)"
        fi
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo ""
echo "Generating comparison report..."

python scripts/compare_results.py

echo ""
echo "Done! Check comparison/ directory for results."
