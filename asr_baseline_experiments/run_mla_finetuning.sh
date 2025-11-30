#!/bin/bash
#
# Fine-tune Conformer with MLA attention
#
# Usage:
#   bash run_mla_finetuning.sh [quick|full]
#
# Options:
#   quick - Fast test with 1000 samples (~30 min)
#   full  - Full fine-tuning on 28k samples (~5-8 hours)

set -e

MODE=${1:-full}

echo "=========================================="
echo "MLA Conformer Fine-tuning"
echo "=========================================="
echo "Mode: $MODE"
echo ""

# Activate environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate speech

cd /home/ubuntu/speech_mha2mla/asr_baseline_experiments

if [ "$MODE" == "quick" ]; then
    echo "Running QUICK test (1000 samples, 5 epochs)"
    python scripts/finetune_mla_conformer.py \
        --base_model facebook/wav2vec2-conformer-rel-pos-large-960h-ft \
        --mla_weights checkpoints/conformer_mla_from_hf/converted_model.pt \
        --output_dir results/conformer_mla_finetuned_quick \
        --num_epochs 5 \
        --learning_rate 1e-5 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_train_samples 1000 \
        --save_steps 200 \
        --eval_steps 200
else
    echo "Running FULL fine-tuning (28k samples, 10 epochs)"
    python scripts/finetune_mla_conformer.py \
        --base_model facebook/wav2vec2-conformer-rel-pos-large-960h-ft \
        --mla_weights checkpoints/conformer_mla_from_hf/converted_model.pt \
        --output_dir results/conformer_mla_finetuned \
        --num_epochs 10 \
        --learning_rate 1e-5 \
        --batch_size 8 \
        --gradient_accumulation_steps 2 \
        --warmup_steps 500 \
        --save_steps 1000 \
        --eval_steps 1000
fi

echo ""
echo "=========================================="
echo "Fine-tuning complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate: python scripts/evaluate_hf_model.py --model_dir results/conformer_mla_finetuned --output_file results/conformer_mla_eval.json"
echo "  2. Compare with baseline: diff results/conformer_mha_baseline_full.json results/conformer_mla_eval.json"
echo ""

