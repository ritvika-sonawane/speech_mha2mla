# Quick Start Guide

This guide will help you get started with the ASR baseline experiments.

## Prerequisites

- Linux system (Ubuntu 20.04+ recommended)
- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 100GB free disk space for LibriSpeech data
- 16GB+ RAM

## Step-by-Step Setup

### 1. Setup Environment (5 minutes)

```bash
# Make setup script executable
chmod +x setup_environment.sh

# Run setup
bash setup_environment.sh
```

This will:
- Install all Python dependencies
- Clone the MHA2MLA repository
- Create necessary directories

### 2. Download Data (30-60 minutes)

```bash
# Download and prepare LibriSpeech 100h
bash scripts/prepare_data.sh
```

This downloads:
- LibriSpeech train-clean-100 (6.3 GB)
- LibriSpeech test-clean (346 MB)
- LibriSpeech test-other (328 MB)

### 3. Run Experiments

#### Option A: Run All Baselines (Recommended for complete evaluation)

```bash
# This will train and evaluate all 8 model combinations
# Estimated time: 12-24 hours depending on GPU
bash run_all_baselines.sh
```

This trains and evaluates:
- Conformer: MHA, MLA, GQA, Linear
- Branchformer: MHA, MLA, GQA, Linear

#### Option B: Run Individual Models

Train a specific model:
```bash
bash scripts/train_model.sh conformer mha
```

Evaluate a specific model:
```bash
bash scripts/evaluate_model.sh conformer mha
```

### 4. View Results

After running experiments, view the comparison report:

```bash
cat comparison/summary_report.txt
```

Or explore individual results:
```bash
ls results/
# Each directory contains:
# - best_model.pt: trained model
# - config.yaml: model configuration
# - vocab.json: vocabulary
# - evaluation_results.json: metrics
# - logs/: tensorboard logs
```

## Quick Commands Reference

```bash
# Train Conformer with different attentions
bash scripts/train_model.sh conformer mha
bash scripts/train_model.sh conformer mla
bash scripts/train_model.sh conformer gqa
bash scripts/train_model.sh conformer linear

# Train Branchformer with different attentions
bash scripts/train_model.sh branchformer mha
bash scripts/train_model.sh branchformer mla
bash scripts/train_model.sh branchformer gqa
bash scripts/train_model.sh branchformer linear

# Evaluate specific model
bash scripts/evaluate_model.sh conformer mha

# Evaluate all trained models
bash scripts/evaluate_all.sh

# Compare all results
python scripts/compare_results.py
```

## Expected Training Time

On a single V100 GPU:
- One model (50 epochs): ~1.5-2 hours
- All 8 models: ~12-16 hours

On a RTX 3090:
- One model: ~2-3 hours
- All 8 models: ~16-24 hours

## Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir results/conformer_mha/logs
```

Check training logs:
```bash
tail -f logs/train_conformer_mha.log
```

## Expected Results

Based on similar experiments, you should expect:

**KV Cache Compression (relative to MHA baseline):**
- MLA: ~50-60% reduction
- GQA: ~70-75% reduction
- Linear: ~0-10% reduction

**WER (on LibriSpeech test-clean):**
- All models: ~10-20% WER (depends on training)
- Differences between attention types: <5% relative

**Inference Time:**
- Linear attention: fastest
- MHA: baseline
- MLA: similar to MHA
- GQA: faster than MHA

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config files
# Edit configs/*.yaml and change batch_size: 16 to batch_size: 8 or 4
```

### Data Download Issues
```bash
# Download manually from: http://www.openslr.org/12/
# Place files in data/librispeech/
```

### Import Errors
```bash
# Reinstall dependencies
pip install --break-system-packages --upgrade torch torchaudio transformers datasets
```

## Next Steps

1. Review results in `comparison/summary_report.txt`
2. Analyze plots in `comparison/` directory
3. Check detailed metrics in `results/*/evaluation_results.json`
4. Experiment with different hyperparameters in config files
5. Try MHA2MLA conversion on trained MHA models (see MHA2MLA/ directory)

## Support

For issues:
1. Check logs in `logs/` directory
2. Review error messages
3. Verify data downloaded correctly
4. Ensure sufficient disk space and memory
