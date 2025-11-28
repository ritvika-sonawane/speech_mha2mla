# ASR Baseline Experiments: MHA-to-MLA Conversion for KV Cache Compression

This repository contains scripts to evaluate baseline ASR models with different attention mechanisms for your project on KV cache compression in Branchformer.

## Attention Variants Implemented

1. **MHA (Multi-Head Attention)** - Standard baseline
2. **MLA (Multi-Head Latent Attention)** - Low-rank KV cache compression
3. **GQA (Grouped Query Attention)** - Grouped key-value heads
4. **Linear Attention** - Linear complexity attention mechanism

## Architectures

- **Conformer**: Convolution-augmented Transformer
- **Branchformer**: Parallel branches with MLP and attention

## Quick Start

```bash
# 1. Setup environment
bash setup_environment.sh

# 2. Download and prepare LibriSpeech 100h
bash scripts/prepare_data.sh

# 3. Run all baseline experiments
bash run_all_baselines.sh

# 4. Compare results
python scripts/compare_results.py
```

## Detailed Usage

### Setup

```bash
# Install dependencies
bash setup_environment.sh
```

### Individual Model Training

```bash
# Train Conformer with MHA
bash scripts/train_model.sh conformer mha

# Train Branchformer with MLA
bash scripts/train_model.sh branchformer mla

# Train Conformer with GQA
bash scripts/train_model.sh conformer gqa
```

### Evaluation

```bash
# Evaluate specific model
bash scripts/evaluate_model.sh conformer mha

# Evaluate all models
bash scripts/evaluate_all.sh
```

### KV Cache Analysis

```bash
# Profile KV cache size and inference metrics
python scripts/profile_kv_cache.py --model_path results/conformer_mha/best_model.pt --attention_type mha
```

## Metrics Tracked

- **Word Error Rate (WER)**: Primary ASR accuracy metric
- **KV Cache Size**: Memory footprint per token
- **Inference Time**: Time per utterance
- **Training Time**: Time to convergence
- **Model Parameters**: Total parameter count
- **Memory Usage**: Peak GPU memory during inference

## Directory Structure

```
asr_baseline_experiments/
├── setup_environment.sh          # Environment setup
├── run_all_baselines.sh          # Master script to run all experiments
├── models/                       # Model implementations
│   ├── conformer.py
│   ├── branchformer.py
│   └── attention_variants.py     # MHA, MLA, GQA, Linear
├── scripts/
│   ├── prepare_data.sh           # Download and prepare LibriSpeech
│   ├── train_model.sh            # Training script
│   ├── evaluate_model.sh         # Evaluation script
│   ├── profile_kv_cache.py       # KV cache profiling
│   └── compare_results.py        # Results comparison
├── configs/                      # Model configurations
│   ├── conformer_mha.yaml
│   ├── conformer_mla.yaml
│   ├── conformer_gqa.yaml
│   ├── branchformer_mha.yaml
│   └── ...
└── results/                      # Experiment results
    └── [model_architecture]_[attention_type]/
```

## Expected Results

The experiments will generate:
- Trained model checkpoints
- WER on LibriSpeech test-clean and test-other
- KV cache size comparisons
- Inference speed benchmarks
- Memory usage profiles

## References

- LibriSpeech: http://www.openslr.org/12
- MHA2MLA: https://github.com/JT-Ushio/MHA2MLA
- Conformer: https://arxiv.org/abs/2005.08100
- Branchformer: https://arxiv.org/abs/2207.02971
