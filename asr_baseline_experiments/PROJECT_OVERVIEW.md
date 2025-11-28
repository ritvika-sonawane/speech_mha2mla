# ASR Baseline Experiments - Complete Project Overview

## Project Goal

Evaluate different attention mechanisms for KV cache compression in ASR models, specifically:
- Compare cache sizes across attention variants
- Measure WER performance trade-offs
- Analyze inference speed and memory usage
- Provide baseline results for MHA-to-MLA conversion project

## Attention Mechanisms Implemented

### 1. Multi-Head Attention (MHA) - Baseline
- **Description**: Standard attention with full key-value caching
- **Cache Size**: Largest (baseline)
- **Complexity**: O(N²)
- **Use Case**: Baseline for comparison

### 2. Multi-Head Latent Attention (MLA)
- **Description**: Compresses KV into low-dimensional latent vectors
- **Cache Size**: ~50% of MHA (configurable via latent_dim)
- **Complexity**: O(N²) but with reduced memory
- **Use Case**: Memory-efficient inference with minimal accuracy loss

### 3. Grouped Query Attention (GQA)
- **Description**: Uses fewer KV heads than query heads
- **Cache Size**: ~25-30% of MHA (8 Q heads, 2 KV heads)
- **Complexity**: O(N²) with reduced memory
- **Use Case**: Balance between cache compression and accuracy

### 4. Linear Attention
- **Description**: Linear complexity attention using kernel approximation
- **Cache Size**: Similar to MHA
- **Complexity**: O(N) time and space
- **Use Case**: Fast inference on very long sequences

## Model Architectures

### Conformer
- Convolution-augmented Transformer
- Sequential: FF → Attention → Conv → FF
- Strong ASR baseline architecture
- Best for accuracy-focused experiments

### Branchformer  
- Parallel branches: Attention || MLP
- Better parameter efficiency
- Faster training than Conformer
- Best for efficiency-focused experiments

## File Structure

```
asr_baseline_experiments/
│
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── setup_environment.sh        # Environment setup
├── run_all_baselines.sh       # Master script (runs everything)
├── test_setup.sh              # Setup verification
│
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── attention_variants.py  # MHA, MLA, GQA, Linear implementations
│   ├── conformer.py           # Conformer architecture
│   └── branchformer.py        # Branchformer architecture
│
├── scripts/                    # Utility scripts
│   ├── __init__.py
│   ├── prepare_data.sh        # Download LibriSpeech
│   ├── train.py               # Training script
│   ├── train_model.sh         # Training wrapper
│   ├── evaluate_model.sh      # Evaluation wrapper
│   ├── evaluate_all.sh        # Evaluate all models
│   ├── profile_kv_cache.py    # Cache profiling + WER
│   └── compare_results.py     # Results comparison
│
├── configs/                    # Model configurations (8 files)
│   ├── conformer_mha.yaml
│   ├── conformer_mla.yaml
│   ├── conformer_gqa.yaml
│   ├── conformer_linear.yaml
│   ├── branchformer_mha.yaml
│   ├── branchformer_mla.yaml
│   ├── branchformer_gqa.yaml
│   └── branchformer_linear.yaml
│
├── data/                       # Data directory (created by scripts)
│   └── librispeech/           # LibriSpeech dataset
│
├── results/                    # Model outputs (created during training)
│   ├── conformer_mha/
│   │   ├── best_model.pt
│   │   ├── config.yaml
│   │   ├── vocab.json
│   │   ├── evaluation_results.json
│   │   └── logs/
│   └── ... (7 more model directories)
│
├── comparison/                 # Comparison outputs
│   ├── summary_report.txt
│   ├── comparison_table.csv
│   ├── cache_comparison.png
│   ├── wer_vs_cache.png
│   └── inference_time_comparison.png
│
├── logs/                       # Training logs
└── MHA2MLA/                    # Cloned MHA2MLA repository
```

## Metrics Tracked

### Primary Metrics
1. **KV Cache Size (KB/MB)**
   - Actual measured cache size during inference
   - Theoretical cache size calculation
   - Per-token cache overhead

2. **Word Error Rate (WER)**
   - Primary ASR accuracy metric
   - Evaluated on LibriSpeech test-clean and test-other
   - Character-level CTC decoding

3. **Inference Time (ms)**
   - Average time per utterance
   - Includes feature extraction and decoding
   - Measured on GPU

4. **Memory Usage (MB)**
   - Peak GPU memory during inference
   - Helps identify memory bottlenecks
   - Important for deployment

### Secondary Metrics
- Model parameters count
- Training time
- Training loss curves
- Convergence speed

## Experimental Setup

### Dataset
- **Training**: LibriSpeech train-clean-100 (100 hours)
- **Evaluation**: LibriSpeech test-clean, test-other
- **Features**: 80-dimensional mel-spectrograms
- **Sample Rate**: 16 kHz

### Model Configuration (Default)
- **d_model**: 512
- **num_heads**: 8
- **num_layers**: 12
- **dropout**: 0.1
- **vocab**: Character-level (26 chars + special tokens)

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 16
- **Epochs**: 50
- **Loss**: CTC Loss
- **Scheduler**: ReduceLROnPlateau

### Attention-Specific Parameters

**MLA**:
- `latent_dim`: 256 (50% compression)

**GQA**:
- `num_kv_heads`: 2 (4x compression from 8 heads)

**Linear**:
- `feature_dim`: 256

## Usage Workflows

### Workflow 1: Complete Baseline Evaluation
```bash
# Setup
bash setup_environment.sh
bash test_setup.sh
bash scripts/prepare_data.sh

# Run all experiments
bash run_all_baselines.sh

# View results
cat comparison/summary_report.txt
```

### Workflow 2: Individual Model Development
```bash
# Train specific model
bash scripts/train_model.sh conformer mla

# Evaluate
bash scripts/evaluate_model.sh conformer mla

# View results
cat results/conformer_mla/evaluation_results.json
```

### Workflow 3: Custom Configuration
```bash
# Edit config
nano configs/conformer_mla.yaml

# Train with custom config
python scripts/train.py \
    --config configs/conformer_mla.yaml \
    --output_dir results/conformer_mla_custom

# Profile
python scripts/profile_kv_cache.py \
    --checkpoint results/conformer_mla_custom/best_model.pt \
    --eval_wer
```

## Expected Outcomes

### Cache Size Comparison (Relative to MHA)
| Attention | Cache Size | Reduction |
|-----------|-----------|-----------|
| MHA       | 100%      | 0%        |
| MLA       | ~45-55%   | ~45-55%   |
| GQA       | ~25-30%   | ~70-75%   |
| Linear    | ~95-105%  | ~0-5%     |

### Performance Trade-offs
- **MLA**: Best balance (significant cache reduction, <1% WER degradation)
- **GQA**: Maximum compression (largest cache reduction, potential WER drop)
- **Linear**: Fast inference (O(N) complexity, may affect accuracy)

### Inference Speed
- Linear: Fastest on long sequences
- GQA: Faster than MHA
- MLA: Similar to MHA
- MHA: Baseline

## Integration with MHA2MLA

The project includes the MHA2MLA repository for fine-tuning conversion:

1. Train standard MHA model
2. Use MHA2MLA to convert to MLA
3. Fine-tune with minimal data
4. Compare with end-to-end MLA training

Example:
```bash
# Train MHA baseline
bash scripts/train_model.sh conformer mha

# Convert using MHA2MLA (see MHA2MLA/ directory for instructions)
cd MHA2MLA
python convert.py --checkpoint ../results/conformer_mha/best_model.pt

# Compare with our MLA results
python scripts/compare_results.py
```

## Troubleshooting Common Issues

### 1. Out of Memory
**Symptom**: CUDA out of memory error
**Solution**: 
- Reduce batch_size in config files (try 8 or 4)
- Reduce model size (d_model, num_layers)
- Use gradient checkpointing (add to training script)

### 2. Slow Training
**Symptom**: Training takes very long
**Solution**:
- Reduce num_epochs
- Use smaller model (num_layers=6)
- Check GPU utilization
- Reduce num_workers in dataloader

### 3. Poor WER
**Symptom**: WER >50%
**Solution**:
- Train longer (increase num_epochs)
- Check data preprocessing
- Verify vocabulary is correct
- Try different learning rates

### 4. Data Download Fails
**Symptom**: wget errors
**Solution**:
- Download manually from http://www.openslr.org/12/
- Place in data/librispeech/
- Extract manually

### 5. Import Errors
**Symptom**: Module not found
**Solution**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Performance Optimization Tips

1. **Use Mixed Precision Training**
   - Add torch.cuda.amp to training script
   - 2x faster training, same accuracy

2. **Optimize Data Loading**
   - Increase num_workers
   - Use pin_memory=True
   - Pre-process features

3. **Model Optimization**
   - Use torch.jit.script for inference
   - Quantize models for deployment
   - Enable CUDNN benchmarking

4. **Distributed Training**
   - Use torch.distributed for multi-GPU
   - Modify training script for DDP

## Extending the Framework

### Adding New Attention Mechanisms
1. Implement in `models/attention_variants.py`
2. Add factory registration in `get_attention_module()`
3. Create config file in `configs/`
4. Run training and evaluation

### Adding New Architectures
1. Create model file in `models/`
2. Implement build function
3. Update train.py to recognize new model
4. Create config files

### Custom Metrics
1. Add metric calculation to `profile_kv_cache.py`
2. Update comparison script to display new metrics
3. Add visualization in `compare_results.py`

## Research Questions Addressed

1. **How much can MLA reduce KV cache?**
   - Theoretical: 50% (with latent_dim=256)
   - Practical: Measured in experiments

2. **What's the WER cost of cache compression?**
   - Compare WER across all attention types
   - Identify acceptable trade-offs

3. **Is GQA better than MLA for cache compression?**
   - GQA: More compression, possible accuracy drop
   - MLA: Balanced compression and accuracy

4. **Does Linear attention work well for ASR?**
   - O(N) complexity benefit on long sequences
   - Check if accuracy is maintained

5. **Conformer vs Branchformer: which is better?**
   - Compare accuracy, speed, memory
   - Identify best architecture for each use case

## Citation

If you use this code for your research, please cite:

```bibtex
@misc{asr_baseline_experiments,
  title={ASR Baseline Experiments: KV Cache Compression with Attention Variants},
  author={Sonawane, Ritvika and Gokhale, Hrishikesh and Tian, Jinchuan},
  year={2025},
  institution={Carnegie Mellon University}
}
```

## License

MIT License - See individual files for details

## Contact

For questions or issues:
- Check logs in `logs/` directory
- Review error messages carefully
- Verify all dependencies installed
- Ensure sufficient disk space and memory

## Acknowledgments

- LibriSpeech: Panayotov et al.
- Conformer: Gulati et al., 2020
- Branchformer: Peng et al., 2022
- MHA2MLA: JT-Ushio, 2024
- PyTorch and HuggingFace teams
