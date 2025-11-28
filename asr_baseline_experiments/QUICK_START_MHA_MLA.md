# Quick Start: MHA and MLA for ASR

## TL;DR

✅ **Everything is implemented and ready!**

Your Conformer and Branchformer models now support:
- **MHA** (Multi-Head Attention) - baseline
- **MLA** (Multi-Head Latent Attention) - 50% cache reduction
- **MLA-Simple** - faster MLA variant
- **GQA** and **Linear** attention for comparisons

## Quick Commands

### 1. Train MHA Baseline
```bash
cd asr_baseline_experiments
bash scripts/train_model.sh configs/conformer_mha.yaml experiments/conformer_mha
```

### 2. Train MLA from Scratch
```bash
bash scripts/train_model.sh configs/conformer_mla.yaml experiments/conformer_mla
```

### 3. Convert MHA → MLA (Recommended Workflow)
```bash
# Step 1: Train MHA first
bash scripts/train_model.sh configs/conformer_mha.yaml experiments/conformer_mha

# Step 2: Convert using SVD
python scripts/convert_mha_to_mla.py \
  --input_checkpoint experiments/conformer_mha/best_model.pt \
  --output_checkpoint experiments/conformer_mla/init_model.pt \
  --config configs/conformer_mha.yaml \
  --latent_dim 512 \
  --mla_variant simple \
  --test_quality

# Step 3: Fine-tune (3-5 epochs)
bash scripts/train_model.sh configs/conformer_mla.yaml experiments/conformer_mla
```

## What's Available

### Models
- `models/conformer.py` - Conformer with any attention type
- `models/branchformer.py` - Branchformer with any attention type

### Attention Mechanisms
- `models/attention_variants.py`:
  - `MultiHeadAttention` - Standard MHA
  - `MultiHeadLatentAttention` - Full MLA (per-head projections)
  - `MultiHeadLatentAttentionSimple` - Efficient MLA (shared projections)
  - `GroupedQueryAttention` - GQA
  - `LinearAttention` - Linear complexity

### Conversion Tools
- `models/mha_to_mla_conversion.py`:
  - `convert_mha_to_mla_simple_svd()` - Convert using SVD
  - `convert_mha_to_mla_svd()` - Full MLA conversion
  - `analyze_compression_quality()` - Check conversion quality

### Scripts
- `scripts/convert_mha_to_mla.py` - CLI tool for conversion
- `test_mha_mla.py` - Test suite

### Configs (Ready to Use)
```
configs/
├── conformer_mha.yaml        # Baseline
├── conformer_mla.yaml        # 50% cache reduction
├── branchformer_mha.yaml     # Baseline
└── branchformer_mla.yaml     # 50% cache reduction
```

## Key Parameters

### MLA Configuration
```yaml
attention_type: mla  # or 'mla_simple' for faster variant
attention_kwargs:
  latent_dim: 512    # Set to d_model for 50% compression
                     # 256 for 75% compression
                     # 768 for 25% compression
```

### Compression Ratios
| latent_dim | Cache Reduction | Trade-off |
|------------|-----------------|-----------|
| 128 | 87.5% | Highest compression, may affect accuracy |
| 256 | 75% | Good balance |
| 512 (d_model) | **50%** | **Recommended - minimal accuracy loss** |
| 768 | 25% | Lower compression, higher accuracy |

## Expected Results (from Paper)

### Baselines (Already Achieved)
- **Conformer + MHA**: 1.18% WER on LibriSpeech 100h
- **Branchformer + MHA**: 8.11% WER on TED-LIUM2

### Targets (After MLA Conversion)
- **ΔWE R < 5%** (i.e., < 1.24% for Conformer, < 8.5% for Branchformer)
- **50% KV cache reduction**
- **Similar inference time**

### Cache Sizes (1000-frame sequence)
| Model | MHA | MLA | Reduction |
|-------|-----|-----|-----------|
| Conformer | 187.5 MB | 93.75 MB | **50%** |
| Branchformer | 70.31 MB | 35.16 MB | **50%** |

## File Structure
```
asr_baseline_experiments/
├── models/
│   ├── attention_variants.py          # ✅ MHA, MLA implementations
│   ├── conformer.py                   # ✅ Supports all attention types
│   ├── branchformer.py                # ✅ Supports all attention types
│   ├── mha_to_mla_conversion.py       # ✅ NEW: Conversion utilities
│   ├── ATTENTION_MECHANISMS.md        # ✅ NEW: Full documentation
│   └── __init__.py                    # ✅ Updated exports
├── scripts/
│   ├── train.py                       # Training loop
│   ├── train_model.sh                 # Training script
│   ├── evaluate_model.sh              # Evaluation
│   ├── convert_mha_to_mla.py          # ✅ NEW: CLI conversion tool
│   └── compare_results.py             # Result comparison
├── configs/
│   ├── conformer_mha.yaml             # ✅ Ready
│   ├── conformer_mla.yaml             # ✅ Ready
│   ├── branchformer_mha.yaml          # ✅ Ready
│   └── branchformer_mla.yaml          # ✅ Ready
├── test_mha_mla.py                    # ✅ NEW: Test suite
├── IMPLEMENTATION_SUMMARY.md          # ✅ NEW: Detailed summary
└── QUICK_START_MHA_MLA.md             # ✅ NEW: This file
```

## Python Usage Examples

### Use MLA in Code
```python
from models import build_conformer

config = {
    'attention_type': 'mla',
    'attention_kwargs': {'latent_dim': 512},
    'd_model': 512,
    'num_heads': 8,
    # ... other params
}

model = build_conformer(config)
```

### Convert Existing Model
```python
from models import convert_mha_to_mla_simple_svd
import torch

# Load trained MHA model
model = build_conformer(mha_config)
model.load_state_dict(torch.load('checkpoint.pt'))

# Convert attention layers
for i, layer in enumerate(model.layers):
    layer.attention = convert_mha_to_mla_simple_svd(
        layer.attention,
        latent_dim=512
    )

# Save converted model
torch.save(model.state_dict(), 'converted_mla.pt')
```

### Analyze Conversion Quality
```python
from models import analyze_compression_quality

metrics = analyze_compression_quality(
    mha_module=mha_attention,
    mla_module=mla_attention,
    test_input=test_tensor,
    verbose=True
)

if metrics['relative_error'] < 0.05:
    print("Excellent conversion!")
elif metrics['relative_error'] < 0.10:
    print("Good conversion - fine-tune 3-5 epochs")
else:
    print("Needs more fine-tuning")
```

## Documentation

📖 **Full details**: `models/ATTENTION_MECHANISMS.md`
📊 **Implementation summary**: `IMPLEMENTATION_SUMMARY.md`
📄 **Research paper**: `Reports/rsonawan_hgokhale_checkpoint.pdf`

## Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the right directory
cd asr_baseline_experiments

# Check Python path
python -c "import sys; print(sys.path)"

# Try importing
python -c "from models import build_conformer; print('OK')"
```

### Issue: PyTorch not found
```bash
# Check installation
python -c "import torch; print(torch.__version__)"

# If missing, install:
pip install torch torchaudio
```

### Issue: Out of memory
- Reduce `batch_size` in config
- Use `mla_simple` instead of full `mla`
- Enable gradient checkpointing

### Issue: High WER after conversion
- Fine-tune for more epochs (5-10 instead of 3)
- Use lower learning rate (1e-5)
- Try higher `latent_dim` (512 or 768)

## Next Steps

1. ✅ **Implementation Complete** - All code ready
2. 🔄 **Train Baselines** - Run MHA training
3. 🔄 **Convert to MLA** - Use conversion script
4. 🔄 **Fine-tune** - 3-5 epochs on MLA
5. 🔄 **Evaluate** - Compare WER and cache sizes
6. 🔄 **Ablation Studies** - Test different latent_dim values

## Questions?

- See detailed docs: `models/ATTENTION_MECHANISMS.md`
- Check paper: `Reports/rsonawan_hgokhale_checkpoint.pdf`
- Review implementation: `IMPLEMENTATION_SUMMARY.md`

## Summary

✅ **All implementations are complete and ready to use!**

The models directory now contains:
- Full MHA and MLA implementations
- Conversion utilities with SVD factorization
- Support for Conformer and Branchformer
- Comprehensive documentation and examples

**You can now run your experiments as described in the paper!**

Start with:
```bash
bash scripts/train_model.sh configs/conformer_mha.yaml experiments/conformer_mha
```

Good luck with your experiments! 🚀

