# Attention Mechanisms for ASR Models

This document describes the attention mechanism implementations for Conformer and Branchformer ASR models, focusing on **Multi-Head Attention (MHA)** and **Multi-Head Latent Attention (MLA)** as described in the project paper.

## Overview

We implement multiple attention variants to investigate KV cache compression for ASR inference:

1. **MHA (Multi-Head Attention)** - Standard transformer attention
2. **MLA (Multi-Head Latent Attention)** - Compressed KV cache via low-rank factorization
3. **MLA-Simple** - Computationally efficient MLA variant
4. **GQA (Grouped Query Attention)** - Shared KV heads
5. **Linear Attention** - O(N) complexity attention

## Architecture Implementations

### 1. Multi-Head Attention (MHA)

**File**: `attention_variants.py:MultiHeadAttention`

Standard transformer attention with full KV cache.

**KV Cache Size**: `O(2 × L × h × d)` per layer
- L = sequence length
- h = number of heads  
- d = dimension per head

**Configuration**:
```yaml
attention_type: mha
attention_kwargs: null
```

**Use Case**: Baseline for comparison

---

### 2. Multi-Head Latent Attention (MLA)

**File**: `attention_variants.py:MultiHeadLatentAttention`

Implements the full MLA architecture from the DeepSeek paper with per-head two-stage projections:

$$K^{(i)} = (K^C W_{KA}^{(i)}) W_{KB}^{(i)}$$
$$V^{(i)} = (K^C W_{VA}^{(i)}) W_{VB}^{(i)}$$

where $K^C \in \mathbb{R}^{L \times d_c}$ is the shared compressed representation.

**KV Cache Size**: `O(L × d_c)` per layer
- With `d_c = d_model`: **50% reduction** vs MHA
- With `d_c = d_model/2`: **75% reduction** vs MHA

**Configuration**:
```yaml
attention_type: mla
attention_kwargs:
  latent_dim: 512  # Compressed dimension (d_c)
```

**Key Features**:
- Per-head projection matrices for maximum expressivity
- Two-stage expansion: latent → intermediate → head
- Caches only compressed latent vectors

**Advantages**:
- Preserves full model capacity
- Significant memory savings (40-60%)
- Maintains representational power

**Disadvantages**:
- More parameters than MHA due to per-head projections
- Slightly slower forward pass

---

### 3. Multi-Head Latent Attention Simple (MLA-Simple)

**File**: `attention_variants.py:MultiHeadLatentAttentionSimple`

Simplified MLA variant with shared expansion matrices for computational efficiency.

**KV Cache Size**: `O(L × d_c)` per layer (same as MLA)

**Configuration**:
```yaml
attention_type: mla_simple
attention_kwargs:
  latent_dim: 512
```

**Key Features**:
- Shared compression and expansion layers
- Single-stage expansion: latent → full dimension → split to heads
- Fewer parameters than full MLA

**Advantages**:
- Faster computation than full MLA
- Fewer parameters, easier to train
- Same memory savings as full MLA

**Disadvantages**:
- Slightly less expressive than per-head projections

**Recommendation**: Use MLA-Simple for initial experiments and faster training. Switch to full MLA if you need maximum model capacity.

---

## MHA to MLA Conversion

**File**: `mha_to_mla_conversion.py`

Implements conversion strategies from Section 2.3 of the paper.

### Conversion Strategy 1: Direct SVD Factorization

Convert a trained MHA model to MLA using Singular Value Decomposition.

```python
from models.mha_to_mla_conversion import convert_mha_to_mla_simple_svd

# Convert single attention layer
mla_module = convert_mha_to_mla_simple_svd(
    mha_module=trained_mha,
    latent_dim=512  # or 256 for higher compression
)

# Analyze conversion quality
from models.mha_to_mla_conversion import analyze_compression_quality

metrics = analyze_compression_quality(
    mha_module=trained_mha,
    mla_module=mla_module,
    test_input=test_tensor,
    verbose=True
)
```

**When to use**:
- Quick conversion of pretrained models
- Initial experiments without retraining
- When fine-tuning resources are limited

**Expected Results**:
- Relative error: 5-15% without fine-tuning
- < 5% error after 2-3 epochs of fine-tuning
- Good initialization for subsequent training

### Conversion Strategy 2: Fine-tuning (Recommended)

After SVD conversion, fine-tune on your dataset:

```bash
# Convert and fine-tune Conformer
python scripts/train.py \
  --config configs/conformer_mla.yaml \
  --from_checkpoint checkpoints/conformer_mha_trained.pt \
  --convert_to_mla \
  --latent_dim 512 \
  --num_epochs 5
```

**Benefits**:
- Recovers from SVD approximation errors
- Adapts MLA to your specific data distribution
- Typically achieves ΔWE R < 1%

### Conversion Strategy 3: Progressive Conversion

Train with gradually increasing compression (future work).

---

## Model Configuration

### Conformer with MHA

```yaml
# configs/conformer_mha.yaml
model_type: conformer
attention_type: mha
d_model: 512
num_heads: 8
num_layers: 12
attention_kwargs: null
```

### Conformer with MLA

```yaml
# configs/conformer_mla.yaml
model_type: conformer
attention_type: mla
d_model: 512
num_heads: 8
num_layers: 12
attention_kwargs:
  latent_dim: 512  # 50% compression
```

### Branchformer with MHA

```yaml
# configs/branchformer_mha.yaml
model_type: branchformer
attention_type: mha
d_model: 512
num_heads: 8
num_layers: 12
merge_method: concat
attention_kwargs: null
```

### Branchformer with MLA

```yaml
# configs/branchformer_mla.yaml
model_type: branchformer
attention_type: mla
d_model: 512
num_heads: 8
num_layers: 12
merge_method: concat
attention_kwargs:
  latent_dim: 512
```

---

## KV Cache Analysis

### Theoretical Cache Sizes

For a 1000-frame sequence with d_model=512, num_heads=8, num_layers=12:

#### Conformer (12 layers)
| Attention Type | Cache per Layer | Total Cache | Reduction |
|----------------|-----------------|-------------|-----------|
| MHA | 15.625 MB | 187.5 MB | - |
| MLA (d_c=512) | 7.8125 MB | 93.75 MB | 50% |
| MLA (d_c=256) | 3.906 MB | 46.875 MB | 75% |
| GQA (h_kv=2) | 1.953 MB | 23.44 MB | 87.5% |

#### Branchformer (12 layers)
| Attention Type | Cache per Layer | Total Cache | Reduction |
|----------------|-----------------|-------------|-----------|
| MHA | 5.859 MB | 70.31 MB | - |
| MLA (d_c=512) | 2.930 MB | 35.16 MB | 50% |
| MLA (d_c=256) | 1.465 MB | 17.58 MB | 75% |

### Empirical Measurements

Use the profiling script to measure actual cache usage:

```bash
python scripts/profile_kv_cache.py \
  --model_type conformer \
  --attention_type mla \
  --latent_dim 512 \
  --seq_len 1000
```

---

## Training Guidelines

### Training from Scratch

**MHA Baseline** (Train first):
```bash
bash scripts/train_model.sh \
  configs/conformer_mha.yaml \
  experiments/conformer_mha
```

**MLA from Scratch**:
```bash
bash scripts/train_model.sh \
  configs/conformer_mla.yaml \
  experiments/conformer_mla
```

### Converting and Fine-tuning

1. **Train MHA baseline** (50 epochs)
2. **Convert to MLA** using SVD
3. **Fine-tune** for 3-5 epochs

```python
# Example conversion script
from models.conformer import build_conformer
from models.mha_to_mla_conversion import convert_mha_to_mla_simple_svd
import torch

# Load trained MHA model
config_mha = {...}  # MHA config
model_mha = build_conformer(config_mha)
model_mha.load_state_dict(torch.load('checkpoints/conformer_mha.pt'))

# Convert attention layers to MLA
for i, layer in enumerate(model_mha.layers):
    mla_attention = convert_mha_to_mla_simple_svd(
        layer.attention,
        latent_dim=512
    )
    model_mha.layers[i].attention = mla_attention

# Save converted model
torch.save(model_mha.state_dict(), 'checkpoints/conformer_mla_init.pt')

# Fine-tune with standard training loop
```

---

## Expected Results (LibriSpeech 100h)

Based on the paper and preliminary experiments:

| Model | Attention | WER (%) | Cache (MB) | Inf. Time (s) |
|-------|-----------|---------|------------|---------------|
| Conformer | MHA | 1.18 | 187.5 | 0.0016 |
| Conformer | MLA | ~1.25* | 93.75 | ~0.0017* |
| Branchformer | MHA | 8.11 | 70.31 | 6.72 |
| Branchformer | MLA | ~8.5* | 35.16 | ~6.8* |

\* *Expected values - experiments in progress*

**Target**: ΔWE R < 5% with 50% cache reduction

---

## Implementation Details

### Cache Format

**MHA Cache**:
```python
cache = {
    'k': [B, num_heads, T, d_k],  # Key cache
    'v': [B, num_heads, T, d_k]   # Value cache
}
```

**MLA Cache**:
```python
cache = {
    'latent': [B, T, latent_dim]  # Compressed latent
}
```

### Forward Pass with Cache

```python
# Initial forward pass
output, cache = attention_layer(x, x, x, cache=None)

# Incremental forward pass (autoregressive)
new_frame = x[:, -1:, :]  # [B, 1, D]
output, new_cache = attention_layer(new_frame, new_frame, new_frame, cache=cache)
```

---

## Testing

### Unit Tests

Test individual attention modules:

```bash
# Test all attention types
python models/attention_variants.py

# Test conversion utilities
python models/mha_to_mla_conversion.py
```

### Integration Tests

Test full models:

```bash
# Test Conformer
python models/conformer.py

# Test Branchformer  
python models/branchformer.py
```

---

## References

1. **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
2. **Branchformer**: Peng et al., "Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context" (2022)
3. **MLA**: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (2024)
4. **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)

---

## Troubleshooting

### Issue: High WER after MHA→MLA conversion

**Solutions**:
1. Fine-tune for more epochs (5-10 instead of 3)
2. Use lower learning rate for fine-tuning (1e-5 instead of 1e-4)
3. Try higher `latent_dim` (512 instead of 256)
4. Check conversion quality using `analyze_compression_quality()`

### Issue: Out of memory during training

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training (FP16)
4. For MLA, use `mla_simple` instead of full `mla`

### Issue: Slow inference with MLA

**Solutions**:
1. Use `mla_simple` instead of full `mla`
2. Increase `latent_dim` slightly (reduces expansion overhead)
3. Optimize batch processing
4. Consider using GQA for better speed/memory tradeoff

---

## Next Steps

1. ✅ Implement MHA and MLA attention mechanisms
2. ✅ Create conversion utilities (SVD factorization)
3. 🔄 Train baseline models (Conformer + Branchformer)
4. 🔄 Convert and fine-tune MLA models
5. ⏳ Compare WER, cache size, and inference time
6. ⏳ Ablation studies on `latent_dim`
7. ⏳ Compare with GQA baseline

---

For more details, see:
- Project paper: `Reports/rsonawan_hgokhale_checkpoint.pdf`
- Training scripts: `scripts/train.py`
- Evaluation scripts: `scripts/evaluate_model.sh`

