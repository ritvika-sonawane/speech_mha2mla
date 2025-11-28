# Configuration Summary

✅ **YAML configs updated to match JSON baseline files**

---

## Conformer Configurations

**JSON Baseline** (`conformer_baseline.json`):
- Model: `facebook/wav2vec2-large-960h`
- d_model: 1024
- num_heads: 16
- num_layers: 24
- batch_size: 16

**Updated YAML files to match:**
- ✅ `configs/conformer_mha.yaml` - d_model=1024, num_heads=16, num_layers=24
- ✅ `configs/conformer_mla.yaml` - d_model=1024, num_heads=16, num_layers=24, latent_dim=1024
- ✅ `configs/conformer_gqa.yaml` - d_model=1024, num_heads=16, num_layers=24
- ✅ `configs/conformer_linear.yaml` - d_model=1024, num_heads=16, num_layers=24

---

## Branchformer Configurations

**JSON Baseline** (`branchformer_baseline.json`):
- Model: `speechbrain/asr-branchformer-large-tedlium2`
- d_model: 512
- num_heads: 8
- num_layers: 18
- batch_size: 1

**Updated YAML files to match:**
- ✅ `configs/branchformer_mha.yaml` - d_model=512, num_heads=8, num_layers=18, batch_size=1
- ✅ `configs/branchformer_mla.yaml` - d_model=512, num_heads=8, num_layers=18, batch_size=1, latent_dim=512
- ✅ `configs/branchformer_gqa.yaml` - d_model=512, num_heads=8, num_layers=18, batch_size=1
- ✅ `configs/branchformer_linear.yaml` - d_model=512, num_heads=8, num_layers=18, batch_size=1

---

## MLA latent_dim Settings

For 50% KV cache reduction, `latent_dim` should equal `d_model`:

- **Conformer MLA**: `latent_dim=1024` (matches d_model=1024) → 50% cache reduction
- **Branchformer MLA**: `latent_dim=512` (matches d_model=512) → 50% cache reduction

---

## KV Cache Analysis (from JSON baselines)

### Conformer (1000-frame sequence)
- MHA: 187.5 MB
- MLA: 93.75 MB (50% reduction)
- GQA: 23.4375 MB (87.5% reduction)

### Branchformer (1000-frame sequence)
- MHA: 70.3125 MB
- MLA: 35.15625 MB (50% reduction)
- GQA: 17.578125 MB (75% reduction)

---

## Changes Summary

| File | Old d_model | New d_model | Old num_layers | New num_layers | Old batch_size | New batch_size |
|------|-------------|-------------|----------------|----------------|----------------|----------------|
| conformer_mha.yaml | 512 | **1024** | 12 | **24** | 16 | 16 |
| conformer_mla.yaml | 512 | **1024** | 12 | **24** | 16 | 16 |
| conformer_gqa.yaml | 512 | **1024** | 12 | **24** | 16 | 16 |
| conformer_linear.yaml | 512 | **1024** | 12 | **24** | 16 | 16 |
| branchformer_mha.yaml | 512 | 512 | 12 | **18** | 16 | **1** |
| branchformer_mla.yaml | 512 | 512 | 12 | **18** | 16 | **1** |
| branchformer_gqa.yaml | 512 | 512 | 12 | **18** | 16 | **1** |
| branchformer_linear.yaml | 512 | 512 | 12 | **18** | 16 | **1** |

---

## All configurations are now aligned! ✅

