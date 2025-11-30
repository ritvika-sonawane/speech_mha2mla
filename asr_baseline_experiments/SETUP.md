# Setup Guide for ASR MHA→MLA Pipeline

## Prerequisites

- **GPU**: NVIDIA GPU with at least 16GB VRAM (T4, V100, A100, or similar)
- **CUDA**: Version 12.1 or higher
- **Python**: 3.8 or higher
- **Conda**: Recommended for environment management

## Quick Setup

### 1. Create Conda Environment

```bash
# Create and activate environment
conda create -n speech python=3.10
conda activate speech
```

### 2. Install PyTorch with CUDA Support

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if needed)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Required Packages

```bash
cd /home/ubuntu/speech_mha2mla/asr_baseline_experiments
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
```

## Running the Pipeline

### Quick Test (1-2 hours)
```bash
bash run_complete_pipeline.sh quick
```

### Full Pipeline (6-10 hours on T4)
```bash
bash run_complete_pipeline.sh full
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in the script:
- Edit `run_complete_pipeline.sh`
- Change `BATCH_SIZE=8` to `BATCH_SIZE=4` or lower

### Issue: "evaluate module not found" or WER calculation fails
**Solution**: 
```bash
pip install evaluate jiwer
```

### Issue: "datasets cannot find librispeech"
**Solution**: The dataset will be downloaded automatically on first run. Ensure you have:
- Internet connection
- At least 50GB free disk space for LibriSpeech dataset
- Patience for the initial download (~10-15 minutes)

### Issue: "transformers cannot load model"
**Solution**: 
```bash
# Clear HuggingFace cache and retry
rm -rf ~/.cache/huggingface/transformers
python scripts/convert_hf_conformer_to_mla.py --help  # Should trigger re-download
```

## Expected Resource Usage

### Disk Space
- LibriSpeech dataset: ~35GB
- Model checkpoints: ~5GB per checkpoint
- Results and logs: ~1GB
- **Total**: ~50-70GB

### Memory
- System RAM: 32GB+ recommended
- GPU VRAM: 16GB minimum (24GB+ for full batch sizes)

### Time (on T4 GPU)
- **Quick mode**: 1-2 hours
- **Full mode**: 6-10 hours
  - Conversion: ~5 minutes
  - MHA baseline eval: ~30 minutes
  - MLA fine-tuning: 5-8 hours
  - MLA evaluation: ~30 minutes
  - Report generation: ~1 minute

## Directory Structure After Setup

```
asr_baseline_experiments/
├── requirements.txt          # Python dependencies (created)
├── SETUP.md                  # This file
├── run_complete_pipeline.sh  # Main pipeline script
├── scripts/                  # Python scripts
│   ├── convert_hf_conformer_to_mla.py
│   ├── evaluate_hf_model.py
│   ├── benchmark_model.py
│   ├── finetune_mla_conformer.py
│   └── evaluate_finetuned_mla.py
├── checkpoints/              # Converted MLA models (created during run)
└── results/                  # Pipeline results (created during run)
    └── pipeline_TIMESTAMP/
        ├── COMPARISON_REPORT.txt
        ├── pipeline.log
        ├── conformer_mla_finetuned/
        └── *.json
```

## Next Steps

1. **Activate environment**: `conda activate speech`
2. **Run quick test**: `bash run_complete_pipeline.sh quick`
3. **Check results**: `cat results/pipeline_*/COMPARISON_REPORT.txt`
4. **View training**: `tensorboard --logdir results/pipeline_*/conformer_mla_finetuned`

## Support

If you encounter issues:
1. Check the logs in `results/pipeline_*/pipeline.log`
2. Verify GPU is available: `nvidia-smi`
3. Ensure all dependencies are installed: `pip list | grep -E "torch|transformers|datasets"`

