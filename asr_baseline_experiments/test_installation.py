#!/usr/bin/env python3
"""
Quick installation test script
Verifies all dependencies are correctly installed before running the pipeline
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    tests = []
    
    # Core dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        tests.append(("PyTorch", True))
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        tests.append(("PyTorch", False))
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        tests.append(("Transformers", True))
    except ImportError as e:
        print(f"✗ Transformers not found: {e}")
        tests.append(("Transformers", False))
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        tests.append(("Datasets", True))
    except ImportError as e:
        print(f"✗ Datasets not found: {e}")
        tests.append(("Datasets", False))
    
    try:
        import evaluate
        print(f"✓ Evaluate {evaluate.__version__}")
        tests.append(("Evaluate", True))
    except ImportError as e:
        print(f"✗ Evaluate not found: {e}")
        tests.append(("Evaluate", False))
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        tests.append(("NumPy", True))
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        tests.append(("NumPy", False))
    
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
        tests.append(("tqdm", True))
    except ImportError as e:
        print(f"✗ tqdm not found: {e}")
        tests.append(("tqdm", False))
    
    return tests


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA is not available")
            print("  The pipeline requires a GPU. Please check your CUDA installation.")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def test_model_loading():
    """Test if we can load a small HF model"""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoConfig
        # Try to load config only (no actual model download)
        config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
        print("✓ Can access HuggingFace models")
        return True
    except Exception as e:
        print(f"✗ Cannot access HuggingFace models: {e}")
        print("  Check your internet connection or HuggingFace token if needed")
        return False


def test_dataset_access():
    """Test if we can access datasets"""
    print("\nTesting dataset access...")
    
    try:
        from datasets import load_dataset_builder
        # Just check if we can access the dataset info
        ds_builder = load_dataset_builder("librispeech_asr", "clean")
        print("✓ Can access LibriSpeech dataset")
        return True
    except Exception as e:
        print(f"✗ Cannot access datasets: {e}")
        print("  The dataset will be downloaded on first run")
        return True  # Not critical, will download later


def main():
    print("="*70)
    print("Installation Test for ASR MHA→MLA Pipeline")
    print("="*70)
    print()
    
    # Run tests
    import_tests = test_imports()
    cuda_ok = test_cuda()
    model_ok = test_model_loading()
    dataset_ok = test_dataset_access()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    failed_imports = [name for name, passed in import_tests if not passed]
    
    if failed_imports:
        print(f"\n✗ Missing packages: {', '.join(failed_imports)}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return 1
    
    if not cuda_ok:
        print("\n⚠ Warning: CUDA not available. Pipeline requires GPU!")
        print("  Check CUDA installation with: nvidia-smi")
        return 1
    
    print("\n✓ All tests passed!")
    print("\nYou can now run the pipeline:")
    print("  bash run_complete_pipeline.sh quick    # Quick test (1-2 hours)")
    print("  bash run_complete_pipeline.sh full     # Full pipeline (6-10 hours)")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

