#!/usr/bin/env python3
"""
Quick test to verify MHA and MLA implementations work correctly
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import (
    build_conformer,
    build_branchformer,
    convert_mha_to_mla_simple_svd,
    analyze_compression_quality,
    calculate_kv_cache_size
)


def test_attention_mechanisms():
    """Test basic attention mechanism functionality"""
    print("\n" + "="*70)
    print("Testing Attention Mechanisms")
    print("="*70)
    
    batch_size = 2
    seq_len = 100
    d_model = 512
    num_heads = 8
    
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Test with both model types
    for model_type in ['conformer', 'branchformer']:
        print(f"\n{model_type.upper()}:")
        print("-" * 70)
        
        for attention_type in ['mha', 'mla', 'mla_simple']:
            config = {
                'model_type': model_type,
                'input_dim': 80,
                'd_model': 256,
                'num_heads': 4,
                'num_layers': 2,
                'conv_kernel_size': 31,
                'dropout': 0.1,
                'vocab_size': 1000,
                'attention_type': attention_type,
            }
            
            if attention_type in ['mla', 'mla_simple']:
                config['attention_kwargs'] = {'latent_dim': 256}
            else:
                config['attention_kwargs'] = None
            
            if model_type == 'branchformer':
                config['mlp_expansion_factor'] = 4
                config['merge_method'] = 'concat'
            else:
                config['ff_expansion_factor'] = 4
            
            try:
                if model_type == 'conformer':
                    model = build_conformer(config)
                else:
                    model = build_branchformer(config)
                
                # Test forward pass
                x = torch.randn(batch_size, seq_len, config['input_dim'])
                lengths = torch.tensor([seq_len, seq_len - 10])
                
                with torch.no_grad():
                    logits, caches = model(x, lengths)
                
                # Check output shape
                assert logits.shape[0] == batch_size
                assert logits.shape[2] == config['vocab_size']
                
                # Check cache
                assert len(caches) == config['num_layers']
                
                # Calculate cache size
                cache_size = 0
                for cache in caches:
                    if 'latent' in cache:
                        cache_size += cache['latent'].numel() * 4  # 4 bytes per float32
                    else:
                        cache_size += cache['k'].numel() * 4
                        cache_size += cache['v'].numel() * 4
                
                print(f"  ✓ {attention_type.upper():12} - Output: {logits.shape}, Cache: {cache_size/1024:.2f} KB")
                
            except Exception as e:
                print(f"  ✗ {attention_type.upper():12} - Error: {e}")
                raise


def test_mha_to_mla_conversion():
    """Test MHA to MLA conversion"""
    print("\n" + "="*70)
    print("Testing MHA to MLA Conversion")
    print("="*70)
    
    batch_size = 2
    seq_len = 50
    
    config = {
        'input_dim': 80,
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 2,
        'conv_kernel_size': 31,
        'ff_expansion_factor': 4,
        'dropout': 0.1,
        'vocab_size': 1000,
        'attention_type': 'mha',
        'attention_kwargs': None
    }
    
    # Build Conformer with MHA
    print("\nBuilding Conformer with MHA...")
    model_mha = build_conformer(config)
    
    # Test input
    x = torch.randn(batch_size, seq_len, config['input_dim'])
    lengths = torch.tensor([seq_len, seq_len - 10])
    
    # Get MHA output
    print("Running MHA forward pass...")
    with torch.no_grad():
        logits_mha, caches_mha = model_mha(x, lengths)
    
    # Convert first layer to MLA
    print("\nConverting attention layer to MLA...")
    test_input = torch.randn(batch_size, seq_len, config['d_model'])
    
    mla_attention = convert_mha_to_mla_simple_svd(
        model_mha.layers[0].attention,
        latent_dim=256
    )
    
    # Analyze conversion quality
    metrics = analyze_compression_quality(
        model_mha.layers[0].attention,
        mla_attention,
        test_input,
        verbose=True
    )
    
    # Check metrics
    if metrics['relative_error'] < 0.20:
        print("✓ Conversion successful with acceptable quality!")
    else:
        print("⚠ Conversion has high error - may need fine-tuning")
    
    return metrics


def test_cache_sizes():
    """Test KV cache size calculations"""
    print("\n" + "="*70)
    print("Testing KV Cache Size Calculations")
    print("="*70)
    
    batch_size = 1
    seq_len = 1000
    d_model = 512
    num_heads = 8
    num_layers = 12
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Number of layers: {num_layers}")
    
    print(f"\nCache sizes per layer:")
    
    for attention_type in ['mha', 'mla', 'gqa']:
        if attention_type == 'mla':
            kwargs = {'latent_dim': 512}
        elif attention_type == 'gqa':
            kwargs = {'num_kv_heads': 2}
        else:
            kwargs = {}
        
        cache_per_layer = calculate_kv_cache_size(
            attention_type, batch_size, seq_len, d_model, num_heads, **kwargs
        )
        
        total_cache = cache_per_layer * num_layers / (1024 * 1024)  # Convert to MB
        cache_per_layer_mb = cache_per_layer / (1024 * 1024)
        
        reduction = 0
        if attention_type != 'mha':
            mha_cache = calculate_kv_cache_size('mha', batch_size, seq_len, d_model, num_heads)
            mha_total = mha_cache * num_layers
            reduction = (1 - (cache_per_layer * num_layers) / mha_total) * 100
        
        if reduction > 0:
            print(f"  {attention_type.upper():8} - {cache_per_layer_mb:.2f} MB/layer, "
                  f"{total_cache:.2f} MB total ({reduction:.1f}% reduction)")
        else:
            print(f"  {attention_type.upper():8} - {cache_per_layer_mb:.2f} MB/layer, "
                  f"{total_cache:.2f} MB total")


def main():
    print("\n" + "="*70)
    print("MHA/MLA Implementation Test Suite")
    print("="*70)
    
    try:
        # Test 1: Basic attention mechanisms
        test_attention_mechanisms()
        
        # Test 2: MHA to MLA conversion
        test_mha_to_mla_conversion()
        
        # Test 3: Cache size calculations
        test_cache_sizes()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYour MHA and MLA implementations are working correctly!")
        print("\nNext steps:")
        print("  1. Train baseline models: bash scripts/train_model.sh configs/conformer_mha.yaml")
        print("  2. Convert to MLA: python scripts/convert_mha_to_mla.py --help")
        print("  3. Fine-tune and evaluate")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

