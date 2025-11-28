#!/usr/bin/env python3
"""
Convert trained MHA model to MLA using SVD factorization

Usage:
    python scripts/convert_mha_to_mla.py \
        --input_checkpoint checkpoints/conformer_mha_trained.pt \
        --output_checkpoint checkpoints/conformer_mla_init.pt \
        --config configs/conformer_mha.yaml \
        --latent_dim 512 \
        --mla_variant simple
"""

import argparse
import torch
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    build_conformer,
    build_branchformer,
    convert_mha_to_mla_simple_svd,
    convert_mha_to_mla_svd,
    analyze_compression_quality
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def convert_model(model, latent_dim, mla_variant='simple', test_input=None):
    """
    Convert all attention layers in a model from MHA to MLA
    
    Args:
        model: Model with MHA attention layers
        latent_dim: Compressed dimension for MLA
        mla_variant: 'simple' or 'full'
        test_input: Optional test input for quality analysis
        
    Returns:
        model: Model with MLA attention layers
        metrics: List of conversion quality metrics per layer
    """
    convert_fn = (convert_mha_to_mla_simple_svd if mla_variant == 'simple' 
                  else convert_mha_to_mla_svd)
    
    metrics_list = []
    
    print(f"\nConverting {len(model.layers)} layers from MHA to MLA-{mla_variant.upper()}...")
    print("="*70)
    
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i+1}/{len(model.layers)}:")
        
        # Get the attention module
        if hasattr(layer, 'attention'):
            old_attention = layer.attention
        else:
            print(f"  ⚠ Layer {i} has no 'attention' attribute, skipping")
            continue
        
        # Convert
        new_attention = convert_fn(old_attention, latent_dim)
        
        # Analyze quality if test input provided
        if test_input is not None:
            metrics = analyze_compression_quality(
                old_attention, 
                new_attention, 
                test_input,
                verbose=False
            )
            metrics_list.append(metrics)
            
            print(f"  Relative Error: {metrics['relative_error']:.4%}")
            print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
            
            if metrics['relative_error'] < 0.05:
                print(f"  ✓ Excellent conversion quality")
            elif metrics['relative_error'] < 0.10:
                print(f"  ✓ Good conversion quality")
            else:
                print(f"  ⚠ Moderate quality - fine-tuning recommended")
        
        # Replace the attention module
        layer.attention = new_attention
    
    print("\n" + "="*70)
    print("Conversion complete!")
    
    if metrics_list:
        avg_error = sum(m['relative_error'] for m in metrics_list) / len(metrics_list)
        avg_cosine = sum(m['cosine_similarity'] for m in metrics_list) / len(metrics_list)
        print(f"\nAverage across all layers:")
        print(f"  Relative Error: {avg_error:.4%}")
        print(f"  Cosine Similarity: {avg_cosine:.6f}")
    
    return model, metrics_list


def main():
    parser = argparse.ArgumentParser(description='Convert MHA model to MLA')
    parser.add_argument('--input_checkpoint', type=str, required=True,
                        help='Path to input MHA model checkpoint')
    parser.add_argument('--output_checkpoint', type=str, required=True,
                        help='Path to save converted MLA model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent dimension for MLA (default: d_model for 50%% compression)')
    parser.add_argument('--mla_variant', type=str, default='simple',
                        choices=['simple', 'full'],
                        help='MLA variant to use (simple is faster)')
    parser.add_argument('--test_quality', action='store_true',
                        help='Test conversion quality with random input')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MHA to MLA Model Conversion")
    print("="*70)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    model_type = config.get('model_type', 'conformer')
    
    # Determine latent_dim
    latent_dim = args.latent_dim if args.latent_dim else config.get('d_model', 512)
    d_model = config.get('d_model', 512)
    compression_ratio = latent_dim / d_model
    
    print(f"Model type: {model_type}")
    print(f"d_model: {d_model}")
    print(f"latent_dim: {latent_dim}")
    print(f"Compression ratio: {compression_ratio:.1%} (cache reduction: {(1-compression_ratio/2):.1%})")
    print(f"MLA variant: {args.mla_variant}")
    
    # Build model
    print(f"\nBuilding {model_type} model with MHA...")
    if model_type == 'conformer':
        model = build_conformer(config)
    elif model_type == 'branchformer':
        model = build_branchformer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(args.device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.input_checkpoint}")
    checkpoint = torch.load(args.input_checkpoint, map_location=args.device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    print("  ✓ Checkpoint loaded successfully")
    
    # Create test input if requested
    test_input = None
    if args.test_quality:
        print("\nGenerating test input for quality analysis...")
        batch_size = 2
        seq_len = 100
        input_dim = config.get('input_dim', 80)
        test_input = torch.randn(batch_size, seq_len, d_model).to(args.device)
    
    # Convert model
    model.eval()
    with torch.no_grad():
        model, metrics = convert_model(model, latent_dim, args.mla_variant, test_input)
    
    # Update config for MLA
    config['attention_type'] = f'mla' if args.mla_variant == 'full' else 'mla_simple'
    config['attention_kwargs'] = {'latent_dim': latent_dim}
    
    # Save converted model
    print(f"\nSaving converted model to: {args.output_checkpoint}")
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'conversion_info': {
            'original_checkpoint': args.input_checkpoint,
            'latent_dim': latent_dim,
            'mla_variant': args.mla_variant,
            'conversion_metrics': metrics if metrics else None
        }
    }
    
    torch.save(save_dict, output_path)
    print("  ✓ Model saved successfully")
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Input:  {args.input_checkpoint}")
    print(f"Output: {args.output_checkpoint}")
    print(f"Model:  {model_type.capitalize()} with MLA-{args.mla_variant.upper()}")
    print(f"Cache reduction: {(1-compression_ratio/2):.1%}")
    
    if metrics:
        avg_error = sum(m['relative_error'] for m in metrics) / len(metrics)
        if avg_error < 0.10:
            print(f"\n✓ Good conversion quality!")
            print(f"  Recommended: Fine-tune for 3-5 epochs")
        else:
            print(f"\n⚠ Moderate conversion quality")
            print(f"  Recommended: Fine-tune for 5-10 epochs with lower learning rate")
    else:
        print(f"\nℹ Run with --test_quality to analyze conversion quality")
    
    print("\nNext steps:")
    print(f"  1. Fine-tune: bash scripts/train_model.sh configs/{model_type}_mla.yaml")
    print(f"  2. Evaluate: bash scripts/evaluate_model.sh {args.output_checkpoint}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

