#!/usr/bin/env python3
"""
Convert HuggingFace pre-trained Conformer (MHA) to MLA

This script:
1. Loads a pre-trained Conformer model from HuggingFace
2. Extracts the attention weights from each layer
3. Converts MHA attention to MLA using SVD
4. Saves the converted model

Usage:
    python scripts/convert_hf_conformer_to_mla.py \
        --model_name facebook/wav2vec2-conformer-rel-pos-large-960h-ft \
        --output_dir checkpoints/conformer_mla_from_hf \
        --latent_dim 512 \
        --mla_variant simple
"""

import argparse
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC, AutoConfig, AutoModelForCTC
from pathlib import Path
import json


def convert_attention_weights_to_mla(attention_module, latent_dim, variant='simple'):
    """
    Convert HuggingFace Wav2Vec2Conformer attention weights to MLA using SVD
    
    Args:
        attention_module: HF attention module with q/k/v projections
        latent_dim: Compressed dimension for MLA
        variant: 'simple' or 'full'
    
    Returns:
        Dictionary with converted MLA weights
    """
    # Get original weights
    # HF Conformer uses: linear_q, linear_k, linear_v (each is d_model -> d_model)
    q_weight = attention_module.linear_q.weight.data  # [d_model, d_model]
    k_weight = attention_module.linear_k.weight.data  # [d_model, d_model]
    v_weight = attention_module.linear_v.weight.data  # [d_model, d_model]
    
    d_model = q_weight.shape[0]
    
    print(f"  Original shapes: Q={q_weight.shape}, K={k_weight.shape}, V={v_weight.shape}")
    
    # Apply SVD to compress K and V projections
    # K compression: K = U_k @ S_k @ V_k^T -> K_down @ K_up
    U_k, S_k, Vt_k = torch.linalg.svd(k_weight, full_matrices=False)
    k_down = (U_k[:, :latent_dim] @ torch.diag(S_k[:latent_dim])).t()  # [latent_dim, d_model]
    k_up = Vt_k[:latent_dim, :]  # [latent_dim, d_model]
    
    # V compression: V = U_v @ S_v @ V_v^T -> V_down @ V_up
    U_v, S_v, Vt_v = torch.linalg.svd(v_weight, full_matrices=False)
    v_down = (U_v[:, :latent_dim] @ torch.diag(S_v[:latent_dim])).t()  # [latent_dim, d_model]
    v_up = Vt_v[:latent_dim, :]  # [latent_dim, d_model]
    
    # Q can also be compressed for full MLA variant
    if variant == 'full':
        U_q, S_q, Vt_q = torch.linalg.svd(q_weight, full_matrices=False)
        q_down = (U_q[:, :latent_dim] @ torch.diag(S_q[:latent_dim])).t()
        q_up = Vt_q[:latent_dim, :]
    else:
        q_down = None
        q_up = q_weight  # Keep Q as is for simple variant
    
    # Calculate compression quality
    k_reconstructed = torch.matmul(k_down.t(), k_up)
    v_reconstructed = torch.matmul(v_down.t(), v_up)
    
    k_error = torch.norm(k_weight - k_reconstructed) / torch.norm(k_weight)
    v_error = torch.norm(v_weight - v_reconstructed) / torch.norm(v_weight)
    
    print(f"  Compression quality: K error={k_error:.4f}, V error={v_error:.4f}")
    print(f"  New shapes: K_down={k_down.shape}, K_up={k_up.shape}")
    
    mla_weights = {
        'k_down': k_down,
        'k_up': k_up,
        'v_down': v_down,
        'v_up': v_up,
        'q_proj': q_up if variant == 'simple' else None,
        'q_down': q_down,
        'q_up': q_up if variant == 'full' else None,
        'out_proj': attention_module.linear_out.weight.data.clone(),
    }
    
    # Include biases if they exist
    if attention_module.linear_q.bias is not None:
        mla_weights['q_bias'] = attention_module.linear_q.bias.data.clone()
    if attention_module.linear_k.bias is not None:
        mla_weights['k_bias'] = attention_module.linear_k.bias.data.clone()
    if attention_module.linear_v.bias is not None:
        mla_weights['v_bias'] = attention_module.linear_v.bias.data.clone()
    if attention_module.linear_out.bias is not None:
        mla_weights['out_bias'] = attention_module.linear_out.bias.data.clone()
    
    return mla_weights, {'k_error': k_error.item(), 'v_error': v_error.item()}


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace Conformer to MLA')
    parser.add_argument('--model_name', type=str, 
                        default='facebook/wav2vec2-conformer-rel-pos-large-960h-ft',
                        help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for converted model')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension for MLA compression')
    parser.add_argument('--mla_variant', type=str, default='simple',
                        choices=['simple', 'full'],
                        help='MLA variant')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HuggingFace Conformer MHA → MLA Conversion")
    print("="*70)
    print(f"\nModel: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Variant: {args.mla_variant}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\nLoading pre-trained Conformer model...")
    model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_name)
    config = model.config
    model_type = "Conformer"
    
    print(f"  ✓ Model loaded")
    print(f"  Model type: {model_type}")
    print(f"  Architecture: {config.architectures}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    
    # Get encoder layers
    encoder_layers = model.wav2vec2_conformer.encoder.layers
    
    # Convert each attention layer
    print(f"\n{'='*70}")
    print(f"Converting {config.num_hidden_layers} attention layers...")
    print(f"{'='*70}\n")
    
    converted_layers = []
    all_metrics = []
    
    for i, layer in enumerate(encoder_layers):
        print(f"Layer {i+1}/{config.num_hidden_layers}:")
        
        # Get the attention module (self_attn for Conformer)
        attention = layer.self_attn
        
        # Convert to MLA
        mla_weights, metrics = convert_attention_weights_to_mla(
            attention, 
            args.latent_dim,
            args.mla_variant
        )
        
        converted_layers.append(mla_weights)
        all_metrics.append(metrics)
        
        print(f"  ✓ Layer {i+1} converted\n")
    
    # Calculate average metrics
    avg_k_error = sum(m['k_error'] for m in all_metrics) / len(all_metrics)
    avg_v_error = sum(m['v_error'] for m in all_metrics) / len(all_metrics)
    
    print(f"{'='*70}")
    print("Conversion Summary")
    print(f"{'='*70}")
    print(f"Average K reconstruction error: {avg_k_error:.4%}")
    print(f"Average V reconstruction error: {avg_v_error:.4%}")
    print(f"Compression ratio: {args.latent_dim / config.hidden_size:.1%}")
    print(f"KV cache reduction: ~{(1 - args.latent_dim / config.hidden_size):.1%}")
    
    # Save converted weights
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving converted model to: {output_path}")
    
    # Save the full model state dict with converted attention layers
    save_dict = {
        'original_model': args.model_name,
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'latent_dim': args.latent_dim,
            'mla_variant': args.mla_variant,
        },
        'converted_attention_layers': converted_layers,
        'conversion_metrics': all_metrics,
        'full_model_state_dict': model.state_dict(),
    }
    
    torch.save(save_dict, output_path / 'converted_model.pt')
    
    # Save config as JSON
    with open(output_path / 'config.json', 'w') as f:
        json.dump(save_dict['config'], f, indent=2)
    
    print("  ✓ Saved converted model")
    print("  ✓ Saved config.json")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print(f"  1. Fine-tune the converted model")
    print(f"  2. Compare performance with original MHA model")
    print(f"  3. Measure inference speed and memory usage")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

