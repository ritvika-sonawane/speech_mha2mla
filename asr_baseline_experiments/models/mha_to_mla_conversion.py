"""
MHA to MLA Conversion Utilities

Implements three conversion strategies as mentioned in the paper:
1. Direct weight matrix factorization via SVD
2. Data-efficient fine-tuning (requires training loop)
3. Progressive conversion during training (requires training loop)

Reference: Section 2.3 - KV Cache and Conversion Strategies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from .attention_variants import (
    MultiHeadAttention, 
    MultiHeadLatentAttention,
    MultiHeadLatentAttentionSimple
)


def svd_factorize_matrix(weight_matrix: torch.Tensor, target_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factorize a weight matrix using SVD for low-rank approximation.
    
    Args:
        weight_matrix: [out_dim, in_dim] weight matrix
        target_rank: desired rank for low-rank approximation
        
    Returns:
        W_A: [out_dim, target_rank] - first projection
        W_B: [target_rank, in_dim] - second projection
    """
    # Move to CPU for SVD (more stable)
    device = weight_matrix.device
    dtype = weight_matrix.dtype
    W = weight_matrix.detach().cpu().float().numpy()
    
    # Perform SVD: W ≈ U @ S @ V^T
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    # Keep only top target_rank components
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    Vt_r = Vt[:target_rank, :]
    
    # Create factorized matrices: W ≈ W_A @ W_B
    # where W_A = U_r @ sqrt(S_r) and W_B = sqrt(S_r) @ Vt_r
    sqrt_S_r = np.sqrt(S_r)
    W_A = U_r * sqrt_S_r[np.newaxis, :]  # [out_dim, target_rank]
    W_B = sqrt_S_r[:, np.newaxis] * Vt_r  # [target_rank, in_dim]
    
    # Convert back to torch tensors
    W_A = torch.from_numpy(W_A).to(dtype).to(device)
    W_B = torch.from_numpy(W_B).to(dtype).to(device)
    
    return W_A, W_B


def convert_mha_to_mla_simple_svd(
    mha_module: MultiHeadAttention,
    latent_dim: Optional[int] = None
) -> MultiHeadLatentAttentionSimple:
    """
    Convert MHA to MLA-Simple using SVD factorization.
    
    This is the "direct weight matrix factorization" approach from the paper.
    Works well for initial conversion but may need fine-tuning for best results.
    
    Args:
        mha_module: Trained MHA module
        latent_dim: Compressed dimension (default: d_model for 50% compression)
        
    Returns:
        mla_module: Initialized MLA-Simple module with factorized weights
    """
    d_model = mha_module.d_model
    num_heads = mha_module.num_heads
    latent_dim = latent_dim if latent_dim is not None else d_model
    
    # Create new MLA module
    mla_module = MultiHeadLatentAttentionSimple(
        d_model=d_model,
        num_heads=num_heads,
        latent_dim=latent_dim,
        dropout=mha_module.dropout.p
    )
    
    # Copy query projection directly (no compression)
    mla_module.q_proj.weight.data.copy_(mha_module.q_proj.weight.data)
    mla_module.q_proj.bias.data.copy_(mha_module.q_proj.bias.data)
    
    # Copy output projection directly
    mla_module.out_proj.weight.data.copy_(mha_module.out_proj.weight.data)
    mla_module.out_proj.bias.data.copy_(mha_module.out_proj.bias.data)
    
    # Factorize key projection: k_proj = k_expand @ kv_compress
    # We want: W_k ≈ W_k_expand @ W_kv_compress
    k_weight = mha_module.k_proj.weight  # [d_model, d_model]
    
    # Factor as: kv_compress [d_model, latent_dim] @ k_expand [latent_dim, d_model]
    # SVD gives us: W ≈ U @ S @ V^T, we need W ≈ (U @ sqrt(S)) @ (sqrt(S) @ V^T)
    W_compress_k, W_expand_k = svd_factorize_matrix(k_weight.t(), latent_dim)
    
    mla_module.kv_compress.weight.data.copy_(W_compress_k.t())
    mla_module.k_expand.weight.data.copy_(W_expand_k.t())
    
    # Factorize value projection similarly
    v_weight = mha_module.v_proj.weight
    W_compress_v, W_expand_v = svd_factorize_matrix(v_weight.t(), latent_dim)
    
    # Average the compression weights (shared between K and V)
    mla_module.kv_compress.weight.data = 0.5 * (mla_module.kv_compress.weight.data + W_compress_v.t())
    mla_module.v_expand.weight.data.copy_(W_expand_v.t())
    
    # Initialize biases
    mla_module.kv_compress.bias.data.zero_()
    mla_module.k_expand.bias.data.copy_(mha_module.k_proj.bias.data)
    mla_module.v_expand.bias.data.copy_(mha_module.v_proj.bias.data)
    
    return mla_module


def convert_mha_to_mla_svd(
    mha_module: MultiHeadAttention,
    latent_dim: Optional[int] = None
) -> MultiHeadLatentAttention:
    """
    Convert MHA to full MLA using SVD factorization with per-head projections.
    
    This creates a more faithful MLA implementation with separate projection
    matrices for each head, following the paper's formulation.
    
    Args:
        mha_module: Trained MHA module
        latent_dim: Compressed dimension (default: d_model for 50% compression)
        
    Returns:
        mla_module: Initialized MLA module with factorized weights
    """
    d_model = mha_module.d_model
    num_heads = mha_module.num_heads
    d_k = d_model // num_heads
    latent_dim = latent_dim if latent_dim is not None else d_model
    
    # Create new MLA module
    mla_module = MultiHeadLatentAttention(
        d_model=d_model,
        num_heads=num_heads,
        latent_dim=latent_dim,
        dropout=mha_module.dropout.p
    )
    
    # Copy query projection
    mla_module.q_proj.weight.data.copy_(mha_module.q_proj.weight.data)
    mla_module.q_proj.bias.data.copy_(mha_module.q_proj.bias.data)
    
    # Copy output projection
    mla_module.out_proj.weight.data.copy_(mha_module.out_proj.weight.data)
    mla_module.out_proj.bias.data.copy_(mha_module.out_proj.bias.data)
    
    # Get the full K and V projection matrices
    k_weight = mha_module.k_proj.weight  # [d_model, d_model]
    v_weight = mha_module.v_proj.weight  # [d_model, d_model]
    k_bias = mha_module.k_proj.bias  # [d_model]
    v_bias = mha_module.v_proj.bias  # [d_model]
    
    # Reshape to per-head weights: [num_heads, d_k, d_model]
    k_weight_heads = k_weight.view(num_heads, d_k, d_model)
    v_weight_heads = v_weight.view(num_heads, d_k, d_model)
    k_bias_heads = k_bias.view(num_heads, d_k)
    v_bias_heads = v_bias.view(num_heads, d_k)
    
    # Initialize compression layer with average of all heads
    compress_weight_sum = torch.zeros(d_model, latent_dim, device=k_weight.device, dtype=k_weight.dtype)
    
    for i in range(num_heads):
        # For each head, factorize: W_head^T ≈ W_A @ W_B
        # where W_head is [d_k, d_model]
        W_compress, W_expand = svd_factorize_matrix(k_weight_heads[i].t(), latent_dim)
        compress_weight_sum += W_compress
        
        # Set per-head expansion weights
        # W_{KA}^(i): [d_k, latent_dim]
        mla_module.k_expand_A[i].weight.data.copy_(W_expand.t())
        mla_module.k_expand_A[i].bias.data.zero_()
        
        # W_{KB}^(i): [d_k, d_k] - initialize as identity
        mla_module.k_expand_B[i].weight.data.copy_(torch.eye(d_k, device=k_weight.device, dtype=k_weight.dtype))
        mla_module.k_expand_B[i].bias.data.copy_(k_bias_heads[i])
        
        # Same for value projections
        W_compress_v, W_expand_v = svd_factorize_matrix(v_weight_heads[i].t(), latent_dim)
        compress_weight_sum += W_compress_v
        
        mla_module.v_expand_A[i].weight.data.copy_(W_expand_v.t())
        mla_module.v_expand_A[i].bias.data.zero_()
        
        mla_module.v_expand_B[i].weight.data.copy_(torch.eye(d_k, device=v_weight.device, dtype=v_weight.dtype))
        mla_module.v_expand_B[i].bias.data.copy_(v_bias_heads[i])
    
    # Average compression weights across heads
    mla_module.kv_compress.weight.data.copy_((compress_weight_sum / (2 * num_heads)).t())
    mla_module.kv_compress.bias.data.zero_()
    
    return mla_module


def analyze_compression_quality(
    mha_module: MultiHeadAttention,
    mla_module: nn.Module,
    test_input: torch.Tensor,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze the quality of MHA-to-MLA conversion.
    
    Args:
        mha_module: Original MHA module
        mla_module: Converted MLA module
        test_input: Test input tensor [B, T, D]
        verbose: Whether to print results
        
    Returns:
        metrics: Dictionary of quality metrics
    """
    mha_module.eval()
    mla_module.eval()
    
    with torch.no_grad():
        mha_output, _ = mha_module(test_input, test_input, test_input)
        mla_output, _ = mla_module(test_input, test_input, test_input)
        
        # Compute metrics
        mse = torch.mean((mha_output - mla_output) ** 2).item()
        mae = torch.mean(torch.abs(mha_output - mla_output)).item()
        
        # Relative error
        output_norm = torch.norm(mha_output).item()
        diff_norm = torch.norm(mha_output - mla_output).item()
        relative_error = diff_norm / (output_norm + 1e-8)
        
        # Cosine similarity
        mha_flat = mha_output.reshape(-1)
        mla_flat = mla_output.reshape(-1)
        cosine_sim = torch.nn.functional.cosine_similarity(
            mha_flat.unsqueeze(0),
            mla_flat.unsqueeze(0)
        ).item()
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'relative_error': relative_error,
        'cosine_similarity': cosine_sim
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MHA to MLA Conversion Quality Analysis")
        print("="*60)
        print(f"Mean Squared Error (MSE):     {mse:.6f}")
        print(f"Mean Absolute Error (MAE):    {mae:.6f}")
        print(f"Relative Error:               {relative_error:.4%}")
        print(f"Cosine Similarity:            {cosine_sim:.6f}")
        print("="*60)
        
        if relative_error < 0.05:
            print("✓ Excellent conversion quality (< 5% error)")
        elif relative_error < 0.10:
            print("✓ Good conversion quality (< 10% error)")
        elif relative_error < 0.20:
            print("⚠ Moderate conversion quality (< 20% error) - consider fine-tuning")
        else:
            print("✗ Poor conversion quality (> 20% error) - fine-tuning recommended")
        print()
    
    return metrics


if __name__ == "__main__":
    # Test conversion
    print("Testing MHA to MLA Conversion")
    print("="*60)
    
    batch_size = 2
    seq_len = 100
    d_model = 512
    num_heads = 8
    latent_dim = 256
    
    # Create and initialize MHA
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Convert using SVD
    print("\n1. Converting MHA to MLA-Simple using SVD...")
    mla_simple = convert_mha_to_mla_simple_svd(mha, latent_dim)
    metrics_simple = analyze_compression_quality(mha, mla_simple, x)
    
    print("\n2. Converting MHA to full MLA using SVD...")
    mla_full = convert_mha_to_mla_svd(mha, latent_dim)
    metrics_full = analyze_compression_quality(mha, mla_full, x)
    
    # Compare parameter counts
    mha_params = sum(p.numel() for p in mha.parameters())
    mla_simple_params = sum(p.numel() for p in mla_simple.parameters())
    mla_full_params = sum(p.numel() for p in mla_full.parameters())
    
    print("\nParameter Comparison:")
    print(f"  MHA:        {mha_params:,} parameters")
    print(f"  MLA-Simple: {mla_simple_params:,} parameters ({mla_simple_params/mha_params:.2%})")
    print(f"  MLA-Full:   {mla_full_params:,} parameters ({mla_full_params/mha_params:.2%})")

