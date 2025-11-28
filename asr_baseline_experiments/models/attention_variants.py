"""
Attention Variants for ASR Models
Implements: MHA, MLA, GQA, and Linear Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA)"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None, cache=None):
        batch_size = query.size(0)
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Handle KV cache
        if cache is not None:
            K = torch.cat([cache['k'], K], dim=2)
            V = torch.cat([cache['v'], V], dim=2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        new_cache = {'k': K, 'v': V}
        return output, new_cache


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    Compresses KV cache using low-rank decomposition as described in DeepSeek paper.
    
    For each head i:
        K^(i) = (K^C W_{KA}^(i)) W_{KB}^(i)
        V^(i) = (K^C W_{VA}^(i)) W_{VB}^(i)
    
    where K^C is the shared compressed representation of dimension latent_dim.
    This reduces KV cache from O(2Lhd) to O(L*latent_dim).
    """
    
    def __init__(self, d_model: int, num_heads: int, latent_dim: int = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Default latent_dim to d_model for 50% compression as per paper
        self.latent_dim = latent_dim if latent_dim is not None else d_model
        
        # Query projection (standard)
        self.q_proj = nn.Linear(d_model, d_model)
        
        # Compression to shared latent space K^C
        self.kv_compress = nn.Linear(d_model, self.latent_dim)
        
        # Per-head two-stage expansion for keys
        # First stage: W_{KA}^(i) for each head
        self.k_expand_A = nn.ModuleList([
            nn.Linear(self.latent_dim, self.d_k) for _ in range(num_heads)
        ])
        # Second stage: W_{KB}^(i) for each head
        self.k_expand_B = nn.ModuleList([
            nn.Linear(self.d_k, self.d_k) for _ in range(num_heads)
        ])
        
        # Per-head two-stage expansion for values
        # First stage: W_{VA}^(i) for each head
        self.v_expand_A = nn.ModuleList([
            nn.Linear(self.latent_dim, self.d_k) for _ in range(num_heads)
        ])
        # Second stage: W_{VB}^(i) for each head
        self.v_expand_B = nn.ModuleList([
            nn.Linear(self.d_k, self.d_k) for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None, cache=None):
        batch_size = query.size(0)
        
        # Standard query projection
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compress key-value into shared latent representation K^C
        latent = self.kv_compress(key)  # [B, T, latent_dim]
        
        # Handle cache with compressed latent
        if cache is not None:
            latent = torch.cat([cache['latent'], latent], dim=1)
        
        seq_len = latent.size(1)
        
        # Expand latent to keys and values using per-head two-stage projections
        K_heads = []
        V_heads = []
        
        for i in range(self.num_heads):
            # K^(i) = (K^C W_{KA}^(i)) W_{KB}^(i)
            k_intermediate = self.k_expand_A[i](latent)  # [B, T, d_k]
            k_head = self.k_expand_B[i](k_intermediate)  # [B, T, d_k]
            K_heads.append(k_head)
            
            # V^(i) = (K^C W_{VA}^(i)) W_{VB}^(i)
            v_intermediate = self.v_expand_A[i](latent)  # [B, T, d_k]
            v_head = self.v_expand_B[i](v_intermediate)  # [B, T, d_k]
            V_heads.append(v_head)
        
        # Stack heads: [B, num_heads, T, d_k]
        K = torch.stack(K_heads, dim=1)
        V = torch.stack(V_heads, dim=1)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        # Cache only the compressed latent representation - key memory savings!
        new_cache = {'latent': latent}
        return output, new_cache


class MultiHeadLatentAttentionSimple(nn.Module):
    """
    Simplified Multi-Head Latent Attention (MLA-Simple)
    More computationally efficient variant that uses shared expansion matrices.
    
    This version compresses to latent space and uses shared expansions,
    then splits into heads. More efficient than per-head projections.
    """
    
    def __init__(self, d_model: int, num_heads: int, latent_dim: int = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Default latent_dim to d_model for 50% compression
        self.latent_dim = latent_dim if latent_dim is not None else d_model
        
        # Query projection (standard)
        self.q_proj = nn.Linear(d_model, d_model)
        
        # Compressed KV projections (simplified)
        self.kv_compress = nn.Linear(d_model, self.latent_dim)
        self.k_expand = nn.Linear(self.latent_dim, d_model)
        self.v_expand = nn.Linear(self.latent_dim, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None, cache=None):
        batch_size = query.size(0)
        
        # Standard query projection
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compress key-value into latent representation
        latent = self.kv_compress(key)  # [B, T, latent_dim]
        
        # Handle cache with compressed latent
        if cache is not None:
            latent = torch.cat([cache['latent'], latent], dim=1)
        
        # Expand latent to full key and value
        K = self.k_expand(latent).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_expand(latent).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        # Cache only the compressed latent representation
        new_cache = {'latent': latent}
        return output, new_cache


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    Uses fewer KV heads than Q heads for cache reduction
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads // 4
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_k = d_model // num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None, cache=None):
        batch_size = query.size(0)
        
        # Project queries (all heads)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Project keys and values (fewer heads)
        K = self.k_proj(key).view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Handle cache
        if cache is not None:
            K = torch.cat([cache['k'], K], dim=2)
            V = torch.cat([cache['v'], V], dim=2)
        
        # Repeat KV heads to match Q heads
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        # Cache only the KV heads (not repeated)
        new_cache = {
            'k': K[:, ::self.num_queries_per_kv],
            'v': V[:, ::self.num_queries_per_kv]
        }
        return output, new_cache


class LinearAttention(nn.Module):
    """
    Linear Attention with kernel feature maps
    O(N) complexity instead of O(N^2)
    """
    
    def __init__(self, d_model: int, num_heads: int, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_dim = feature_dim
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def feature_map(self, x):
        """Apply ELU+1 feature map for positive values"""
        return F.elu(x) + 1
        
    def forward(self, query, key, value, mask=None, cache=None):
        batch_size = query.size(0)
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply feature map
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # Handle cache
        if cache is not None:
            K = torch.cat([cache['k'], K], dim=2)
            V = torch.cat([cache['v'], V], dim=2)
        
        # Linear attention: O(N) complexity
        # Compute K^T @ V
        KV = torch.einsum('bhnd,bhnm->bhmd', K, V)
        # Compute normalizer
        Z = torch.einsum('bhnd,bhd->bhn', Q, K.sum(dim=2))
        Z = Z.unsqueeze(-1) + 1e-6
        
        # Compute output
        output = torch.einsum('bhnd,bhmd->bhnm', Q, KV) / Z
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        new_cache = {'k': K, 'v': V}
        return output, new_cache


def get_attention_module(attention_type: str, d_model: int, num_heads: int, **kwargs):
    """Factory function to get attention module by name"""
    
    attention_types = {
        'mha': MultiHeadAttention,
        'mla': MultiHeadLatentAttention,
        'mla_simple': MultiHeadLatentAttentionSimple,
        'gqa': GroupedQueryAttention,
        'linear': LinearAttention
    }
    
    if attention_type.lower() not in attention_types:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available types: {list(attention_types.keys())}")
    
    return attention_types[attention_type.lower()](d_model, num_heads, **kwargs)


def calculate_kv_cache_size(attention_type: str, batch_size: int, seq_len: int, 
                            d_model: int, num_heads: int, **kwargs):
    """Calculate KV cache size in bytes for different attention types"""
    
    d_k = d_model // num_heads
    bytes_per_element = 4  # float32
    
    if attention_type.lower() == 'mha':
        # Full KV cache: batch_size * num_heads * seq_len * d_k * 2 (K and V)
        cache_size = batch_size * num_heads * seq_len * d_k * 2 * bytes_per_element
        
    elif attention_type.lower() == 'mla':
        # Compressed latent cache
        latent_dim = kwargs.get('latent_dim', 512)
        cache_size = batch_size * seq_len * latent_dim * bytes_per_element
        
    elif attention_type.lower() == 'gqa':
        # Reduced KV heads
        num_kv_heads = kwargs.get('num_kv_heads', num_heads // 4)
        cache_size = batch_size * num_kv_heads * seq_len * d_k * 2 * bytes_per_element
        
    elif attention_type.lower() == 'linear':
        # Similar to MHA but can be optimized
        cache_size = batch_size * num_heads * seq_len * d_k * 2 * bytes_per_element
        
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return cache_size


if __name__ == "__main__":
    # Test different attention mechanisms
    batch_size = 2
    seq_len = 100
    d_model = 512
    num_heads = 8
    
    print("Testing Attention Mechanisms")
    print("=" * 50)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    for attn_type in ['mha', 'mla', 'gqa', 'linear']:
        print(f"\nTesting {attn_type.upper()}...")
        
        if attn_type == 'mla':
            attn = get_attention_module(attn_type, d_model, num_heads, latent_dim=256)
        elif attn_type == 'gqa':
            attn = get_attention_module(attn_type, d_model, num_heads, num_kv_heads=2)
        else:
            attn = get_attention_module(attn_type, d_model, num_heads)
        
        output, cache = attn(x, x, x)
        print(f"Output shape: {output.shape}")
        
        # Calculate cache size
        if attn_type == 'mla':
            cache_size = calculate_kv_cache_size(attn_type, batch_size, seq_len, d_model, num_heads, latent_dim=256)
        elif attn_type == 'gqa':
            cache_size = calculate_kv_cache_size(attn_type, batch_size, seq_len, d_model, num_heads, num_kv_heads=2)
        else:
            cache_size = calculate_kv_cache_size(attn_type, batch_size, seq_len, d_model, num_heads)
        
        print(f"KV Cache Size: {cache_size / 1024:.2f} KB")
        print(f"Cache keys: {list(cache.keys())}")
