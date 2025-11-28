"""
ASR Models Package

Includes:
- Conformer and Branchformer architectures
- Multiple attention mechanisms (MHA, MLA, GQA, Linear)
- MHA-to-MLA conversion utilities
"""

from .attention_variants import (
    MultiHeadAttention,
    MultiHeadLatentAttention,
    MultiHeadLatentAttentionSimple,
    GroupedQueryAttention,
    LinearAttention,
    get_attention_module,
    calculate_kv_cache_size
)

from .conformer import (
    Conformer,
    ConformerBlock,
    build_conformer
)

from .branchformer import (
    Branchformer,
    BranchformerBlock,
    build_branchformer
)

from .mha_to_mla_conversion import (
    convert_mha_to_mla_simple_svd,
    convert_mha_to_mla_svd,
    analyze_compression_quality,
    svd_factorize_matrix
)

__all__ = [
    # Attention mechanisms
    'MultiHeadAttention',
    'MultiHeadLatentAttention',
    'MultiHeadLatentAttentionSimple',
    'GroupedQueryAttention',
    'LinearAttention',
    'get_attention_module',
    'calculate_kv_cache_size',
    
    # Conformer
    'Conformer',
    'ConformerBlock',
    'build_conformer',
    
    # Branchformer
    'Branchformer',
    'BranchformerBlock',
    'build_branchformer',
    
    # Conversion utilities
    'convert_mha_to_mla_simple_svd',
    'convert_mha_to_mla_svd',
    'analyze_compression_quality',
    'svd_factorize_matrix',
]

