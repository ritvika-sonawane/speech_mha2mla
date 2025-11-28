"""
Conformer: Convolution-augmented Transformer for Speech Recognition
Supports different attention mechanisms (MHA, MLA, GQA, Linear)
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention_variants import get_attention_module


class ConvolutionModule(nn.Module):
    """Convolution module in Conformer"""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, D]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [B, T, D]
        return residual + x


class FeedForwardModule(nn.Module):
    """Feed-forward module in Conformer"""
    
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, expansion_factor * d_model)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion_factor * d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return 0.5 * residual + x


class ConformerBlock(nn.Module):
    """Single Conformer block"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        attention_type: str = 'mha',
        attention_kwargs: dict = None
    ):
        super().__init__()
        
        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        
        # Self-attention with configurable attention type
        self.attention_layer_norm = nn.LayerNorm(d_model)
        kwargs = attention_kwargs or {}
        self.attention = get_attention_module(attention_type, d_model, num_heads, dropout=dropout, **kwargs)
        self.attention_dropout = nn.Dropout(dropout)
        
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None, cache=None):
        # First feed-forward
        x = self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.attention_layer_norm(x)
        attn_out, new_cache = self.attention(x, x, x, mask, cache)
        x = residual + self.attention_dropout(attn_out)
        
        # Convolution module
        x = self.conv_module(x)
        
        # Second feed-forward
        x = self.ff2(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x, new_cache


class Conformer(nn.Module):
    """Complete Conformer model for ASR"""
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        vocab_size: int = 5000,
        attention_type: str = 'mha',
        attention_kwargs: dict = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model, num_heads, conv_kernel_size,
                ff_expansion_factor, dropout,
                attention_type, attention_kwargs
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, lengths=None, cache_list=None):
        """
        Args:
            x: [B, T, input_dim] - input features
            lengths: [B] - sequence lengths
            cache_list: list of caches for each layer
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_dropout(x)
        
        # Create mask from lengths
        mask = None
        if lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        
        # Process through Conformer blocks
        new_cache_list = []
        for i, layer in enumerate(self.layers):
            cache = cache_list[i] if cache_list is not None else None
            x, new_cache = layer(x, mask, cache)
            new_cache_list.append(new_cache)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits, new_cache_list
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_conformer(config):
    """Build Conformer model from config"""
    return Conformer(
        input_dim=config.get('input_dim', 80),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 12),
        conv_kernel_size=config.get('conv_kernel_size', 31),
        ff_expansion_factor=config.get('ff_expansion_factor', 4),
        dropout=config.get('dropout', 0.1),
        vocab_size=config.get('vocab_size', 5000),
        attention_type=config.get('attention_type', 'mha'),
        attention_kwargs=config.get('attention_kwargs', None)
    )


if __name__ == "__main__":
    # Test Conformer with different attention types
    batch_size = 2
    seq_len = 100
    input_dim = 80
    
    print("Testing Conformer with different attention mechanisms")
    print("=" * 60)
    
    for attn_type in ['mha', 'mla', 'gqa', 'linear']:
        print(f"\n{attn_type.upper()}:")
        
        if attn_type == 'mla':
            attn_kwargs = {'latent_dim': 256}
        elif attn_type == 'gqa':
            attn_kwargs = {'num_kv_heads': 2}
        else:
            attn_kwargs = None
        
        model = Conformer(
            input_dim=input_dim,
            d_model=256,
            num_heads=4,
            num_layers=4,
            attention_type=attn_type,
            attention_kwargs=attn_kwargs
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len, seq_len - 10])
        
        logits, caches = model(x, lengths)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Number of parameters: {model.get_num_params():,}")
        print(f"  Number of caches: {len(caches)}")
