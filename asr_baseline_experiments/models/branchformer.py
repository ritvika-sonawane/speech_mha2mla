"""
Branchformer: Parallel-branch architecture with attention and MLP
Supports different attention mechanisms (MHA, MLA, GQA, Linear)
"""

import torch
import torch.nn as nn
from typing import Optional
from .attention_variants import get_attention_module


class MultiLayerPerceptron(nn.Module):
    """MLP branch in Branchformer"""
    
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, expansion_factor * d_model)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion_factor * d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConvolutionModule(nn.Module):
    """Depthwise separable convolution"""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class BranchformerBlock(nn.Module):
    """
    Branchformer block with parallel attention and MLP branches
    Combined with convolution module
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        merge_method: str = 'concat',  # 'concat', 'add', or 'learned_add'
        attention_type: str = 'mha',
        attention_kwargs: dict = None
    ):
        super().__init__()
        
        self.merge_method = merge_method
        
        # Layer normalization before branches
        self.norm1 = nn.LayerNorm(d_model)
        
        # Attention branch
        kwargs = attention_kwargs or {}
        self.attention = get_attention_module(attention_type, d_model, num_heads, dropout=dropout, **kwargs)
        
        # MLP branch
        self.mlp = MultiLayerPerceptron(d_model, mlp_expansion_factor, dropout)
        
        # Merge branches
        if merge_method == 'concat':
            self.merge_proj = nn.Linear(2 * d_model, d_model)
        elif merge_method == 'learned_add':
            self.merge_weight = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Final feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = MultiLayerPerceptron(d_model, mlp_expansion_factor, dropout)
        
    def forward(self, x, mask=None, cache=None):
        # First residual block: parallel branches
        residual = x
        x = self.norm1(x)
        
        # Attention branch
        attn_out, new_cache = self.attention(x, x, x, mask, cache)
        
        # MLP branch
        mlp_out = self.mlp(x)
        
        # Merge branches
        if self.merge_method == 'concat':
            branch_out = self.merge_proj(torch.cat([attn_out, mlp_out], dim=-1))
        elif self.merge_method == 'add':
            branch_out = attn_out + mlp_out
        elif self.merge_method == 'learned_add':
            branch_out = self.merge_weight * attn_out + (1 - self.merge_weight) * mlp_out
        
        x = residual + self.dropout(branch_out)
        
        # Second residual block: convolution
        residual = x
        x = self.norm2(x)
        x = residual + self.conv(x)
        
        # Third residual block: feed-forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)
        
        return x, new_cache


class Branchformer(nn.Module):
    """Complete Branchformer model for ASR"""
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        mlp_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        vocab_size: int = 5000,
        merge_method: str = 'concat',
        attention_type: str = 'mha',
        attention_kwargs: dict = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Input projection with subsampling
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate input dimension after conv layers
        conv_output_dim = (input_dim // 4) * 32
        self.linear_proj = nn.Linear(conv_output_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Branchformer blocks
        self.layers = nn.ModuleList([
            BranchformerBlock(
                d_model, num_heads, mlp_expansion_factor,
                conv_kernel_size, dropout, merge_method,
                attention_type, attention_kwargs
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, lengths=None, cache_list=None):
        """
        Args:
            x: [B, T, input_dim] - input features
            lengths: [B] - sequence lengths
            cache_list: list of caches for each layer
        """
        # Input projection with subsampling
        # Add channel dimension
        x = x.unsqueeze(1)  # [B, 1, T, input_dim]
        x = self.input_proj(x)  # [B, 32, T//4, input_dim//4]
        
        # Flatten spatial dimensions
        batch_size, channels, time, freq = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T//4, input_dim//4, 32]
        x = x.view(batch_size, time, -1)  # [B, T//4, (input_dim//4)*32]
        
        x = self.linear_proj(x)
        x = self.input_dropout(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Adjust lengths for subsampling
        if lengths is not None:
            lengths = lengths // 4
        
        # Create mask from lengths
        mask = None
        if lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        
        # Process through Branchformer blocks
        new_cache_list = []
        for i, layer in enumerate(self.layers):
            cache = cache_list[i] if cache_list is not None else None
            x, new_cache = layer(x, mask, cache)
            new_cache_list.append(new_cache)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits, new_cache_list
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def build_branchformer(config):
    """Build Branchformer model from config"""
    return Branchformer(
        input_dim=config.get('input_dim', 80),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 12),
        mlp_expansion_factor=config.get('mlp_expansion_factor', 4),
        conv_kernel_size=config.get('conv_kernel_size', 31),
        dropout=config.get('dropout', 0.1),
        vocab_size=config.get('vocab_size', 5000),
        merge_method=config.get('merge_method', 'concat'),
        attention_type=config.get('attention_type', 'mha'),
        attention_kwargs=config.get('attention_kwargs', None)
    )


if __name__ == "__main__":
    # Test Branchformer with different attention types
    batch_size = 2
    seq_len = 100
    input_dim = 80
    
    print("Testing Branchformer with different attention mechanisms")
    print("=" * 60)
    
    for attn_type in ['mha', 'mla', 'gqa', 'linear']:
        print(f"\n{attn_type.upper()}:")
        
        if attn_type == 'mla':
            attn_kwargs = {'latent_dim': 256}
        elif attn_type == 'gqa':
            attn_kwargs = {'num_kv_heads': 2}
        else:
            attn_kwargs = None
        
        model = Branchformer(
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
