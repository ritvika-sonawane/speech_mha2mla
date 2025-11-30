#!/usr/bin/env python3
"""
Fine-tune Conformer with converted MLA attention

This script:
1. Loads the original HF Conformer model
2. Loads converted MLA weights
3. Replaces MHA attention with MLA in each layer
4. Fine-tunes on LibriSpeech

Usage:
    python scripts/finetune_mla_conformer.py \
        --base_model facebook/wav2vec2-conformer-rel-pos-large-960h-ft \
        --mla_weights checkpoints/conformer_mla_from_hf/converted_model.pt \
        --output_dir results/conformer_mla_finetuned \
        --num_epochs 10
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    Wav2Vec2ConformerForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from dataclasses import dataclass
from typing import Dict, List, Union
import json
import soundfile as sf
import numpy as np
from tqdm import tqdm


class MLAAttention(nn.Module):
    """
    Multi-Latent Attention module to replace standard MHA
    Uses compressed K/V projections via low-rank factorization
    """
    def __init__(self, d_model, num_heads, latent_dim, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_size = d_model // num_heads
        
        # Compressed K/V projections (two-stage)
        self.k_down = nn.Linear(d_model, latent_dim, bias=True)
        self.k_up = nn.Linear(latent_dim, d_model, bias=False)
        
        self.v_down = nn.Linear(d_model, latent_dim, bias=True)
        self.v_up = nn.Linear(latent_dim, d_model, bias=False)
        
        # Q projection (full rank for simple MLA variant)
        self.linear_q = nn.Linear(d_model, d_model, bias=True)
        
        # Output projection
        self.linear_out = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Position embeddings (copy from original)
        self.pos_bias_u = None
        self.pos_bias_v = None
        self.linear_pos = None
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        relative_position_embeddings=None,
        output_attentions=False
    ):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Q projection (full rank)
        Q = self.linear_q(hidden_states)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_size)
        Q = Q.transpose(1, 2)  # [B, H, T, head_size]
        
        # K projection (compressed)
        K_latent = self.k_down(hidden_states)  # [B, T, latent_dim]
        K = self.k_up(K_latent)  # [B, T, d_model]
        K = K.view(batch_size, seq_length, self.num_heads, self.head_size)
        K = K.transpose(1, 2)  # [B, H, T, head_size]
        
        # V projection (compressed)
        V_latent = self.v_down(hidden_states)  # [B, T, latent_dim]
        V = self.v_up(V_latent)  # [B, T, d_model]
        V = V.view(batch_size, seq_length, self.num_heads, self.head_size)
        V = V.transpose(1, 2)  # [B, H, T, head_size]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, H, T, head_size]
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        output = self.linear_out(context)
        
        # Always return 2 values to match HF API: (hidden_states, attn_weights)
        # attn_weights is None when output_attentions is False
        return (output, attn_weights if output_attentions else None)


def load_mla_weights_into_attention(mla_attention, converted_weights):
    """Load converted MLA weights into our custom MLA module"""
    with torch.no_grad():
        # K projections
        # k_down: Linear(1024->512) expects [512, 1024], converted has [512, 1024] ✓
        mla_attention.k_down.weight.copy_(converted_weights['k_down'])
        # k_up: Linear(512->1024) expects [1024, 512], converted has [512, 1024] - needs transpose
        mla_attention.k_up.weight.copy_(converted_weights['k_up'].t())
        
        # V projections
        # v_down: Linear(1024->512) expects [512, 1024], converted has [512, 1024] ✓
        mla_attention.v_down.weight.copy_(converted_weights['v_down'])
        # v_up: Linear(512->1024) expects [1024, 512], converted has [512, 1024] - needs transpose
        mla_attention.v_up.weight.copy_(converted_weights['v_up'].t())
        
        # Q projection (full)
        mla_attention.linear_q.weight.copy_(converted_weights['q_proj'])
        
        # Output projection
        mla_attention.linear_out.weight.copy_(converted_weights['out_proj'])
        
        # Biases
        # Q and output biases match dimensions, so we can copy them
        if 'q_bias' in converted_weights:
            mla_attention.linear_q.bias.copy_(converted_weights['q_bias'])
        if 'out_bias' in converted_weights:
            mla_attention.linear_out.bias.copy_(converted_weights['out_bias'])
        
        # K/V biases from original projections don't match compressed dimensions
        # Initialize k_down and v_down biases to zero (they'll be learned during fine-tuning)
        # k_up and v_up don't have biases (bias=False in their definition)


def replace_attention_with_mla(model, converted_layers, latent_dim):
    """
    Replace all MHA attention layers with MLA in the model
    
    Args:
        model: HuggingFace Conformer model
        converted_layers: List of converted MLA weight dicts
        latent_dim: Latent dimension for MLA
    """
    config = model.config
    encoder_layers = model.wav2vec2_conformer.encoder.layers
    
    print(f"\nReplacing {len(encoder_layers)} attention layers with MLA...")
    
    for i, (layer, mla_weights) in enumerate(zip(encoder_layers, converted_layers)):
        # Get original attention config
        old_attn = layer.self_attn
        
        # Create new MLA attention
        mla_attn = MLAAttention(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            latent_dim=latent_dim,
            dropout=config.attention_dropout
        )
        
        # Copy position embeddings from original
        if hasattr(old_attn, 'pos_bias_u'):
            mla_attn.pos_bias_u = old_attn.pos_bias_u
        if hasattr(old_attn, 'pos_bias_v'):
            mla_attn.pos_bias_v = old_attn.pos_bias_v
        if hasattr(old_attn, 'linear_pos'):
            mla_attn.linear_pos = old_attn.linear_pos
        
        # Load converted MLA weights
        load_mla_weights_into_attention(mla_attn, mla_weights)
        
        # Replace attention module
        layer.self_attn = mla_attn
        
        if (i + 1) % 6 == 0:
            print(f"  ✓ Replaced layers 1-{i+1}")
    
    print(f"  ✓ All {len(encoder_layers)} layers replaced with MLA\n")
    
    return model


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC training
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: int = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Use tokenizer directly instead of deprecated as_target_processor()
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        
        return batch


def load_librispeech_offline(data_dir, split='train-clean-100', max_samples=None):
    """
    Load LibriSpeech data from offline directory
    
    Args:
        data_dir: Path to the base LibriSpeech directory (e.g., 'data/librispeech/LibriSpeech')
        split: One of 'train-clean-100', 'dev-clean', 'dev-other', 'test-clean', 'test-other'
        max_samples: Maximum number of samples to load (optional)
    
    Returns:
        HuggingFace Dataset object with audio and text
    """
    data_path = Path(data_dir) / split
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    print(f"  Loading offline data from: {data_path}")
    
    # Collect all samples
    samples = []
    
    # Iterate through speaker directories
    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        # Iterate through chapter directories
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            # Find transcript file
            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            
            if not trans_file.exists():
                continue
            
            # Read transcripts
            with open(trans_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(' ', 1)
                    if len(parts) != 2:
                        continue
                    
                    audio_id, text = parts
                    audio_file = chapter_dir / f"{audio_id}.flac"
                    
                    if not audio_file.exists():
                        continue
                    
                    samples.append({
                        'audio_id': audio_id,
                        'audio_path': str(audio_file),
                        'text': text
                    })
                    
                    # Stop early if we have enough samples
                    if max_samples and len(samples) >= max_samples:
                        break
            
            if max_samples and len(samples) >= max_samples:
                break
        
        if max_samples and len(samples) >= max_samples:
            break
    
    print(f"  Found {len(samples)} samples")
    
    # Load audio data
    audio_data = []
    texts = []
    
    print(f"  Loading audio files...")
    for sample in tqdm(samples, desc="  Loading audio"):
        try:
            audio_array, sample_rate = sf.read(sample['audio_path'])
            
            # Ensure mono and correct sample rate
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Convert to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            audio_data.append({
                'array': audio_array,
                'sampling_rate': sample_rate,
                'path': sample['audio_path']
            })
            texts.append(sample['text'])
        except Exception as e:
            print(f"  Error loading {sample['audio_path']}: {e}")
            continue
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({
        'audio': audio_data,
        'text': texts
    })
    
    return dataset


def prepare_dataset(batch, processor):
    """Prepare audio and text for training"""
    audio = batch["audio"]
    
    # Process audio
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    # Process text using tokenizer directly instead of deprecated as_target_processor()
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MLA Conformer')
    parser.add_argument('--base_model', type=str, 
                        default='facebook/wav2vec2-conformer-rel-pos-large-960h-ft',
                        help='Base HuggingFace model')
    parser.add_argument('--mla_weights', type=str, required=True,
                        help='Path to converted MLA weights (.pt file)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate (low for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Max training samples (for testing)')
    parser.add_argument('--offline_data_dir', type=str, 
                        default='/home/ubuntu/speech_mha2mla/asr_baseline_experiments/data/librispeech/LibriSpeech',
                        help='Path to offline LibriSpeech data directory')
    parser.add_argument('--use_hf_dataset', action='store_true',
                        help='Use HuggingFace dataset instead of offline data')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Fine-tuning Conformer with MLA Attention")
    print("="*70)
    print(f"\nBase model: {args.base_model}")
    print(f"MLA weights: {args.mla_weights}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Load processor
    print("\nLoading processor...")
    processor = Wav2Vec2Processor.from_pretrained(args.base_model)
    
    # Load base model
    print("Loading base model...")
    model = Wav2Vec2ConformerForCTC.from_pretrained(args.base_model)
    
    # Load converted MLA weights
    print("Loading converted MLA weights...")
    checkpoint = torch.load(args.mla_weights, map_location='cpu')
    converted_layers = checkpoint['converted_attention_layers']
    config = checkpoint['config']
    latent_dim = config['latent_dim']
    
    print(f"  Loaded {len(converted_layers)} converted layers")
    print(f"  Latent dim: {latent_dim}")
    
    # Replace attention with MLA
    model = replace_attention_with_mla(model, converted_layers, latent_dim)
    
    # Freeze feature extractor
    model.freeze_feature_encoder()
    print("✓ Frozen feature extractor (only fine-tuning attention)\n")
    
    # Load datasets
    print("Loading LibriSpeech dataset...")
    
    if args.use_hf_dataset:
        # Use HuggingFace dataset
        print("  Using HuggingFace dataset loader...")
        train_dataset = load_dataset("librispeech_asr", "clean", split="train.100")
        eval_dataset = load_dataset("librispeech_asr", "clean", split="validation")
        
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
            print(f"  Using {args.max_train_samples} training samples for testing")
    else:
        # Use offline dataset
        print("  Using offline LibriSpeech data...")
        train_dataset = load_librispeech_offline(
            args.offline_data_dir, 
            'train-clean-100',
            max_samples=args.max_train_samples
        )
        eval_dataset = load_librispeech_offline(
            args.offline_data_dir, 
            'dev-clean'
        )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval: {len(eval_dataset)} samples")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_dataset.column_names,
        num_proc=4
    )
    eval_dataset = eval_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=eval_dataset.column_names,
        num_proc=4
    )
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=args.num_epochs,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=4,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting fine-tuning...")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save metadata
    metadata = {
        'base_model': args.base_model,
        'mla_config': config,
        'training_args': vars(args),
        'final_eval_loss': trainer.state.best_metric,
    }
    
    with open(Path(args.output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Best eval loss: {trainer.state.best_metric:.4f}")
    print("\nNext steps:")
    print("  1. Evaluate on test set")
    print("  2. Compare WER with original MHA model")
    print("  3. Measure inference speed and memory usage")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

