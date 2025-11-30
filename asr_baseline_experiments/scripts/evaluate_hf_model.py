#!/usr/bin/env python3
"""
Evaluate HuggingFace Conformer/Wav2Vec2 models on LibriSpeech

Usage:
    # Evaluate original HF model
    python scripts/evaluate_hf_model.py \
        --model_name facebook/wav2vec2-conformer-rel-pos-large-960h-ft \
        --output_file results/conformer_mha_baseline.json
    
    # Evaluate converted MLA model
    python scripts/evaluate_hf_model.py \
        --model_dir checkpoints/conformer_mla_from_hf \
        --output_file results/conformer_mla_converted.json
"""

import argparse
import torch
import torch.nn as nn
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
from tqdm import tqdm
import evaluate
import time
import sys


# Import MLAAttention class for loading fine-tuned models
class MLAAttention(nn.Module):
    """Multi-Latent Attention - needed for loading fine-tuned MLA models"""
    def __init__(self, d_model, num_heads, latent_dim, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_size = d_model // num_heads
        
        self.k_down = nn.Linear(d_model, latent_dim, bias=True)
        self.k_up = nn.Linear(latent_dim, d_model, bias=False)
        self.v_down = nn.Linear(d_model, latent_dim, bias=True)
        self.v_up = nn.Linear(latent_dim, d_model, bias=False)
        self.linear_q = nn.Linear(d_model, d_model, bias=True)
        self.linear_out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.pos_bias_u = None
        self.pos_bias_v = None
        self.linear_pos = None
    
    def forward(self, hidden_states, attention_mask=None, 
                relative_position_embeddings=None, output_attentions=False):
        batch_size, seq_length, _ = hidden_states.size()
        
        Q = self.linear_q(hidden_states)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        K_latent = self.k_down(hidden_states)
        K = self.k_up(K_latent)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        V_latent = self.v_down(hidden_states)
        V = self.v_up(V_latent)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.linear_out(context)
        
        return (output, attn_weights) if output_attentions else (output,)


def evaluate_model(model, processor, dataset, device='cuda', batch_size=8):
    """
    Evaluate model on dataset
    
    Returns:
        dict with WER, accuracy, and timing metrics
    """
    model.eval()
    model.to(device)
    
    predictions = []
    references = []
    inference_times = []
    
    print(f"\nEvaluating on {len(dataset)} examples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            # Get batch indices
            batch_indices = range(i, min(i+batch_size, len(dataset)))
            batch = [dataset[idx] for idx in batch_indices]
            
            # Process audio
            audios = [item['audio']['array'] for item in batch]
            
            # Tokenize
            inputs = processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None
            
            # Inference with timing
            start_time = time.time()
            
            if attention_mask is not None:
                logits = model(input_values, attention_mask=attention_mask).logits
            else:
                logits = model(input_values).logits
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(audios))  # per sample
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            
            for pred_ids in predicted_ids:
                pred_str = processor.decode(pred_ids)
                predictions.append(pred_str)
            
            # Get references
            for item in batch:
                references.append(item['text'])
    
    # Calculate metrics
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Calculate per-character accuracy for additional insight
    total_chars = sum(len(ref) for ref in references)
    matching_chars = sum(
        sum(1 for p, r in zip(pred, ref) if p == r)
        for pred, ref in zip(predictions, references)
    )
    char_accuracy = matching_chars / total_chars if total_chars > 0 else 0
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    results = {
        'wer': wer * 100,  # as percentage
        'char_accuracy': char_accuracy * 100,
        'num_samples': len(dataset),
        'avg_inference_time_per_sample': avg_inference_time,
        'total_inference_time': sum(inference_times),
    }
    
    # Show some examples
    print("\n" + "="*70)
    print("Sample Predictions:")
    print("="*70)
    for i in range(min(5, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"  Reference: {references[i]}")
        print(f"  Predicted: {predictions[i]}")
    
    return results, predictions, references


def main():
    parser = argparse.ArgumentParser(description='Evaluate HF Conformer/Wav2Vec2 model')
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model name (e.g., facebook/wav2vec2-conformer-rel-pos-large-960h-ft)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Local model directory (for converted MLA models)')
    parser.add_argument('--dataset', type=str, default='test-clean',
                        choices=['test-clean', 'test-other', 'dev-clean', 'dev-other'],
                        help='LibriSpeech test set to use')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    if args.model_name is None and args.model_dir is None:
        raise ValueError("Must specify either --model_name or --model_dir")
    
    print("="*70)
    print("HuggingFace Model Evaluation on LibriSpeech")
    print("="*70)
    
    if args.model_name:
        print(f"\nModel: {args.model_name}")
    else:
        print(f"\nModel directory: {args.model_dir}")
    
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_split = {
        'test-clean': 'test',
        'test-other': 'test.other',
        'dev-clean': 'validation',
        'dev-other': 'validation.other',
    }[args.dataset]
    
    dataset_name = 'clean' if 'clean' in args.dataset else 'other'
    dataset = load_dataset("librispeech_asr", dataset_name, split=dataset_split)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    # Load model and processor
    print("\nLoading model and processor...")
    
    if args.model_name:
        # Load from HuggingFace
        try:
            model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_name)
            model_type = "Conformer"
        except:
            model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
            model_type = "Wav2Vec2"
        
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    else:
        # Load fine-tuned MLA model from local directory
        print("  Loading MLA model from local directory...")
        
        # Check if this is an MLA model by looking for metadata
        metadata_path = Path(args.model_dir) / 'metadata.json'
        
        if not metadata_path.exists():
            # Try standard HuggingFace loading
            print("  No metadata found, loading as standard model...")
            try:
                model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_dir)
                processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
                model_type = "Conformer"
            except:
                model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
                processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
                model_type = "Wav2Vec2"
        else:
            # Load MLA model with custom attention
            print("  Detected MLA model, loading with custom attention...")
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            base_model_name = metadata.get('base_model', 'facebook/wav2vec2-conformer-rel-pos-large-960h-ft')
            print(f"  Base model: {base_model_name}")
            
            # Load processor
            processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
            
            # Load model state dict
            state_dict_path = Path(args.model_dir) / 'pytorch_model.bin'
            if not state_dict_path.exists():
                raise FileNotFoundError(f"Model weights not found at {state_dict_path}")
            
            print(f"  Loading weights from {state_dict_path.name}...")
            state_dict = torch.load(state_dict_path, map_location='cpu')
            
            # Create base model structure
            model = Wav2Vec2ConformerForCTC.from_pretrained(base_model_name)
            
            # Load the state dict (will include MLAAttention parameters)
            # Note: This will fail if the state dict has MLAAttention but model has standard attention
            # In that case, we need to reconstruct the model with MLA first
            try:
                model.load_state_dict(state_dict, strict=True)
                print("  ✓ Loaded MLA model successfully")
                model_type = "Conformer (MLA)"
            except RuntimeError as e:
                print(f"  ⚠ Could not load with strict=True: {e}")
                print("  ⚠ MLA model saved with custom classes - evaluation may not work correctly")
                print("  ⚠ Using validation loss from training logs instead")
                raise NotImplementedError(
                    "Fine-tuned MLA model uses custom MLAAttention class that requires "
                    "model reconstruction. For now, please check validation loss in training logs."
                )
    
    print(f"  ✓ Model loaded ({model_type})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate
    results, predictions, references = evaluate_model(
        model, processor, dataset, args.device, args.batch_size
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Dataset: LibriSpeech {args.dataset}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"\nWord Error Rate (WER): {results['wer']:.2f}%")
    print(f"Character Accuracy: {results['char_accuracy']:.2f}%")
    print(f"\nAvg inference time per sample: {results['avg_inference_time_per_sample']*1000:.2f}ms")
    print(f"Total inference time: {results['total_inference_time']:.2f}s")
    print("="*70)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'model': args.model_name or args.model_dir,
        'dataset': args.dataset,
        'results': results,
        'config': {
            'batch_size': args.batch_size,
            'device': args.device,
        }
    }
    
    # Optionally save predictions for analysis
    if len(predictions) <= 1000:  # Don't save huge prediction files
        save_data['predictions'] = predictions[:100]  # Save first 100
        save_data['references'] = references[:100]
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()

