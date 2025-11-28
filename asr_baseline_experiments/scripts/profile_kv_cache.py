"""
Profile KV Cache Size and Inference Metrics
"""

import os
import sys
import argparse
import torch
import yaml
import json
import time
import numpy as np
from pathlib import Path
import torchaudio
from datasets import load_dataset
from jiwer import wer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.conformer import build_conformer
from models.branchformer import build_branchformer
from models.attention_variants import calculate_kv_cache_size


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Build model
    if config['model_type'] == 'conformer':
        model = build_conformer(config)
    elif config['model_type'] == 'branchformer':
        model = build_branchformer(config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def extract_features(audio, n_mels=80):
    """Extract mel-spectrogram features"""
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160
    )
    
    features = mel_spec(audio)
    features = torch.log(features + 1e-9)
    return features.transpose(1, 2)


def decode_predictions(logits, vocab):
    """Decode logits to text"""
    # Greedy decoding
    predictions = torch.argmax(logits, dim=-1)
    
    # Convert to text
    inv_vocab = {v: k for k, v in vocab.items()}
    texts = []
    
    for pred in predictions:
        chars = []
        prev_char = None
        for idx in pred:
            idx = idx.item()
            if idx == 0:  # blank
                prev_char = None
                continue
            if idx == prev_char:
                continue
            if idx in [1, 2, 3]:  # special tokens
                continue
            chars.append(inv_vocab.get(idx, ''))
            prev_char = idx
        texts.append(''.join(chars))
    
    return texts


def measure_kv_cache_size(model, config, device, num_samples=100):
    """Measure actual KV cache size during inference"""
    
    # Load test dataset
    test_dataset = load_dataset("librispeech_asr", "clean", split="test")
    
    cache_sizes = []
    inference_times = []
    memory_usage = []
    
    # Sample random utterances
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    print(f"Measuring KV cache for {len(indices)} samples...")
    
    for idx in indices:
        item = test_dataset[int(idx)]
        audio = torch.tensor(item['audio']['array']).float().unsqueeze(0).to(device)
        
        # Extract features
        features = extract_features(audio).to(device)
        seq_len = features.size(1)
        
        # Measure memory before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            logits, caches = model(features, torch.tensor([seq_len]).to(device))
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Measure cache size
        total_cache_size = 0
        for cache in caches:
            if cache is not None:
                for key, value in cache.items():
                    if isinstance(value, torch.Tensor):
                        total_cache_size += value.element_size() * value.nelement()
        
        cache_sizes.append(total_cache_size)
        
        # Measure memory after inference
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            memory_usage.append(peak_mem - mem_before)
    
    # Calculate theoretical cache size
    attention_type = config.get('attention_type', 'mha')
    d_model = config.get('d_model', 512)
    num_heads = config.get('num_heads', 8)
    
    # Average sequence length
    avg_seq_len = int(np.mean([test_dataset[int(i)]['audio']['array'].shape[0] for i in indices]) / 160)
    
    if attention_type == 'mla':
        attention_kwargs = {'latent_dim': config.get('attention_kwargs', {}).get('latent_dim', 512)}
    elif attention_type == 'gqa':
        attention_kwargs = {'num_kv_heads': config.get('attention_kwargs', {}).get('num_kv_heads', num_heads // 4)}
    else:
        attention_kwargs = {}
    
    theoretical_cache_size = calculate_kv_cache_size(
        attention_type, 1, avg_seq_len, d_model, num_heads, **attention_kwargs
    )
    
    results = {
        'attention_type': attention_type,
        'num_samples': len(indices),
        'avg_cache_size_bytes': float(np.mean(cache_sizes)),
        'std_cache_size_bytes': float(np.std(cache_sizes)),
        'avg_cache_size_kb': float(np.mean(cache_sizes) / 1024),
        'avg_cache_size_mb': float(np.mean(cache_sizes) / (1024 * 1024)),
        'theoretical_cache_size_bytes': theoretical_cache_size,
        'theoretical_cache_size_kb': theoretical_cache_size / 1024,
        'avg_inference_time_ms': float(np.mean(inference_times) * 1000),
        'std_inference_time_ms': float(np.std(inference_times) * 1000),
        'avg_memory_usage_mb': float(np.mean(memory_usage) / (1024 * 1024)) if memory_usage else 0,
        'avg_sequence_length': avg_seq_len
    }
    
    return results


def evaluate_wer(model, config, vocab, device, num_samples=500):
    """Evaluate Word Error Rate"""
    
    # Load test dataset
    test_dataset = load_dataset("librispeech_asr", "clean", split="test")
    
    references = []
    hypotheses = []
    
    # Sample random utterances
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    print(f"Evaluating WER on {len(indices)} samples...")
    
    for idx in indices:
        item = test_dataset[int(idx)]
        audio = torch.tensor(item['audio']['array']).float().unsqueeze(0).to(device)
        
        # Extract features
        features = extract_features(audio).to(device)
        seq_len = features.size(1)
        
        # Inference
        with torch.no_grad():
            logits, _ = model(features, torch.tensor([seq_len]).to(device))
        
        # Decode
        hypothesis = decode_predictions(logits, vocab)[0]
        reference = item['text'].lower()
        
        references.append(reference)
        hypotheses.append(hypothesis)
    
    # Calculate WER
    word_error_rate = wer(references, hypotheses)
    
    return word_error_rate, references[:10], hypotheses[:10]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    
    # Load vocabulary
    vocab_path = Path(args.checkpoint).parent / 'vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"Model: {config['model_type']}")
    print(f"Attention: {config['attention_type']}")
    print(f"Parameters: {model.get_num_params():,}")
    
    # Profile KV cache
    print("\n" + "="*60)
    print("Profiling KV Cache")
    print("="*60)
    
    cache_results = measure_kv_cache_size(model, config, device, args.num_cache_samples)
    
    print(f"\nKV Cache Results:")
    print(f"  Average Cache Size: {cache_results['avg_cache_size_kb']:.2f} KB")
    print(f"  Theoretical Cache Size: {cache_results['theoretical_cache_size_kb']:.2f} KB")
    print(f"  Average Inference Time: {cache_results['avg_inference_time_ms']:.2f} ms")
    if cache_results['avg_memory_usage_mb'] > 0:
        print(f"  Average Memory Usage: {cache_results['avg_memory_usage_mb']:.2f} MB")
    
    # Evaluate WER
    if args.eval_wer:
        print("\n" + "="*60)
        print("Evaluating Word Error Rate")
        print("="*60)
        
        word_error_rate, refs, hyps = evaluate_wer(model, config, vocab, device, args.num_wer_samples)
        
        print(f"\nWord Error Rate: {word_error_rate * 100:.2f}%")
        
        print("\nSample Predictions:")
        for i, (ref, hyp) in enumerate(zip(refs[:5], hyps[:5])):
            print(f"\n  Example {i+1}:")
            print(f"    Reference:  {ref}")
            print(f"    Hypothesis: {hyp}")
        
        cache_results['wer'] = float(word_error_rate)
        cache_results['wer_percent'] = float(word_error_rate * 100)
    
    # Save results
    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / 'profile_results.json'
    with open(output_path, 'w') as f:
        json.dump(cache_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile KV cache and evaluate model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--num_cache_samples', type=int, default=100, help='Number of samples for cache profiling')
    parser.add_argument('--num_wer_samples', type=int, default=500, help='Number of samples for WER evaluation')
    parser.add_argument('--eval_wer', action='store_true', help='Evaluate WER')
    
    args = parser.parse_args()
    main(args)
