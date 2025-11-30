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
from datasets import load_dataset, Dataset
from jiwer import wer
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.conformer import build_conformer
from models.branchformer import build_branchformer
from models.attention_variants import calculate_kv_cache_size


def load_librispeech_offline(data_dir, split='test-clean', max_samples=None):
    """
    Load LibriSpeech data from offline directory
    
    Args:
        data_dir: Path to the base LibriSpeech directory
        split: One of 'test-clean', 'test-other', 'dev-clean', 'dev-other'
        max_samples: Maximum number of samples to load
    
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
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Convert to float32
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


def measure_kv_cache_size(model, config, device, test_dataset, num_samples=100):
    """Measure actual KV cache size during inference"""
    
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


def evaluate_wer(model, config, vocab, device, test_dataset, num_samples=500):
    """Evaluate Word Error Rate"""
    
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
    
    # Load dataset
    print("\nLoading dataset...")
    if args.use_hf_dataset:
        print("  Using HuggingFace dataset loader...")
        dataset_name = 'clean' if 'clean' in args.dataset else 'other'
        dataset_split = 'test' if 'test' in args.dataset else 'validation'
        test_dataset = load_dataset("librispeech_asr", dataset_name, split=dataset_split)
    else:
        print("  Using offline LibriSpeech data...")
        # Load enough samples for both cache and WER evaluation
        max_samples = max(args.num_cache_samples, args.num_wer_samples if args.eval_wer else 0)
        test_dataset = load_librispeech_offline(args.offline_data_dir, args.dataset, max_samples=max_samples)
    
    print(f"  ✓ Loaded {len(test_dataset)} samples")
    
    # Profile KV cache
    print("\n" + "="*60)
    print("Profiling KV Cache")
    print("="*60)
    
    cache_results = measure_kv_cache_size(model, config, device, test_dataset, args.num_cache_samples)
    
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
        
        word_error_rate, refs, hyps = evaluate_wer(model, config, vocab, device, test_dataset, args.num_wer_samples)
        
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
    parser.add_argument('--dataset', type=str, default='test-clean',
                        choices=['test-clean', 'test-other', 'dev-clean', 'dev-other'],
                        help='LibriSpeech test set to use')
    parser.add_argument('--offline_data_dir', type=str, 
                        default='/home/ubuntu/speech_mha2mla/asr_baseline_experiments/data/librispeech/LibriSpeech',
                        help='Path to offline LibriSpeech data directory')
    parser.add_argument('--use_hf_dataset', action='store_true',
                        help='Use HuggingFace dataset instead of offline data')
    
    args = parser.parse_args()
    main(args)
