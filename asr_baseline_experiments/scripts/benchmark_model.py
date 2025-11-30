#!/usr/bin/env python3
"""
Benchmark model inference speed and memory usage

Usage:
    python scripts/benchmark_model.py \
        --model_name facebook/wav2vec2-conformer-rel-pos-large-960h-ft \
        --output_file results/benchmark_mha.json
"""

import argparse
import torch
import json
import time
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor
from tqdm import tqdm


def benchmark_model(model, processor, dataset, device='cuda', num_samples=100):
    """
    Benchmark model speed and memory
    
    Returns:
        dict with timing and memory metrics
    """
    model.eval()
    model.to(device)
    
    # Warmup
    print("Warming up...")
    sample = dataset[0]
    inputs = processor(sample['audio']['array'], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        for _ in range(5):
            _ = model(inputs.input_values.to(device))
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"\nBenchmarking on {num_samples} samples...")
    
    inference_times = []
    memory_allocated = []
    memory_reserved = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            
            # Process audio
            inputs = processor(
                sample['audio']['array'],
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_values = inputs.input_values.to(device)
            
            # Measure memory before
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            
            # Time inference
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                logits = model(input_values).logits
                end_event.record()
                
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000.0  # to seconds
            else:
                start_time = time.time()
                logits = model(input_values).logits
                inference_time = time.time() - start_time
            
            # Measure memory after
            if device == 'cuda':
                mem_after = torch.cuda.memory_allocated()
                mem_reserved = torch.cuda.memory_reserved()
                
                memory_allocated.append((mem_after - mem_before) / 1e9)  # GB
                memory_reserved.append(mem_reserved / 1e9)  # GB
            
            inference_times.append(inference_time)
    
    # Calculate statistics
    results = {
        'num_samples': num_samples,
        'inference_time': {
            'mean': float(np.mean(inference_times)),
            'std': float(np.std(inference_times)),
            'median': float(np.median(inference_times)),
            'min': float(np.min(inference_times)),
            'max': float(np.max(inference_times)),
            'p95': float(np.percentile(inference_times, 95)),
            'p99': float(np.percentile(inference_times, 99)),
        },
        'throughput': {
            'samples_per_second': float(1.0 / np.mean(inference_times)),
        }
    }
    
    if device == 'cuda' and memory_allocated:
        results['memory'] = {
            'allocated_gb': {
                'mean': float(np.mean(memory_allocated)),
                'max': float(np.max(memory_allocated)),
            },
            'reserved_gb': {
                'mean': float(np.mean(memory_reserved)),
                'max': float(np.max(memory_reserved)),
            },
            'peak_allocated_gb': float(torch.cuda.max_memory_allocated() / 1e9),
            'peak_reserved_gb': float(torch.cuda.max_memory_reserved() / 1e9),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model name')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Local model directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file for results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to benchmark')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    if args.model_name is None and args.model_dir is None:
        raise ValueError("Must specify either --model_name or --model_dir")
    
    print("="*70)
    print("Model Benchmarking")
    print("="*70)
    
    if args.model_name:
        print(f"\nModel: {args.model_name}")
        model_id = args.model_name
    else:
        print(f"\nModel directory: {args.model_dir}")
        model_id = args.model_dir
    
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")
    
    # Load model and processor
    print("\nLoading model...")
    if args.model_name:
        model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_name)
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    else:
        # For local models, we'd need to handle this differently
        # For now, just use the base model
        raise NotImplementedError("Benchmarking local models not yet supported")
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    print("\nLoading test dataset...")
    dataset = load_dataset("librispeech_asr", "clean", split="test")
    dataset = dataset.select(range(args.num_samples))
    
    # Benchmark
    results = benchmark_model(model, processor, dataset, args.device, args.num_samples)
    
    # Print results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"\nInference Time (per sample):")
    print(f"  Mean:   {results['inference_time']['mean']*1000:.2f} ms")
    print(f"  Median: {results['inference_time']['median']*1000:.2f} ms")
    print(f"  Std:    {results['inference_time']['std']*1000:.2f} ms")
    print(f"  P95:    {results['inference_time']['p95']*1000:.2f} ms")
    print(f"  P99:    {results['inference_time']['p99']*1000:.2f} ms")
    
    print(f"\nThroughput:")
    print(f"  {results['throughput']['samples_per_second']:.2f} samples/sec")
    
    if 'memory' in results:
        print(f"\nMemory Usage:")
        print(f"  Mean allocated: {results['memory']['allocated_gb']['mean']:.2f} GB")
        print(f"  Max allocated:  {results['memory']['allocated_gb']['max']:.2f} GB")
        print(f"  Peak allocated: {results['memory']['peak_allocated_gb']:.2f} GB")
        print(f"  Peak reserved:  {results['memory']['peak_reserved_gb']:.2f} GB")
    
    print("="*70)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'model': model_id,
        'device': args.device,
        'num_samples': args.num_samples,
        'results': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}\n")


if __name__ == '__main__':
    main()

