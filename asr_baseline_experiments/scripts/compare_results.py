"""
Compare results across all baseline models
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_dir):
    """Load all evaluation results"""
    results = []
    
    results_path = Path(results_dir)
    
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        eval_file = model_dir / 'evaluation_results.json'
        config_file = model_dir / 'config.yaml'
        
        if not eval_file.exists():
            print(f"Warning: No evaluation results found for {model_dir.name}")
            continue
        
        # Load evaluation results
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        # Parse model name
        parts = model_dir.name.split('_')
        if len(parts) >= 2:
            model_type = parts[0]
            attention_type = '_'.join(parts[1:])
        else:
            model_type = model_dir.name
            attention_type = 'unknown'
        
        eval_data['model_type'] = model_type
        eval_data['model_name'] = model_dir.name
        
        results.append(eval_data)
    
    return results


def create_comparison_table(results):
    """Create a comparison table"""
    
    df = pd.DataFrame(results)
    
    # Select key metrics
    columns = [
        'model_name',
        'model_type', 
        'attention_type',
        'avg_cache_size_kb',
        'theoretical_cache_size_kb',
        'avg_inference_time_ms',
        'wer_percent'
    ]
    
    # Filter columns that exist
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    # Sort by model type and attention type
    df = df.sort_values(['model_type', 'attention_type'])
    
    return df


def plot_cache_comparison(results, output_dir):
    """Plot KV cache size comparison"""
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by model type
    x = range(len(df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df['avg_cache_size_kb'], 
           width, label='Actual Cache Size', alpha=0.8)
    ax.bar([i + width/2 for i in x], df['theoretical_cache_size_kb'], 
           width, label='Theoretical Cache Size', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('KV Cache Size (KB)')
    ax.set_title('KV Cache Size Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cache_comparison.png', dpi=300)
    print(f"Saved cache comparison plot to {output_dir / 'cache_comparison.png'}")
    plt.close()


def plot_wer_vs_cache(results, output_dir):
    """Plot WER vs Cache Size"""
    
    df = pd.DataFrame(results)
    
    if 'wer_percent' not in df.columns:
        print("Warning: WER data not available, skipping WER plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by model type
    for model_type in df['model_type'].unique():
        mask = df['model_type'] == model_type
        ax.scatter(df[mask]['avg_cache_size_kb'], 
                  df[mask]['wer_percent'],
                  label=model_type.capitalize(),
                  s=100, alpha=0.7)
        
        # Add labels
        for _, row in df[mask].iterrows():
            ax.annotate(row['attention_type'].upper(),
                       (row['avg_cache_size_kb'], row['wer_percent']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
    
    ax.set_xlabel('KV Cache Size (KB)')
    ax.set_ylabel('Word Error Rate (%)')
    ax.set_title('WER vs KV Cache Size Trade-off')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_vs_cache.png', dpi=300)
    print(f"Saved WER vs cache plot to {output_dir / 'wer_vs_cache.png'}")
    plt.close()


def plot_inference_time(results, output_dir):
    """Plot inference time comparison"""
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df))
    ax.bar(x, df['avg_inference_time_ms'], alpha=0.8, color='steelblue')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Average Inference Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=300)
    print(f"Saved inference time plot to {output_dir / 'inference_time_comparison.png'}")
    plt.close()


def create_summary_report(df, output_dir):
    """Create a summary report"""
    
    report = []
    report.append("=" * 80)
    report.append("ASR BASELINE EXPERIMENTS - SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total models evaluated: {len(df)}")
    report.append(f"Model types: {', '.join(df['model_type'].unique())}")
    report.append(f"Attention types: {', '.join(df['attention_type'].unique())}")
    report.append("")
    
    # Cache size comparison
    report.append("KV CACHE SIZE COMPARISON")
    report.append("-" * 80)
    
    # Find best (smallest) cache
    min_cache_idx = df['avg_cache_size_kb'].idxmin()
    max_cache_idx = df['avg_cache_size_kb'].idxmax()
    
    report.append(f"Smallest cache: {df.loc[min_cache_idx, 'model_name']} "
                 f"({df.loc[min_cache_idx, 'avg_cache_size_kb']:.2f} KB)")
    report.append(f"Largest cache: {df.loc[max_cache_idx, 'model_name']} "
                 f"({df.loc[max_cache_idx, 'avg_cache_size_kb']:.2f} KB)")
    
    # Calculate compression ratio vs MHA baseline
    mha_rows = df[df['attention_type'] == 'mha']
    if not mha_rows.empty:
        mha_cache = mha_rows['avg_cache_size_kb'].mean()
        report.append("")
        report.append("Compression vs MHA baseline:")
        for _, row in df.iterrows():
            if row['attention_type'] != 'mha':
                compression = (1 - row['avg_cache_size_kb'] / mha_cache) * 100
                report.append(f"  {row['model_name']}: {compression:+.1f}% "
                            f"({row['avg_cache_size_kb']:.2f} KB)")
    
    report.append("")
    
    # WER comparison
    if 'wer_percent' in df.columns:
        report.append("WORD ERROR RATE COMPARISON")
        report.append("-" * 80)
        
        min_wer_idx = df['wer_percent'].idxmin()
        max_wer_idx = df['wer_percent'].idxmax()
        
        report.append(f"Best WER: {df.loc[min_wer_idx, 'model_name']} "
                     f"({df.loc[min_wer_idx, 'wer_percent']:.2f}%)")
        report.append(f"Worst WER: {df.loc[max_wer_idx, 'model_name']} "
                     f"({df.loc[max_wer_idx, 'wer_percent']:.2f}%)")
        report.append("")
    
    # Inference time comparison
    report.append("INFERENCE TIME COMPARISON")
    report.append("-" * 80)
    
    min_time_idx = df['avg_inference_time_ms'].idxmin()
    max_time_idx = df['avg_inference_time_ms'].idxmax()
    
    report.append(f"Fastest: {df.loc[min_time_idx, 'model_name']} "
                 f"({df.loc[min_time_idx, 'avg_inference_time_ms']:.2f} ms)")
    report.append(f"Slowest: {df.loc[max_time_idx, 'model_name']} "
                 f"({df.loc[max_time_idx, 'avg_inference_time_ms']:.2f} ms)")
    report.append("")
    
    # Detailed results table
    report.append("DETAILED RESULTS")
    report.append("-" * 80)
    report.append(df.to_string(index=False))
    report.append("")
    report.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report)
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSummary report saved to {output_dir / 'summary_report.txt'}")


def main(args):
    # Load results
    print("Loading evaluation results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("Error: No evaluation results found!")
        print(f"Please run evaluations first using: bash scripts/evaluate_all.sh")
        return
    
    print(f"Found {len(results)} models with evaluation results")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    df.to_csv(output_dir / 'comparison_table.csv', index=False)
    print(f"Saved comparison table to {output_dir / 'comparison_table.csv'}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_cache_comparison(results, output_dir)
    plot_wer_vs_cache(results, output_dir)
    plot_inference_time(results, output_dir)
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(df, output_dir)
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"All results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare baseline results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing model results')
    parser.add_argument('--output_dir', type=str, default='comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    main(args)
