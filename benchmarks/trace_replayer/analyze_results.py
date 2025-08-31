#!/usr/bin/env python3
"""
Analysis script for LMCache trace replayer benchmark results.
Compares performance between different configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
from pathlib import Path
import seaborn as sns
import yaml

def load_results(pattern="*.csv"):
    """Load all CSV result files matching the pattern"""
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    results = {}
    for file in files:
        try:
            df = pd.read_csv(file)
            # Extract scenario name from filename
            scenario = Path(file).stem
            results[scenario] = df
            print(f"Loaded {len(df)} results from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results

def analyze_performance(results):
    """Analyze and compare performance metrics"""
    print("\n=== PERFORMANCE ANALYSIS ===")
    
    summary = {}
    for scenario, df in results.items():
        if len(df) == 0:
            continue
            
        # Calculate metrics
        metrics = {
            'total_requests': len(df),
            'mean_ttft': df['ttft'].mean(),
            'median_ttft': df['ttft'].median(),
            'p95_ttft': df['ttft'].quantile(0.95),
            'p99_ttft': df['ttft'].quantile(0.99),
            'total_input_tokens': df['input_token_len'].sum(),
            'total_output_tokens': df['output_token_len'].sum(),
            'total_duration': df['finish_time'].max() - df['launch_time'].min(),
        }
        
        # Calculate throughput
        metrics['throughput_rps'] = metrics['total_requests'] / metrics['total_duration']
        metrics['input_tokens_per_sec'] = metrics['total_input_tokens'] / metrics['total_duration']
        metrics['output_tokens_per_sec'] = metrics['total_output_tokens'] / metrics['total_duration']
        
        summary[scenario] = metrics
        
        # Print individual scenario stats
        print(f"\n--- {scenario.upper()} ---")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Duration: {metrics['total_duration']:.2f}s")
        print(f"Throughput: {metrics['throughput_rps']:.2f} req/s")
        print(f"Mean TTFT: {metrics['mean_ttft']:.4f}s")
        print(f"Median TTFT: {metrics['median_ttft']:.4f}s")
        print(f"P95 TTFT: {metrics['p95_ttft']:.4f}s")
        print(f"P99 TTFT: {metrics['p99_ttft']:.4f}s")
        print(f"Total Tokens (in/out): {metrics['total_input_tokens']}/{metrics['total_output_tokens']}")
    
    return summary

def compare_scenarios(summary):
    """Compare scenarios and calculate improvements"""
    if len(summary) < 2:
        print("Need at least 2 scenarios for comparison")
        return
    
    print("\n=== SCENARIO COMPARISON ===")
    
    # Find baseline (usually the one with highest TTFT or containing 'baseline')
    baseline_key = None
    for key in summary.keys():
        if 'baseline' in key.lower():
            baseline_key = key
            break
    
    if not baseline_key:
        # Use the scenario with highest mean TTFT as baseline
        baseline_key = max(summary.keys(), key=lambda k: summary[k]['mean_ttft'])
    
    baseline = summary[baseline_key]
    print(f"Using '{baseline_key}' as baseline")
    
    for scenario, metrics in summary.items():
        if scenario == baseline_key:
            continue
            
        print(f"\n--- {scenario.upper()} vs {baseline_key.upper()} ---")
        
        # Calculate improvements
        ttft_improvement = baseline['mean_ttft'] / metrics['mean_ttft']
        throughput_improvement = metrics['throughput_rps'] / baseline['throughput_rps']
        
        print(f"TTFT Improvement: {ttft_improvement:.2f}× faster")
        print(f"Throughput Improvement: {throughput_improvement:.2f}× higher")
        
        # Calculate percentage improvements
        ttft_percent = (baseline['mean_ttft'] - metrics['mean_ttft']) / baseline['mean_ttft'] * 100
        throughput_percent = (metrics['throughput_rps'] - baseline['throughput_rps']) / baseline['throughput_rps'] * 100
        
        print(f"TTFT Reduction: {ttft_percent:.1f}%")
        print(f"Throughput Increase: {throughput_percent:.1f}%")

def plot_results(results, output_dir="plots"):
    """Generate visualization plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # 1. TTFT Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for scenario, df in results.items():
        ax.hist(df['ttft'], bins=30, alpha=0.6, label=scenario, density=True)
    
    ax.set_xlabel('Time to First Token (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('TTFT Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ttft_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Timeline plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, (scenario, df) in enumerate(results.items()):
        # Plot request launches and completions
        launch_times = df['launch_time'] - df['launch_time'].min()
        finish_times = df['finish_time'] - df['launch_time'].min()
        
        # Create a scatter plot for launches
        ax.scatter(launch_times, [i] * len(launch_times), 
                  alpha=0.6, s=20, label=f'{scenario} (launch)')
        
        # Create lines showing request duration
        for j in range(min(50, len(df))):  # Limit to first 50 for clarity
            ax.plot([launch_times.iloc[j], finish_times.iloc[j]], 
                   [i, i], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Scenario')
    ax.set_title('Request Timeline Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance metrics comparison
    if len(results) > 1:
        metrics_data = []
        for scenario, df in results.items():
            metrics_data.append({
                'Scenario': scenario,
                'Mean TTFT': df['ttft'].mean(),
                'P95 TTFT': df['ttft'].quantile(0.95),
                'Throughput': len(df) / (df['finish_time'].max() - df['launch_time'].min())
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # TTFT comparison
        metrics_df.plot(x='Scenario', y='Mean TTFT', kind='bar', ax=axes[0])
        axes[0].set_title('Mean TTFT')
        axes[0].set_ylabel('Seconds')
        
        # P95 TTFT comparison
        metrics_df.plot(x='Scenario', y='P95 TTFT', kind='bar', ax=axes[1])
        axes[1].set_title('P95 TTFT')
        axes[1].set_ylabel('Seconds')
        
        # Throughput comparison
        metrics_df.plot(x='Scenario', y='Throughput', kind='bar', ax=axes[2])
        axes[2].set_title('Throughput')
        axes[2].set_ylabel('Requests/second')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nPlots saved to {output_dir}/ directory")

def main():
    parser = argparse.ArgumentParser(description="Analyze LMCache benchmark results")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Configuration file path")
    parser.add_argument("--pattern", type=str, default="*.csv",
                       help="File pattern to match result files")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Directory for output plots")
    
    args = parser.parse_args()

    # Load configuration from YAML file
    config = {}
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Using default directories.")
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")

    # Directory configuration
    dir_config = config.get('directories', {})
    summary_dir = dir_config.get('summary_dir', 'summaries')
    
    # Load results
    search_pattern = str(Path(summary_dir) / args.pattern)
    results = load_results(search_pattern)
    if not results:
        return
    
    # Analyze performance
    summary = analyze_performance(results)
    
    # Compare scenarios
    compare_scenarios(summary)
    
    # Generate plots if requested
    if args.plot:
        plot_results(results, args.output_dir)
    
    print(f"\nAnalysis complete! Found {len(results)} result files.")

if __name__ == "__main__":
    main()
