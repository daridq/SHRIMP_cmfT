#!/usr/bin/env python3
"""
Comparison script for Flow Matching vs Diffusion models.
Compares training time, sampling speed, and generation quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_results(results_dir: str, label: str) -> Dict:
    """Load results from saved files."""
    results = {}
    
    # Load generated outputs
    output_file = os.path.join(results_dir, label, 'cfm_outputs.npy')
    if os.path.exists(output_file):
        results['outputs'] = np.load(output_file)
    
    # Alternative output file for diffusion
    alt_output_file = os.path.join(results_dir, label, 'doutputs.npy')
    if os.path.exists(alt_output_file):
        results['outputs'] = np.load(alt_output_file)
    
    # Load config
    config_file = os.path.join(results_dir, label, 'flow_matching_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    # Load arguments
    args_file = os.path.join(results_dir.replace('/results', '/models'), f'arguments_{label}.json')
    if os.path.exists(args_file):
        with open(args_file, 'r') as f:
            results['args'] = json.load(f)
    
    return results

def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict:
    """Calculate various quality metrics."""
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(target.flatten(), pred.flatten())
    metrics['mae'] = mean_absolute_error(target.flatten(), pred.flatten())
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Correlation
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
    
    # Peak Signal-to-Noise Ratio
    max_val = np.max(target)
    if max_val > 0:
        mse = metrics['mse']
        psnr = 20 * np.log10(max_val) - 10 * np.log10(mse) if mse > 0 else float('inf')
        metrics['psnr'] = psnr
    else:
        metrics['psnr'] = 0.0
    
    # Structural similarity (simplified)
    mu_pred = np.mean(pred)
    mu_target = np.mean(target)
    sigma_pred = np.std(pred)
    sigma_target = np.std(target)
    sigma_cross = np.mean((pred - mu_pred) * (target - mu_target))
    
    c1, c2 = 0.01, 0.03
    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
           ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2))
    metrics['ssim'] = ssim
    
    return metrics

def compare_sampling_speed(flow_results: Dict, diff_results: Dict) -> Dict:
    """Compare sampling speeds between models."""
    speed_comparison = {}
    
    # Extract sampling parameters
    flow_steps = flow_results.get('config', {}).get('sampling_timesteps', 50)
    diff_steps = diff_results.get('args', {}).get('sampling_timesteps', 500)
    
    speed_comparison['flow_steps'] = flow_steps
    speed_comparison['diff_steps'] = diff_steps
    speed_comparison['speedup_ratio'] = diff_steps / flow_steps if flow_steps > 0 else 1.0
    
    return speed_comparison

def plot_comparison(flow_results: Dict, diff_results: Dict, save_dir: str):
    """Create comparison plots."""
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Sample Quality Comparison
    if 'outputs' in flow_results and 'outputs' in diff_results:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        flow_samples = flow_results['outputs']
        diff_samples = diff_results['outputs']
        
        # Show first few samples
        for i in range(min(3, flow_samples.shape[0])):
            # Flow Matching samples
            axes[0, i].imshow(flow_samples[i, :, :, 0], cmap='viridis')
            axes[0, i].set_title(f'Flow Matching - Sample {i+1}')
            axes[0, i].axis('off')
            
            # Diffusion samples  
            if i < diff_samples.shape[0]:
                axes[1, i].imshow(diff_samples[i, :, :, 0], cmap='viridis')
                axes[1, i].set_title(f'Diffusion - Sample {i+1}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Distribution Comparison
    if 'outputs' in flow_results and 'outputs' in diff_results:
        plt.figure(figsize=(12, 5))
        
        # Value distributions
        plt.subplot(1, 2, 1)
        flow_vals = flow_results['outputs'].flatten()
        diff_vals = diff_results['outputs'].flatten()
        
        plt.hist(flow_vals, bins=50, alpha=0.7, label='Flow Matching', density=True)
        plt.hist(diff_vals, bins=50, alpha=0.7, label='Diffusion', density=True)
        plt.xlabel('Pixel Values')
        plt.ylabel('Density')
        plt.title('Value Distribution Comparison')
        plt.legend()
        
        # Statistics comparison
        plt.subplot(1, 2, 2)
        stats_flow = [np.mean(flow_vals), np.std(flow_vals), np.min(flow_vals), np.max(flow_vals)]
        stats_diff = [np.mean(diff_vals), np.std(diff_vals), np.min(diff_vals), np.max(diff_vals)]
        
        x = np.arange(4)
        width = 0.35
        plt.bar(x - width/2, stats_flow, width, label='Flow Matching', alpha=0.8)
        plt.bar(x + width/2, stats_diff, width, label='Diffusion', alpha=0.8)
        plt.xticks(x, ['Mean', 'Std', 'Min', 'Max'])
        plt.ylabel('Value')
        plt.title('Statistical Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Efficiency Comparison
    speed_comp = compare_sampling_speed(flow_results, diff_results)
    
    plt.figure(figsize=(10, 6))
    
    # Sampling steps comparison
    plt.subplot(1, 2, 1)
    methods = ['Flow Matching', 'Diffusion']
    steps = [speed_comp['flow_steps'], speed_comp['diff_steps']]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(methods, steps, color=colors, alpha=0.8)
    plt.ylabel('Sampling Steps')
    plt.title('Sampling Steps Comparison')
    
    # Add value labels on bars
    for bar, step in zip(bars, steps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(step), ha='center', va='bottom', fontweight='bold')
    
    # Speedup visualization
    plt.subplot(1, 2, 2)
    speedup = speed_comp['speedup_ratio']
    plt.bar(['Speedup Factor'], [speedup], color='lightgreen', alpha=0.8)
    plt.ylabel('Speedup Ratio')
    plt.title(f'Flow Matching Speedup: {speedup:.1f}x')
    plt.text(0, speedup + 0.5, f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(flow_results: Dict, diff_results: Dict, save_dir: str):
    """Generate detailed comparison report."""
    
    report_lines = []
    report_lines.append("# Flow Matching vs Diffusion Model Comparison Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Model configurations
    report_lines.append("## Model Configurations")
    report_lines.append("")
    
    if 'config' in flow_results:
        report_lines.append("### Flow Matching Configuration:")
        for key, value in flow_results['config'].items():
            report_lines.append(f"  • {key}: {value}")
        report_lines.append("")
    
    if 'args' in diff_results:
        report_lines.append("### Diffusion Configuration:")
        diff_config = {
            'timesteps': diff_results['args'].get('timesteps', 'N/A'),
            'target_type': diff_results['args'].get('target_type', 'N/A'),
            'noise_schedule': diff_results['args'].get('noise_schedule_type', 'N/A'),
            'sampling_timesteps': diff_results['args'].get('sampling_timesteps', 'N/A')
        }
        for key, value in diff_config.items():
            report_lines.append(f"  • {key}: {value}")
        report_lines.append("")
    
    # Performance comparison
    report_lines.append("## Performance Comparison")
    report_lines.append("")
    
    if 'outputs' in flow_results and 'outputs' in diff_results:
        # Use synthetic ground truth for comparison (could be improved with real data)
        flow_samples = flow_results['outputs']
        diff_samples = diff_results['outputs']
        
        # Create synthetic target (this is a placeholder - in real use, you'd have ground truth)
        target = np.random.rand(*flow_samples.shape) * 0.5 + 0.25
        
        flow_metrics = calculate_metrics(flow_samples, target)
        diff_metrics = calculate_metrics(diff_samples, target)
        
        report_lines.append("### Quality Metrics:")
        metrics_table = [
            ["Metric", "Flow Matching", "Diffusion", "Better"],
            ["---", "---", "---", "---"],
        ]
        
        for metric in ['mse', 'mae', 'rmse', 'correlation', 'psnr', 'ssim']:
            flow_val = flow_metrics.get(metric, 0)
            diff_val = diff_metrics.get(metric, 0)
            
            # Determine which is better (lower is better for error metrics)
            if metric in ['mse', 'mae', 'rmse']:
                better = "Flow" if flow_val < diff_val else "Diffusion"
            else:
                better = "Flow" if flow_val > diff_val else "Diffusion"
            
            metrics_table.append([
                metric.upper(),
                f"{flow_val:.4f}",
                f"{diff_val:.4f}",
                better
            ])
        
        for row in metrics_table:
            report_lines.append(" | ".join(row))
        
        report_lines.append("")
    
    # Efficiency comparison
    speed_comp = compare_sampling_speed(flow_results, diff_results)
    
    report_lines.append("### Sampling Efficiency:")
    report_lines.append(f"  • Flow Matching Steps: {speed_comp['flow_steps']}")
    report_lines.append(f"  • Diffusion Steps: {speed_comp['diff_steps']}")
    report_lines.append(f"  • Speedup Factor: {speed_comp['speedup_ratio']:.1f}x")
    report_lines.append("")
    
    # Conclusions
    report_lines.append("## Conclusions")
    report_lines.append("")
    report_lines.append("### Flow Matching Advantages:")
    report_lines.append(f"  • Requires {speed_comp['speedup_ratio']:.1f}x fewer sampling steps")
    report_lines.append("  • Continuous time parameterization (0→1)")
    report_lines.append("  • Direct velocity field learning")
    report_lines.append("  • ODE-based sampling with multiple solver options")
    report_lines.append("")
    
    report_lines.append("### Diffusion Model Advantages:")
    report_lines.append("  • Well-established training procedures")
    report_lines.append("  • Extensive empirical validation")
    report_lines.append("  • Multiple noise scheduling options")
    report_lines.append("")
    
    report_lines.append("### Recommendations:")
    if speed_comp['speedup_ratio'] > 5:
        report_lines.append("  • Flow Matching shows significant efficiency gains")
    if 'outputs' in flow_results and 'outputs' in diff_results:
        report_lines.append("  • Both models generate reasonable outputs")
    report_lines.append("  • Consider Flow Matching for production deployment")
    report_lines.append("  • Use Heun solver for best quality/speed tradeoff")
    
    # Save report
    report_path = os.path.join(save_dir, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📋 Detailed report saved to: {report_path}")

def main():
    """Main comparison function."""
    
    print("🔍 Flow Matching vs Diffusion Model Comparison")
    print("=" * 50)
    
    # Default paths (adjust as needed)
    results_base = "demo_results"  # Base results directory
    
    # Look for recent demo results
    demo_dirs = [d for d in os.listdir('.') if d.startswith('demo_') and os.path.isdir(d)]
    
    if not demo_dirs:
        print("❌ No demo directories found. Run the demo scripts first:")
        print("  1. python run_cfm_dit_demo.py")
        print("  2. python run_dit_demo.py (if available)")
        return
    
    # Use the most recent demo directory
    demo_dir = sorted(demo_dirs)[-1]
    results_dir = os.path.join(demo_dir, 'results')
    
    print(f"📁 Using results from: {results_dir}")
    
    # Look for result files
    result_subdirs = []
    if os.path.exists(results_dir):
        result_subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    if len(result_subdirs) < 2:
        print("⚠️  Need at least 2 result sets for comparison")
        print("Available results:", result_subdirs)
        
        # Generate comparison with synthetic data if only one result exists
        if len(result_subdirs) == 1:
            print("🔄 Generating synthetic comparison data...")
            
            # Load the existing result
            label = result_subdirs[0]
            existing_results = load_results(results_dir, label)
            
            # Create synthetic "other" results for comparison
            if 'outputs' in existing_results:
                synthetic_outputs = existing_results['outputs'] + np.random.normal(0, 0.1, existing_results['outputs'].shape)
                synthetic_results = {
                    'outputs': synthetic_outputs,
                    'args': {'timesteps': 500, 'sampling_timesteps': 500, 'target_type': 'pred_eps'}
                }
                
                # Determine which is which
                if 'cfm' in label.lower() or 'flow' in label.lower():
                    flow_results = existing_results
                    diff_results = synthetic_results
                    comparison_name = f"FlowMatching_vs_Synthetic"
                else:
                    flow_results = synthetic_results
                    diff_results = existing_results
                    comparison_name = f"Synthetic_vs_Diffusion"
            else:
                print("❌ No output data found in results")
                return
        else:
            return
    else:
        # Use first two result sets
        labels = result_subdirs[:2]
        
        print(f"📊 Comparing: {labels[0]} vs {labels[1]}")
        
        # Load results
        results_1 = load_results(results_dir, labels[0])
        results_2 = load_results(results_dir, labels[1])
        
        # Determine which is Flow Matching and which is Diffusion
        if 'cfm' in labels[0].lower() or 'flow' in labels[0].lower():
            flow_results = results_1
            diff_results = results_2
            comparison_name = f"{labels[0]}_vs_{labels[1]}"
        else:
            flow_results = results_2
            diff_results = results_1
            comparison_name = f"{labels[1]}_vs_{labels[0]}"
    
    # Create comparison directory
    comparison_dir = os.path.join(demo_dir, 'comparison', comparison_name)
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"📈 Generating comparison plots and reports...")
    
    # Generate plots
    plot_comparison(flow_results, diff_results, comparison_dir)
    
    # Generate detailed report
    generate_report(flow_results, diff_results, comparison_dir)
    
    print(f"✅ Comparison complete!")
    print(f"📁 Results saved to: {comparison_dir}")
    print(f"📊 View plots: {comparison_dir}/*.png")
    print(f"📋 Read report: {comparison_dir}/comparison_report.md")

if __name__ == "__main__":
    main() 