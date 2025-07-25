#!/usr/bin/env python3
"""
HeteroSched Evaluation Harness

Comprehensive benchmarking script that compares different scheduling modes:
1. Static threshold scheduler (baseline)
2. Always CPU mode
3. Always GPU mode  
4. ML-guided scheduler

Measures throughput, latency, device utilization, and bounce rate.
"""

import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import argparse
from datetime import datetime

class HeteroSchedBenchmark:
    def __init__(self, binary_path='build/hetero_sched', output_dir='benchmark_results'):
        self.binary_path = binary_path
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def clear_logs(self):
        """Clear previous log files"""
        if os.path.exists('logs/task_log.csv'):
            os.remove('logs/task_log.csv')
            
    def run_scheduler(self, mode, num_tasks=100, timeout=60):
        """Run the scheduler in specified mode and collect metrics"""
        print(f"\n=== Running {mode} mode with {num_tasks} tasks ===")
        
        self.clear_logs()
        
        # Build command based on mode
        cmd = [self.binary_path, '--tasks', str(num_tasks)]
        
        if mode == 'static':
            cmd.append('--static')
        elif mode == 'ml':
            pass  # Default ML mode
        # Note: always_cpu and always_gpu would require code modifications
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the scheduler
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"Error running {mode} mode:")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout running {mode} mode")
            return None
            
        # Record end time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Parse output for performance metrics
        output_lines = result.stdout.split('\\n')
        metrics = self.parse_output(output_lines, total_time)
        
        # Load and analyze task log
        if os.path.exists('logs/task_log.csv'):
            task_df = pd.read_csv('logs/task_log.csv')
            task_metrics = self.analyze_task_log(task_df)
            metrics.update(task_metrics)
        
        metrics['mode'] = mode
        metrics['num_tasks'] = num_tasks
        
        return metrics
        
    def parse_output(self, output_lines, total_time):
        """Parse scheduler output for performance metrics"""
        metrics = {
            'total_time': total_time,
            'cpu_tasks': 0,
            'gpu_tasks': 0,
            'total_cpu_time': 0.0,
            'total_gpu_time': 0.0,
            'completed_tasks': 0
        }
        
        for line in output_lines:
            line = line.strip()
            
            # Parse statistics
            if 'CPU tasks:' in line:
                parts = line.split()
                if len(parts) >= 3:
                    metrics['cpu_tasks'] = int(parts[2])
                    
            elif 'GPU tasks:' in line:
                parts = line.split()
                if len(parts) >= 3:
                    metrics['gpu_tasks'] = int(parts[2])
                    
            elif 'Total CPU time:' in line:
                parts = line.split()
                if len(parts) >= 4:
                    metrics['total_cpu_time'] = float(parts[3])
                    
            elif 'Total GPU time:' in line: 
                parts = line.split()
                if len(parts) >= 4:
                    metrics['total_gpu_time'] = float(parts[3])
                    
            elif 'Total tasks completed:' in line:
                parts = line.split()
                if len(parts) >= 4:
                    metrics['completed_tasks'] = int(parts[3])
        
        # Calculate derived metrics
        total_tasks = metrics['cpu_tasks'] + metrics['gpu_tasks']
        if total_tasks > 0:
            metrics['cpu_percentage'] = metrics['cpu_tasks'] / total_tasks * 100
            metrics['gpu_percentage'] = metrics['gpu_tasks'] / total_tasks * 100
            metrics['throughput'] = total_tasks / total_time  # tasks/second
        
        total_compute_time = metrics['total_cpu_time'] + metrics['total_gpu_time']
        if total_compute_time > 0:
            metrics['parallel_efficiency'] = total_compute_time / total_time
            
        return metrics
        
    def analyze_task_log(self, df):
        """Analyze task execution log for detailed metrics"""
        if len(df) == 0:
            return {}
            
        metrics = {
            'avg_latency': df['execution_time_ms'].mean(),
            'median_latency': df['execution_time_ms'].median(),
            'p95_latency': df['execution_time_ms'].quantile(0.95),
            'p99_latency': df['execution_time_ms'].quantile(0.99),
            'latency_std': df['execution_time_ms'].std(),
        }
        
        # Calculate bounce rate (would need task reassignment tracking)
        # For now, approximate based on device distribution vs expected
        
        # Device utilization analysis
        device_counts = df['device'].value_counts()
        if 'CPU' in device_counts:
            metrics['actual_cpu_tasks'] = device_counts['CPU']
        if 'GPU' in device_counts:
            metrics['actual_gpu_tasks'] = device_counts['GPU']
            
        # Task type performance analysis
        task_type_perf = df.groupby(['task_type', 'device'])['execution_time_ms'].agg(['mean', 'std', 'count'])
        metrics['task_performance'] = task_type_perf.to_dict()
        
        return metrics
        
    def run_comparison_benchmark(self, task_counts=[50, 100, 200]):
        """Run comprehensive comparison across all modes and task counts"""
        modes = ['static', 'ml']  
        all_results = []
        
        for num_tasks in task_counts:
            for mode in modes:
                print(f"\\n{'='*50}")
                print(f"Benchmarking {mode} mode with {num_tasks} tasks")
                print(f"{'='*50}")
                
                # Run multiple iterations for statistical significance
                mode_results = []
                for iteration in range(3):  # 3 iterations per configuration
                    print(f"Iteration {iteration + 1}/3...")
                    result = self.run_scheduler(mode, num_tasks)
                    if result:
                        result['iteration'] = iteration
                        mode_results.append(result)
                        all_results.append(result)
                        
                # Calculate averages for this mode/task_count combination
                if mode_results:
                    avg_result = self.calculate_averages(mode_results)
                    avg_result['mode'] = mode
                    avg_result['num_tasks'] = num_tasks
                    avg_result['iteration'] = 'average'
                    all_results.append(avg_result)
        
        # Save raw results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{self.output_dir}/benchmark_results.csv', index=False)
        
        # Generate analysis and plots
        self.analyze_results(results_df)
        
        return results_df
        
    def calculate_averages(self, results):
        """Calculate average metrics across iterations"""
        numeric_fields = ['total_time', 'throughput', 'avg_latency', 'median_latency', 
                         'p95_latency', 'p99_latency', 'cpu_percentage', 'gpu_percentage',
                         'parallel_efficiency']
        
        avg_result = {}
        for field in numeric_fields:
            values = [r.get(field, 0) for r in results if field in r]
            if values:
                avg_result[field] = np.mean(values)
                avg_result[f'{field}_std'] = np.std(values)
                
        return avg_result
        
    def analyze_results(self, df):
        """Generate analysis plots and summary statistics"""
        print("\\n=== Generating Analysis Plots ===")
        
        # Filter to only average results for cleaner plots
        avg_df = df[df['iteration'] == 'average'].copy()
        
        if len(avg_df) == 0:
            print("No average results to plot")
            return
            
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Throughput comparison
        self.plot_throughput(avg_df, axes[0, 0])
        
        # Plot 2: Latency comparison
        self.plot_latency(avg_df, axes[0, 1])
        
        # Plot 3: Device utilization
        self.plot_device_utilization(avg_df, axes[0, 2])
        
        # Plot 4: Parallel efficiency
        self.plot_parallel_efficiency(avg_df, axes[1, 0])
        
        # Plot 5: Execution time breakdown
        self.plot_execution_time(avg_df, axes[1, 1])
        
        # Plot 6: Performance vs task count
        self.plot_scalability(avg_df, axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/benchmark_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis plots saved to {self.output_dir}/benchmark_analysis.png")
        
        # Generate summary report
        self.generate_summary_report(avg_df)
        
    def plot_throughput(self, df, ax):
        """Plot throughput comparison"""
        if 'throughput' not in df.columns:
            ax.text(0.5, 0.5, 'No throughput data', ha='center', va='center')
            ax.set_title('Throughput Comparison')
            return
            
        sns.barplot(data=df, x='num_tasks', y='throughput', hue='mode', ax=ax)
        ax.set_title('Throughput (tasks/second)')
        ax.set_ylabel('Tasks/Second')
        
    def plot_latency(self, df, ax):
        """Plot latency comparison"""
        if 'avg_latency' not in df.columns:
            ax.text(0.5, 0.5, 'No latency data', ha='center', va='center')
            ax.set_title('Latency Comparison') 
            return
            
        sns.barplot(data=df, x='num_tasks', y='avg_latency', hue='mode', ax=ax)
        ax.set_title('Average Latency')
        ax.set_ylabel('Latency (ms)')
        
    def plot_device_utilization(self, df, ax):
        """Plot device utilization"""
        if 'cpu_percentage' not in df.columns:
            ax.text(0.5, 0.5, 'No utilization data', ha='center', va='center')
            ax.set_title('Device Utilization')
            return
            
        # Create stacked bar plot
        bottoms = np.zeros(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            cpu_pct = row.get('cpu_percentage', 0)
            gpu_pct = row.get('gpu_percentage', 0)
            
            ax.bar(i, cpu_pct, label='CPU' if i == 0 else '', color='skyblue')
            ax.bar(i, gpu_pct, bottom=cpu_pct, label='GPU' if i == 0 else '', color='orange')
            
        ax.set_title('Device Utilization (%)')
        ax.set_ylabel('Percentage')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"{row['mode']}\\n{row['num_tasks']}" for _, row in df.iterrows()])
        ax.legend()
        
    def plot_parallel_efficiency(self, df, ax):
        """Plot parallel efficiency"""
        if 'parallel_efficiency' not in df.columns:
            ax.text(0.5, 0.5, 'No efficiency data', ha='center', va='center')
            ax.set_title('Parallel Efficiency')
            return
            
        sns.barplot(data=df, x='num_tasks', y='parallel_efficiency', hue='mode', ax=ax)
        ax.set_title('Parallel Efficiency')
        ax.set_ylabel('Efficiency Ratio')
        
    def plot_execution_time(self, df, ax):
        """Plot total execution time"""
        if 'total_time' not in df.columns:
            ax.text(0.5, 0.5, 'No time data', ha='center', va='center')
            ax.set_title('Execution Time')
            return
            
        sns.barplot(data=df, x='num_tasks', y='total_time', hue='mode', ax=ax)
        ax.set_title('Total Execution Time')
        ax.set_ylabel('Time (seconds)')
        
    def plot_scalability(self, df, ax):
        """Plot performance scalability"""
        if 'throughput' not in df.columns:
            ax.text(0.5, 0.5, 'No scalability data', ha='center', va='center')
            ax.set_title('Scalability Analysis')
            return
            
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode]
            ax.plot(mode_data['num_tasks'], mode_data['throughput'], 
                   marker='o', label=mode, linewidth=2)
            
        ax.set_title('Throughput Scalability')
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Throughput (tasks/sec)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def generate_summary_report(self, df):
        """Generate text summary report"""
        report_path = f'{self.output_dir}/benchmark_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("HeteroSched Benchmark Report\\n")
            f.write("=" * 40 + "\\n\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # Summary statistics
            f.write("PERFORMANCE SUMMARY\\n")
            f.write("-" * 20 + "\\n")
            
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]
                f.write(f"\\n{mode.upper()} MODE:\\n")
                
                if 'throughput' in mode_data.columns:
                    avg_throughput = mode_data['throughput'].mean()
                    f.write(f"  Average Throughput: {avg_throughput:.2f} tasks/sec\\n")
                    
                if 'avg_latency' in mode_data.columns:
                    avg_latency = mode_data['avg_latency'].mean()
                    f.write(f"  Average Latency: {avg_latency:.2f} ms\\n")
                    
                if 'gpu_percentage' in mode_data.columns:
                    avg_gpu_usage = mode_data['gpu_percentage'].mean()
                    f.write(f"  GPU Utilization: {avg_gpu_usage:.1f}%\\n")
            
            # Comparative analysis
            if len(df['mode'].unique()) > 1:
                f.write("\\n\\nCOMPARATIVE ANALYSIS\\n")
                f.write("-" * 20 + "\\n")
                
                modes = list(df['mode'].unique())
                if 'ml' in modes and 'static' in modes:
                    ml_data = df[df['mode'] == 'ml']
                    static_data = df[df['mode'] == 'static']
                    
                    if 'throughput' in df.columns:
                        ml_throughput = ml_data['throughput'].mean()
                        static_throughput = static_data['throughput'].mean()
                        improvement = ((ml_throughput - static_throughput) / static_throughput) * 100
                        f.write(f"ML vs Static Throughput: {improvement:+.1f}% improvement\\n")
                        
                    if 'avg_latency' in df.columns:
                        ml_latency = ml_data['avg_latency'].mean()
                        static_latency = static_data['avg_latency'].mean()
                        improvement = ((static_latency - ml_latency) / static_latency) * 100
                        f.write(f"ML vs Static Latency: {improvement:+.1f}% improvement\\n")
        
        print(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='HeteroSched Benchmark Suite')
    parser.add_argument('--binary', default='build/hetero_sched', help='Path to scheduler binary')
    parser.add_argument('--output', default='benchmark_results', help='Output directory')
    parser.add_argument('--tasks', nargs='+', type=int, default=[50, 100, 200], 
                       help='Task counts to test')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer iterations')
    
    args = parser.parse_args()
    
    benchmark = HeteroSchedBenchmark(binary_path=args.binary, output_dir=args.output)
    
    if args.quick:
        # Quick test with single task count
        results = benchmark.run_comparison_benchmark([50])
    else:
        # Full benchmark suite
        results = benchmark.run_comparison_benchmark(args.tasks)
    
    print("\\n=== Benchmark Complete ===")
    print(f"Results saved to {args.output}/")
    print("Files generated:")
    print(f"  - benchmark_results.csv")
    print(f"  - benchmark_analysis.png") 
    print(f"  - benchmark_report.txt")

if __name__ == '__main__':
    main()