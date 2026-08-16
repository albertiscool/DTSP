import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from tqdm import tqdm
import time
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# Import necessary functions from existing modules
from utils.ortools_tsp import or_solve
from utils.heading_opt import aa_solve
from utils.glkh_dtsp import glkh_dtsp_solve_originfixed
from utils.dubins.dubins_length_only import dubins_path_length


def calculate_aa_cost(instance, turning_radius):
    """Calculate cost using AA solver for single vehicle DTSP"""
    try:
        # Solve TSP sequence using ortools
        route = or_solve(instance)
        
        # Optimize headings using alternating algorithm
        route_headings = aa_solve(instance, route)
        
        # Fix first waypoint heading angle as 0 (consistent with training)
        route_headings[0] = 0
        
        # Make dubins waypoints
        dubins_waypoints = np.hstack((np.array(instance[route[:-1]]), 
                                      np.array(route_headings).reshape(-1, 1)))
        
        # Calculate total dubins path length
        total_length = 0
        curvature = 1.0 / turning_radius  # Convert turning radius to curvature
        
        for i in range(len(dubins_waypoints)):
            start_x = dubins_waypoints[i, 0]
            start_y = dubins_waypoints[i, 1]
            start_yaw = dubins_waypoints[i, 2]

            if i < len(dubins_waypoints) - 1:
                end_x = dubins_waypoints[i+1, 0]
                end_y = dubins_waypoints[i+1, 1]
                end_yaw = dubins_waypoints[i+1, 2]
            else:
                # Return to start for closed tour
                end_x = dubins_waypoints[0, 0]
                end_y = dubins_waypoints[0, 1]
                end_yaw = dubins_waypoints[0, 2]

            length = dubins_path_length(start_x, start_y, start_yaw, 
                                        end_x, end_y, end_yaw, curvature)
            total_length += length
            
        return total_length, True
        
    except Exception as e:
        print(f"AA solver failed: {e}")
        return None, False


def calculate_glkh_cost(instance, turning_radius):
    """Calculate cost using GLKH solver for single vehicle DTSP"""
    try:
        curvature = 1.0 / turning_radius  # Convert turning radius to curvature
        
        # Solve DTSP using GLKH with fixed origin heading
        route, route_headings = glkh_dtsp_solve_originfixed(instance, 8, curvature)
        
        # Make dubins waypoints
        dubins_waypoints = np.hstack((np.array(instance[route]), 
                                      np.array(route_headings).reshape(-1, 1)))
        
        # Calculate total dubins path length
        total_length = 0
        for i in range(len(dubins_waypoints)):
            start_x = dubins_waypoints[i, 0]
            start_y = dubins_waypoints[i, 1]
            start_yaw = dubins_waypoints[i, 2]

            if i < len(dubins_waypoints) - 1:
                end_x = dubins_waypoints[i+1, 0]
                end_y = dubins_waypoints[i+1, 1]
                end_yaw = dubins_waypoints[i+1, 2]
            else:
                # Return to start for closed tour
                end_x = dubins_waypoints[0, 0]
                end_y = dubins_waypoints[0, 1]
                end_yaw = dubins_waypoints[0, 2]

            length = dubins_path_length(start_x, start_y, start_yaw, 
                                        end_x, end_y, end_yaw, curvature)
            total_length += length
            
        return total_length, True
        
    except Exception as e:
        print(f"GLKH solver failed: {e}")
        return None, False


def process_single_experiment(args):
    """Process a single experiment - designed for multiprocessing"""
    n_nodes, turning_radius, instance_idx, seed = args
    
    try:
        # Generate random instance
        np.random.seed(seed)
        instance = np.random.rand(n_nodes, 2)
        
        # Calculate AA cost
        aa_start_time = time.time()
        aa_cost, aa_success = calculate_aa_cost(instance, turning_radius)
        aa_time = time.time() - aa_start_time
        
        # Calculate GLKH cost
        glkh_start_time = time.time()
        glkh_cost, glkh_success = calculate_glkh_cost(instance, turning_radius)
        glkh_time = time.time() - glkh_start_time
        
        # Return results
        if aa_success and glkh_success and aa_cost is not None and glkh_cost is not None:
            return {
                'n_nodes': n_nodes,
                'turning_radius': turning_radius,
                'instance_idx': instance_idx,
                'aa_cost': aa_cost,
                'glkh_cost': glkh_cost,
                'aa_time': aa_time,
                'glkh_time': glkh_time,
                'relative_error': abs(aa_cost - glkh_cost) / glkh_cost if glkh_cost > 0 else 0,
                'cost_ratio': aa_cost / glkh_cost if glkh_cost > 0 else 1,
                'seed': seed,
                'success': True
            }
        else:
            return {
                'n_nodes': n_nodes,
                'turning_radius': turning_radius,
                'instance_idx': instance_idx,
                'seed': seed,
                'success': False,
                'aa_success': aa_success,
                'glkh_success': glkh_success
            }
    except Exception as e:
        return {
            'n_nodes': n_nodes,
            'turning_radius': turning_radius,
            'instance_idx': instance_idx,
            'seed': seed,
            'success': False,
            'error': str(e)
        }


def analyze_correlation(n_workers=None):
    """Main function to analyze AA-GLKH correlation with multiprocessing"""
    
    # Create output directory
    output_dir = "AA_GLKH_correlation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Use at most 8 cores to avoid overwhelming the system
    
    print(f"Using {n_workers} parallel workers")
    
    # Experimental parameters - use specific turning radii
    node_counts = range(10, 50, 5)  # 10, 15, 20, 25, 30, 35, 40, 45
    turning_radii = [0.05, 0.1, 0.2]  # Desired turning radius values: R = 0.5, 1, 2
    instances_per_case = 64
    
    # Prepare all experiment parameters
    experiment_args = []
    experiment_count = 0
    
    for n_nodes in node_counts:
        for turning_radius in turning_radii:
            for instance_idx in range(instances_per_case):
                experiment_count += 1
                seed = experiment_count  # For reproducibility
                experiment_args.append((n_nodes, turning_radius, instance_idx, seed))
    
    total_experiments = len(experiment_args)
    
    print("Starting AA-GLKH correlation analysis with multiprocessing...")
    print(f"Node counts: {list(node_counts)}")
    print(f"Turning radii: {turning_radii}")
    print(f"Instances per case: {instances_per_case}")
    print(f"Total experiments: {total_experiments}")
    print(f"Parallel workers: {n_workers}")
    
    # Run experiments in parallel with progress bar
    print("\nRunning experiments in parallel...")
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_single_experiment, experiment_args),
            total=total_experiments,
            desc="Processing experiments",
            ncols=100
        ))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nParallel processing completed in {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time/total_experiments:.3f} seconds")
    
    # Process results
    successful_results = []
    failed_results = []
    
    for result in results:
        if result['success']:
            successful_results.append(result)
        else:
            failed_results.append(result)
    
    print(f"Successful experiments: {len(successful_results)} / {total_experiments}")
    print(f"Failed experiments: {len(failed_results)}")
    
    if failed_results:
        print("\nFailure analysis:")
        failure_types = {}
        for fail in failed_results:
            if 'error' in fail:
                error_type = fail['error']
            else:
                aa_success = fail.get('aa_success', False)
                glkh_success = fail.get('glkh_success', False)
                if not aa_success and not glkh_success:
                    error_type = "Both solvers failed"
                elif not aa_success:
                    error_type = "AA solver failed"
                else:
                    error_type = "GLKH solver failed"
            
            failure_types[error_type] = failure_types.get(error_type, 0) + 1
        
        for error_type, count in failure_types.items():
            print(f"  {error_type}: {count}")
    
    # Convert successful results to DataFrame
    df = pd.DataFrame(successful_results)
    
    if df.empty:
        print("No successful experiments. Check solver implementations.")
        return None
    
    print(f"\nSuccessful experiments: {len(df)} / {total_experiments}")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    print(f"Sequential time estimate: {df['aa_time'].sum() + df['glkh_time'].sum():.2f} seconds")
    print(f"Actual parallel time: {total_time:.2f} seconds")
    print(f"Speedup: {(df['aa_time'].sum() + df['glkh_time'].sum()) / total_time:.1f}x")
    
    # Overall correlation analysis
    print("\n" + "="*60)
    print("OVERALL COST CORRELATION ANALYSIS")
    print("="*60)
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(df['aa_cost'], df['glkh_cost'])
    print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    
    # Spearman correlation (main metric for conclusion)
    spearman_r, spearman_p = spearmanr(df['aa_cost'], df['glkh_cost'])
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}")
    
    # Error metrics
    mae = mean_absolute_error(df['glkh_cost'], df['aa_cost'])
    mse = mean_squared_error(df['glkh_cost'], df['aa_cost'])
    rmse = np.sqrt(mse)
    mean_rel_error = df['relative_error'].mean()
    median_rel_error = df['relative_error'].median()
    
    print(f"\nError Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Relative Error: {mean_rel_error:.4f} ({mean_rel_error*100:.2f}%)")
    print(f"Median Relative Error: {median_rel_error:.4f} ({median_rel_error*100:.2f}%)")
    
    # Cost ratio statistics
    print(f"\nCost Ratio Statistics (AA/GLKH):")
    print(f"Mean ratio: {df['cost_ratio'].mean():.4f}")
    print(f"Median ratio: {df['cost_ratio'].median():.4f}")
    print(f"Std ratio: {df['cost_ratio'].std():.4f}")
    print(f"Min ratio: {df['cost_ratio'].min():.4f}")
    print(f"Max ratio: {df['cost_ratio'].max():.4f}")
    
    # Timing comparison
    print(f"\nTiming Comparison:")
    print(f"AA mean time: {df['aa_time'].mean():.4f}s")
    print(f"GLKH mean time: {df['glkh_time'].mean():.4f}s")
    print(f"Speed ratio (GLKH/AA): {df['glkh_time'].mean() / df['aa_time'].mean():.1f}x")
    
    # Correlation by node count and turning radius
    print("\n" + "="*60)
    print("CORRELATION BY PARAMETERS")
    print("="*60)
    
    print("\nSpearman correlation by node count:")
    for n_nodes in node_counts:
        subset = df[df['n_nodes'] == n_nodes]
        if len(subset) >= 3:
            r, p = spearmanr(subset['aa_cost'], subset['glkh_cost'])
            print(f"  {n_nodes} nodes: ρ = {r:.3f}, p = {p:.3f}, n = {len(subset)}")
    
    print("\nSpearman correlation by turning radius:")
    for turning_radius in turning_radii:
        subset = df[df['turning_radius'] == turning_radius]
        if len(subset) >= 3:
            r, p = spearmanr(subset['aa_cost'], subset['glkh_cost'])
            print(f"  R = {turning_radius}: ρ = {r:.3f}, p = {p:.3f}, n = {len(subset)}")
    
    # Create visualizations
    create_correlation_visualizations(df, node_counts, turning_radii, output_dir)
    
    # Save results
    csv_path = os.path.join(output_dir, 'aa_glkh_correlation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to '{csv_path}'")
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'correlation_summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("AA-GLKH Cost Correlation Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experimental Setup:\n")
        f.write(f"- Node counts: {list(node_counts)}\n")
        f.write(f"- Turning radii: {turning_radii}\n")
        f.write(f"- Instances per case: {instances_per_case}\n")
        f.write(f"- Total experiments: {len(node_counts) * len(turning_radii) * instances_per_case}\n")
        f.write(f"- Successful experiments: {len(df)}\n")
        f.write(f"- Processing time: {total_time:.2f} seconds\n")
        f.write(f"- Parallel workers: {n_workers}\n\n")
        
        f.write("Overall Cost Correlation Results:\n")
        f.write(f"- Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}\n")
        f.write(f"- Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}\n\n")
        
        f.write("Error Metrics:\n")
        f.write(f"- Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"- Root Mean Square Error (RMSE): {rmse:.4f}\n")
        f.write(f"- Mean Relative Error: {mean_rel_error:.4f} ({mean_rel_error*100:.2f}%)\n")
        f.write(f"- Median Relative Error: {median_rel_error:.4f} ({median_rel_error*100:.2f}%)\n\n")
        
        f.write("Cost Ratio Statistics (AA/GLKH):\n")
        f.write(f"- Mean ratio: {df['cost_ratio'].mean():.4f}\n")
        f.write(f"- Median ratio: {df['cost_ratio'].median():.4f}\n")
        f.write(f"- Std ratio: {df['cost_ratio'].std():.4f}\n")
        f.write(f"- Min ratio: {df['cost_ratio'].min():.4f}\n")
        f.write(f"- Max ratio: {df['cost_ratio'].max():.4f}\n\n")
        
        f.write("Timing Comparison:\n")
        f.write(f"- AA mean time: {df['aa_time'].mean():.4f}s\n")
        f.write(f"- GLKH mean time: {df['glkh_time'].mean():.4f}s\n")
        f.write(f"- Speed ratio (GLKH/AA): {df['glkh_time'].mean() / df['aa_time'].mean():.1f}x\n\n")
        
        f.write("Spearman Correlation by Node Count:\n")
        for n_nodes in node_counts:
            subset = df[df['n_nodes'] == n_nodes]
            if len(subset) >= 3:
                r, p = spearmanr(subset['aa_cost'], subset['glkh_cost'])
                f.write(f"- {n_nodes} nodes: ρ = {r:.3f}, p = {p:.3f}, n = {len(subset)}\n")
        
        f.write("\nSpearman Correlation by Turning Radius:\n")
        for turning_radius in turning_radii:
            subset = df[df['turning_radius'] == turning_radius]
            if len(subset) >= 3:
                r, p = spearmanr(subset['aa_cost'], subset['glkh_cost'])
                f.write(f"- R = {turning_radius}: ρ = {r:.3f}, p = {p:.3f}, n = {len(subset)}\n")
    
    print(f"Summary report saved to '{summary_path}'")
    
    return df


def create_correlation_visualizations(df, node_counts, turning_radii, output_dir):
    """Create correlation visualization plots (first row focus)"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(18, 6))  # Wide figure for 3 plots in a row
    
    # 1. Overall cost scatter plot (no perfect correlation line)
    ax1 = plt.subplot(1, 3, 1)
    plt.scatter(df['glkh_cost'], df['aa_cost'], alpha=0.6, s=30, color='steelblue')
    
    # Add regression line only
    z = np.polyfit(df['glkh_cost'], df['aa_cost'], 1)
    p = np.poly1d(z)
    x_sorted = df['glkh_cost'].sort_values()
    plt.plot(x_sorted, p(x_sorted), "red", alpha=0.8, linewidth=2, label='Linear fit')
    
    # Use Spearman correlation for the title
    spearman_r, _ = spearmanr(df['aa_cost'], df['glkh_cost'])
    plt.title(f'Overall Makespan Correlation', fontsize=14)
    plt.xlabel('GLKH Makespan', fontsize=12)
    plt.ylabel('AA Makespan', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Cost correlation by node count
    ax2 = plt.subplot(1, 3, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(node_counts)))
    for i, n_nodes in enumerate(node_counts):
        subset = df[df['n_nodes'] == n_nodes]
        if len(subset) > 0:
            plt.scatter(subset['glkh_cost'], subset['aa_cost'], 
                       c=[colors[i]], label=f'{n_nodes} nodes', alpha=0.7, s=30)
    
    plt.title('Makespan Correlation by Node Count', fontsize=14)
    plt.xlabel('GLKH Makespan', fontsize=12)
    plt.ylabel('AA Makespan', fontsize=12)
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 3. Cost correlation by turning radius
    ax3 = plt.subplot(1, 3, 3)
    colors = ['red', 'blue', 'green']
    for i, turning_radius in enumerate(turning_radii):
        subset = df[df['turning_radius'] == turning_radius]
        if len(subset) > 0:
            plt.scatter(subset['glkh_cost'], subset['aa_cost'], 
                       c=colors[i], label=f'R = {turning_radius}', alpha=0.7, s=30)
    
    plt.title('Makespan Correlation by Turning Radius', fontsize=14)
    plt.xlabel('GLKH Makespan', fontsize=12)
    plt.ylabel('AA Makespan', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot with 600 dpi
    plot_path = os.path.join(output_dir, 'aa_glkh_correlation_analysis.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.show()
    
    print(f"Correlation visualizations saved to '{plot_path}' at 600 dpi")


if __name__ == "__main__":
    # Run the correlation analysis with multiprocessing
    import argparse
    
    parser = argparse.ArgumentParser(description='AA-GLKH Cost Correlation Analysis')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of parallel workers (default: auto-detect)')
    args = parser.parse_args()
    
    results_df = analyze_correlation(n_workers=args.workers)
    
    if results_df is not None and not results_df.empty:
        print("\n" + "="*60)
        print("CORRELATION CONCLUSION")
        print("="*60)
        
        pearson_r, _ = pearsonr(results_df['aa_cost'], results_df['glkh_cost'])
        spearman_r, _ = spearmanr(results_df['aa_cost'], results_df['glkh_cost'])
        mean_rel_error = results_df['relative_error'].mean()
        
        print(f"Based on {len(results_df)} single-vehicle experiments:")
        print(f"• Pearson correlation: {pearson_r:.3f}")
        print(f"• Spearman correlation: {spearman_r:.3f}")
        print(f"• Mean relative error: {mean_rel_error:.1%}")
        print(f"• Speed advantage: {results_df['glkh_time'].mean() / results_df['aa_time'].mean():.1f}x faster")
        
        # Conclusion based on Spearman correlation
        if spearman_r > 0.8:
            print("\n✅ STRONG CORRELATION: AA and GLKH show excellent agreement.")
            print("   Using AA for training and GLKH for testing is well justified.")
        elif spearman_r > 0.6:
            print("\n⚠️  MODERATE CORRELATION: AA and GLKH show reasonable agreement.")
            print("   Using AA for training is acceptable but monitor performance differences.")
        else:
            print("\n❌ WEAK CORRELATION: AA and GLKH show poor agreement.")
            print("   Consider alternative approaches for training-testing consistency.")
