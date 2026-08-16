#!/usr/bin/env python3
"""
Data Analysis Script for TSPLIB Multi-Run Results
Loads data from 01_revision_multiple_run_TSPLIB_mj_data directory and presents
comprehensive statistics in table format.

This script addresses the reviewer comment:
"MinMax makespan is primary, but include secondary metrics (total distance, 
balance across vehicles, infeasibility rate) and report mean±std over ≥5 runs 
for TSPLIB as well; include runtime breakdown (allocation vs routing)"
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy for statistical tests, fallback if not available
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using manual statistical calculations.")
    # Create a dummy stats module for basic functionality
    class DummyStats:
        class t:
            @staticmethod
            def ppf(q, df):
                # Approximate critical values for 95% CI
                if df >= 30:
                    return 1.96
                elif df >= 10:
                    return 2.23
                else:
                    return 2.78
        
        @staticmethod
        def ttest_rel(a, b):
            """Manual implementation of paired t-test."""
            a = np.array(a)
            b = np.array(b)
            differences = a - b
            n = len(differences)
            
            if n <= 1:
                return 0.0, 1.0
            
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            
            # t-statistic
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            
            # Approximate p-value using normal distribution for large samples
            # or conservative estimates for small samples
            if n >= 30:
                # Use normal approximation
                p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / np.sqrt(2))))
            else:
                # Conservative estimate - if |t| > 2, likely significant
                if abs(t_stat) > 2.0:
                    p_value = 0.05
                elif abs(t_stat) > 1.5:
                    p_value = 0.1
                else:
                    p_value = 0.2
            
            return t_stat, p_value
    
    stats = DummyStats()

def create_table(data, headers, tablefmt='grid'):
    """Create a formatted table without external dependencies."""
    if not data or not headers:
        return ""
    
    # Calculate column widths
    col_widths = [len(str(header)) for header in headers]
    for row in data:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Add padding
    col_widths = [w + 2 for w in col_widths]
    
    # Create format string
    format_str = "|".join([f" {{:<{w-1}}}" for w in col_widths])
    
    # Create separator
    separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"
    
    # Build table
    lines = [separator]
    
    # Header
    header_line = "|" + format_str.format(*headers) + "|"
    lines.append(header_line)
    lines.append(separator)
    
    # Data rows
    for row in data:
        row_line = "|" + format_str.format(*[str(cell) for cell in row]) + "|"
        lines.append(row_line)
    
    lines.append(separator)
    
    return "\n".join(lines)

def load_complete_results(data_dir):
    """Load all results from individual pickle files."""
    import glob
    
    # Find all individual result files
    pattern = os.path.join(data_dir, "*_results.pkl")
    result_files = glob.glob(pattern)
    
    # Filter out complete_results.pkl and summary_statistics.pkl
    result_files = [f for f in result_files if not f.endswith('complete_results.pkl') and not f.endswith('summary_statistics.pkl')]
    
    if not result_files:
        print(f"Error: No individual result files found in {data_dir}")
        return None
    
    results = {}
    print(f"Loading {len(result_files)} individual result files:")
    
    for file_path in sorted(result_files):
        # Extract instance name from filename
        filename = os.path.basename(file_path)
        instance_name = filename.replace('_results.pkl', '')
        
        try:
            with open(file_path, 'rb') as f:
                instance_results = pickle.load(f)
                results[instance_name] = instance_results
                print(f"  ✓ {instance_name}")
        except Exception as e:
            print(f"  ✗ Failed to load {instance_name}: {e}")
    
    return results

def calculate_statistics(values):
    """Calculate mean, standard deviation, and 95% confidence interval for a list of values."""
    values = np.array(values)
    if len(values) == 0:
        return 0.0, 0.0, (0.0, 0.0)
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    # Calculate 95% confidence interval
    if len(values) > 1:
        # Use t-distribution for small samples
        confidence_level = 0.95
        degrees_freedom = len(values) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_error = t_critical * (std_val / np.sqrt(len(values)))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        confidence_interval = (ci_lower, ci_upper)
    else:
        confidence_interval = (mean_val, mean_val)
    
    return mean_val, std_val, confidence_interval

def format_mean_std(mean_val, std_val, decimals=2):
    """Format mean±std string."""
    return f"{mean_val:.{decimals}f}±{std_val:.{decimals}f}"

def format_confidence_interval(ci, decimals=2):
    """Format confidence interval string."""
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"

def calculate_cohens_d(group1, group2, paired=True):
    """Calculate Cohen's d effect size."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if paired:
        # Paired samples Cohen's d
        differences = group1 - group2
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    else:
        # Independent samples Cohen's d
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d

def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def perform_pairwise_comparison(results, case_names, algorithms):
    """Perform comprehensive pairwise comparisons between algorithms."""
    algorithm_pairs = [
        ('HyDR-TR', 'KM-GLKH'),
        ('HyDR-TR', 'GIN-GLKH'), 
        ('KM-GLKH', 'GIN-GLKH')
    ]
    
    comparison_results = []
    
    for alg1, alg2 in algorithm_pairs:
        # Collect paired makespan data across all instances
        alg1_makespans = []
        alg2_makespans = []
        
        for case_name in case_names:
            case_results = results[case_name]
            alg1_makespans.extend(case_results[alg1]['makespan'])
            alg2_makespans.extend(case_results[alg2]['makespan'])
        
        # Perform paired t-test
        if len(alg1_makespans) == len(alg2_makespans) and len(alg1_makespans) > 1:
            t_stat, p_value = stats.ttest_rel(alg1_makespans, alg2_makespans)
            
            # Effect size (Cohen's d for paired samples)
            cohens_d = calculate_cohens_d(alg1_makespans, alg2_makespans, paired=True)
            effect_interpretation = interpret_effect_size(cohens_d)
            
            # Determine significance
            alpha = 0.05
            is_significant = "Yes" if p_value < alpha else "No"
            
            # Better algorithm (lower makespan is better)
            mean_diff = np.mean(np.array(alg1_makespans) - np.array(alg2_makespans))
            better_alg = alg2 if mean_diff > 0 else alg1
            
            # Calculate mean makespans for context
            mean_alg1 = np.mean(alg1_makespans)
            mean_alg2 = np.mean(alg2_makespans)
            improvement_pct = abs(mean_diff) / max(mean_alg1, mean_alg2) * 100
            
            comparison_results.append({
                'comparison': f"{alg1} vs {alg2}",
                'alg1_mean': mean_alg1,
                'alg2_mean': mean_alg2,
                'mean_difference': mean_diff,
                'improvement_pct': improvement_pct,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'cohens_d': cohens_d,
                'effect_size': effect_interpretation,
                'better_algorithm': better_alg
            })
    
    return comparison_results

def perform_instancewise_comparison(results, case_names):
    """Perform t-tests for each instance separately with 30 runs per algorithm."""
    comparison_pairs = [
        ('HyDR-TR', 'KM-GLKH'),
        ('HyDR-TR', 'GIN-GLKH')
    ]
    
    instance_comparisons = []
    
    for case_name in case_names:
        case_results = results[case_name]
        
        for alg1, alg2 in comparison_pairs:
            # Get data for this specific instance (30 runs each)
            alg1_makespans = case_results[alg1]['makespan']
            alg2_makespans = case_results[alg2]['makespan']
            alg1_total_distances = case_results[alg1]['total_distance']
            alg2_total_distances = case_results[alg2]['total_distance']
            
            # Calculate CV values for each run
            alg1_cv_values = []
            alg2_cv_values = []
            
            for costs in case_results[alg1]['individual_costs']:
                costs = np.array(costs)
                if len(costs) > 0 and np.mean(costs) > 0:
                    cv = np.std(costs) / np.mean(costs)
                    alg1_cv_values.append(cv)
            
            for costs in case_results[alg2]['individual_costs']:
                costs = np.array(costs)
                if len(costs) > 0 and np.mean(costs) > 0:
                    cv = np.std(costs) / np.mean(costs)
                    alg2_cv_values.append(cv)
            
            # Perform comparisons for each metric
            metrics_data = {
                'makespan': {
                    'alg1_values': alg1_makespans,
                    'alg2_values': alg2_makespans,
                    'better_direction': 'lower'  # Lower is better
                },
                'total_distance': {
                    'alg1_values': alg1_total_distances,
                    'alg2_values': alg2_total_distances,
                    'better_direction': 'lower'  # Lower is better
                },
                'cv_balance': {
                    'alg1_values': alg1_cv_values,
                    'alg2_values': alg2_cv_values,
                    'better_direction': 'lower'  # Lower CV means better balance
                }
            }
            
            for metric_name, metric_info in metrics_data.items():
                alg1_values = metric_info['alg1_values']
                alg2_values = metric_info['alg2_values']
                
                if len(alg1_values) == 0 or len(alg2_values) == 0:
                    continue
                
                # Calculate means
                mean_alg1 = np.mean(alg1_values)
                mean_alg2 = np.mean(alg2_values)
                
                # Performance gap (%) - using alg1 as baseline
                performance_gap = ((mean_alg2 - mean_alg1) / mean_alg1) * 100 if mean_alg1 != 0 else 0.0
                
                # Perform paired t-test (30 runs vs 30 runs)
                if len(alg1_values) == len(alg2_values) and len(alg1_values) > 1:
                    t_stat, p_value = stats.ttest_rel(alg1_values, alg2_values)
                    
                    # Calculate Cohen's d for paired samples
                    cohens_d = calculate_cohens_d(alg1_values, alg2_values, paired=True)
                    
                    # Determine significance
                    is_significant = "Yes" if p_value < 0.05 else "No"
                    
                    instance_comparisons.append({
                        'instance': case_name,
                        'metric': metric_name,
                        'comparison': f"{alg1} vs {alg2}",
                        'alg1_mean': mean_alg1,
                        'alg2_mean': mean_alg2,
                        'performance_gap': performance_gap,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'cohens_d': cohens_d,
                        'effect_size': interpret_effect_size(cohens_d),
                        'better_direction': metric_info['better_direction']
                    })
                else:
                    # Fallback for insufficient data
                    instance_comparisons.append({
                        'instance': case_name,
                        'metric': metric_name,
                        'comparison': f"{alg1} vs {alg2}",
                        'alg1_mean': mean_alg1,
                        'alg2_mean': mean_alg2,
                        'performance_gap': performance_gap,
                        't_statistic': 0.0,
                        'p_value': 1.0,
                        'is_significant': "No",
                        'cohens_d': 0.0,
                        'effect_size': "negligible",
                        'better_direction': metric_info['better_direction']
                    })
    
    return instance_comparisons

def calculate_balance_metrics(individual_costs_list):
    """Calculate balance metrics across all runs."""
    cv_values = []
    jain_values = []
    
    for costs in individual_costs_list:
        costs = np.array(costs)
        if len(costs) > 0 and np.mean(costs) > 0:
            cv = np.std(costs) / np.mean(costs)
            cv_values.append(cv)
            
            # Jain's Fairness Index
            n = len(costs)
            sum_squares = np.sum(costs ** 2)
            square_sum = (np.sum(costs)) ** 2
            if sum_squares > 0:
                jain_index = square_sum / (n * sum_squares)
                jain_values.append(jain_index)
    
    cv_mean, cv_std, cv_ci = calculate_statistics(cv_values)
    jain_mean, jain_std, jain_ci = calculate_statistics(jain_values)
    
    return cv_mean, cv_std, cv_ci, jain_mean, jain_std, jain_ci

def analyze_infeasibility(individual_costs_list, routes_list):
    """Analyze infeasibility rate (vehicles with no assigned waypoints)."""
    total_runs = len(individual_costs_list)
    infeasible_count = 0
    
    for costs in individual_costs_list:
        # Count vehicles with zero cost (no waypoints assigned)
        zero_cost_vehicles = sum(1 for cost in costs if cost == 0.0)
        if zero_cost_vehicles > 0:
            infeasible_count += 1
    
    infeasibility_rate = (infeasible_count / total_runs) * 100 if total_runs > 0 else 0.0
    return infeasibility_rate

def print_comprehensive_analysis(results):
    """Print comprehensive analysis in table format."""
    
    # Get all case names and sort them
    case_names = sorted(results.keys())
    algorithms = ['HyDR-TR', 'KM-GLKH', 'GIN-GLKH']
    
    print("="*150)
    print("COMPREHENSIVE TSPLIB MULTI-RUN ANALYSIS")
    print("Primary: MinMax Makespan | Secondary: Total Distance, Balance, Infeasibility | Runtime Breakdown")
    print("="*150)
    
    # 1. Main Results Table
    print("\n" + "="*120)
    print("1. MAIN PERFORMANCE METRICS (Mean±Std over 30 runs)")
    print("="*120)
    
    main_data = []
    for case_name in case_names:
        case_results = results[case_name]
        
        for algorithm in algorithms:
            data = case_results[algorithm]
            
            # Primary metric: Makespan
            makespan_mean, makespan_std, makespan_ci = calculate_statistics(data['makespan'])
            
            # Secondary metrics
            total_dist_mean, total_dist_std, total_dist_ci = calculate_statistics(data['total_distance'])
            
            # Balance metrics
            cv_mean, cv_std, cv_ci, jain_mean, jain_std, jain_ci = calculate_balance_metrics(data['individual_costs'])
            
            # Infeasibility rate
            infeasibility_rate = analyze_infeasibility(data['individual_costs'], data['routes'])
            
            main_data.append([
                case_name,
                algorithm,
                format_mean_std(makespan_mean, makespan_std, 2),
                format_mean_std(total_dist_mean, total_dist_std, 2),
                format_mean_std(cv_mean, cv_std, 4),
                format_mean_std(jain_mean, jain_std, 4),
                f"{infeasibility_rate:.1f}%"
            ])
    
    main_headers = [
        'Instance', 'Algorithm', 
        'Makespan\n(Primary)', 'Total Distance', 
        'CV (Balance)', 'Jain Index', 'Infeasible Rate'
    ]
    
    print(create_table(main_data, main_headers))
    
    # 1.5. Confidence Intervals Table
    print("\n" + "="*120)
    print("1.5. 95% CONFIDENCE INTERVALS")
    print("="*120)
    
    ci_data = []
    for case_name in case_names:
        case_results = results[case_name]
        
        for algorithm in algorithms:
            data = case_results[algorithm]
            
            # Primary metric: Makespan
            makespan_mean, makespan_std, makespan_ci = calculate_statistics(data['makespan'])
            
            # Secondary metrics
            total_dist_mean, total_dist_std, total_dist_ci = calculate_statistics(data['total_distance'])
            
            # Balance metrics
            cv_mean, cv_std, cv_ci, jain_mean, jain_std, jain_ci = calculate_balance_metrics(data['individual_costs'])
            
            ci_data.append([
                case_name,
                algorithm,
                format_confidence_interval(makespan_ci, 2),
                format_confidence_interval(total_dist_ci, 2),
                format_confidence_interval(cv_ci, 4),
                format_confidence_interval(jain_ci, 4)
            ])
    
    ci_headers = [
        'Instance', 'Algorithm', 
        'Makespan CI', 'Total Distance CI', 
        'CV CI', 'Jain Index CI'
    ]
    
    print(create_table(ci_data, ci_headers))
    
    # 2. Runtime Breakdown Table
    print("\n" + "="*100)
    print("2. RUNTIME BREAKDOWN (Mean±Std over 30 runs)")
    print("="*100)
    
    runtime_data = []
    for case_name in case_names:
        case_results = results[case_name]
        
        for algorithm in algorithms:
            data = case_results[algorithm]
            
            # Convert clustering times to milliseconds for better readability
            clustering_times_ms = [t * 1000 for t in data['clustering_times']]
            clustering_mean, clustering_std, clustering_ci = calculate_statistics(clustering_times_ms)
            
            routing_mean, routing_std, routing_ci = calculate_statistics(data['routing_times'])
            total_mean, total_std, total_ci = calculate_statistics(data['total_times'])
            
            # Calculate percentage breakdown
            if total_mean > 0:
                clustering_pct = (clustering_mean / 1000) / total_mean * 100
                routing_pct = routing_mean / total_mean * 100
            else:
                clustering_pct = routing_pct = 0.0
            
            runtime_data.append([
                case_name,
                algorithm,
                format_mean_std(clustering_mean, clustering_std, 2) + "ms",
                format_mean_std(routing_mean, routing_std, 3) + "s",
                format_mean_std(total_mean, total_std, 3) + "s",
                f"{clustering_pct:.1f}% / {routing_pct:.1f}%"
            ])
    
    runtime_headers = [
        'Instance', 'Algorithm', 
        'Allocation Time', 'Routing Time', 
        'Total Time', 'Allocation/Routing %'
    ]
    
    print(create_table(runtime_data, runtime_headers))
    
    # 3. Best Performance Summary
    print("\n" + "="*100)
    print("3. BEST PERFORMANCE SUMMARY (Minimum values across 30 runs)")
    print("="*100)
    
    best_data = []
    for case_name in case_names:
        case_results = results[case_name]
        
        row = [case_name]
        makespan_values = {}
        
        for algorithm in algorithms:
            data = case_results[algorithm]
            best_makespan = np.min(data['makespan'])
            makespan_values[algorithm] = best_makespan
            row.append(f"{best_makespan:.2f}")
        
        # Find the best algorithm for this instance
        best_algorithm = min(makespan_values, key=makespan_values.get)
        best_value = makespan_values[best_algorithm]
        row.append(f"{best_algorithm} ({best_value:.2f})")
        
        best_data.append(row)
    
    best_headers = ['Instance', 'HyDR-TR', 'KM-GLKH', 'GIN-GLKH', 'Best (Algorithm)']
    print(create_table(best_data, best_headers))
    
    # 4. Statistical Significance Analysis
    print("\n" + "="*120)
    print("4. ALGORITHM COMPARISON SUMMARY")
    print("="*120)
    
    overall_stats = {}
    for algorithm in algorithms:
        all_makespans = []
        all_total_distances = []
        all_clustering_times = []
        all_routing_times = []
        all_cv_values = []
        all_jain_values = []
        all_infeasibility = []
        
        for case_name in case_names:
            data = results[case_name][algorithm]
            all_makespans.extend(data['makespan'])
            all_total_distances.extend(data['total_distance'])
            all_clustering_times.extend([t * 1000 for t in data['clustering_times']])  # ms
            all_routing_times.extend(data['routing_times'])
            
            # Calculate balance metrics for each run
            for costs in data['individual_costs']:
                costs = np.array(costs)
                if len(costs) > 0 and np.mean(costs) > 0:
                    cv = np.std(costs) / np.mean(costs)
                    all_cv_values.append(cv)
                    
                    # Jain's Fairness Index
                    n = len(costs)
                    sum_squares = np.sum(costs ** 2)
                    square_sum = (np.sum(costs)) ** 2
                    if sum_squares > 0:
                        jain_index = square_sum / (n * sum_squares)
                        all_jain_values.append(jain_index)
            
            # Infeasibility
            infeasibility_rate = analyze_infeasibility(data['individual_costs'], data['routes'])
            all_infeasibility.append(infeasibility_rate)
        
        overall_stats[algorithm] = {
            'makespan': calculate_statistics(all_makespans),
            'total_distance': calculate_statistics(all_total_distances),
            'clustering_time': calculate_statistics(all_clustering_times),
            'routing_time': calculate_statistics(all_routing_times),
            'cv': calculate_statistics(all_cv_values),
            'jain_index': calculate_statistics(all_jain_values),
            'infeasibility': calculate_statistics(all_infeasibility)
        }
    
    summary_data = []
    for algorithm in algorithms:
        stats = overall_stats[algorithm]
        summary_data.append([
            algorithm,
            format_mean_std(stats['makespan'][0], stats['makespan'][1], 2),
            format_mean_std(stats['total_distance'][0], stats['total_distance'][1], 2),
            format_mean_std(stats['cv'][0], stats['cv'][1], 4),
            format_mean_std(stats['jain_index'][0], stats['jain_index'][1], 4),
            format_mean_std(stats['infeasibility'][0], stats['infeasibility'][1], 1) + "%",
            format_mean_std(stats['clustering_time'][0], stats['clustering_time'][1], 2) + "ms",
            format_mean_std(stats['routing_time'][0], stats['routing_time'][1], 3) + "s"
        ])
    
    summary_headers = [
        'Algorithm', 'Makespan\n(Primary)', 'Total Distance', 
        'CV (Balance)', 'Jain Index', 'Infeasible Rate',
        'Allocation Time', 'Routing Time'
    ]
    
    print(create_table(summary_data, summary_headers))
    
    # 4.5. Overall Confidence Intervals
    print("\n" + "="*120)
    print("4.5. OVERALL 95% CONFIDENCE INTERVALS (Across all instances)")
    print("="*120)
    
    ci_summary_data = []
    for algorithm in algorithms:
        stats = overall_stats[algorithm]
        ci_summary_data.append([
            algorithm,
            format_confidence_interval(stats['makespan'][2], 2),
            format_confidence_interval(stats['total_distance'][2], 2),
            format_confidence_interval(stats['cv'][2], 4),
            format_confidence_interval(stats['jain_index'][2], 4),
            format_confidence_interval(stats['infeasibility'][2], 1) + "%",
            format_confidence_interval(stats['clustering_time'][2], 2) + "ms",
            format_confidence_interval(stats['routing_time'][2], 3) + "s"
        ])
    
    ci_summary_headers = [
        'Algorithm', 'Makespan CI', 'Total Distance CI', 
        'CV CI', 'Jain Index CI', 'Infeasible Rate CI',
        'Allocation Time CI', 'Routing Time CI'
    ]
    
    print(create_table(ci_summary_data, ci_summary_headers))
    
    # 5. Win Rate Analysis
    print("\n" + "="*80)
    print("5. WIN RATE ANALYSIS (Best makespan per instance)")
    print("="*80)
    
    wins = {algorithm: 0 for algorithm in algorithms}
    ties = 0
    
    for case_name in case_names:
        case_results = results[case_name]
        makespan_values = {}
        
        for algorithm in algorithms:
            data = case_results[algorithm]
            best_makespan = np.min(data['makespan'])
            makespan_values[algorithm] = best_makespan
        
        min_makespan = min(makespan_values.values())
        winners = [alg for alg, val in makespan_values.items() if abs(val - min_makespan) < 1e-6]
        
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1
    
    win_data = []
    for algorithm in algorithms:
        win_rate = (wins[algorithm] / len(case_names)) * 100
        win_data.append([algorithm, wins[algorithm], f"{win_rate:.1f}%"])
    
    win_data.append(['Ties', ties, f"{(ties / len(case_names)) * 100:.1f}%"])
    
    win_headers = ['Algorithm', 'Wins', 'Win Rate']
    print(create_table(win_data, win_headers))
    
    # 6. Instance-wise Statistical Significance Testing
    print("\n" + "="*120)
    print("6. INSTANCE-WISE STATISTICAL SIGNIFICANCE TESTING (30 runs per instance)")
    print("="*120)
    
    # Perform instance-wise comparisons
    instance_comparisons = perform_instancewise_comparison(results, case_names)
    
    # Group by metric and comparison type for better presentation
    metrics = ['makespan', 'total_distance', 'cv_balance']
    metric_names = {
        'makespan': 'Makespan (Primary)',
        'total_distance': 'Total Distance',
        'cv_balance': 'CV Balance'
    }
    
    for metric in metrics:
        print(f"\n6.{metrics.index(metric)+1}. {metric_names[metric].upper()}")
        print("="*80)
        
        # Filter comparisons for this metric
        metric_comparisons = [comp for comp in instance_comparisons if comp['metric'] == metric]
        
        # Group by comparison type
        hydr_vs_km_data = []
        hydr_vs_gin_data = []
        
        for comp in metric_comparisons:
            # Format the values based on metric type
            if metric == 'makespan' or metric == 'total_distance':
                gap_format = f"{comp['performance_gap']:+.2f}%"
                mean1_format = f"{comp['alg1_mean']:.2f}"
                mean2_format = f"{comp['alg2_mean']:.2f}"
            else:  # cv_balance
                gap_format = f"{comp['performance_gap']:+.2f}%"
                mean1_format = f"{comp['alg1_mean']:.4f}"
                mean2_format = f"{comp['alg2_mean']:.4f}"
            
            row_data = [
                comp['instance'],
                gap_format,
                f"{comp['t_statistic']:.3f}",
                f"{comp['p_value']:.4f}",
                f"{comp['cohens_d']:+.3f}",
                comp['is_significant']
            ]
            
            if 'KM-GLKH' in comp['comparison']:
                hydr_vs_km_data.append(row_data)
            else:
                hydr_vs_gin_data.append(row_data)
        
        comparison_headers = [
            'Instance', 'Performance Gap', 't-statistic', 
            'p-value', 'Cohen\'s d', 'Significant?'
        ]
        
        print(f"\n6.{metrics.index(metric)+1}.1. HyDR-TR vs KM-GLKH ({metric_names[metric]})")
        print("-" * 80)
        print(create_table(hydr_vs_km_data, comparison_headers))
        
        print(f"\n6.{metrics.index(metric)+1}.2. HyDR-TR vs GIN-GLKH ({metric_names[metric]})")
        print("-" * 80)
        print(create_table(hydr_vs_gin_data, comparison_headers))
        
        # Summary statistics for this metric
        print(f"\n6.{metrics.index(metric)+1}.3. SUMMARY STATISTICS ({metric_names[metric]})")
        print("-" * 60)
        
        # Calculate summary for each comparison
        for comparison_name, data_list in [("HyDR-TR vs KM-GLKH", hydr_vs_km_data), 
                                           ("HyDR-TR vs GIN-GLKH", hydr_vs_gin_data)]:
            
            if not data_list:
                continue
                
            # Extract performance gaps (remove % sign and convert to float)
            gaps = [float(row[1].replace('%', '').replace('+', '')) for row in data_list]
            p_values = [float(row[3]) for row in data_list]
            
            significant_count = sum(1 for row in data_list if row[5] == "Yes")
            total_instances = len(data_list)
            
            print(f"\n{comparison_name} ({metric_names[metric]}):")
            print(f"  Mean Performance Gap: {np.mean(gaps):+.2f}%")
            print(f"  Std Performance Gap: ±{np.std(gaps, ddof=1):.2f}%")
            print(f"  Significant instances: {significant_count}/{total_instances} ({significant_count/total_instances*100:.1f}%)")
            print(f"  Mean p-value: {np.mean(p_values):.4f}")
            
            # Count wins (negative gap means HyDR-TR is better for all metrics since lower is better)
            hydr_wins = sum(1 for gap in gaps if gap < 0)
            print(f"  HyDR-TR wins: {hydr_wins}/{total_instances} ({hydr_wins/total_instances*100:.1f}%)")
    
    print("\n" + "="*80)
    print("6.4. INTERPRETATION GUIDE")
    print("="*80)
    print("- Performance Gap: Positive = HyDR-TR is worse, Negative = HyDR-TR is better")
    print("- p < 0.05: Statistically significant difference")
    print("- Cohen's d: negligible(<0.2), small(0.2-0.5), medium(0.5-0.8), large(>0.8)")
    print("- Positive Cohen's d: HyDR-TR has higher values (worse)")
    print("- Negative Cohen's d: HyDR-TR has lower values (better)")
    print("- All metrics: Lower values are better (makespan, distance, CV balance)")
    
    if not SCIPY_AVAILABLE:
        print("\nNote: Using manual statistical calculations. Install scipy for more accurate p-values.")
    
    # 7. Overall Statistical Significance Testing (for reference)
    print("\n" + "="*120)
    print("7. OVERALL STATISTICAL SIGNIFICANCE TESTING (Pooled across all instances)")
    print("="*120)
    
    # Perform comprehensive pairwise comparisons
    comparison_results = perform_pairwise_comparison(results, case_names, algorithms)
    
    # Display results in table format
    significance_data = []
    for comp in comparison_results:
        significance_data.append([
            comp['comparison'],
            f"{comp['alg1_mean']:.2f}",
            f"{comp['alg2_mean']:.2f}",
            f"{comp['mean_difference']:+.2f}",
            f"{comp['improvement_pct']:.1f}%",
            f"{comp['t_statistic']:.3f}",
            f"{comp['p_value']:.4f}",
            comp['is_significant'],
            f"{comp['cohens_d']:+.3f}",
            comp['effect_size'],
            comp['better_algorithm']
        ])
    
    significance_headers = [
        'Comparison', 'Alg1 Mean', 'Alg2 Mean', 'Mean Diff', 
        'Improvement', 't-stat', 'p-value', 'Significant?', 
        'Cohen\'s d', 'Effect Size', 'Better Algorithm'
    ]
    
    print(create_table(significance_data, significance_headers))
    
    print("\nInterpretation:")
    print("- p < 0.05: Statistically significant difference")
    print("- Cohen's d: negligible(<0.2), small(0.2-0.5), medium(0.5-0.8), large(>0.8)")
    print("- Positive Cohen's d: First algorithm has higher values")
    print("- Negative Cohen's d: Second algorithm has higher values")
    print("- Better Algorithm: Algorithm with lower mean makespan")
    
    if not SCIPY_AVAILABLE:
        print("\nNote: Using manual statistical calculations. Install scipy for more accurate p-values.")
    
    # 7.5. Effect Size Summary
    print("\n" + "="*100)
    print("7.5. EFFECT SIZE SUMMARY")
    print("="*100)
    
    effect_summary_data = []
    for comp in comparison_results:
        abs_d = abs(comp['cohens_d'])
        direction = "↑" if comp['cohens_d'] > 0 else "↓"
        winner = comp['comparison'].split(' vs ')[0] if comp['cohens_d'] < 0 else comp['comparison'].split(' vs ')[1]
        
        effect_summary_data.append([
            comp['comparison'],
            f"{abs_d:.3f}",
            comp['effect_size'],
            f"{winner} {direction}",
            f"{comp['improvement_pct']:.1f}%"
        ])
    
    effect_headers = ['Comparison', '|Cohen\'s d|', 'Effect Size', 'Direction', 'Improvement']
    print(create_table(effect_summary_data, effect_headers))
    
    print("\n" + "="*150)
    print("ANALYSIS COMPLETE")
    print(f"Total instances: {len(case_names)}")
    print(f"Runs per instance: 30")
    print(f"Total experiments: {len(case_names) * 30 * len(algorithms)}")
    print(f"Instance-wise comparisons: {len(case_names)} instances × 3 metrics × 2 comparisons = {len(case_names) * 3 * 2} tests")
    print("Metrics analyzed: Makespan (primary), Total Distance, CV Balance")
    print("="*150)

def main():
    """Main function to run the analysis."""
    data_dir = "TSPLIB_30runs"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please run the main experiment script first to generate the data.")
        return
    
    print("Loading experimental results...")
    results = load_complete_results(data_dir)
    
    if results is None:
        print("Failed to load results!")
        return
    
    print(f"Successfully loaded results for {len(results)} TSPLIB instances")
    print("Starting comprehensive analysis...\n")
    
    print_comprehensive_analysis(results)

if __name__ == "__main__":
    main()
