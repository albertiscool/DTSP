import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle
import os
import time
import random
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans

import torch

from torch_geometric.data import Data
from torch_geometric.data import Batch

# Import policies and helper functions
from policy_HyDR_TR import Policy_HyDR_TR, action_sample as action_sample_tf
from policy_GIN_GLKH import Policy_GIN_GLKH, Policy_GIN_GLKH_ORIGIN, action_sample as action_sample_gin

def get_route_and_cost_gtsp_with_costs(assignments:np.ndarray, data:np.ndarray, n_agent:int, curvature:float=10.0, heading_num=8, fix_origin_heading=False):
    """Modified version of get_route_and_cost_gtsp that returns routes and individual costs.
    
    Args:
        assignments: cluster assignments for each waypoint
        data: coordinate data with depot and waypoints
        n_agent: number of agents/vehicles
        curvature: curvature parameter for Dubins path calculation
        heading_num: number of heading discretizations
        fix_origin_heading: whether to fix origin heading
        
    Returns:
        tuple: (routes_dict, individual_costs)
            routes_dict: dictionary of {vehicle_id: route_array}
            individual_costs: list of individual route costs
    """
    from utils.glkh_dtsp import glkh_dtsp_solve, glkh_dtsp_solve_originfixed
    from utils.dubins.dubins_path import plan_dubins_path
    
    depot = data[0, :].tolist()
    sub_tours = [[] for _ in range(n_agent)]
    for tour in sub_tours:
        tour.append(depot)

    for n, m in zip(assignments.tolist(), data.tolist()[1:]):
        sub_tours[n].append(m)

    rl_lengths = []
    rl_routes = {}

    for a in range(n_agent):
        instance = np.array(sub_tours[a])
        agent_route = []

        if instance.shape[0] == 1:
            # Empty route - vehicle has no waypoints
            rl_lengths.append(0.0)
            continue
        
        if fix_origin_heading:
            sub_route, sub_route_headings = glkh_dtsp_solve_originfixed(instance, heading_num, curvature)
        else:
            sub_route, sub_route_headings = glkh_dtsp_solve(instance, heading_num, curvature)

        # make dubins waypoints
        dubins_waypoints = np.hstack((np.array(instance[sub_route]), 
                                        np.array(sub_route_headings).reshape(-1, 1)))

        # calculate dubins path length
        sub_tour_length = 0
        for i in range(len(sub_route)):
            start_x = dubins_waypoints[i, 0]
            start_y = dubins_waypoints[i, 1]
            start_yaw = dubins_waypoints[i, 2]

            if i <= len(sub_route)-2:
                end_x = dubins_waypoints[i+1, 0]
                end_y = dubins_waypoints[i+1, 1]
                end_yaw = dubins_waypoints[i+1, 2]
            else:
                end_x = dubins_waypoints[0, 0]
                end_y = dubins_waypoints[0, 1]
                end_yaw = dubins_waypoints[0, 2]

            path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(start_x, start_y, start_yaw, 
                                                                        end_x, end_y, end_yaw, curvature)
            path = np.hstack((np.array(path_x).reshape(-1, 1), 
                              np.array(path_y).reshape(-1, 1), 
                              np.array(path_yaw).reshape(-1, 1))) # [n_nodes, 3]
            
            agent_route.append(path)
            sub_tour_length += sum(lengths)

        rl_lengths.append(sub_tour_length)
        agent_route = np.vstack(agent_route)
        rl_routes[a] = agent_route

    return rl_routes, rl_lengths

def set_random_seeds(seed):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_metrics_from_costs(individual_costs):
    """Calculate comprehensive metrics from individual route costs.
    
    Args:
        individual_costs: list or array of individual route costs
        
    Returns:
        dict: metrics including makespan, total_distance, CV, jain_index, individual_costs
    """
    individual_costs = np.array(individual_costs)
    
    # Calculate metrics
    makespan = np.max(individual_costs) if len(individual_costs) > 0 else 0.0
    total_distance = np.sum(individual_costs)
    
    # Calculate Coefficient of Variation (CV)
    if len(individual_costs) > 0 and np.mean(individual_costs) > 0:
        cv = np.std(individual_costs) / np.mean(individual_costs)
    else:
        cv = 0.0
    
    # Calculate Jain's Fairness Index
    if len(individual_costs) > 0:
        sum_squares = np.sum(individual_costs)**2
        if sum_squares > 0:
            jain_index = sum_squares / (len(individual_costs) * np.sum(individual_costs**2))
        else:
            jain_index = 1.0
    else:
        jain_index = 1.0
    
    return {
        'makespan': makespan,
        'total_distance': total_distance,
        'cv': cv,
        'jain_index': jain_index,
        'individual_costs': individual_costs.tolist(),
        'num_vehicles': len(individual_costs)
    }

# kroA100 specific parameters
case_name = "kroA100"
base_turning_radius = 200  # From the original database

# Experimental parameters - 30 runs like in the mj version
base_seeds = [42, 123, 456, 789, 999, 2024, 31415, 271828, 161803, 123456,
              654321, 111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999,
              13579, 24680, 11223, 44556, 77889, 10101, 20202, 30303, 40404, 50505]  # 30 unique seeds
n_runs = 30
n_agent = 3
heading_discretizations = [4, 8, 16]  # Different heading discretization levels
turning_radius_factors = [0.5, 1.0, 2.0]  # Half, original, double

# Create data directory if it doesn't exist
data_dir = "sensitivity"
os.makedirs(data_dir, exist_ok=True)

# Store all results
all_results = {}

def load_models():
    """Load both HyDR-TR (TF) and GIN models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load HyDR-TR model
    hydr_model_path = "./saved_model/a3_n10to50_c10_ortools_seed2_exp250307_HyDR_TR.pth"
    hydr_config_path = hydr_model_path.replace(".pth", ".yaml")
    with open(hydr_config_path, 'r') as file:
        hydr_config = yaml.safe_load(file)
    
    hydr_policy = Policy_HyDR_TR(in_chnl=hydr_config["in_chnl"], hid_chnl=hydr_config["hid_chnl"], 
                       n_agent=hydr_config["n_agent"], key_size_embd=hydr_config["key_size_embd"], 
                       key_size_policy=hydr_config["key_size_policy"], val_size=hydr_config["val_size"], 
                       clipping=hydr_config["clipping"], dev=device, 
                       n_heads=hydr_config["n_heads"],ff_dim=hydr_config["ff_dim"], norm_eps=hydr_config["norm_eps"],
                       batch_first=hydr_config["batch_first"], n_enc_layer=hydr_config["n_enc_layer"])
    
    hydr_policy.load_state_dict(torch.load(hydr_model_path, map_location=torch.device(device)))
    hydr_policy.eval()
    
    # Load GIN model
    gin_model_path = "./saved_model/a3_n10to30_c10_glkh_seed1_exp250227_GIN_GLKH.pth"
    gin_config_path = gin_model_path.replace(".pth", ".yaml")
    with open(gin_config_path, 'r') as file:
        gin_config = yaml.safe_load(file)
    
    try:
        gin_policy = Policy_GIN_GLKH(in_chnl=gin_config["in_chnl"], hid_chnl=gin_config["hid_chnl"], 
                               n_agent=gin_config["n_agent"], key_size_embd=gin_config["key_size_embd"],
                               key_size_policy=gin_config["key_size_policy"], val_size=gin_config["val_size"], 
                               clipping=gin_config["clipping"], dev=device)
        gin_policy.load_state_dict(torch.load(gin_model_path, map_location=torch.device(device)))
        gin_policy.eval()
    except:  # alternative import
        gin_policy = Policy_GIN_GLKH_ORIGIN(in_chnl=gin_config["in_chnl"], hid_chnl=gin_config["hid_chnl"], 
                               n_agent=gin_config["n_agent"], key_size_embd=gin_config["key_size_embd"],
                               key_size_policy=gin_config["key_size_policy"], val_size=gin_config["val_size"], 
                               clipping=gin_config["clipping"], dev=device)
        gin_policy.load_state_dict(torch.load(gin_model_path, map_location=torch.device(device)))
        gin_policy.eval()
    
    return hydr_policy, gin_policy, device

def run_hydr_algorithm(policy, test_data, device, seed):
    """Run HyDR-TR algorithm and return assignments"""
    set_random_seeds(seed)
    
    with torch.no_grad():
        data = torch.FloatTensor(test_data).unsqueeze(0).to(device)
        pi = policy(data)
        action, _ = action_sample_tf(pi)
        assignments = action.cpu().numpy().squeeze()
    
    return assignments

def run_gin_algorithm(policy, test_data, device, seed):
    """Run GIN algorithm and return assignments"""
    set_random_seeds(seed)
    
    with torch.no_grad():
        # Prepare data for PyTorch Geometric GIN model
        data = torch.FloatTensor(test_data).unsqueeze(0).to(device)
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # Fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)
        
        # Get assignments from the policy
        pi = policy(batch_graph, n_nodes=data.shape[1], n_batch=1)
        action, _ = action_sample_gin(pi)
        assignments = action.cpu().numpy().squeeze()
    
    return assignments

def run_kmeans_algorithm(test_data, seed):
    """Run K-means algorithm and return assignments"""
    set_random_seeds(seed)
    
    depot = test_data[0]
    waypoints = test_data[1:]
    
    kmeans = KMeans(n_clusters=n_agent, random_state=seed, n_init="auto").fit(waypoints)
    assignments = kmeans.labels_
    
    return assignments

# Load models once
print("Loading models...")
hydr_policy, gin_policy, device = load_models()
print(f"Models loaded successfully on {device}")

print(f"\n{'='*60}")
print(f"Processing {case_name} - kroA100 Sensitivity Analysis")
print(f"{'='*60}")

# Initialize results storage for this case
case_results = {
    'HyDR-TR': {},
    'GIN-GLKH': {},
    'KM-GLKH': {}
}

# Load and parse TSPLIB data
file_path = f'TSPLIB/{case_name}.tsp'
with open(file_path, 'r') as file:
    raw_data = file.readlines()

# Parse coordinates from the file
coordinates = []
parsing_coords = False
for line in raw_data:
    if line.strip() == 'NODE_COORD_SECTION':
        parsing_coords = True
        continue
    if parsing_coords:
        if line.strip() == 'EOF':
            break
        parts = line.split()
        if len(parts) == 3:
            _, x, y = parts
            coordinates.append((float(x), float(y)))

test_data = np.array(coordinates)
original_data = np.array(coordinates)

# Normalize coordinates for model input
x_min, x_max = test_data[:, 0].min(), test_data[:, 0].max()
y_min, y_max = test_data[:, 1].min(), test_data[:, 1].max()
scale = max(x_max - x_min, y_max - y_min)
test_data[:, 0] = ((test_data[:, 0] - x_min) / scale) + 0.01
test_data[:, 1] = ((test_data[:, 1] - y_min) / scale) + 0.01

# Loop over turning radius factors
for tr_factor in turning_radius_factors:
    minimum_turning_radius = base_turning_radius * tr_factor
    curvature = 1 / minimum_turning_radius
    
    print(f"\nTurning radius factor: {tr_factor} (radius: {minimum_turning_radius})")
    
    # Loop over heading discretizations
    for heading_num in heading_discretizations:
        print(f"  Heading discretization: {heading_num}")
        
        # Initialize storage for this configuration
        config_key = f"tr_{tr_factor}_hd_{heading_num}"
        
        for algorithm in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
            case_results[algorithm][config_key] = {
                'makespans': [],
                'total_distances': [],
                'cvs': [],
                'jain_indices': [],
                'individual_costs': [],
                'clustering_times': [],
                'routing_times': [],
                'total_times': [],
                'routes': [],
                'seeds': []
            }
        
        # Run experiments with different seeds
        for run_idx in range(n_runs):
            seed = base_seeds[run_idx]
            print(f"    Run {run_idx + 1}/{n_runs} (seed: {seed})")
            
            # ===== HyDR-TR Algorithm =====
            start_time = time.perf_counter()
            
            clustering_start = time.perf_counter()
            hydr_assignments = run_hydr_algorithm(hydr_policy, test_data, device, seed)
            clustering_time = time.perf_counter() - clustering_start
            
            routing_start = time.perf_counter()
            hydr_route, hydr_individual_costs = get_route_and_cost_gtsp_with_costs(hydr_assignments, original_data, n_agent=n_agent, 
                                                           curvature=curvature, heading_num=heading_num, 
                                                           fix_origin_heading=True)
            routing_time = time.perf_counter() - routing_start
            total_time = time.perf_counter() - start_time
            
            # Calculate comprehensive metrics
            hydr_metrics = calculate_metrics_from_costs(hydr_individual_costs)
            
            # Store HyDR-TR results
            case_results['HyDR-TR'][config_key]['makespans'].append(hydr_metrics['makespan'])
            case_results['HyDR-TR'][config_key]['total_distances'].append(hydr_metrics['total_distance'])
            case_results['HyDR-TR'][config_key]['cvs'].append(hydr_metrics['cv'])
            case_results['HyDR-TR'][config_key]['jain_indices'].append(hydr_metrics['jain_index'])
            case_results['HyDR-TR'][config_key]['individual_costs'].append(hydr_metrics['individual_costs'])
            case_results['HyDR-TR'][config_key]['clustering_times'].append(clustering_time)
            case_results['HyDR-TR'][config_key]['routing_times'].append(routing_time)
            case_results['HyDR-TR'][config_key]['total_times'].append(total_time)
            case_results['HyDR-TR'][config_key]['routes'].append(hydr_route)
            case_results['HyDR-TR'][config_key]['seeds'].append(seed)
            
            print(f"      HyDR-TR - Makespan: {hydr_metrics['makespan']:.2f}, Total: {hydr_metrics['total_distance']:.2f}, CV: {hydr_metrics['cv']:.4f}, Jain: {hydr_metrics['jain_index']:.4f}, Time: {total_time:.4f}s")
            
            # ===== GIN Algorithm =====
            start_time = time.perf_counter()
            
            clustering_start = time.perf_counter()
            gin_assignments = run_gin_algorithm(gin_policy, test_data, device, seed)
            clustering_time = time.perf_counter() - clustering_start
            
            routing_start = time.perf_counter()
            gin_route, gin_individual_costs = get_route_and_cost_gtsp_with_costs(gin_assignments, original_data, n_agent=n_agent, 
                                                         curvature=curvature, heading_num=heading_num, 
                                                         fix_origin_heading=True)
            routing_time = time.perf_counter() - routing_start
            total_time = time.perf_counter() - start_time
            
            # Calculate comprehensive metrics
            gin_metrics = calculate_metrics_from_costs(gin_individual_costs)
            
            # Store GIN results
            case_results['GIN-GLKH'][config_key]['makespans'].append(gin_metrics['makespan'])
            case_results['GIN-GLKH'][config_key]['total_distances'].append(gin_metrics['total_distance'])
            case_results['GIN-GLKH'][config_key]['cvs'].append(gin_metrics['cv'])
            case_results['GIN-GLKH'][config_key]['jain_indices'].append(gin_metrics['jain_index'])
            case_results['GIN-GLKH'][config_key]['individual_costs'].append(gin_metrics['individual_costs'])
            case_results['GIN-GLKH'][config_key]['clustering_times'].append(clustering_time)
            case_results['GIN-GLKH'][config_key]['routing_times'].append(routing_time)
            case_results['GIN-GLKH'][config_key]['total_times'].append(total_time)
            case_results['GIN-GLKH'][config_key]['routes'].append(gin_route)
            case_results['GIN-GLKH'][config_key]['seeds'].append(seed)
            
            print(f"      GIN-GLKH - Makespan: {gin_metrics['makespan']:.2f}, Total: {gin_metrics['total_distance']:.2f}, CV: {gin_metrics['cv']:.4f}, Jain: {gin_metrics['jain_index']:.4f}, Time: {total_time:.4f}s")
            
            # ===== K-means Algorithm =====
            start_time = time.perf_counter()
            
            clustering_start = time.perf_counter()
            kmeans_assignments = run_kmeans_algorithm(test_data, seed)
            clustering_time = time.perf_counter() - clustering_start
            
            routing_start = time.perf_counter()
            kmeans_route, kmeans_individual_costs = get_route_and_cost_gtsp_with_costs(kmeans_assignments, original_data, n_agent=n_agent, 
                                                               curvature=curvature, heading_num=heading_num, 
                                                               fix_origin_heading=True)
            routing_time = time.perf_counter() - routing_start
            total_time = time.perf_counter() - start_time
            
            # Calculate comprehensive metrics
            kmeans_metrics = calculate_metrics_from_costs(kmeans_individual_costs)
            
            # Store K-means results
            case_results['KM-GLKH'][config_key]['makespans'].append(kmeans_metrics['makespan'])
            case_results['KM-GLKH'][config_key]['total_distances'].append(kmeans_metrics['total_distance'])
            case_results['KM-GLKH'][config_key]['cvs'].append(kmeans_metrics['cv'])
            case_results['KM-GLKH'][config_key]['jain_indices'].append(kmeans_metrics['jain_index'])
            case_results['KM-GLKH'][config_key]['individual_costs'].append(kmeans_metrics['individual_costs'])
            case_results['KM-GLKH'][config_key]['clustering_times'].append(clustering_time)
            case_results['KM-GLKH'][config_key]['routing_times'].append(routing_time)
            case_results['KM-GLKH'][config_key]['total_times'].append(total_time)
            case_results['KM-GLKH'][config_key]['routes'].append(kmeans_route)
            case_results['KM-GLKH'][config_key]['seeds'].append(seed)
            
            print(f"      KM-GLKH - Makespan: {kmeans_metrics['makespan']:.2f}, Total: {kmeans_metrics['total_distance']:.2f}, CV: {kmeans_metrics['cv']:.4f}, Jain: {kmeans_metrics['jain_index']:.4f}, Time: {total_time:.4f}s")

# Store results for this case
all_results[case_name] = case_results

# Save individual case results
case_pickle_path = os.path.join(data_dir, f"{case_name}_sensitivity_results.pkl")
with open(case_pickle_path, 'wb') as f:
    pickle.dump(case_results, f)

# Save complete results
complete_results_path = os.path.join(data_dir, "complete_sensitivity_results.pkl")
with open(complete_results_path, 'wb') as f:
    pickle.dump(all_results, f)

# ===== ANALYSIS AND VISUALIZATION =====
print("\n" + "="*80)
print("GENERATING ANALYSIS AND VISUALIZATIONS")
print("="*80)

# Create comprehensive analysis
analysis_results = []

for algorithm in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
    for config_key, config_data in case_results[algorithm].items():
        # Parse configuration
        tr_factor = float(config_key.split('_')[1])
        heading_num = int(config_key.split('_')[3])
        
        # Calculate statistics
        makespans = config_data['makespans']
        total_distances = config_data['total_distances']
        cvs = config_data['cvs']
        jain_indices = config_data['jain_indices']
        total_times = config_data['total_times']
        clustering_times = config_data['clustering_times']
        routing_times = config_data['routing_times']
        
        analysis_results.append({
            'case': case_name,
            'algorithm': algorithm,
            'tr_factor': tr_factor,
            'heading_num': heading_num,
            'makespan_mean': np.mean(makespans),
            'makespan_std': np.std(makespans),
            'makespan_min': np.min(makespans),
            'makespan_max': np.max(makespans),
            'total_distance_mean': np.mean(total_distances),
            'total_distance_std': np.std(total_distances),
            'cv_mean': np.mean(cvs),
            'cv_std': np.std(cvs),
            'jain_mean': np.mean(jain_indices),
            'jain_std': np.std(jain_indices),
            'time_mean': np.mean(total_times),
            'time_std': np.std(total_times),
            'clustering_time_mean': np.mean(clustering_times),
            'routing_time_mean': np.mean(routing_times),
        })

# Convert to DataFrame for easier analysis
df = pd.DataFrame(analysis_results)

# Save analysis results
df.to_csv(os.path.join(data_dir, "sensitivity_analysis_results.csv"), index=False)

# Generate comprehensive visualizations
plt.style.use('default')

# 1. Main sensitivity analysis plots
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle(f'{case_name.upper()} - Heading Discretization & Turning Radius Sensitivity Analysis', fontsize=16)

# Define colors and markers for algorithms
algorithm_styles = {
    'HyDR-TR': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
    'GIN-GLKH': {'color': 'red', 'marker': 's', 'linestyle': '--'},
    'KM-GLKH': {'color': 'green', 'marker': '^', 'linestyle': '-.'}
}

# Row 1: Effect of Heading Discretization (averaged across turning radius factors)
metrics = ['makespan_mean', 'total_distance_mean', 'routing_time_mean']
metric_names = ['Makespan', 'Total Distance', 'Routing Time (s)']

for col, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    df_hd = df.groupby(['algorithm', 'heading_num'])[metric].mean().reset_index()
    for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
        alg_data = df_hd[df_hd['algorithm'] == alg]
        style = algorithm_styles[alg]
        axes[0, col].plot(alg_data['heading_num'], alg_data[metric], 
                         color=style['color'], marker=style['marker'], 
                         linestyle=style['linestyle'], label=alg, 
                         linewidth=2, markersize=8)
    axes[0, col].set_xlabel('Heading Discretization')
    axes[0, col].set_ylabel(metric_name)
    axes[0, col].set_title(f'{metric_name} vs Heading Discretization')
    axes[0, col].legend()
    axes[0, col].grid(True, alpha=0.3)

# Row 2: Effect of Turning Radius Factor (averaged across heading discretizations)
for col, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    df_tr = df.groupby(['algorithm', 'tr_factor'])[metric].mean().reset_index()
    for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
        alg_data = df_tr[df_tr['algorithm'] == alg]
        style = algorithm_styles[alg]
        axes[1, col].plot(alg_data['tr_factor'], alg_data[metric], 
                         color=style['color'], marker=style['marker'], 
                         linestyle=style['linestyle'], label=alg, 
                         linewidth=2, markersize=8)
    axes[1, col].set_xlabel('Turning Radius Factor')
    axes[1, col].set_ylabel(metric_name)
    axes[1, col].set_title(f'{metric_name} vs Turning Radius Factor')
    axes[1, col].legend()
    axes[1, col].grid(True, alpha=0.3)

# Row 3: Detailed interaction effects
for col, (tr_factor) in enumerate(turning_radius_factors):
    for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
        alg_data = df[(df['algorithm'] == alg) & (df['tr_factor'] == tr_factor)]
        style = algorithm_styles[alg]
        axes[2, col].plot(alg_data['heading_num'], alg_data['makespan_mean'], 
                         color=style['color'], marker=style['marker'], 
                         linestyle=style['linestyle'], label=alg, 
                         linewidth=2, markersize=8)
    axes[2, col].set_xlabel('Heading Discretization')
    axes[2, col].set_ylabel('Makespan')
    axes[2, col].set_title(f'Makespan vs HD (TR×{tr_factor})')
    axes[2, col].legend()
    axes[2, col].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(data_dir, f"{case_name}_comprehensive_sensitivity_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Detailed statistical analysis plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'{case_name.upper()} - Statistical Analysis (Mean ± Std)', fontsize=16)

# Makespan with error bars
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    alg_data = df[df['algorithm'] == alg]
    
    # Heading discretization effect
    hd_data = alg_data.groupby('heading_num').agg({
        'makespan_mean': 'mean',
        'makespan_std': 'mean'
    }).reset_index()
    
    axes[0, 0].errorbar(hd_data['heading_num'], hd_data['makespan_mean'], 
                       yerr=hd_data['makespan_std'], label=alg, 
                       marker=algorithm_styles[alg]['marker'], 
                       color=algorithm_styles[alg]['color'],
                       linestyle=algorithm_styles[alg]['linestyle'],
                       linewidth=2, markersize=8, capsize=5)

axes[0, 0].set_xlabel('Heading Discretization')
axes[0, 0].set_ylabel('Makespan (Mean ± Std)')
axes[0, 0].set_title('Makespan vs Heading Discretization')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Turning radius effect on makespan
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    alg_data = df[df['algorithm'] == alg]
    
    tr_data = alg_data.groupby('tr_factor').agg({
        'makespan_mean': 'mean',
        'makespan_std': 'mean'
    }).reset_index()
    
    axes[0, 1].errorbar(tr_data['tr_factor'], tr_data['makespan_mean'], 
                       yerr=tr_data['makespan_std'], label=alg, 
                       marker=algorithm_styles[alg]['marker'], 
                       color=algorithm_styles[alg]['color'],
                       linestyle=algorithm_styles[alg]['linestyle'],
                       linewidth=2, markersize=8, capsize=5)

axes[0, 1].set_xlabel('Turning Radius Factor')
axes[0, 1].set_ylabel('Makespan (Mean ± Std)')
axes[0, 1].set_title('Makespan vs Turning Radius Factor')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Routing time vs heading discretization
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    alg_data = df[df['algorithm'] == alg]
    
    hd_data = alg_data.groupby('heading_num')['routing_time_mean'].mean().reset_index()
    
    axes[0, 2].plot(hd_data['heading_num'], hd_data['routing_time_mean'], 
                   label=alg, marker=algorithm_styles[alg]['marker'], 
                   color=algorithm_styles[alg]['color'],
                   linestyle=algorithm_styles[alg]['linestyle'],
                   linewidth=2, markersize=8)

axes[0, 2].set_xlabel('Heading Discretization')
axes[0, 2].set_ylabel('Routing Time (s)')
axes[0, 2].set_title('Routing Time vs Heading Discretization')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Coefficient of Variation analysis
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    alg_data = df[df['algorithm'] == alg]
    
    hd_data = alg_data.groupby('heading_num')['cv_mean'].mean().reset_index()
    
    axes[1, 0].plot(hd_data['heading_num'], hd_data['cv_mean'], 
                   label=alg, marker=algorithm_styles[alg]['marker'], 
                   color=algorithm_styles[alg]['color'],
                   linestyle=algorithm_styles[alg]['linestyle'],
                   linewidth=2, markersize=8)

axes[1, 0].set_xlabel('Heading Discretization')
axes[1, 0].set_ylabel('Coefficient of Variation')
axes[1, 0].set_title('Load Balance vs Heading Discretization')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Jain's Fairness Index analysis
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    alg_data = df[df['algorithm'] == alg]
    
    hd_data = alg_data.groupby('heading_num')['jain_mean'].mean().reset_index()
    
    axes[1, 1].plot(hd_data['heading_num'], hd_data['jain_mean'], 
                   label=alg, marker=algorithm_styles[alg]['marker'], 
                   color=algorithm_styles[alg]['color'],
                   linestyle=algorithm_styles[alg]['linestyle'],
                   linewidth=2, markersize=8)

axes[1, 1].set_xlabel('Heading Discretization')
axes[1, 1].set_ylabel("Jain's Fairness Index")
axes[1, 1].set_title('Fairness vs Heading Discretization')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Total computation time
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    alg_data = df[df['algorithm'] == alg]
    
    hd_data = alg_data.groupby('heading_num')['time_mean'].mean().reset_index()
    
    axes[1, 2].plot(hd_data['heading_num'], hd_data['time_mean'], 
                   label=alg, marker=algorithm_styles[alg]['marker'], 
                   color=algorithm_styles[alg]['color'],
                   linestyle=algorithm_styles[alg]['linestyle'],
                   linewidth=2, markersize=8)

axes[1, 2].set_xlabel('Heading Discretization')
axes[1, 2].set_ylabel('Total Time (s)')
axes[1, 2].set_title('Total Time vs Heading Discretization')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(data_dir, f"{case_name}_statistical_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Heatmaps for each algorithm
metrics_for_heatmap = ['makespan_mean', 'routing_time_mean', 'cv_mean']
metric_titles = ['Makespan', 'Routing Time (s)', 'Coefficient of Variation']

for metric, title in zip(metrics_for_heatmap, metric_titles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{case_name.upper()} - {title} Heatmap by Algorithm', fontsize=16)
    
    for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
        alg_data = df[df['algorithm'] == alg]
        pivot_data = alg_data.pivot(index='tr_factor', columns='heading_num', values=metric)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[i])
        axes[i].set_title(f'{alg}')
        axes[i].set_xlabel('Heading Discretization')
        axes[i].set_ylabel('Turning Radius Factor')
    
    plt.tight_layout()
    fig.savefig(os.path.join(data_dir, f"{case_name}_heatmap_{metric}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 4. Box plots for variance analysis
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'{case_name.upper()} - Distribution Analysis', fontsize=16)

# Create data for box plots
box_data_hd = {}
box_data_tr = {}

for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
    box_data_hd[alg] = {}
    box_data_tr[alg] = {}
    
    alg_results = case_results[alg]
    
    for config_key, config_data in alg_results.items():
        tr_factor = float(config_key.split('_')[1])
        heading_num = int(config_key.split('_')[3])
        
        if heading_num not in box_data_hd[alg]:
            box_data_hd[alg][heading_num] = []
        if tr_factor not in box_data_tr[alg]:
            box_data_tr[alg][tr_factor] = []
        
        box_data_hd[alg][heading_num].extend(config_data['makespans'])
        box_data_tr[alg][tr_factor].extend(config_data['makespans'])

# Makespan distributions by heading discretization
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    data_to_plot = [box_data_hd[alg][hd] for hd in heading_discretizations]
    bp = axes[0, i].boxplot(data_to_plot, labels=heading_discretizations, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(algorithm_styles[alg]['color'])
        patch.set_alpha(0.7)
    axes[0, i].set_title(f'{alg} - Makespan by Heading Discretization')
    axes[0, i].set_xlabel('Heading Discretization')
    axes[0, i].set_ylabel('Makespan')
    axes[0, i].grid(True, alpha=0.3)

# Makespan distributions by turning radius factor
for i, alg in enumerate(['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']):
    data_to_plot = [box_data_tr[alg][tr] for tr in turning_radius_factors]
    bp = axes[1, i].boxplot(data_to_plot, labels=turning_radius_factors, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(algorithm_styles[alg]['color'])
        patch.set_alpha(0.7)
    axes[1, i].set_title(f'{alg} - Makespan by Turning Radius Factor')
    axes[1, i].set_xlabel('Turning Radius Factor')
    axes[1, i].set_ylabel('Makespan')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(data_dir, f"{case_name}_distribution_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# Generate summary statistics report
print("\n" + "="*80)
print(f"kroA100 SENSITIVITY ANALYSIS SUMMARY - {n_runs} RUNS")
print("="*80)

summary_stats = {}

# Overall effect of heading discretization
print("\n--- Effect of Heading Discretization (averaged across all turning radii) ---")
hd_effect = df.groupby(['algorithm', 'heading_num']).agg({
    'makespan_mean': ['mean', 'std'],
    'routing_time_mean': ['mean', 'std'],
    'cv_mean': ['mean', 'std'],
    'jain_mean': ['mean', 'std']
}).round(4)

for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
    print(f"\n{alg}:")
    alg_data = hd_effect.loc[alg]
    for hd in heading_discretizations:
        if hd in alg_data.index:
            makespan_mean = alg_data.loc[hd, ('makespan_mean', 'mean')]
            makespan_std = alg_data.loc[hd, ('makespan_mean', 'std')]
            time_mean = alg_data.loc[hd, ('routing_time_mean', 'mean')]
            time_std = alg_data.loc[hd, ('routing_time_mean', 'std')]
            cv_mean = alg_data.loc[hd, ('cv_mean', 'mean')]
            jain_mean = alg_data.loc[hd, ('jain_mean', 'mean')]
            print(f"  HD={hd}: Makespan={makespan_mean:.2f}±{makespan_std:.2f}, Time={time_mean:.4f}±{time_std:.4f}s, CV={cv_mean:.4f}, Jain={jain_mean:.4f}")

# Overall effect of turning radius factor
print("\n--- Effect of Turning Radius Factor (averaged across all heading discretizations) ---")
tr_effect = df.groupby(['algorithm', 'tr_factor']).agg({
    'makespan_mean': ['mean', 'std'],
    'routing_time_mean': ['mean', 'std'],
    'cv_mean': ['mean', 'std'],
    'jain_mean': ['mean', 'std']
}).round(4)

for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
    print(f"\n{alg}:")
    alg_data = tr_effect.loc[alg]
    for tr in turning_radius_factors:
        if tr in alg_data.index:
            makespan_mean = alg_data.loc[tr, ('makespan_mean', 'mean')]
            makespan_std = alg_data.loc[tr, ('makespan_mean', 'std')]
            time_mean = alg_data.loc[tr, ('routing_time_mean', 'mean')]
            time_std = alg_data.loc[tr, ('routing_time_mean', 'std')]
            cv_mean = alg_data.loc[tr, ('cv_mean', 'mean')]
            jain_mean = alg_data.loc[tr, ('jain_mean', 'mean')]
            print(f"  TR×{tr}: Makespan={makespan_mean:.2f}±{makespan_std:.2f}, Time={time_mean:.4f}±{time_std:.4f}s, CV={cv_mean:.4f}, Jain={jain_mean:.4f}")

# Detailed configuration analysis
print("\n--- Detailed Configuration Analysis ---")
for alg in ['HyDR-TR', 'GIN-GLKH', 'KM-GLKH']:
    print(f"\n{alg} - All Configurations:")
    alg_data = df[df['algorithm'] == alg].sort_values(['tr_factor', 'heading_num'])
    for _, row in alg_data.iterrows():
        print(f"  TR×{row['tr_factor']}, HD={row['heading_num']}: "
              f"Makespan={row['makespan_mean']:.2f}±{row['makespan_std']:.2f}, "
              f"Time={row['routing_time_mean']:.4f}s, "
              f"CV={row['cv_mean']:.4f}, "
              f"Jain={row['jain_mean']:.4f}")

# Save summary statistics
summary_stats = {
    'heading_discretization_effect': hd_effect.to_dict(),
    'turning_radius_effect': tr_effect.to_dict(),
    'detailed_results': df.to_dict('records')
}

summary_pickle_path = os.path.join(data_dir, "sensitivity_summary_statistics.pkl")
with open(summary_pickle_path, 'wb') as f:
    pickle.dump(summary_stats, f)

print(f"\n{'='*80}")
print(f"kroA100 SENSITIVITY ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"All results and visualizations saved to: {data_dir}/")
print("Files generated:")
print("- complete_sensitivity_results.pkl: All raw experimental results")
print("- sensitivity_analysis_results.csv: Tabulated results for further analysis")
print("- sensitivity_summary_statistics.pkl: Summary statistics")
print(f"- {case_name}_comprehensive_sensitivity_analysis.png: Main sensitivity plots")
print(f"- {case_name}_statistical_analysis.png: Statistical analysis with error bars")
print(f"- {case_name}_heatmap_*.png: Heatmap visualizations for each metric")
print(f"- {case_name}_distribution_analysis.png: Box plots showing distributions")
print(f"- {case_name}_sensitivity_results.pkl: Individual case results")

print(f"\nKey findings summary:")
print(f"- Tested {len(heading_discretizations)} heading discretization levels: {heading_discretizations}")
print(f"- Tested {len(turning_radius_factors)} turning radius factors: {turning_radius_factors}")
print(f"- Total configurations per algorithm: {len(heading_discretizations) * len(turning_radius_factors)}")
print(f"- Runs per configuration: {n_runs}")
print(f"- Total experiments: {len(heading_discretizations) * len(turning_radius_factors) * n_runs * 3} (3 algorithms)")
