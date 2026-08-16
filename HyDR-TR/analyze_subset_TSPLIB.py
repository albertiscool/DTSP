import numpy as np
import yaml
import pickle
import os
import time
import random
import argparse

from sklearn.cluster import KMeans

import torch

from torch_geometric.data import Data
from torch_geometric.data import Batch

# Import functions from both policy files
from policy_HyDR_TR import Policy_HyDR_TR, action_sample as action_sample_tf
from policy_GIN_GLKH import Policy_GIN_GLKH, Policy_GIN_GLKH_ORIGIN, action_sample as action_sample_gin



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
        n = len(individual_costs)
        sum_squares = np.sum(individual_costs ** 2)
        square_sum = (np.sum(individual_costs)) ** 2
        if sum_squares > 0:
            jain_index = square_sum / (n * sum_squares)
        else:
            jain_index = 1.0  # Perfect fairness if all costs are zero
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

# Database of TSPLIB instances and their corresponding minimum turning radii
database = {
    "ulysses16": 2,
    "ulysses22": 2,
    "att48": 394,
    "eil51": 4,
    "berlin52": 55,
    "st70": 6,
    "pr76": 980,
    "eil76": 4,
    "gr96": 4,
    "rat99": 5,
    "rd100": 57,
    "kroA100": 200,
    "eil101": 4,
    "ch130": 39,
    "pr136": 516,
    "gr137": 4
}

def save_checkpoint(all_results, checkpoint_file):
    """Save current progress to checkpoint file"""
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"    Checkpoint saved: {len(all_results)} instances completed")
    except Exception as e:
        print(f"    Warning: Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file):
    """Load previous progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                all_results = pickle.load(f)
            print(f"Checkpoint loaded: {len(all_results)} instances already completed")
            return all_results
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return {}
    return {}

def is_instance_complete(case_results, n_runs):
    """Check if an instance has completed all runs"""
    if not case_results:
        return False
    
    # Check if all algorithms have the required number of runs
    for algorithm in ['HyDR-TR', 'KM-GLKH', 'GIN-GLKH']:
        if algorithm not in case_results:
            return False
        if len(case_results[algorithm]['makespan']) < n_runs:
            return False
    
    return True

def get_completed_runs(case_results):
    """Get the number of completed runs for an instance"""
    if not case_results:
        return 0
    
    min_runs = float('inf')
    for algorithm in ['HyDR-TR', 'KM-GLKH', 'GIN-GLKH']:
        if algorithm not in case_results:
            return 0
        min_runs = min(min_runs, len(case_results[algorithm]['makespan']))
    
    return min_runs if min_runs != float('inf') else 0

def parse_tsplib_instances(instances_str):
    """Parse comma-separated list of TSPLIB instances"""
    if instances_str.lower() == 'all':
        return list(database.keys())
    
    instances = [inst.strip() for inst in instances_str.split(',')]
    
    # Validate instances
    invalid_instances = [inst for inst in instances if inst not in database]
    if invalid_instances:
        print(f"Warning: Invalid TSPLIB instances will be skipped: {invalid_instances}")
    
    valid_instances = [inst for inst in instances if inst in database]
    return valid_instances

def parse_seed_list(seed_str):
    """Parse comma-separated list of seeds or ranges"""
    if not seed_str:
        return None
    
    seeds = []
    parts = seed_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # Range format like "1-5"
            try:
                start, end = map(int, part.split('-'))
                seeds.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{part}', skipping")
        else:
            # Single seed
            try:
                seeds.append(int(part))
            except ValueError:
                print(f"Warning: Invalid seed '{part}', skipping")
    
    return seeds if seeds else None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run TSPLIB analysis with customizable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (30 runs, default output)
  python analyze_subset_TSPLIB.py
  
  # Run specific instances with 10 runs
  python analyze_subset_TSPLIB.py --instances berlin52,gr96 --n_runs 10
  
  # Run all instances with custom output location
  python analyze_subset_TSPLIB.py --instances all --output TSPLIB_custom_output
  
  # Run with custom seeds (up to 30)
  python analyze_subset_TSPLIB.py --instances ulysses16 --n_runs 5 --seeds 42,123,456,789,999
  
  # Run with seed range
  python analyze_subset_TSPLIB.py --instances berlin52 --n_runs 10 --seeds 100-109

Available TSPLIB instances:
  ulysses16, ulysses22, att48, eil51, berlin52, st70, pr76, eil76,
  gr96, rat99, rd100, kroA100, eil101, ch130, pr136, gr137
        """
    )
    
    parser.add_argument('--instances', type=str, default='all',
                        help='Comma-separated list of TSPLIB instances or "all" (default: all)')
    parser.add_argument('--n_runs', type=int, default=30,
                        help='Number of runs per instance (default: 30, max: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory name (default: TSPLIB_{n_runs}runs)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of random seeds or ranges (e.g., "42,123,456" or "1-10"). If not specified, uses default base_seeds.')
    parser.add_argument('--n_agent', type=int, default=3,
                        help='Number of agents/vehicles (default: 3)')
    
    args = parser.parse_args()
    
    # Parse instances
    instances_to_run = parse_tsplib_instances(args.instances)
    if not instances_to_run:
        print("Error: No valid TSPLIB instances specified")
        return
    
    # Validate and set n_runs
    n_runs = args.n_runs
    if n_runs < 1:
        print("Error: n_runs must be at least 1")
        return
    if n_runs > 30:
        print("Warning: n_runs exceeds maximum of 30, setting to 30")
        n_runs = 30
    
    # Parse seeds
    custom_seeds = parse_seed_list(args.seeds)
    if custom_seeds:
        if len(custom_seeds) < n_runs:
            print(f"Error: Provided {len(custom_seeds)} seeds but need {n_runs} for n_runs={n_runs}")
            return
        base_seeds = custom_seeds[:n_runs]
        print(f"Using custom seeds: {base_seeds}")
    else:
        # Default base_seeds from analyze_full_TSPLIB.py
        base_seeds = [42, 123, 456, 789, 999, 2024, 31415, 271828, 161803, 123456,
                      654321, 111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999,
                      13579, 24680, 11223, 44556, 77889, 10101, 20202, 30303, 40404, 50505]
        print(f"Using default seeds (first {n_runs})")
    
    n_agent = args.n_agent
    
    # Set output directory
    if args.output:
        data_dir = args.output
    else:
        data_dir = f"TSPLIB_{n_runs}runs"
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Define checkpoint file
    checkpoint_file = os.path.join(data_dir, "checkpoint.pkl")
    
    # Print configuration
    print("="*80)
    print("TSPLIB Analysis Configuration")
    print("="*80)
    print(f"Instances to run: {', '.join(instances_to_run)}")
    print(f"Number of runs per instance: {n_runs}")
    print(f"Number of agents: {n_agent}")
    print(f"Output directory: {data_dir}")
    print(f"Random seeds: {base_seeds[:n_runs]}")
    print("="*80)
    print()
    
    # Load existing checkpoint
    all_results = load_checkpoint(checkpoint_file)
    
    # Load models once before processing all cases
    print("Loading models...")
    
    # Load HyDR-TR model
    hydr_model_path = "./saved_model/a3_n10to50_c10_ortools_seed1_exp250307_HyDR_TR.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    print(f"  Loaded HyDR-TR model from {hydr_model_path}")
    
    # Load GIN model
    gin_model_path = "./saved_model/a3_n10to50_c10_ortools_seed1_exp250227_GIN_GLKH.pth"
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
        gin_policy = Policy_GIN_GLKH_ORIGIN(in_chnl=gin_config["in_chnl"], hid_chnl=gin_config["hid_chnl"], n_agent=gin_config["n_agent"], 
                          key_size_embd=gin_config["key_size_embd"], key_size_policy=gin_config["key_size_policy"], 
                          val_size=gin_config["val_size"], clipping=gin_config["clipping"], dev=device)
        gin_policy.load_state_dict(torch.load(gin_model_path, map_location=torch.device(device)))
        gin_policy.eval()
    
    print(f"  Loaded GIN model from {gin_model_path}")
    
    for case_name in instances_to_run:
        turning_radius = database[case_name]
        print(f"\n=== Processing {case_name} ===")
        
        # Check if this instance is already complete
        if case_name in all_results and is_instance_complete(all_results[case_name], n_runs):
            print(f"  {case_name} already completed ({n_runs} runs), skipping...")
            continue
        
        # Check for partial completion
        completed_runs = 0
        if case_name in all_results:
            completed_runs = get_completed_runs(all_results[case_name])
            if completed_runs > 0:
                print(f"  {case_name} partially completed ({completed_runs}/{n_runs} runs), resuming...")
        
        test_case = case_name
        minimum_turning_radius = turning_radius
        curvature = 1 / minimum_turning_radius
        
        # Initialize or load existing results storage for this case
        if case_name in all_results:
            case_results = all_results[case_name]
        else:
            case_results = {
                'HyDR-TR': {
                    'makespan': [],
                    'total_distance': [],
                    'cv': [],
                    'jain_index': [],
                    'individual_costs': [],
                    'clustering_times': [],
                    'routing_times': [],
                    'total_times': [],
                    'routes': [],
                    'seeds': []
                },
                'KM-GLKH': {
                    'makespan': [],
                    'total_distance': [],
                    'cv': [],
                    'jain_index': [],
                    'individual_costs': [],
                    'clustering_times': [],
                    'routing_times': [],
                    'total_times': [],
                    'routes': [],
                    'seeds': []
                },
                'GIN-GLKH': {
                    'makespan': [],
                    'total_distance': [],
                    'cv': [],
                    'jain_index': [],
                    'individual_costs': [],
                    'clustering_times': [],
                    'routing_times': [],
                    'total_times': [],
                    'routes': [],
                    'seeds': []
                }
            }
        
        # Load and parse TSPLIB data
        file_path = f'TSPLIB/{test_case}.tsp'
        with open(file_path, 'r') as file:
            raw_data = file.readlines()
        
        # Parse the coordinates from the file
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
        
        # Normalize the coordinates to the range [0 1]
        x_min, x_max = test_data[:, 0].min(), test_data[:, 0].max()
        y_min, y_max = test_data[:, 1].min(), test_data[:, 1].max()
        scale = max(x_max - x_min, y_max - y_min)
        test_data[:, 0] = ((test_data[:, 0] - x_min) / scale) + 0.01
        test_data[:, 1] = ((test_data[:, 1] - y_min) / scale) + 0.01
        
        # Run multiple experiments with different seeds
        for run_idx in range(completed_runs, n_runs):  # Start from completed_runs instead of 0
            seed = base_seeds[run_idx]
            random.seed(seed)
            np.random.seed(seed)
            print(f"  Run {run_idx + 1}/{n_runs} (seed: {seed})")
            
            # ===== HyDR-TR Algorithm =====
            with torch.no_grad():
                # Start timing for HyDR-TR
                hydr_start_time = time.perf_counter()
                
                # Clustering phase (RL assignment)
                clustering_start = time.perf_counter()
                data = torch.FloatTensor(test_data).unsqueeze(0).to(device)
                pi = hydr_policy(data)
                action, _ = action_sample_tf(pi)
                assignments = action.cpu().numpy().squeeze()
                clustering_time = time.perf_counter() - clustering_start
                
                # Routing phase (pure routing timing)
                routing_start = time.perf_counter()
                hydr_route, hydr_individual_costs = get_route_and_cost_gtsp_with_costs(assignments, original_data, n_agent=n_agent, 
                                                                                      curvature=curvature, heading_num=8, fix_origin_heading=True)
                routing_time = time.perf_counter() - routing_start
                
                total_hydr_time = time.perf_counter() - hydr_start_time
            
            # Calculate comprehensive metrics (outside routing timing)
            hydr_metrics = calculate_metrics_from_costs(hydr_individual_costs)
            
            # Store HyDR-TR results
            case_results['HyDR-TR']['makespan'].append(hydr_metrics['makespan'])
            case_results['HyDR-TR']['total_distance'].append(hydr_metrics['total_distance'])
            case_results['HyDR-TR']['cv'].append(hydr_metrics['cv'])
            case_results['HyDR-TR']['jain_index'].append(hydr_metrics['jain_index'])
            case_results['HyDR-TR']['individual_costs'].append(hydr_metrics['individual_costs'])
            case_results['HyDR-TR']['clustering_times'].append(clustering_time)
            case_results['HyDR-TR']['routing_times'].append(routing_time)
            case_results['HyDR-TR']['total_times'].append(total_hydr_time)
            case_results['HyDR-TR']['routes'].append(hydr_route)
            case_results['HyDR-TR']['seeds'].append(seed)
            
            print(f"    HyDR-TR - Makespan: {hydr_metrics['makespan']:.2f}, Total Distance: {hydr_metrics['total_distance']:.2f}, CV: {hydr_metrics['cv']:.4f}, Jain: {hydr_metrics['jain_index']:.4f}")
            
            # ===== K-means + GLKH Algorithm =====
            depot = test_data[0]
            waypoints = test_data[1:]
            
            # Start timing for K-means
            kmeans_start_time = time.perf_counter()
            
            # Clustering phase
            clustering_start = time.perf_counter()
            kmeans = KMeans(n_clusters=n_agent, random_state=seed, n_init="auto").fit(waypoints)
            labels = kmeans.labels_
            clustering_time = time.perf_counter() - clustering_start
            
            # Routing phase (pure routing timing)
            routing_start = time.perf_counter()
            kmeans_route, kmeans_individual_costs = get_route_and_cost_gtsp_with_costs(labels, original_data, n_agent=n_agent, 
                                                                                      curvature=curvature, heading_num=8, fix_origin_heading=True)
            routing_time = time.perf_counter() - routing_start
            
            total_kmeans_time = time.perf_counter() - kmeans_start_time
            
            # Calculate comprehensive metrics (outside routing timing)
            kmeans_metrics = calculate_metrics_from_costs(kmeans_individual_costs)
            
            # Store K-means results
            case_results['KM-GLKH']['makespan'].append(kmeans_metrics['makespan'])
            case_results['KM-GLKH']['total_distance'].append(kmeans_metrics['total_distance'])
            case_results['KM-GLKH']['cv'].append(kmeans_metrics['cv'])
            case_results['KM-GLKH']['jain_index'].append(kmeans_metrics['jain_index'])
            case_results['KM-GLKH']['individual_costs'].append(kmeans_metrics['individual_costs'])
            case_results['KM-GLKH']['clustering_times'].append(clustering_time)
            case_results['KM-GLKH']['routing_times'].append(routing_time)
            case_results['KM-GLKH']['total_times'].append(total_kmeans_time)
            case_results['KM-GLKH']['routes'].append(kmeans_route)
            case_results['KM-GLKH']['seeds'].append(seed)
            
            print(f"    KM-GLKH - Makespan: {kmeans_metrics['makespan']:.2f}, Total Distance: {kmeans_metrics['total_distance']:.2f}, CV: {kmeans_metrics['cv']:.4f}, Jain: {kmeans_metrics['jain_index']:.4f}")
            
            # ===== GIN + GLKH Algorithm =====
            with torch.no_grad():
                # Start timing for GIN
                gin_start_time = time.perf_counter()
                
                # Clustering phase (GIN assignment)
                clustering_start = time.perf_counter()
                
                # Prepare data for PyTorch Geometric GIN model
                data = torch.FloatTensor(test_data).unsqueeze(0).to(device)
                adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]]) # Fully connected
                data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
                batch_graph = Batch.from_data_list(data_list=data_list).to(device)
                
                # Get assignments from the policy
                pi = gin_policy(batch_graph, n_nodes=data.shape[1], n_batch=1)
                action, _ = action_sample_gin(pi)
                assignments = action.cpu().numpy().squeeze()
                clustering_time = time.perf_counter() - clustering_start
                
                # Routing phase (pure routing timing)
                routing_start = time.perf_counter()
                gin_route, gin_individual_costs = get_route_and_cost_gtsp_with_costs(assignments, original_data, n_agent=n_agent, 
                                                                                    curvature=curvature, heading_num=8, fix_origin_heading=True)
                routing_time = time.perf_counter() - routing_start
                
                total_gin_time = time.perf_counter() - gin_start_time
            
            # Calculate comprehensive metrics (outside routing timing)
            gin_metrics = calculate_metrics_from_costs(gin_individual_costs)
            
            # Store GIN results
            case_results['GIN-GLKH']['makespan'].append(gin_metrics['makespan'])
            case_results['GIN-GLKH']['total_distance'].append(gin_metrics['total_distance'])
            case_results['GIN-GLKH']['cv'].append(gin_metrics['cv'])
            case_results['GIN-GLKH']['jain_index'].append(gin_metrics['jain_index'])
            case_results['GIN-GLKH']['individual_costs'].append(gin_metrics['individual_costs'])
            case_results['GIN-GLKH']['clustering_times'].append(clustering_time)
            case_results['GIN-GLKH']['routing_times'].append(routing_time)
            case_results['GIN-GLKH']['total_times'].append(total_gin_time)
            case_results['GIN-GLKH']['routes'].append(gin_route)
            case_results['GIN-GLKH']['seeds'].append(seed)
            
            print(f"    GIN-GLKH - Makespan: {gin_metrics['makespan']:.2f}, Total Distance: {gin_metrics['total_distance']:.2f}, CV: {gin_metrics['cv']:.4f}, Jain: {gin_metrics['jain_index']:.4f}")
            
            # Save checkpoint after every run to prevent data loss
            all_results[case_name] = case_results
            if (run_idx + 1) % 5 == 0 or run_idx == n_runs - 1:  # Save every 5 runs or at the end
                save_checkpoint(all_results, checkpoint_file)
        
        # Save results for this case
        all_results[case_name] = case_results
        
        # Save individual case results as pickle
        case_pickle_path = os.path.join(data_dir, f"{case_name}_results.pkl")
        with open(case_pickle_path, 'wb') as f:
            pickle.dump(case_results, f)
        
        print(f"  ✓ {case_name} completed and saved")
        
        # Save final checkpoint for this instance
        save_checkpoint(all_results, checkpoint_file)
    
    # Save complete results
    complete_results_path = os.path.join(data_dir, "complete_results.pkl")
    with open(complete_results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Remove checkpoint file since everything is complete
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed - all instances completed")
    
    print(f"\nAll instances processed successfully!")
    print(f"Total instances requested: {len(instances_to_run)}")
    print(f"Total instances completed: {len(all_results)}")
    
    # Generate summary statistics
    print("\n" + "="*120)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*120)
    
    summary_data = {}
    for case_name in instances_to_run:
        if case_name not in all_results:
            continue
            
        case_results = all_results[case_name]
        
        summary_data[case_name] = {}
        
        for algorithm in ['HyDR-TR', 'KM-GLKH', 'GIN-GLKH']:
            data = case_results[algorithm]
            
            summary_data[case_name][algorithm] = {
                'makespan': {
                    'mean': np.mean(data['makespan']),
                    'std': np.std(data['makespan']),
                    'min': np.min(data['makespan']),
                    'max': np.max(data['makespan'])
                },
                'total_distance': {
                    'mean': np.mean(data['total_distance']),
                    'std': np.std(data['total_distance']),
                    'min': np.min(data['total_distance']),
                    'max': np.max(data['total_distance'])
                },
                'cv': {
                    'mean': np.mean(data['cv']),
                    'std': np.std(data['cv']),
                    'min': np.min(data['cv']),
                    'max': np.max(data['cv'])
                },
                'jain_index': {
                    'mean': np.mean(data['jain_index']),
                    'std': np.std(data['jain_index']),
                    'min': np.min(data['jain_index']),
                    'max': np.max(data['jain_index'])
                },
                'total_time': {
                    'mean': np.mean(data['total_times']),
                    'std': np.std(data['total_times'])
                }
            }
        
        print(f"\n{case_name}:")
        for algorithm in ['HyDR-TR', 'KM-GLKH', 'GIN-GLKH']:
            stats = summary_data[case_name][algorithm]
            print(f"  {algorithm}:")
            print(f"    Makespan: {stats['makespan']['mean']:.2f} ± {stats['makespan']['std']:.2f} "
                  f"[{stats['makespan']['min']:.2f}, {stats['makespan']['max']:.2f}]")
            print(f"    Total Distance: {stats['total_distance']['mean']:.2f} ± {stats['total_distance']['std']:.2f} "
                  f"[{stats['total_distance']['min']:.2f}, {stats['total_distance']['max']:.2f}]")
            print(f"    CV: {stats['cv']['mean']:.4f} ± {stats['cv']['std']:.4f} "
                  f"[{stats['cv']['min']:.4f}, {stats['cv']['max']:.4f}]")
            print(f"    Jain Index: {stats['jain_index']['mean']:.4f} ± {stats['jain_index']['std']:.4f} "
                  f"[{stats['jain_index']['min']:.4f}, {stats['jain_index']['max']:.4f}]")
            print(f"    Total Time: {stats['total_time']['mean']:.4f} ± {stats['total_time']['std']:.4f} seconds")
    
    # Save summary statistics
    summary_path = os.path.join(data_dir, "summary_statistics.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(summary_data, f)
    
    print(f"\nAll results saved to: {data_dir}/")
    print("Files generated:")
    print("- complete_results.pkl: All raw results with comprehensive metrics")
    print("- summary_statistics.pkl: Summary statistics for all metrics")
    print("- {case_name}_results.pkl: Individual case results")

if __name__ == "__main__":
    main()
