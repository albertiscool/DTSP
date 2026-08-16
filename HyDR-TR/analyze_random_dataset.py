import random
import numpy as np
import os
import pickle
import yaml
import time
from scipy import stats
from scipy.stats import ttest_rel
import argparse

from sklearn.cluster import KMeans
from policy_HyDR_TR import Policy_HyDR_TR
from policy_GIN_GLKH import Policy_GIN_GLKH, Policy_GIN_GLKH_ORIGIN, action_sample, parallel_get_reward

import torch
from torch_geometric.data import Data, Batch

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a dataset."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - h, mean + h

def cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def perform_statistical_analysis(hydr_results, comparison_results, algorithm_name):
    """Perform t-test and calculate effect size between HyDR-TR and comparison algorithm."""
    # Paired t-test
    t_stat, p_value = ttest_rel(hydr_results, comparison_results)
    
    # Cohen's d effect size
    effect_size = cohens_d(hydr_results, comparison_results)
    
    # Confidence intervals
    hydr_mean, hydr_ci_low, hydr_ci_high = calculate_confidence_interval(hydr_results)
    comp_mean, comp_ci_low, comp_ci_high = calculate_confidence_interval(comparison_results)
    
    return {
        'hydr_mean': hydr_mean,
        'hydr_ci': [hydr_ci_low, hydr_ci_high],
        'comp_mean': comp_mean,
        'comp_ci': [comp_ci_low, comp_ci_high],
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': effect_size,
        'comparison_algorithm': algorithm_name
    }

def save_checkpoint(base_path, agent_num, test_result, completed_tests):
    """Save checkpoint data for recovery."""
    checkpoint_path = f"{base_path}/checkpoint_a{agent_num}.pkl"
    checkpoint_data = {
        'test_result': test_result,
        'completed_tests': completed_tests,
        'timestamp': time.time()
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved for agent {agent_num} at {checkpoint_path}")

def load_checkpoint(base_path, agent_num):
    """Load checkpoint data if exists."""
    checkpoint_path = f"{base_path}/checkpoint_a{agent_num}.pkl"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        print(f"Checkpoint loaded for agent {agent_num} from {checkpoint_path}")
        return checkpoint_data['test_result'], checkpoint_data['completed_tests']
    return None, None

def main():
    # Test configuration
    test_case = {
        "nodes": [25, 50, 100],
        "curvature": [10],
        "agent_configs": [
            {
                "agent_num": 3,
                "models": {
                    "HyDR-TR": {
                        "path": "a3_n10to50_c10_ortools_seed1_exp250913_HyDR_TR.pth",
                        "type": "HyDR-TR"
                    },
                    "GIN-GLKH": {
                        "path": "a3_n10to50_c10_ortools_seed1_exp250915_GIN_GLKH.pth", 
                        "type": "GIN-GLKH"
                    }
                }
            },
            {

                "agent_num": 4,
                "models": {
                    "HyDR-TR": {
                        "path": "a4_n10to50_c10_ortools_seed1_exp250913_HyDR_TR.pth",
                        "type": "HyDR-TR"
                    },
                    "GIN-GLKH": {
                        "path": "a4_n10to50_c10_ortools_seed1_exp250915_GIN_Origin.pth", 
                        "type": "GIN-GLKH"
                    }
                }
            },
            {
                "agent_num": 5,
                "models": {
                    "HyDR-TR": {
                        "path": "a5_n10to50_c10_ortools_seed1_exp250913_HyDR_TR.pth",
                        "type": "HyDR-TR"
                    },
                    "GIN-GLKH": {
                        "path": "a5_n10to50_c10_ortools_seed1_exp250916_GIN_GLKH.pth",
                        "type": "GIN-GLKH"
                    }
                }
            }
        ]
    }

    test_data_num = 100
    test_heading_num = 8
    test_routing_alg = "glkh"
    test_worker = 8
    test_date = time.strftime("%Y%m%d")
    random_seed = 77

    random.seed(random_seed)
    np.random.seed(random_seed)

    test_algorithms = ["KM-GLKH", "GIN-GLKH", "HyDR-TR"]
    test_rl_device = "cuda"

    # Create base directory
    base_path = f"random_dataset_test/{test_date}"
    os.makedirs(base_path, exist_ok=True)

    # Save test configuration
    with open(f"{base_path}/test_config.yaml", "w") as f:
        yaml.dump(test_case, f)
        f.write(f"test_date: {test_date}\n")
        f.write(f"test_data_num: {test_data_num}\n")
        f.write(f"test_heading_num: {test_heading_num}\n")
        f.write(f"test_routing_alg: {test_routing_alg}\n")
        f.write(f"test_worker: {test_worker}\n")
        f.write(f"random_seed: {random_seed}\n")
        f.write(f"test_algorithms: {test_algorithms}\n")
        f.write(f"test_rl_device: {test_rl_device}\n")

    # Main testing loop for each agent configuration
    for agent_config in test_case["agent_configs"]:
        agent_num = agent_config["agent_num"]
        print(f"\n{'='*50}")
        print(f"Starting tests for {agent_num} agents")
        print(f"{'='*50}")
        
        # Load or initialize test results
        test_result, completed_tests = load_checkpoint(base_path, agent_num)
        if test_result is None:
            test_result = {}
            completed_tests = set()
            
            # Initialize result structure
            for alg in test_algorithms:
                test_result[alg] = {}
                for nodes in test_case["nodes"]:
                    test_result[alg][nodes] = {}
                    for curvature in test_case["curvature"]:
                        test_result[alg][nodes][curvature] = {
                            "cost": [],
                            "compute_time": []
                        }
        
        # Load models for this agent configuration
        models = {}
        for alg_name, model_info in agent_config["models"].items():
            model_path = model_info["path"]
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found!")
                continue
                
            config_path = model_path.replace(".pth", ".yaml")
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if model_info["type"] == "TF":
                policy = Policy_HyDR_TR(
                    in_chnl=config["in_chnl"], 
                    hid_chnl=config["hid_chnl"], 
                    n_agent=config["n_agent"],
                    key_size_embd=config["key_size_embd"],
                    key_size_policy=config["key_size_policy"], 
                    val_size=config["val_size"], 
                    clipping=config["clipping"], 
                    dev=test_rl_device,
                    n_heads=config["n_heads"],
                    ff_dim=config["ff_dim"],
                    norm_eps=config["norm_eps"],
                    batch_first=config["batch_first"],
                    n_enc_layer=config["n_enc_layer"]
                )
            elif model_info["type"] == "GIN":
                try:
                    policy = Policy_GIN_GLKH(
                        in_chnl=config["in_chnl"], 
                        hid_chnl=config["hid_chnl"], 
                        n_agent=config["n_agent"],
                        key_size_embd=config["key_size_embd"],
                        key_size_policy=config["key_size_policy"], 
                        val_size=config["val_size"], 
                        clipping=config["clipping"], 
                        dev=test_rl_device
                    )
                except:  # alternative import
                    policy = Policy_GIN_GLKH_ORIGIN(
                        in_chnl=config["in_chnl"], 
                        hid_chnl=config["hid_chnl"], 
                        n_agent=config["n_agent"],
                        key_size_embd=config["key_size_embd"],
                        key_size_policy=config["key_size_policy"], 
                        val_size=config["val_size"], 
                        clipping=config["clipping"], 
                        dev=test_rl_device
                    )
            
            policy.load_state_dict(torch.load(model_path, map_location=torch.device(test_rl_device)))
            models[alg_name] = {"policy": policy, "type": model_info["type"]}
            print(f"Loaded {alg_name} model: {model_path}")

        # Generate test datasets for each node count
        test_datasets = {}
        for nodes in test_case["nodes"]:
            dataset_path = f"{base_path}/a{agent_num}_{nodes}_nodes.pkl"
            if os.path.exists(dataset_path):
                with open(dataset_path, "rb") as f:
                    test_datasets[nodes] = pickle.load(f)
                print(f"Loaded existing dataset for {nodes} nodes")
            else:
                test_datasets[nodes] = np.random.uniform(0, 1, (test_data_num, nodes+1, 2))
                with open(dataset_path, "wb") as f:
                    pickle.dump(test_datasets[nodes], f)
                print(f"Generated new dataset for {nodes} nodes")

        # Run tests
        total_tests = len(test_case["nodes"]) * len(test_case["curvature"]) * len(test_algorithms)
        current_test = 0
        
        for nodes in test_case["nodes"]:
            test_dataset = test_datasets[nodes]
            
            for curvature in test_case["curvature"]:
                for alg in test_algorithms:
                    test_key = f"{nodes}_{curvature}_{alg}"
                    current_test += 1
                    
                    if test_key in completed_tests:
                        print(f"[{current_test}/{total_tests}] Skipping {alg} with {nodes} nodes and {curvature} curvature (already completed)")
                        continue
                    
                    print(f"[{current_test}/{total_tests}] Running {alg} with {nodes} nodes and {curvature} curvature")
                    
                    start_time = time.time()
                    
                    if alg == "KM-GLKH":
                        # K-means clustering
                        assignments = np.zeros((test_data_num, nodes))
                        for i, data in enumerate(test_dataset):
                            depot = data[0]
                            waypoints = data[1:]
                            kmeans = KMeans(n_clusters=agent_num, random_state=0, n_init="auto").fit(waypoints)
                            labels = kmeans.labels_
                            assignments[i, :] = labels
                        assignments = assignments.astype(int)
                        
                    elif alg in models:
                        # Neural network models
                        model_info = models[alg]
                        policy = model_info["policy"]
                        model_type = model_info["type"]
                        
                        if model_type == "TF":
                            data = torch.FloatTensor(test_dataset).to(test_rl_device)
                            pi = policy(data)
                        elif model_type == "GIN":
                            fea = torch.FloatTensor(test_dataset)
                            adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
                            data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(fea.shape[0])]
                            batch_graph = Batch.from_data_list(data_list=data_list).to(test_rl_device)
                            pi = policy(batch_graph, n_nodes=fea.shape[1], n_batch=fea.shape[0])
                        
                        action, _ = action_sample(pi)
                        assignments = action.cpu().numpy().squeeze()
                    
                    else:
                        raise ValueError(f"Invalid algorithm: {alg}")
                    
                    # Calculate costs
                    costs = parallel_get_reward(test_routing_alg, assignments, test_dataset, agent_num, curvature, test_worker)
                    compute_time = time.time() - start_time
                    
                    # Store results
                    test_result[alg][nodes][curvature]["cost"] = costs
                    test_result[alg][nodes][curvature]["compute_time"] = compute_time
                    
                    # Mark as completed and save checkpoint
                    completed_tests.add(test_key)
                    save_checkpoint(base_path, agent_num, test_result, completed_tests)
                    
                    print(f"  Mean cost: {np.mean(costs):.4f}, Compute time: {compute_time:.2f}s")

        # Save final results for this agent configuration
        with open(f"{base_path}/result_a{agent_num}.pkl", "wb") as f:
            pickle.dump(test_result, f)
        
        # Perform statistical analysis
        print(f"\nPerforming statistical analysis for {agent_num} agents...")
        statistical_results = {}
        
        for nodes in test_case["nodes"]:
            statistical_results[nodes] = {}
            for curvature in test_case["curvature"]:
                statistical_results[nodes][curvature] = {}
                
                hydr_costs = test_result["HyDR-TR"][nodes][curvature]["cost"]
                
                # Compare with each other algorithm
                for comp_alg in ["KM-GLKH", "GIN-GLKH"]:
                    comp_costs = test_result[comp_alg][nodes][curvature]["cost"]
                    stats_result = perform_statistical_analysis(hydr_costs, comp_costs, comp_alg)
                    statistical_results[nodes][curvature][comp_alg] = stats_result
                    
                    print(f"  {nodes} nodes, curvature {curvature}, HyDR-TR vs {comp_alg}:")
                    print(f"    HyDR-TR: {stats_result['hydr_mean']:.4f} ± {(stats_result['hydr_ci'][1] - stats_result['hydr_ci'][0])/2:.4f}")
                    print(f"    {comp_alg}: {stats_result['comp_mean']:.4f} ± {(stats_result['comp_ci'][1] - stats_result['comp_ci'][0])/2:.4f}")
                    print(f"    p-value: {stats_result['p_value']:.6f}")
                    print(f"    Cohen's d: {stats_result['cohens_d']:.4f}")
        
        # Save statistical results
        with open(f"{base_path}/statistical_analysis_a{agent_num}.pkl", "wb") as f:
            pickle.dump(statistical_results, f)
        
        print(f"Completed analysis for {agent_num} agents")
    
    print(f"\nAll analyses completed! Results saved in {base_path}")

if __name__ == "__main__":
    main()