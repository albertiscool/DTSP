from gin import Net, Net_origin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.distributions import Categorical
from utils.ortools_tsp import or_solve
from utils.heading_opt import aa_solve
from utils.glkh_dtsp import glkh_dtsp_solve, glkh_dtsp_solve_originfixed
import numpy as np
from utils.dubins.dubins_length_only import dubins_path_length
from utils.dubins.dubins_path import plan_dubins_path
from multiprocessing import Pool


class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
        self.q_agent = nn.Linear(2 * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent

        # gin
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h, g_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)

        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = nodes_h.mean(dim=1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy_GIN_GLKH(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy_GIN_GLKH, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)

    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob
    

class AgentAndNode_embedding_origin(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding_origin, self).__init__()

        self.n_agent = n_agent

        # gin
        self.gin = Net_origin(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index)

        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = nodes_h.mean(dim=1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy_GIN_GLKH_ORIGIN(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy_GIN_GLKH_ORIGIN, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding_origin(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)

    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob


def action_sample(pi):
    dist = Categorical(pi.transpose(2, 1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def get_reward(action:np.ndarray, data:np.ndarray, n_agent:int, curvature:float=10.0):
    
    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    depot = data[:, 0, :].tolist()
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for tour in sub_tours[i]:
            tour.append(depot[i])

        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)

    for k in range(data.shape[0]):
        for a in range(n_agent):
            instance = np.array(sub_tours[k][a])

            if instance.shape[0] == 1:
                continue

            sub_route = or_solve(instance)
            sub_route_headings = aa_solve(instance, sub_route)
            # fix first waypoint heading angle as 0
            sub_route_headings[0] = 0
            # make dubins waypoints
            dubins_waypoints = np.hstack((np.array(instance[sub_route[:-1]]), 
                                          np.array(sub_route_headings).reshape(-1, 1))) # [n_nodes, 3]
            
            # calculate dubins path length
            sub_tour_length = 0
            for i in range(len(sub_route)-1):
                start_x = dubins_waypoints[i, 0]
                start_y = dubins_waypoints[i, 1]
                start_yaw = dubins_waypoints[i, 2]

                if i < len(sub_route)-2:
                    end_x = dubins_waypoints[i+1, 0]
                    end_y = dubins_waypoints[i+1, 1]
                    end_yaw = dubins_waypoints[i+1, 2]
                else:
                    end_x = dubins_waypoints[0, 0]
                    end_y = dubins_waypoints[0, 1]
                    end_yaw = dubins_waypoints[0, 2]

                lengths = dubins_path_length(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)
                sub_tour_length += lengths

            if sub_tour_length >= subtour_max_lengths[k]:
                subtour_max_lengths[k] = sub_tour_length

    return subtour_max_lengths

def get_reward_glkh(action:np.ndarray, data:np.ndarray, n_agent:int, curvature:float=10.0):

    heading_num = 8
    
    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    depot = data[:, 0, :].tolist()
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for tour in sub_tours[i]:
            tour.append(depot[i])

        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)

    for k in range(data.shape[0]):
        for a in range(n_agent):
            instance = np.array(sub_tours[k][a])

            if instance.shape[0] == 1:
                continue

            sub_route, sub_route_headings = glkh_dtsp_solve_originfixed(instance, heading_num, curvature)

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

                lengths = dubins_path_length(start_x, start_y, start_yaw, 
                                             end_x, end_y, end_yaw, curvature)
                
                sub_tour_length += lengths

            if sub_tour_length >= subtour_max_lengths[k]:
                subtour_max_lengths[k] = sub_tour_length

    return subtour_max_lengths

def get_route_and_cost(assignments:np.ndarray, data:np.ndarray, n_agent:int, curvature:float=10.0):
    
    depot = data[0, :].tolist()
    sub_tours = [[] for _ in range(n_agent)]
    for tour in sub_tours:
        tour.append(depot)

    for n, m in zip(assignments.tolist(), data.tolist()[1:]):
        sub_tours[n].append(m)

    or_routes = {}
    rl_lengths = []
    # make rl_route dictionary for each agent
    rl_routes = {}

    for a in range(n_agent):
        instance = np.array(sub_tours[a])
        agent_route = []

        if instance.shape[0] == 1:
            continue
        sub_route = or_solve(instance)
        sub_route_headings = aa_solve(instance, sub_route)
        # make dubins waypoints
        dubins_waypoints = np.hstack((np.array(instance[sub_route[:-1]]), 
                                        np.array(sub_route_headings).reshape(-1, 1)))
        or_routes[a+1]=instance[sub_route]
        # calculate dubins path length
        sub_tour_length = 0
        for i in range(len(sub_route)-1):
            start_x = dubins_waypoints[i, 0]
            start_y = dubins_waypoints[i, 1]
            start_yaw = dubins_waypoints[i, 2]

            if i < len(sub_route)-2:
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

    max_length = max(rl_lengths)

    return rl_routes, max_length

def get_route_and_cost_gtsp(assignments:np.ndarray, data:np.ndarray, n_agent:int, curvature:float=10.0, heading_num=8, fix_origin_heading=False):
    
    depot = data[0, :].tolist()
    sub_tours = [[] for _ in range(n_agent)]
    for tour in sub_tours:
        tour.append(depot)

    for n, m in zip(assignments.tolist(), data.tolist()[1:]):
        sub_tours[n].append(m)


    rl_lengths = []
    # make rl_route dictionary for each agent
    rl_routes = {}

    for a in range(n_agent):
        instance = np.array(sub_tours[a])
        agent_route = []

        if instance.shape[0] == 1:
            continue
        
        if fix_origin_heading:
            sub_route, sub_route_headings = glkh_dtsp_solve_originfixed(instance, heading_num, curvature)
        else:
            sub_route, sub_route_headings = glkh_dtsp_solve(instance, heading_num, curvature)

        # make dubins waypoints
        dubins_waypoints = np.hstack((np.array(instance[sub_route]), 
                                        np.array(sub_route_headings).reshape(-1, 1)))
        
        # fix first waypoint heading angle as 0
        # dubins_waypoints[0, 2] = 0

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

    max_length = max(rl_lengths)

    return rl_routes, max_length

def worker(chunk):
    # Unpack the chunk information
    routing_algorithm, chunk_action, chunk_data, n_agent, curvature = chunk

    if routing_algorithm == 'ortools':
        # Call the existing get_reward function or a modified version that works on a chunk
        return get_reward(chunk_action, chunk_data, n_agent, curvature)
    elif routing_algorithm == 'glkh':
        return get_reward_glkh(chunk_action, chunk_data, n_agent, curvature)

def parallel_get_reward(routing_algorithm, action, data, n_agent, curvature=5.0, n_workers=None):
    # Determine the number of workers (processes) if not specified 
    if n_workers is None:
        raise ValueError("The number of workers must be specified")

    # Split the action and data arrays into chunks
    n_batches = action.shape[0]
    chunk_size = int(np.ceil(n_batches / n_workers))
    chunks = [
        (routing_algorithm, action[i:i + chunk_size], data[i:i + chunk_size], n_agent, curvature)
        for i in range(0, n_batches, chunk_size)
    ]

    # Set up a pool of worker processes
    with Pool(processes=n_workers) as pool:
        # Process the chunks in parallel
        results = pool.map(worker, chunks)
    
    pool.close()
    pool.join()

    # Aggregate the results from all chunks
    aggregated_results = np.concatenate(results)

    return aggregated_results



if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.data import Batch
    import torch

    import time

    start_t= time.time()

    dev = 'cuda'
    torch.manual_seed(2)

    n_agent = 5
    n_nodes = 50
    n_batch = 4
    curvature = 10.0

    network_construct_start_t = time.time()
    # test policy
    policy = Policy_GIN_GLKH(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    print("Network construction time: ", time.time()-network_construct_start_t)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # get batch graphs data list
    fea = torch.rand(size=[n_batch, n_nodes, 2])  # [batch, nodes, fea]
    adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
    data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(fea.shape[0])]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data_list).to(dev)

    inference_start_t = time.time()
    pi = policy(batch_graph, n_nodes=fea.shape[1], n_batch=n_batch)

    # grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])

    action, log_prob = action_sample(pi)
    # print(log_prob)
    # print("Inference time: ", time.time()-inference_start_t)

    reward_cal_start_t = time.time()
    rewards = get_reward(action.detach().cpu().numpy(), 
                         fea.detach().cpu().numpy(), 
                         n_agent, curvature)
    # print("Reward calculation time: ", time.time()-reward_cal_start_t)

    parallel_reward_cal_start_t = time.time()
    parallel_rewards = parallel_get_reward(action.detach().cpu().numpy(), 
                                           fea.detach().cpu().numpy(), 
                                           n_agent, curvature, 4)
    print("Parallel reward calculation time: ", time.time()-parallel_reward_cal_start_t)

    grad_cal_start_t = time.time()
    loss = torch.mul(torch.tensor(parallel_rewards, device=dev), log_prob.sum(dim=1)).sum()


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Gradient calculation time: ", time.time()-grad_cal_start_t)
    
    print(rewards)
    print(parallel_rewards)

    print("Device: ", dev)
    print("Total time: ",time.time()-start_t)