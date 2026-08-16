from policy_GIN_GLKH import Policy_GIN_GLKH, action_sample, parallel_get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation_GIN_GLKH import validate

import yaml
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter



def train(routing_algorithm:str, train_batch_size:int, test_data_size:int, train_node_n_min:int, train_node_n_max:int, test_node_n:int, policy_net:Policy_GIN_GLKH, l_r:float, no_agent:int, curvature:float, 
          iterations:int, device:str, n_worker:int, writer: SummaryWriter, seed:int, exp_num:int):

    # prepare validation data
    validation_data = torch.load('validation_data/validation_'+str(test_node_n)+'_'+str(test_data_size))
    # a large start point
    best_so_far = np.inf
    validation_results = []

    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    policy_net.train()
    
    total_itr = 1

    for itr in tqdm(range(iterations)):
        # prepare training data with variable node sizes
        no_nodes = np.random.randint(train_node_n_min, train_node_n_max+1)
        
        data = torch.rand(size=[train_batch_size, no_nodes, 2])  # [batch, nodes, fea], fea is 2D location
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=train_batch_size)
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        # get reward for each batch
        reward = parallel_get_reward(
                                     routing_algorithm,
                                     action.detach().cpu().numpy(), 
                                     data.detach().cpu().numpy(), 
                                     no_agent, curvature, n_worker)  # reward: tensor [batch, 1]
        # compute loss
        loss = torch.mul(torch.tensor(reward, device=device), log_prob.sum(dim=1)).sum()

        avg_reward = sum(reward) / train_batch_size
        writer.add_scalar('train/loss', loss / train_batch_size, total_itr)
        writer.add_scalar('train/reward', avg_reward, total_itr)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validate and save best nets
        if (itr+1) % 1000 == 0:
            validation_result = validate(routing_algorithm, validation_data, policy_net, no_agent, device, curvature, n_worker)
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), './saved_model/a{}_n{}to{}_c{}_{}_seed{}_exp{}_GIN_GLKH.pth'.format(str(no_agent), str(train_node_n_min), str(train_node_n_max), str(curvature), routing_algorithm, str(seed), str(exp_num)))
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
                writer.add_scalar('validation/reward', validation_result, total_itr)

        total_itr += 1
    return validation_results


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    routing_algorithm = 'ortools'

    # DMTSP configuration
    n_agent = 3
    train_node_n_min = 10
    train_node_n_max = 50
    test_node_n = 51
    curvature = 10 # minmum turning radius == 1/curvature
    
    # DRL configuration
    train_batch_size = 512
    iterations = 10000 
    test_data_size = 512
    lr = 1e-4
    n_worker = 8
    
    # GIN_GLKH hyperparameters
    in_chnl = 2
    hid_chnl = 64
    key_size_embd = 64
    key_size_policy = 64
    val_size = 64
    clipping = 10

    # Experiment configuration
    seed = 1
    exp_num = 250604

    torch.manual_seed(seed)

    log_path = "log/a{}_n{}to{}_c{}_{}_seed{}_exp{}_GIN_GLKH".format(n_agent, train_node_n_min, train_node_n_max, curvature, routing_algorithm, seed, exp_num)
    writer = SummaryWriter(log_path)

    model_path = "./saved_model/a{}_n{}to{}_c{}_{}_seed{}_exp{}_GIN_GLKH.pth".format(n_agent, train_node_n_min, train_node_n_max, curvature, routing_algorithm, seed, exp_num)
    config_path = model_path.replace(".pth", ".yaml")

    # save all configuration as yaml in the log_path directory
    with open(config_path, 'w') as f:
        yaml.dump({'n_agent': n_agent, 'train_node_n_min': train_node_n_min, 'train_node_n_max': train_node_n_max, 'test_node_n': test_node_n, 'curvature': curvature, 'batch_size': train_batch_size,
                   'lr': lr, 'iterations': iterations, 'n_worker': n_worker, 'seed': seed, 'validation_data_size': test_data_size, 'device': dev,
                   'in_chnl': in_chnl, 'hid_chnl': hid_chnl, 'key_size_embd': key_size_embd, 'key_size_policy': key_size_policy, 'val_size': val_size, 'clipping': clipping, 'train_batch_size': train_batch_size}, f)

    policy = Policy_GIN_GLKH(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent, key_size_embd=key_size_embd,
                    key_size_policy=key_size_policy, val_size=val_size, clipping=clipping, dev=dev)

    best_results = train(routing_algorithm, train_batch_size, test_data_size, train_node_n_min, train_node_n_max, test_node_n, policy, lr, n_agent, curvature, iterations, dev, n_worker, writer, seed, exp_num)
    print(min(best_results))