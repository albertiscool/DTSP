from policy_HyDR_TR import Policy_HyDR_TR, action_sample, parallel_get_reward
from validation_HyDR_TR import validate

import yaml
import numpy as np
from tqdm import tqdm

import torch

from torch.utils.tensorboard import SummaryWriter



def train(iterations:int, routing_algorithm:str, train_batch_size:int, 
          test_data_size:int, train_node_n_min:int, train_node_n_max, test_node_n:int, 
          policy_net:Policy_HyDR_TR, l_r:float, no_agent:int, curvature:float, 
          device:str, n_worker:int, writer: SummaryWriter, seed:int, exp_num:int):

    # Prepare validation data
    validation_data = torch.load('validation_data/validation_' + str(test_node_n) + '_' + str(test_data_size))
    best_so_far = np.inf
    validation_results = []
    
    # Optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    policy_net.train()
    
    total_itr = 1

    for itr in tqdm(range(iterations)):
        no_nodes = train_node_n_min + int((total_itr)%(train_node_n_max - train_node_n_min + 1))
        # Prepare training data
        data = torch.rand(size=[train_batch_size, no_nodes, 2]).to(device)
        pi = policy_net(data)
        action, log_prob = action_sample(pi)
        reward = parallel_get_reward(
            routing_algorithm,
            action.detach().cpu().numpy(), 
            data.detach().cpu().numpy(), 
            no_agent, curvature, n_worker)  # reward: tensor [batch, 1]

        loss = torch.mul(torch.tensor(reward, device=device), log_prob.sum(dim=1)).sum()

        avg_reward = sum(reward) / train_batch_size
        writer.add_scalar('train/loss', loss / train_batch_size, total_itr)
        writer.add_scalar('train/reward', avg_reward, total_itr)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Run validation every 1000 iterations
        if (itr + 1) % 1000 == 0:
            validation_result = validate(routing_algorithm, validation_data, policy_net, no_agent, device, curvature, n_worker)
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), 
                    './saved_model/a{}_n{}to{}_c{}_{}_seed{}_exp{}_HyDR_TR.pth'.format(
                        str(no_agent), str(train_node_n_min), str(train_node_n_max),
                        str(curvature), routing_algorithm, str(seed), str(exp_num)))
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
    
    # HyDR-TR hyperparameters
    in_chnl = 2
    hid_chnl = 64
    key_size_embd = 64
    key_size_policy = 64
    val_size = 64
    clipping = 10
    n_heads = 8
    ff_dim = 4 * hid_chnl  # 256
    norm_eps = 1e-5
    batch_first = True
    n_enc_layer = 3

    # Experiment configuration
    seed = 1
    exp_num = 250307

    torch.manual_seed(seed)

    log_path = "log/a{}_n{}to{}_c{}_{}_seed{}_exp{}_HyDR_TR".format(n_agent, train_node_n_min, train_node_n_max, curvature, routing_algorithm, seed, exp_num)
    writer = SummaryWriter(log_path)

    model_path = "./saved_model/a{}_n{}to{}_c{}_{}_seed{}_exp{}_HyDR_TR.pth".format(n_agent, train_node_n_min, train_node_n_max, curvature, routing_algorithm, seed, exp_num)
    config_path = model_path.replace(".pth", ".yaml")

    # save all configuration as yaml in the log_path directory
    with open(config_path, 'w') as f:
        yaml.dump({'n_agent': n_agent, 'train_node_n_min': train_node_n_min, 'train_node_n_max': train_node_n_max, 'test_node_n': test_node_n, 'curvature': curvature, 'batch_size': train_batch_size,
                   'lr': lr,'n_worker': n_worker, 'seed': seed, 'validation_data_size': test_data_size, 'device': dev,
                   'in_chnl': in_chnl, 'hid_chnl': hid_chnl, 'key_size_embd': key_size_embd, 'key_size_policy': key_size_policy, 'val_size': val_size, 'clipping': clipping,'train_batch_size': train_batch_size, 'iterations':iterations,
                   'n_heads': n_heads, 'ff_dim': ff_dim, 'norm_eps': norm_eps, 'batch_first': batch_first, 'n_enc_layer': n_enc_layer}, f)

    policy = Policy_HyDR_TR(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent, key_size_embd=key_size_embd,
                    key_size_policy=key_size_policy, val_size=val_size, clipping=clipping, dev=dev,
                    n_heads=n_heads, ff_dim=ff_dim, norm_eps=norm_eps, 
                    batch_first=batch_first, n_enc_layer=n_enc_layer)

    best_results = train(iterations, routing_algorithm, train_batch_size, test_data_size, train_node_n_min, train_node_n_max, test_node_n, 
                         policy, lr,n_agent, curvature, dev, n_worker, writer, seed, exp_num)
    print(min(best_results))
