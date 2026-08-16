from policy_HyDR_TR import Policy_HyDR_TR, action_sample, get_reward, parallel_get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def validate(routing_algorithm, instances, p_net, no_agent, device, curvature, n_worker):

    batch_size = instances.shape[0]
    instances = instances.to(device)

    # get pi
    pi = p_net(instances)

    # sample action and calculate log probs
    action, _ = action_sample(pi)

    # get reward for each batch
    reward = parallel_get_reward(routing_algorithm,
                                 action.detach().cpu().numpy(), 
                                 instances.detach().cpu().numpy(), 
                                 no_agent,curvature,n_worker)  # reward: tensor [batch, 1]
    # print('Validation result:', format(sum(reward)/batch_size, '.4f'))

    return sum(reward)/batch_size


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    n_agent = 5
    n_nodes = 50
    n_batch = 256

    data = torch.load('./validation_data_'+str(n_nodes)+'_'+str(n_batch))

    policy = Policy_HyDR_TR(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    path = './{}.pth'.format(str(n_nodes) + '_' + str(n_agent))
    policy.load_state_dict(torch.load(path))
    validate(data, policy, n_agent, dev)
