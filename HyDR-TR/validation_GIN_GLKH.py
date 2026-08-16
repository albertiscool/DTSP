from policy_GIN_GLKH import Policy_GIN_GLKH, action_sample, get_reward, parallel_get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def validate(routing_algorithm, instances, p_net, no_agent, device, curvature, n_worker):

    batch_size = instances.shape[0]
    adj = torch.ones([batch_size, instances.shape[1], instances.shape[1]])  # adjacent matrix

    # get batch graphs instances list
    instances_list = [Data(x=instances[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(batch_size)]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=instances_list).to(device)

    # get pi
    pi = p_net(batch_graph, n_nodes=instances.shape[1], n_batch=batch_size)

    # sample action and calculate log probs
    action, log_prob = action_sample(pi)

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

    policy = Policy_GIN_GLKH(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    path = './{}.pth'.format(str(n_nodes) + '_' + str(n_agent))
    policy.load_state_dict(torch.load(path))
    validate(data, policy, n_agent, dev)
