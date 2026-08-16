import torch


def data_gen(no_nodes, batch_size, flag):
    if flag == 'validation':
        torch.save(torch.rand(size=[batch_size, no_nodes, 2]), 'validation_data/validation_'+str(no_nodes)+'_'+str(batch_size))
    elif flag == 'test':
        torch.save(torch.rand(size=[batch_size, no_nodes, 2]), 'validation_data/test_'+str(no_nodes)+'_'+str(batch_size))
    else:
        print('flag should be "testing", or "validation".')


if __name__ == '__main__':
    n_nodes = 51
    b_size = 512
    flag = 'validation'
    torch.manual_seed(n_nodes+b_size)

    data_gen(n_nodes, b_size, flag)