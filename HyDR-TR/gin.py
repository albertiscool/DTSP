import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_max_pool 

class Net_origin(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl):
        super(Net_origin, self).__init__()

        ## init projection
        # 1st mlp layer
        self.lin1_1 = torch.nn.Linear(in_chnl, hid_chnl)

        ## GIN conv layers
        self.conv1 = GINConv(Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl)))
        self.conv2 = GINConv(Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl)))
        self.conv3 = GINConv(Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl)))

    def forward(self, x, edge_index):

        # init projection
        h = F.relu(self.lin1_1(x))

        # GIN conv
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))


        return h

class Net(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, dropout=0.5, learn_eps=True):
        super().__init__()
        self.dropout = dropout

        # 입력 projection
        self.lin1_1 = nn.Linear(in_chnl, hid_chnl)
        self.bn1_1  = nn.BatchNorm1d(hid_chnl)
        self.lin1_2 = nn.Linear(hid_chnl, hid_chnl)

        # GIN 3-layer
        mlp1 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv1 = GINConv(mlp1, train_eps=learn_eps, aggr='add')
        self.bn1   = nn.BatchNorm1d(hid_chnl)

        mlp2 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv2 = GINConv(mlp2, train_eps=learn_eps, aggr='add')
        self.bn2   = nn.BatchNorm1d(hid_chnl)

        mlp3 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv3 = GINConv(mlp3, train_eps=learn_eps, aggr='add')
        self.bn3   = nn.BatchNorm1d(hid_chnl)


    def forward(self, x, edge_index, batch):

        h = self.lin1_2(F.relu(self.bn1_1(self.lin1_1(x))))

        # GIN layer 1
        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GIN layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GIN layer 3
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)  # ← h3 (node embedding)


        g = global_max_pool(h, batch)

        return h, g
