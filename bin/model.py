import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import BatchNorm1d
from torch.nn import ModuleList


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)
        x = x.relu()
        
        return x


class GCN2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, fcl1, drop_rate):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)

        self.dropout = drop_rate 
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, fcl1)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(fcl1)
        self.lin = Linear(fcl1, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
    

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.bn1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.bn2(self.lin1(x))
        x = x.relu()
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        x = x.relu()
        
        
        return x

# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/example.py
class GCN_NEW(torch.nn.Module):
    def __init__(self, num_node_features, num_of_gcn_layers, num_of_ff_layers, gcn_hidden_neurons, ff_hidden_neurons, dropout):
        super(GCN_NEW, self).__init__()
        torch.manual_seed(12345)

        self.dropout = dropout

        # GCN list
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        self.convs.append(GCNConv(num_node_features, gcn_hidden_neurons))
        self.batch_norms.append(BatchNorm1d(gcn_hidden_neurons))
        for _ in range(num_of_gcn_layers-1):
            self.convs.append(GCNConv(gcn_hidden_neurons, gcn_hidden_neurons))
            self.batch_norms.append(BatchNorm1d(gcn_hidden_neurons))

        # fully connected layer list
        self.ff_layers = ModuleList()
        self.ff_batch_norms = ModuleList()

        self.ff_layers.append(Linear(gcn_hidden_neurons, ff_hidden_neurons))
        self.ff_batch_norms.append(BatchNorm1d(ff_hidden_neurons))
        for _ in range(num_of_ff_layers-1):
            self.ff_layers.append(Linear(ff_hidden_neurons, ff_hidden_neurons))
            self.ff_batch_norms.append(BatchNorm1d(ff_hidden_neurons))

        self.lin = Linear(ff_hidden_neurons, 1)

    def forward(self, x, edge_index, batch):

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            # x = h + x  # residual#
            x = F.dropout(x, self.dropout, training=self.training)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]


        for ff_l, batch_norm in zip(self.ff_layers, self.ff_batch_norms):
            x = F.relu(batch_norm(ff_l(x)))
            # x = h + x  # residual#
            x = F.dropout(x, self.dropout, training=self.training)

    
        x = self.lin(x)
        x = x.relu()
        
        return x

