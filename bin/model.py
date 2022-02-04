import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import BatchNorm1d


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

