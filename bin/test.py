import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
"""
Although the graph has only two edges, we need to define four index tuples to account for both directions of a edge.

Note

You can print out your data object anytime and receive a short information about its attributes and their shapes.

Besides holding a number of node-level, edge-level or graph-level attributes, Data provides a number of useful utility functions, e.g.:
"""
print("keys", data.keys)


print(data['x'])

for key, item in data:
    print(f'{key} found in data')


print('edge_attr' in data)


print("num_nodes",data.num_nodes)


print("num_edges", data.num_edges)


print("num_node_features", data.num_node_features)


print("has_isolated_nodes", data.has_isolated_nodes())


print("has_self_loops", data.has_self_loops())


print("is_directed", data.is_directed())

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# >>> ENZYMES(600)
data = dataset[0]
print(data.keys)
print("len(dataset)", len(dataset))
# >>> 600

print("num_classes", dataset.num_classes)
# >>> 6

print("num_node_features", dataset.num_node_features)

# to shuffle the datasets
dataset = dataset.shuffle()


from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    data
    # >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    data.num_graphs
    # >>> 32

    x = scatter_mean(data.x, data.batch, dim=0)
    x.size()
    # >>> torch.Size([32, 21])
