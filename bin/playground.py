import os
import torch
import pickle
from torch_geometric.data import Data
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from data_processing import get_dataset_from_csv, get_cell_count_df
RAW_DATA_PATH = os.path.join("../data", "raw")

with open(os.path.join(RAW_DATA_PATH, '4_18_features.pickle'), 'rb') as handle:
    feature_arr = pickle.load(handle)

with open(os.path.join(RAW_DATA_PATH, '4_18_edge_index_length.pickle'), 'rb') as handle:
    edge_index_arr, edge_length_arr = pickle.load(handle)

with open(os.path.join(RAW_DATA_PATH, '4_18_coordinates.pickle'), 'rb') as handle:
    coordinates_arr = pickle.load(handle)

edge_index = torch.tensor(edge_index_arr, dtype=torch.long)

feature_arr = torch.tensor(feature_arr, dtype=torch.float)

df_dataset = get_dataset_from_csv()
df_cell_count = get_cell_count_df(700)

print(feature_arr)
print(edge_index_arr)
data = Data(x=feature_arr, edge_index=edge_index.t().contiguous())

print(data)

g = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g, pos=coordinates_arr)
plt.show()
