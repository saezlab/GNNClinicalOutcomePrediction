from tabnanny import check
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torch_sparse import coalesce
from torch_geometric import utils
import pickle5 as pickle
import os
import argparse
import pytorch_lightning as pl
from dataset import TissueDataset
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from model_dgermen import CustomGCN
from torch_geometric.utils import degree

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

parser = argparse.ArgumentParser(description='GNN Arguments')
parser.add_argument(
    '--bs',
    type=int,
    default=32,
    metavar='BS',
    help='batch size (default: 32)')

parser.add_argument(
    '--idx',
    type=int,
    default=15,
    metavar='IDX',
    help='Index Number of the graph (default: 14)')

parser.add_argument(
    '--exp_epoch',
    type=int,
    default=50,
    metavar='EXEPC',
    help='Number of epochs (default: 50)')

parser.add_argument(
    '--exp_lr',
    type=float,
    default=0.001,
    metavar='ELR',
    help='Explainer learning rate (default: 0.001)')

parser.add_argument(
    '--retun_type',
    type=str,
    default="regression",
    metavar='RT',
    help='return type of the explainer (default: regression)')

parser.add_argument(
    '--feat_mask_type',
    type=str,
    default="individual_feature",
    metavar='FMT',
    help='feature mask type of the explainer (default: individual_feature)')

parser.add_argument(
    '--relevant_edges',
    type=float,
    default=0.001,
    metavar='RE',
    help='Get the highest values inside the edge mask (relevant edges) (default: 0.001)')

parser.add_argument(
    '--model',
    type=str,
    default="PNAConv",
    metavar='mn',
    help='model name (default: PNAConv)')

parser.add_argument(
    '--num_of_gcn_layers',
    type=int,
    default=3,
    metavar='NGCNL',
    help='Number of GCN layers (default: 2)')

parser.add_argument(
    '--num_of_ff_layers',
    type=int,
    default=3,
    metavar='NFFL',
    help='Number of FF layers (default: 2)')

parser.add_argument(
    '--gcn_h',
    type=int,
    default=128,
    metavar='GCNH',
    help='GCN hidden channel (default: 128)')

parser.add_argument(
    '--fcl',
    type=int,
    default=128,
    metavar='FCL',
    help='Number of neurons in fully-connected layer (default: 128)')

parser.add_argument(
    '--dropout',
    type=float,
    default=0.20, 
    metavar='DO',
    help='dropout rate (default: 0.20)')

parser.add_argument(
    '--aggregators',
    nargs='+',
    type = str,
    default= ["sum","mean"], 
    metavar='AGR',
    help= "aggregator list for PNAConv")

parser.add_argument(
    '--scalers',
    nargs='+',
    type= str,
    default= ["amplification","identity"],
    metavar='SCL',
    help='Set of scaling function identifiers,')

S_PATH = os.path.dirname(__file__)
pl.seed_everything(42)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

device = "cpu"

if use_gpu:
    print("GPU is available on this device!")
    device = "cuda"
else:
    print("CPU is available on this device!")


dataset = TissueDataset(os.path.join(S_PATH,"../data"))

dataset = dataset.shuffle()

num_of_train = int(len(dataset)*0.80)
num_of_val = int(len(dataset)*0.10)

train_dataset = dataset[:num_of_train]
validation_dataset = dataset[num_of_train:num_of_train+num_of_val]
test_dataset = dataset[num_of_train+num_of_val:]

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

idx = args.idx
test_graph = test_dataset[idx]
coordinates_arr = None
with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
        coordinates_arr = pickle.load(handle)

deg = 1

if args.model == "PNAConv":
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

def load_checkpoint(filepath):
    
    model = CustomGCN(type = args.model,
                num_node_features = dataset.num_node_features,
                num_gcn_layers=args.num_of_gcn_layers, 
                num_ff_layers=args.num_of_ff_layers, 
                gcn_hidden_neurons=args.gcn_h, 
                ff_hidden_neurons=args.fcl, 
                dropout=args.dropout,
                aggregators=args.aggregators,
                scalers=args.scalers,
                deg = deg).to(device)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_checkpoint('../data/models/model.pth')


def plot_original_graph():

    pos_1 = coordinates_arr
    g = utils.to_networkx(test_graph, to_undirected=True)
    
    nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g,edge_color='b', pos=pos_1)

    return plt.savefig(f'../plots/original_graphs/original_graph_{args.idx}_{args.exp_epoch}_{args.exp_lr}_{args.retun_type}_{args.feat_mask_type}.png', dpi=1000)

def GNNExplainer():

    explainer = pyg.nn.GNNExplainer(model, epochs = args.exp_epoch, lr = args.exp_lr, 
                                    return_type = args.retun_type, feat_mask_type = args.feat_mask_type).to(device)
    
    result = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))
    (feature_mask, edge_mask) = result
    edges_idx = edge_mask > args.relevant_edges
    explanation = pyg.data.Data(test_graph.x, test_graph.edge_index[:, edges_idx])

    explanation = pyg.transforms.RemoveIsolatedNodes()(pyg.transforms.ToUndirected()(explanation))
    return explanation

explanation = GNNExplainer()

def plot_subgraph():
    options = ['r','b']
    colors = []

    graph_exp = utils.to_networkx(explanation, to_undirected=True)

    g = utils.to_networkx(test_graph, to_undirected=True)

    gexp_edges = graph_exp.edges

    for e_idx in g.edges:
        for g_exp in gexp_edges:
            if g_exp==e_idx:
                colors.append(options[0])
            else:
                colors.append(options[1])    
            
        
    pos_1 = coordinates_arr

    nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g,edge_color=colors, pos=pos_1)

    return plt.savefig(f'../plots/subgraphs/subgraph_{args.idx}_{args.exp_epoch}_{args.exp_lr}_{args.retun_type}_{args.feat_mask_type}.png', dpi=1000)


plot_original_graph()
plot_subgraph()

##########################################################################
#                             GNNExplainer                               #
##########################################################################
"""
import torch

import torch_geometric as ptgeom
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torch_sparse import coalesce


idx = args.idx
test_graph = test_dataset[idx]
coordinates_arr = None
with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
        coordinates_arr = pickle.load(handle)


def plot_original_graph():

    pos_1 = coordinates_arr
    g = utils.to_networkx(test_graph, to_undirected=True)
    
    nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g,edge_color='b', pos=pos_1)

    return plt.savefig(f'../plots/original_graphs/original_graph_{args.idx}_{args.exp_epoch}_{args.exp_lr}_{args.retun_type}_{args.feat_mask_type}.png', dpi=1000)

def GNNExplainer():

    explainer = ptgeom.nn.GNNExplainer(model, epochs = args.exp_epoch, lr = args.exp_lr, 
                                    return_type = args.retun_type, feat_mask_type = args.feat_mask_type).to(device)
    
    result = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))
    (feature_mask, edge_mask) = result
    edges_idx = edge_mask > args.relevant_edges
    explanation = ptgeom.data.Data(test_graph.x, test_graph.edge_index[:, edges_idx])

    explanation = ptgeom.transforms.RemoveIsolatedNodes()(ptgeom.transforms.ToUndirected()(explanation))
    return explanation

explanation = GNNExplainer()


def plot_subgraph():
    options = ['r','b']
    colors = []

    graph_exp = utils.to_networkx(explanation, to_undirected=True)

    g = utils.to_networkx(test_graph, to_undirected=True)

    gexp_edges = graph_exp.edges

    for e_idx in g.edges:
        for g_exp in gexp_edges:
            if g_exp==e_idx:
                colors.append(options[0])
            else:
                colors.append(options[1])    
            
        
    pos_1 = coordinates_arr

    nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
    nx.draw_networkx_edges(g,edge_color=colors, pos=pos_1)

    return plt.savefig(f'../plots/subgraphs/subgraph_{args.idx}_{args.exp_epoch}_{args.exp_lr}_{args.retun_type}_{args.feat_mask_type}.png', dpi=1000)


plot_original_graph()
plot_subgraph()
"""

############################################################################################################
# ALTERNATIVE SOLUTION FOR MAKING UNDIRECTED GRAPH
# REF: https://colab.research.google.com/github/VisiumCH/AMLD-2021-Graphs/blob/master/notebooks/workshop_notebook.ipynb#scrollTo=AHcOi2Aki9Tp
"""
def is_undirected(graph):
  edge_index, num_nodes = graph.edge_index, graph.num_nodes

  edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

  sources, targets = edge_index
  sources_and_targets = torch.cat([sources, targets], dim=0)
  targets_and_sources = torch.cat([targets, sources], dim=0)

  undirected_edge_index = torch.stack([sources_and_targets, targets_and_sources], dim=0)
  undirected_edge_index, _ = coalesce(undirected_edge_index, None, num_nodes, num_nodes)

  return edge_index.size(1) == undirected_edge_index.size(1)

print("Your function works correctly on G1: ", test_graph.is_undirected() == is_undirected(test_graph))
#g = is_undirected(test_graph)
"""

##################################################################################################################

##############################################################################################
##                                          OLD VERSION :D                                  ##     
"""  
pos_1 = coordinates_arr

g = utils.to_networkx(test_graph, to_undirected=True)

nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
nx.draw_networkx_edges(g,edge_color='b', pos=pos_1)

plt.savefig("original_graph.png", dpi=1000)

# PyG has an implementation of GNNExplainer that we will use
# feat_mask_type provides a mask that can remove individual features for each node
explainer = ptgeom.nn.GNNExplainer(model, epochs = args.exp_epoch, lr = args.exp_lr, 
                                    return_type = args.retun_type, feat_mask_type = args.feat_mask_type).to(device)


# Run a gradient descent loop to optimize the masks
result = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))


# These are the resulting masks, we mostly care about the edges here
(feature_mask, edge_mask) = result

# Get the highest values inside the edge mask (relevant edges)
edges_idx = edge_mask > 0.000

# We build the subgraph corresponding to the explanation
explanation = ptgeom.data.Data(test_graph.x, test_graph.edge_index[:, edges_idx])

# Make undirected and remove the isolated nodes 
explanation = ptgeom.transforms.RemoveIsolatedNodes()(ptgeom.transforms.ToUndirected()(explanation))

options = ['r','b']
colors = []

graph_exp = utils.to_networkx(explanation, to_undirected=True)

g = utils.to_networkx(test_graph, to_undirected=True)

#nx.draw(g, pos=coordinates_arr, node_size=0,  with_labels=False)
gexp_edges = graph_exp.edges

for e_idx in g.edges:
    for g_exp in gexp_edges:
        if g_exp==e_idx:
            colors.append(options[0])
            #print('yes')
        else: 
            colors.append(options[1])
    
pos_1 = coordinates_arr

nx.draw_networkx_nodes(g,pos=pos_1, node_size=1)
nx.draw_networkx_edges(g,edge_color=colors, pos=pos_1)

plt.savefig("subgraph.png", dpi=1000)
"""
##############################################################################################